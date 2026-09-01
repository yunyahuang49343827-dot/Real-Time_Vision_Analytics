#!/usr/bin/env python3
"""Run the controlled Stage 21.1A aerial small-object runtime diagnostic."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.detector import (  # noqa: E402
    TARGET_CLASS_NAMES,
    draw_detections,
    filter_detection_candidates,
)
from vision_analytics.detection.schema import (  # noqa: E402
    DETECTION_FIELDS,
    DetectionRecord,
)
from vision_analytics.evaluation.aerial_runtime import (  # noqa: E402
    DiagnosticConfig,
    Experiment,
    aggregate_runtime_metrics,
    experiment_output_directory,
    load_diagnostic_config,
    normalized_bbox_area,
    render_comparison_report,
    validate_runtime_assets,
)
from vision_analytics.tracking.schema import TRACK_FIELDS  # noqa: E402
from vision_analytics.tracking.tracker import build_track_records, draw_tracks  # noqa: E402
from vision_analytics.tracking.trajectory import (  # noqa: E402
    TRAJECTORY_FIELDS,
    TrajectoryEngine,
    draw_trajectory_trails,
)
from vision_analytics.video.pipeline import add_overlay  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage21_1a_aerial_diagnostic.yaml"
DETECTION_DIAGNOSTIC_FIELDS = DETECTION_FIELDS + (
    "normalized_bbox_area",
    "small_bbox",
)
COMPARISON_FIELDS = (
    "experiment",
    "imgsz",
    "conf",
    "total_frames",
    "processing_seconds",
    "processing_fps",
    "total_detection_observations",
    "small_vehicle_detection_observations",
    "low_confidence_015_025_observations",
    "tracking_observations",
    "diagnostic_unique_track_ids",
    "trajectory_observations",
    "frame_gap_fragmentation_candidates",
    "status",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _video_metadata(config: DiagnosticConfig) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(config.input_video))
    if not capture.isOpened():
        raise RuntimeError("aerial diagnostic input cannot be opened")
    metadata = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    if metadata["width"] <= 0 or metadata["height"] <= 0 or metadata["fps"] <= 0:
        raise RuntimeError("aerial diagnostic input has invalid video metadata")
    if config.end_frame >= metadata["frame_count"]:
        raise ValueError("configured frame range exceeds the input video")
    return metadata


def _diagnostic_detection_rows(
    records: list[DetectionRecord],
    *,
    width: int,
    height: int,
    small_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = record.to_row()
        area = normalized_bbox_area(
            record.bbox.x1,
            record.bbox.y1,
            record.bbox.x2,
            record.bbox.y2,
            frame_width=width,
            frame_height=height,
        )
        row["normalized_bbox_area"] = round(area, 8)
        row["small_bbox"] = area < small_threshold
        rows.append(row)
    return rows


def _comparison_frames(config: DiagnosticConfig, fps: float) -> dict[int, float]:
    result: dict[int, float] = {}
    for relative_seconds in config.comparison_relative_seconds:
        absolute_frame = config.start_frame + int(round(relative_seconds * fps))
        if absolute_frame <= config.end_frame:
            result[absolute_frame] = relative_seconds
    if len(result) != len(config.comparison_relative_seconds):
        raise ValueError("comparison timestamps must map to distinct frames inside the range")
    return result


def _run_experiment(
    config: DiagnosticConfig,
    experiment: Experiment,
    metadata: dict[str, float | int],
    governance: dict[str, object],
) -> dict[str, object]:
    output = experiment_output_directory(config.output_directory, experiment.name)
    output.mkdir(parents=True, exist_ok=True)
    frames_directory = output / "comparison_frames"
    frames_directory.mkdir(parents=True, exist_ok=True)
    video_path = output / "processed.mp4"

    model_started = time.perf_counter()
    model = YOLO(str(config.runtime_model))
    model_load_seconds = time.perf_counter() - model_started
    class_names = {int(class_id): name for class_id, name in model.names.items()}
    target_class_ids = sorted(
        class_id for class_id, name in class_names.items() if name in TARGET_CLASS_NAMES
    )
    if {class_names[class_id] for class_id in target_class_ids} != TARGET_CLASS_NAMES:
        raise RuntimeError("runtime model taxonomy is missing a target class")

    capture = cv2.VideoCapture(str(config.input_video))
    if not capture.isOpened():
        raise RuntimeError("aerial diagnostic input cannot be reopened")
    capture.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(metadata["fps"]),
        (int(metadata["width"]), int(metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("diagnostic VideoWriter could not be opened")

    trajectory = TrajectoryEngine(
        max_history_length=config.max_history_length,
        minimum_displacement=config.minimum_displacement_pixels,
    )
    comparison_frames = _comparison_frames(config, float(metadata["fps"]))
    detection_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    frames_processed = 0
    started = time.perf_counter()
    try:
        for absolute_frame in range(config.start_frame, config.end_frame + 1):
            decoded, frame = capture.read()
            if not decoded:
                break
            timestamp_seconds = absolute_frame / float(metadata["fps"])
            result = model.track(
                source=frame,
                persist=True,
                tracker=config.tracker,
                device=config.device,
                imgsz=experiment.imgsz,
                conf=experiment.conf,
                classes=target_class_ids,
                save=False,
                verbose=False,
            )[0]
            boxes = result.boxes
            detections: list[DetectionRecord] = []
            tracks = []
            if boxes is not None and len(boxes) > 0:
                coordinates = boxes.xyxy.detach().cpu().tolist()
                class_ids = boxes.cls.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                candidates = filter_detection_candidates(
                    coordinates,
                    class_ids,
                    confidences,
                    class_names,
                    confidence_threshold=experiment.conf,
                )
                detections = [
                    DetectionRecord(
                        video_id=config.video_id,
                        source_id=config.source_id,
                        frame_index=absolute_frame,
                        timestamp_seconds=timestamp_seconds,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                    )
                    for class_id, class_name, confidence, bbox in candidates
                ]
                if boxes.id is not None:
                    tracks = build_track_records(
                        coordinates,
                        boxes.id.detach().cpu().tolist(),
                        class_ids,
                        confidences,
                        class_names,
                        video_id=config.video_id,
                        source_id=config.source_id,
                        frame_index=absolute_frame,
                        timestamp_seconds=timestamp_seconds,
                    )

            detection_rows.extend(_diagnostic_detection_rows(
                detections,
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                small_threshold=config.small_bbox_area_threshold,
            ))
            track_rows.extend(record.to_row() for record in tracks)
            frame_trajectories = trajectory.update(tracks)
            trajectory_rows.extend(item.to_row() for item in frame_trajectories)

            if tracks:
                draw_trajectory_trails(frame, tracks, trajectory)
                draw_tracks(frame, tracks)
            else:
                draw_detections(frame, detections)
            add_overlay(
                frame,
                video_id=f"{config.video_id}:{experiment.name}",
                frame_index=absolute_frame,
                source_fps=float(metadata["fps"]),
            )
            if absolute_frame in comparison_frames:
                relative_seconds = comparison_frames[absolute_frame]
                frame_path = frames_directory / f"relative_{relative_seconds:05.1f}s_frame_{absolute_frame:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"comparison frame could not be written: {frame_path}")
            writer.write(frame)
            frames_processed += 1
    finally:
        capture.release()
        writer.release()
    elapsed = time.perf_counter() - started

    expected_frames = config.end_frame - config.start_frame + 1
    failures: list[str] = []
    if frames_processed != expected_frames:
        failures.append(f"processed {frames_processed} frames; expected {expected_frames}")
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        failures.append("processed video is missing or empty")
    video_check = cv2.VideoCapture(str(video_path))
    if not video_check.isOpened():
        failures.append("processed video cannot be decoded")
    else:
        output_frames = int(round(video_check.get(cv2.CAP_PROP_FRAME_COUNT)))
        output_size = (
            int(round(video_check.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(video_check.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        if output_frames != frames_processed:
            failures.append(f"output frame count mismatch: {output_frames} != {frames_processed}")
        if output_size != (int(metadata["width"]), int(metadata["height"])):
            failures.append("output resolution differs from input")
    video_check.release()

    metrics = aggregate_runtime_metrics(
        detection_rows,
        track_rows,
        total_frames=frames_processed,
        processing_seconds=elapsed,
        small_area_threshold=config.small_bbox_area_threshold,
        low_confidence_band=config.low_confidence_band,
    )
    metrics.update({
        "experiment": experiment.name,
        "model": governance["runtime_model"],
        "model_sha256": governance["runtime_model_sha256"],
        "device": config.device,
        "imgsz": experiment.imgsz,
        "conf": experiment.conf,
        "tracker": config.tracker,
        "trajectory_observations": len(trajectory_rows),
        "model_load_seconds_excluded_from_processing_fps": round(model_load_seconds, 6),
        "frame_range_start": config.start_frame,
        "frame_range_end": config.end_frame,
        "source_width": metadata["width"],
        "source_height": metadata["height"],
        "source_fps": round(float(metadata["fps"]), 6),
        "processed_video": video_path.relative_to(config.project_root).as_posix(),
        "status": "FAIL" if failures else "PASS",
        "validation_message": "; ".join(failures) if failures else "All diagnostic outputs validated",
        "measurement_warning": (
            "Detection observations are per-frame occurrences; Track IDs are diagnostic and "
            "are not formal unique-object or traffic counts. No detection GT is available."
        ),
    })
    _write_csv(output / "detections.csv", DETECTION_DIAGNOSTIC_FIELDS, detection_rows)
    _write_csv(output / "tracks.csv", TRACK_FIELDS, track_rows)
    _write_csv(output / "trajectories.csv", TRAJECTORY_FIELDS, trajectory_rows)
    (output / "runtime_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps({
            "experiment": experiment.name,
            "config": {
                "model": governance["runtime_model"],
                "model_sha256": governance["runtime_model_sha256"],
                "device": config.device,
                "imgsz": experiment.imgsz,
                "conf": experiment.conf,
                "tracker": config.tracker,
                "trajectory_max_history_length": config.max_history_length,
                "trajectory_minimum_displacement_pixels": config.minimum_displacement_pixels,
            },
            "metrics": metrics,
            "rejected_model_used": 0,
            "training_runs": 0,
            "manual_visual_review": "REQUIRED",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _write_comparison_mosaics(config: DiagnosticConfig, metadata: dict[str, float | int]) -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for absolute_frame, relative_seconds in _comparison_frames(config, float(metadata["fps"])).items():
        images = []
        source_paths = []
        for experiment in config.experiments:
            path = (
                experiment_output_directory(config.output_directory, experiment.name)
                / "comparison_frames"
                / f"relative_{relative_seconds:05.1f}s_frame_{absolute_frame:06d}.jpg"
            )
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"comparison source frame is unreadable: {path}")
            banner_height = 48
            cv2.rectangle(image, (0, 0), (image.shape[1], banner_height), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"{experiment.name}  imgsz={experiment.imgsz} conf={experiment.conf:.2f}",
                (16, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            images.append(image)
            source_paths.append(path.relative_to(config.project_root).as_posix())
        mosaic = cv2.vconcat((cv2.hconcat(images[:2]), cv2.hconcat(images[2:])))
        destination = config.output_directory / f"comparison_relative_{relative_seconds:05.1f}s_frame_{absolute_frame:06d}.jpg"
        if not cv2.imwrite(str(destination), mosaic):
            raise RuntimeError(f"comparison mosaic could not be written: {destination}")
        manifests.append({
            "relative_seconds": relative_seconds,
            "absolute_frame": absolute_frame,
            "mosaic_path": destination.relative_to(config.project_root).as_posix(),
            "source_frames": source_paths,
        })
    return manifests


def _append_detailed_report(
    base: str,
    summaries: list[dict[str, object]],
    mosaics: list[dict[str, object]],
) -> str:
    lines = [base, "## Per-class detection observations", ""]
    classes = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
    lines.extend((
        "| Config | " + " | ".join(classes) + " |",
        "|---|" + "---:|" * len(classes),
    ))
    for summary in summaries:
        counts = summary["per_class_detection_observations"]
        lines.append(
            f"| {summary['experiment']} | "
            + " | ".join(str(counts.get(name, 0)) for name in classes)
            + " |"
        )
    vehicle_classes = ("car", "motorcycle", "bus", "truck")
    lines.extend(("", "## Small-vehicle observations (`normalized bbox area < 0.01`)", ""))
    lines.extend((
        "| Config | " + " | ".join(vehicle_classes) + " | Total |",
        "|---|" + "---:|" * (len(vehicle_classes) + 1),
    ))
    for summary in summaries:
        counts = summary["small_vehicle_per_class_observations"]
        lines.append(
            f"| {summary['experiment']} | "
            + " | ".join(str(counts.get(name, 0)) for name in vehicle_classes)
            + f" | {summary['small_vehicle_detection_observations']} |"
        )
    lines.extend(("", "## Confidence diagnostics", ""))
    for summary in summaries:
        distribution = summary["confidence_distribution"]
        small_distribution = summary["small_vehicle_confidence_distribution"]
        lines.append(
            f"- `{summary['experiment']}`: overall mean/median "
            f"{distribution['mean']:.4f}/{distribution['median']:.4f}; small-vehicle mean/median "
            f"{small_distribution['mean']:.4f}/{small_distribution['median']:.4f}; "
            f"observations in [0.15, 0.25): {summary['low_confidence_015_025_observations']}."
        )
    baseline = summaries[0]
    lines.extend(("", "## Automated findings", ""))
    for summary in summaries[1:]:
        lines.append(
            f"- `{summary['experiment']}` vs baseline: detection observations "
            f"{int(summary['total_detection_observations']) - int(baseline['total_detection_observations']):+d}, "
            f"small-vehicle observations "
            f"{int(summary['small_vehicle_detection_observations']) - int(baseline['small_vehicle_detection_observations']):+d}, "
            f"diagnostic Track IDs "
            f"{int(summary['diagnostic_unique_track_ids']) - int(baseline['diagnostic_unique_track_ids']):+d}, "
            f"FPS {float(summary['processing_fps']) - float(baseline['processing_fps']):+.3f}."
        )
    lines.extend((
        "",
        "These deltas measure emitted observations, not correctness. Added low-confidence boxes may be true small vehicles or false positives; visual review is required.",
    ))
    lines.extend(("", "## Tracking and trajectory diagnostics", ""))
    lines.extend((
        "| Config | Tracking obs | Diagnostic IDs | Frame-gap candidates | Trajectory obs |",
        "|---|---:|---:|---:|---:|",
    ))
    for summary in summaries:
        lines.append(
            f"| {summary['experiment']} | {summary['tracking_observations']} | "
            f"{summary['diagnostic_unique_track_ids']} | "
            f"{summary['frame_gap_fragmentation_candidates']} | "
            f"{summary['trajectory_observations']} |"
        )
    lines.extend((
        "",
        "A frame-gap candidate is one gap greater than one frame within an observed Track ID. It can indicate intermittent detection but is not verified physical-object fragmentation or an ID switch.",
        "",
        "## Sampled visual observations (three synchronized stills)",
        "",
        "- At source frame 310, both 960 configurations recover additional edge/bottom and construction-side vehicles that are visibly present but unboxed at 640. Lowering confidence at 640 changes less in this still.",
        "- At source frame 370, low-confidence 640 adds at least one visible blue vehicle missed by the 640 baseline; 960 configurations cover more of the central/right traffic. The yellow commercial van also illustrates car/truck taxonomy ambiguity across configurations.",
        "- At source frame 430, 960 configurations again cover additional central/edge vehicles. Several visually obvious scooters/motorcycles in the upper intersection remain unboxed across all four configurations, consistent with zero motorcycle observations in the aggregate output.",
        "- The three stills do not show an unequivocal false positive. Low-confidence partial/edge detections and parked/construction-side vehicles still require full-video review before any runtime choice.",
        "",
        "This is a limited sampled-frame review, not Ground Truth annotation or a complete visual audit of all 180 frames.",
    ))
    lines.extend((
        "",
        "## Synchronized comparison frames",
        "",
    ))
    for item in mosaics:
        lines.append(
            f"- Relative t={item['relative_seconds']:.1f}s / source frame {item['absolute_frame']}: "
            f"`{item['mosaic_path']}`"
        )
    lines.extend((
        "",
        "## Limitations",
        "",
        "- No detection or MOT Ground Truth was created; visually obvious misses, false positives, fragmentation, and trajectory usefulness require manual review.",
        "- FPS is an end-to-end local runtime diagnostic for decode + YOLO/ByteTrack + trajectory/overlay + MP4 writing. Model-load time is recorded separately and excluded, matching earlier runtime baselines.",
        "- `small bbox area < 0.01` is a descriptive image-space heuristic, not a physical-size category or accuracy metric.",
        "- Detection observations are Ultralytics tracking-output box occurrences. In these runs every returned box had a Track ID, so detection and tracking observation totals are equal; this does not make Track IDs formal object counts.",
        "- No configuration is automatically selected or promoted.",
        "",
        "`MANUAL_VISUAL_REVIEW_REQUIRED`",
        "",
    ))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_diagnostic_config(args.config, PROJECT_ROOT)
    governance = validate_runtime_assets(config)
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 21.1A requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1
    metadata = _video_metadata(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Input={config.input_video.relative_to(PROJECT_ROOT)} "
        f"frames={config.start_frame}-{config.end_frame} model_sha256={governance['runtime_model_sha256']}",
        flush=True,
    )
    summaries: list[dict[str, object]] = []
    for experiment in config.experiments:
        print(
            f"Running {experiment.name}: imgsz={experiment.imgsz} conf={experiment.conf:.2f}",
            flush=True,
        )
        summary = _run_experiment(config, experiment, metadata, governance)
        summaries.append(summary)
        print(
            f"  {summary['status']}: {summary['total_frames']} frames, "
            f"{summary['processing_seconds']:.3f}s, {summary['processing_fps']:.3f} FPS, "
            f"{summary['total_detection_observations']} detection observations, "
            f"{summary['diagnostic_unique_track_ids']} diagnostic Track IDs",
            flush=True,
        )
        gc.collect()
        torch.mps.empty_cache()

    mosaics = _write_comparison_mosaics(config, metadata)
    _write_csv(
        config.output_directory / "experiment_comparison.csv",
        COMPARISON_FIELDS,
        [{field: summary[field] for field in COMPARISON_FIELDS} for summary in summaries],
    )
    manifest = {
        "stage": "stage21_1a",
        "input_video": config.input_video.relative_to(PROJECT_ROOT).as_posix(),
        "video_id": config.video_id,
        "source_id": config.source_id,
        "frame_range": {"start": config.start_frame, "end": config.end_frame, "inclusive": True},
        "runtime_model": governance,
        "fixed_config": {
            "device": config.device,
            "tracker": config.tracker,
            "scene_config": config.scene_config.relative_to(PROJECT_ROOT).as_posix(),
            "trajectory_max_history_length": config.max_history_length,
            "trajectory_minimum_displacement_pixels": config.minimum_displacement_pixels,
        },
        "experiments": [
            {"name": item.name, "imgsz": item.imgsz, "conf": item.conf}
            for item in config.experiments
        ],
        "comparison_frames": mosaics,
        "formal_accuracy_metrics_computed": False,
        "manual_visual_review": "REQUIRED",
    }
    (config.output_directory / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    report = render_comparison_report(
        config, summaries, str(governance["runtime_model_sha256"]),
    )
    report = _append_detailed_report(report, summaries, mosaics)
    report_path = PROJECT_ROOT / "docs/stage21_1a_aerial_runtime_diagnostic.md"
    report_path.write_text(report, encoding="utf-8")
    failures = sum(summary["status"] == "FAIL" for summary in summaries)
    print(f"Wrote comparison artifacts to {config.output_directory}")
    print(f"Wrote report to {report_path}")
    print("MANUAL_VISUAL_REVIEW_REQUIRED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
