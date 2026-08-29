#!/usr/bin/env python3
"""Run Stage 8 finite-line crossing with Track-ID-based unique counts."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.spatial.line_crossing import (
    CROSSING_FIELDS,
    CrossingRecord,
    LineCrossingEngine,
    load_scene_config,
)
from vision_analytics.tracking.tracker import StatefulByteTracker, draw_tracks
from vision_analytics.tracking.trajectory import (
    TrajectoryEngine,
    draw_trajectory_trails,
)
from vision_analytics.video.pipeline import add_overlay, process_video

MODEL_NAME = "yolo26n.pt"
MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / MODEL_NAME
DEVICE = "mps"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
TRACKER_CONFIG = "bytetrack.yaml"
MAX_HISTORY_LENGTH = 30
MINIMUM_TRAJECTORY_DISPLACEMENT_PIXELS = 5.0
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.csv"
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes.yaml"
STAGE7_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage7_trajectory_benchmark.csv"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage8"
BENCHMARK_PATH = ANALYTICS_DIR / "stage8_line_crossing_benchmark.csv"

STAGE8_BENCHMARK_FIELDS = (
    "video_id",
    "source_id",
    "model",
    "device",
    "imgsz",
    "confidence_threshold",
    "tracker",
    "maximum_frame_gap",
    "minimum_movement_pixels",
    "input_width",
    "input_height",
    "source_fps",
    "expected_frame_count",
    "frames_processed",
    "elapsed_seconds",
    "processing_fps",
    "stage7_processing_fps",
    "fps_ratio_vs_stage7",
    "line_crossing_count",
    "output_path",
    "status",
    "validation_message",
)


def load_runtime_sources() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def load_stage7_benchmark() -> dict[str, dict[str, float]]:
    if not STAGE7_BENCHMARK.is_file():
        return {}
    with STAGE7_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "input_width": float(row["input_width"]),
                "input_height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def draw_counting_lines(frame: object, engine: LineCrossingEngine) -> None:
    """Draw configured lines and live deduplicated counts without event labels."""
    for line in engine.lines:
        start, end = engine.pixel_line(line.line_id)
        start_pixel = (round(start[0]), round(start[1]))
        end_pixel = (round(end[0]), round(end[1]))
        cv2.line(frame, start_pixel, end_pixel, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.line(frame, start_pixel, end_pixel, (0, 255, 255), 3, cv2.LINE_AA)
        label = f"{line.line_id} count:{engine.count_for_line(line.line_id)}"
        label_position = (start_pixel[0] + 8, max(24, start_pixel[1] - 10))
        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def write_crossing_outputs(
    records: list[CrossingRecord],
    *,
    csv_path: Path,
    summary_path: Path,
    benchmark: dict[str, object],
    engine: LineCrossingEngine,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CROSSING_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)

    grouped = Counter(
        (record.line_id, record.class_name, record.crossing_direction)
        for record in records
    )
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "maximum_frame_gap": engine.maximum_frame_gap,
        "minimum_movement_pixels": engine.minimum_movement_pixels,
        "total_line_crossing_count": len(records),
        "counts": [
            {
                "line_id": line_id,
                "class_name": class_name,
                "crossing_direction": direction,
                "line_crossing_count": count,
            }
            for (line_id, class_name, direction), count in sorted(grouped.items())
        ],
        "deduplication_key": "(video_id, line_id, track_id)",
        "counting_warning": (
            "Counts are Track-ID-based finite-line crossings, not perfect Ground Truth "
            "traffic counts; detection misses and tracking fragmentation affect results."
        ),
        "status": benchmark["status"],
        "validation_message": benchmark["validation_message"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    print(
        f"MPS built={torch.backends.mps.is_built()} "
        f"available={torch.backends.mps.is_available()}"
    )
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 8 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_file():
        print(f"Model weights not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    scene_lines, maximum_frame_gap, minimum_movement_pixels = load_scene_config(
        SCENE_CONFIG
    )
    stage7 = load_stage7_benchmark()
    benchmark_rows: list[dict[str, object]] = []
    for source in load_runtime_sources():
        source_id = source["source_id"]
        if source_id not in scene_lines or source_id not in stage7:
            print(f"Missing Stage 8 config or Stage 7 metadata for {source_id}", file=sys.stderr)
            return 1
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage8") / f"{video_id}_stage8.mp4"
        crossing_csv = ANALYTICS_DIR / f"{video_id}_crossings.csv"
        summary_json = ANALYTICS_DIR / f"{video_id}_summary.json"

        tracker = StatefulByteTracker(
            MODEL_PATH,
            device=DEVICE,
            imgsz=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            tracker_config=TRACKER_CONFIG,
        )
        trajectory_engine = TrajectoryEngine(
            max_history_length=MAX_HISTORY_LENGTH,
            minimum_displacement=MINIMUM_TRAJECTORY_DISPLACEMENT_PIXELS,
        )
        line_engine = LineCrossingEngine(
            scene_lines[source_id],
            frame_width=round(stage7[source_id]["input_width"]),
            frame_height=round(stage7[source_id]["input_height"]),
            maximum_frame_gap=maximum_frame_gap,
            minimum_movement_pixels=minimum_movement_pixels,
        )

        def update_and_draw(
            frame: object,
            frame_index: int,
            timestamp_seconds: float,
            source_fps: float,
        ) -> None:
            frame_records = tracker.track_frame(
                frame,
                video_id=video_id,
                source_id=source_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            trajectory_observations = trajectory_engine.update(frame_records)
            line_engine.update(trajectory_observations)
            draw_trajectory_trails(frame, frame_records, trajectory_engine)
            draw_tracks(frame, frame_records)
            draw_counting_lines(frame, line_engine)
            add_overlay(
                frame,
                video_id=video_id,
                frame_index=frame_index,
                source_fps=source_fps,
            )

        print(f"Counting line crossings for {source_id}...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative,
            PROJECT_ROOT / output_relative,
            video_id=video_id,
            source_id=source_id,
            frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_crossing_outputs(
            line_engine.records,
            csv_path=crossing_csv,
            summary_path=summary_json,
            benchmark=benchmark,
            engine=line_engine,
        )

        prior_fps = stage7[source_id]["processing_fps"]
        processing_fps = float(benchmark["processing_fps"])
        benchmark_rows.append(
            {
                "video_id": video_id,
                "source_id": source_id,
                "model": MODEL_NAME,
                "device": DEVICE,
                "imgsz": IMAGE_SIZE,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "tracker": TRACKER_CONFIG,
                "maximum_frame_gap": maximum_frame_gap,
                "minimum_movement_pixels": minimum_movement_pixels,
                "input_width": benchmark["input_width"],
                "input_height": benchmark["input_height"],
                "source_fps": benchmark["source_fps"],
                "expected_frame_count": benchmark["expected_frame_count"],
                "frames_processed": benchmark["frames_processed"],
                "elapsed_seconds": benchmark["elapsed_seconds"],
                "processing_fps": processing_fps,
                "stage7_processing_fps": prior_fps,
                "fps_ratio_vs_stage7": (
                    round(processing_fps / prior_fps, 4) if prior_fps > 0 else 0.0
                ),
                "line_crossing_count": len(line_engine.records),
                "output_path": output_relative.as_posix(),
                "status": benchmark["status"],
                "validation_message": benchmark["validation_message"],
            }
        )
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, "
            f"{len(line_engine.records)} deduplicated crossings",
            flush=True,
        )

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=STAGE8_BENCHMARK_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(benchmark_rows)

    failures = sum(row["status"] == "FAIL" for row in benchmark_rows)
    warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_PATH}")
    print(
        f"Summary: {len(benchmark_rows) - failures - warnings} PASS, "
        f"{warnings} WARNING, {failures} FAIL"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
