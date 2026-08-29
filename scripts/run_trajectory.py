#!/usr/bin/env python3
"""Run Stage 7 tracking with bounded image-space trajectory histories."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.tracking.tracker import StatefulByteTracker, draw_tracks
from vision_analytics.tracking.trajectory import (
    TRAJECTORY_FIELDS,
    TrajectoryEngine,
    TrajectoryObservation,
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
MINIMUM_DISPLACEMENT_PIXELS = 5.0
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.csv"
STAGE6_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage6_tracking_benchmark.csv"
TRAJECTORY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tracks" / "stage7"
BENCHMARK_PATH = PROJECT_ROOT / "outputs" / "analytics" / "stage7_trajectory_benchmark.csv"

STAGE7_BENCHMARK_FIELDS = (
    "video_id",
    "source_id",
    "model",
    "device",
    "imgsz",
    "confidence_threshold",
    "tracker",
    "max_history_length",
    "minimum_displacement_pixels",
    "input_width",
    "input_height",
    "source_fps",
    "expected_frame_count",
    "frames_processed",
    "elapsed_seconds",
    "processing_fps",
    "stage6_processing_fps",
    "fps_ratio_vs_stage6",
    "trajectory_observations",
    "track_ids_observed",
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


def load_stage6_fps() -> dict[str, float]:
    if not STAGE6_BENCHMARK.is_file():
        return {}
    with STAGE6_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: float(row["processing_fps"])
            for row in csv.DictReader(handle)
        }


def write_trajectory_outputs(
    observations: list[TrajectoryObservation],
    *,
    csv_path: Path,
    summary_path: Path,
    benchmark: dict[str, object],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TRAJECTORY_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(observation.to_row() for observation in observations)

    observed_ids = {observation.track_id for observation in observations}
    direction_counts = Counter(observation.direction for observation in observations)
    frame_gap_counts = Counter(observation.frame_gap for observation in observations)
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "model": MODEL_NAME,
        "device": DEVICE,
        "imgsz": IMAGE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "tracker": TRACKER_CONFIG,
        "frames_processed": benchmark["frames_processed"],
        "max_history_length": MAX_HISTORY_LENGTH,
        "minimum_displacement_pixels": MINIMUM_DISPLACEMENT_PIXELS,
        "trajectory_observations": len(observations),
        "track_ids_observed": len(observed_ids),
        "direction_observations": dict(sorted(direction_counts.items())),
        "frame_gap_observations": {
            str(frame_gap): count for frame_gap, count in sorted(frame_gap_counts.items())
        },
        "measurement_warning": (
            "All displacement values are image-space pixels, not meters, speed, "
            "or real-world distance. Directions are movement classifications, not events."
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
        print("Stage 7 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_file():
        print(f"Model weights not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    stage6_fps = load_stage6_fps()
    benchmark_rows: list[dict[str, object]] = []
    for source in load_runtime_sources():
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage7") / f"{video_id}_stage7.mp4"
        trajectory_csv = TRAJECTORY_OUTPUT_DIR / f"{video_id}_trajectories.csv"
        summary_json = TRAJECTORY_OUTPUT_DIR / f"{video_id}_summary.json"
        observations: list[TrajectoryObservation] = []

        tracker = StatefulByteTracker(
            MODEL_PATH,
            device=DEVICE,
            imgsz=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            tracker_config=TRACKER_CONFIG,
        )
        trajectory_engine = TrajectoryEngine(
            max_history_length=MAX_HISTORY_LENGTH,
            minimum_displacement=MINIMUM_DISPLACEMENT_PIXELS,
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
                source_id=source["source_id"],
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            frame_observations = trajectory_engine.update(frame_records)
            observations.extend(frame_observations)
            draw_trajectory_trails(frame, frame_records, trajectory_engine)
            draw_tracks(frame, frame_records)
            add_overlay(
                frame,
                video_id=video_id,
                frame_index=frame_index,
                source_fps=source_fps,
            )

        print(f"Building trajectories for {source['source_id']}...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative,
            PROJECT_ROOT / output_relative,
            video_id=video_id,
            source_id=source["source_id"],
            frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_trajectory_outputs(
            observations,
            csv_path=trajectory_csv,
            summary_path=summary_json,
            benchmark=benchmark,
        )

        observed_ids = {observation.track_id for observation in observations}
        prior_fps = stage6_fps.get(source["source_id"], 0.0)
        processing_fps = float(benchmark["processing_fps"])
        benchmark_rows.append(
            {
                "video_id": video_id,
                "source_id": source["source_id"],
                "model": MODEL_NAME,
                "device": DEVICE,
                "imgsz": IMAGE_SIZE,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "tracker": TRACKER_CONFIG,
                "max_history_length": MAX_HISTORY_LENGTH,
                "minimum_displacement_pixels": MINIMUM_DISPLACEMENT_PIXELS,
                "input_width": benchmark["input_width"],
                "input_height": benchmark["input_height"],
                "source_fps": benchmark["source_fps"],
                "expected_frame_count": benchmark["expected_frame_count"],
                "frames_processed": benchmark["frames_processed"],
                "elapsed_seconds": benchmark["elapsed_seconds"],
                "processing_fps": processing_fps,
                "stage6_processing_fps": prior_fps,
                "fps_ratio_vs_stage6": (
                    round(processing_fps / prior_fps, 4) if prior_fps > 0 else 0.0
                ),
                "trajectory_observations": len(observations),
                "track_ids_observed": len(observed_ids),
                "output_path": output_relative.as_posix(),
                "status": benchmark["status"],
                "validation_message": benchmark["validation_message"],
            }
        )
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(observations)} observations, "
            f"{len(observed_ids)} Track IDs observed (diagnostic only)",
            flush=True,
        )

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=STAGE7_BENCHMARK_FIELDS, lineterminator="\n"
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
