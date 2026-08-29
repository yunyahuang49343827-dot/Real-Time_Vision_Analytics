#!/usr/bin/env python3
"""Run the Stage 4 YOLO26n MPS detection baseline."""

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

from vision_analytics.detection.detector import PretrainedDetector, draw_detections
from vision_analytics.detection.schema import DETECTION_FIELDS, DetectionRecord
from vision_analytics.video.pipeline import add_overlay, process_video

MODEL_NAME = "yolo26n.pt"
MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / MODEL_NAME
DEVICE = "mps"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.csv"
STAGE3_BENCHMARK = (
    PROJECT_ROOT / "outputs" / "analytics" / "stage3_video_benchmark.csv"
)
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "stage4"
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections" / "stage4"
BENCHMARK_PATH = (
    PROJECT_ROOT / "outputs" / "analytics" / "stage4_detection_benchmark.csv"
)

STAGE4_BENCHMARK_FIELDS = (
    "video_id",
    "source_id",
    "model",
    "device",
    "imgsz",
    "confidence_threshold",
    "input_width",
    "input_height",
    "source_fps",
    "expected_frame_count",
    "frames_processed",
    "elapsed_seconds",
    "processing_fps",
    "stage3_processing_fps",
    "fps_ratio_vs_stage3",
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


def load_stage3_fps() -> dict[str, float]:
    if not STAGE3_BENCHMARK.is_file():
        return {}
    with STAGE3_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: float(row["processing_fps"])
            for row in csv.DictReader(handle)
        }


def write_detection_outputs(
    records: list[DetectionRecord],
    *,
    csv_path: Path,
    summary_path: Path,
    benchmark: dict[str, object],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DETECTION_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)

    class_occurrences = Counter(record.class_name for record in records)
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "model": MODEL_NAME,
        "device": DEVICE,
        "imgsz": IMAGE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "frames_processed": benchmark["frames_processed"],
        "total_detection_occurrences": len(records),
        "class_occurrences": dict(sorted(class_occurrences.items())),
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
        print("Stage 4 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_file():
        print(f"Model weights not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    detector = PretrainedDetector(
        MODEL_PATH,
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    print(
        f"Loaded {MODEL_NAME}: device={DEVICE}, imgsz={IMAGE_SIZE}, "
        f"confidence={CONFIDENCE_THRESHOLD}"
    )

    stage3_fps = load_stage3_fps()
    benchmark_rows: list[dict[str, object]] = []
    for source in load_runtime_sources():
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage4") / f"{video_id}_stage4.mp4"
        detection_csv = DETECTION_OUTPUT_DIR / f"{video_id}_detections.csv"
        summary_json = DETECTION_OUTPUT_DIR / f"{video_id}_summary.json"
        records: list[DetectionRecord] = []

        def detect_and_draw(
            frame: object,
            frame_index: int,
            timestamp_seconds: float,
            source_fps: float,
        ) -> None:
            frame_records = detector.detect(
                frame,
                video_id=video_id,
                source_id=source["source_id"],
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            records.extend(frame_records)
            draw_detections(frame, frame_records)
            add_overlay(
                frame,
                video_id=video_id,
                frame_index=frame_index,
                source_fps=source_fps,
            )

        print(f"Detecting {source['source_id']} ({input_relative.name})...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative,
            PROJECT_ROOT / output_relative,
            video_id=video_id,
            source_id=source["source_id"],
            frame_processor=detect_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_detection_outputs(
            records,
            csv_path=detection_csv,
            summary_path=summary_json,
            benchmark=benchmark,
        )

        prior_fps = stage3_fps.get(source["source_id"], 0.0)
        processing_fps = float(benchmark["processing_fps"])
        benchmark_row = {
            "video_id": video_id,
            "source_id": source["source_id"],
            "model": MODEL_NAME,
            "device": DEVICE,
            "imgsz": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "input_width": benchmark["input_width"],
            "input_height": benchmark["input_height"],
            "source_fps": benchmark["source_fps"],
            "expected_frame_count": benchmark["expected_frame_count"],
            "frames_processed": benchmark["frames_processed"],
            "elapsed_seconds": benchmark["elapsed_seconds"],
            "processing_fps": processing_fps,
            "stage3_processing_fps": prior_fps,
            "fps_ratio_vs_stage3": (
                round(processing_fps / prior_fps, 4) if prior_fps > 0 else 0.0
            ),
            "output_path": output_relative.as_posix(),
            "status": benchmark["status"],
            "validation_message": benchmark["validation_message"],
        }
        benchmark_rows.append(benchmark_row)
        class_counts = Counter(record.class_name for record in records)
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(records)} detection occurrences, "
            f"classes={dict(sorted(class_counts.items()))}",
            flush=True,
        )

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=STAGE4_BENCHMARK_FIELDS, lineterminator="\n"
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
