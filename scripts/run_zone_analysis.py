#!/usr/bin/env python3
"""Run Stage 9 polygon-zone state analysis on persistent tracks."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.spatial.zone import (
    ZONE_TRANSITION_FIELDS,
    ZoneEngine,
    ZoneObservation,
    load_zone_config,
)
from vision_analytics.tracking.tracker import StatefulByteTracker, draw_tracks
from vision_analytics.tracking.trajectory import TrajectoryEngine, draw_trajectory_trails
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
STAGE8_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage8" / "stage8_line_crossing_benchmark.csv"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage9"
BENCHMARK_PATH = ANALYTICS_DIR / "stage9_zone_benchmark.csv"

BENCHMARK_FIELDS = (
    "video_id", "source_id", "model", "device", "imgsz", "confidence_threshold",
    "tracker", "input_width", "input_height", "source_fps", "expected_frame_count",
    "frames_processed", "elapsed_seconds", "processing_fps", "stage8_processing_fps",
    "fps_ratio_vs_stage8", "zone_transition_count", "output_path", "status",
    "validation_message",
)


def load_runtime_sources() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["asset_type"] == "video" and row["role"] == "runtime_demo"]


def load_stage8() -> dict[str, dict[str, float]]:
    with STAGE8_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "width": float(row["input_width"]),
                "height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def draw_zones(frame: object, engine: ZoneEngine) -> None:
    overlay = frame.copy()
    for zone in engine.zones:
        polygon = np.rint(engine.polygon(zone.zone_id)).astype(np.int32)
        cv2.fillPoly(overlay, [polygon], (255, 255, 0))
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    for zone in engine.zones:
        polygon = np.rint(engine.polygon(zone.zone_id)).astype(np.int32)
        cv2.polylines(frame, [polygon], True, (255, 255, 0), 3, cv2.LINE_AA)
        x, y = polygon[0]
        label = f"{zone.zone_id} occupancy:{engine.current_occupancy[zone.zone_id]}"
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)


def write_outputs(
    transitions: list[ZoneObservation], *, csv_path: Path, summary_path: Path,
    benchmark: dict[str, object], engine: ZoneEngine,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ZONE_TRANSITION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.to_row() for item in transitions)
    grouped = Counter((item.zone_id, item.class_name, item.transition) for item in transitions)
    summary = {
        "video_id": benchmark["video_id"], "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "transition_counts": [
            {"zone_id": zone_id, "class_name": class_name, "transition": transition, "count": count}
            for (zone_id, class_name, transition), count in sorted(grouped.items())
        ],
        "zone_diagnostics": {
            zone.zone_id: {
                "zone_entry_count": sum(x.zone_id == zone.zone_id and x.transition == "ENTER" for x in transitions),
                "zone_exit_count": sum(x.zone_id == zone.zone_id and x.transition == "EXIT" for x in transitions),
                "tracks_observed_inside": len(engine.tracks_observed_inside[zone.zone_id]),
                "current_observed_occupancy": engine.current_occupancy[zone.zone_id],
                "peak_observed_occupancy": engine.peak_occupancy[zone.zone_id],
            }
            for zone in engine.zones
        },
        "diagnostic_warning": "Zone statistics are observed Track-state diagnostics, not Ground Truth unique visitors or formal traffic analytics.",
        "status": benchmark["status"], "validation_message": benchmark["validation_message"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 9 requires Apple MPS; CPU fallback is disabled", file=sys.stderr); return 1
    zones_by_source = load_zone_config(SCENE_CONFIG)
    stage8 = load_stage8()
    benchmark_rows: list[dict[str, object]] = []
    for source in load_runtime_sources():
        source_id = source["source_id"]
        if source_id not in zones_by_source or source_id not in stage8:
            print(f"Missing Stage 9 config or metadata for {source_id}", file=sys.stderr); return 1
        input_relative = Path(source["local_path"]); video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage9") / f"{video_id}_stage9.mp4"
        transition_csv = ANALYTICS_DIR / f"{video_id}_zone_transitions.csv"
        summary_json = ANALYTICS_DIR / f"{video_id}_summary.json"
        tracker = StatefulByteTracker(MODEL_PATH, device=DEVICE, imgsz=IMAGE_SIZE, confidence_threshold=CONFIDENCE_THRESHOLD, tracker_config=TRACKER_CONFIG)
        trajectory = TrajectoryEngine(max_history_length=MAX_HISTORY_LENGTH, minimum_displacement=MINIMUM_TRAJECTORY_DISPLACEMENT_PIXELS)
        zone_engine = ZoneEngine(zones_by_source[source_id], frame_width=round(stage8[source_id]["width"]), frame_height=round(stage8[source_id]["height"]))

        def update_and_draw(frame: object, frame_index: int, timestamp_seconds: float, source_fps: float) -> None:
            records = tracker.track_frame(frame, video_id=video_id, source_id=source_id, frame_index=frame_index, timestamp_seconds=timestamp_seconds)
            trajectory.update(records)
            zone_engine.update(records)
            draw_zones(frame, zone_engine)
            draw_trajectory_trails(frame, records, trajectory)
            draw_tracks(frame, records)
            add_overlay(frame, video_id=video_id, frame_index=frame_index, source_fps=source_fps)

        print(f"Analyzing zones for {source_id}...", flush=True)
        benchmark = process_video(PROJECT_ROOT / input_relative, PROJECT_ROOT / output_relative, video_id=video_id, source_id=source_id, frame_processor=update_and_draw)
        benchmark["output_path"] = output_relative.as_posix()
        write_outputs(zone_engine.transitions, csv_path=transition_csv, summary_path=summary_json, benchmark=benchmark, engine=zone_engine)
        prior_fps = stage8[source_id]["processing_fps"]; fps = float(benchmark["processing_fps"])
        benchmark_rows.append({
            "video_id": video_id, "source_id": source_id, "model": MODEL_NAME, "device": DEVICE,
            "imgsz": IMAGE_SIZE, "confidence_threshold": CONFIDENCE_THRESHOLD, "tracker": TRACKER_CONFIG,
            "input_width": benchmark["input_width"], "input_height": benchmark["input_height"],
            "source_fps": benchmark["source_fps"], "expected_frame_count": benchmark["expected_frame_count"],
            "frames_processed": benchmark["frames_processed"], "elapsed_seconds": benchmark["elapsed_seconds"],
            "processing_fps": fps, "stage8_processing_fps": prior_fps,
            "fps_ratio_vs_stage8": round(fps / prior_fps, 4) if prior_fps > 0 else 0.0,
            "zone_transition_count": len(zone_engine.transitions), "output_path": output_relative.as_posix(),
            "status": benchmark["status"], "validation_message": benchmark["validation_message"],
        })
        print(f"  {benchmark['status']}: {benchmark['frames_processed']} frames, {benchmark['processing_fps']} FPS, {len(zone_engine.transitions)} ENTER/EXIT transitions", flush=True)

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n"); writer.writeheader(); writer.writerows(benchmark_rows)
    failures = sum(row["status"] == "FAIL" for row in benchmark_rows); warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_PATH}")
    print(f"Summary: {len(benchmark_rows)-failures-warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
