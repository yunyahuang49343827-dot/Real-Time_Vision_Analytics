#!/usr/bin/env python3
"""Run Stage 11 observed dwell and stationary-vehicle rules."""

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

from vision_analytics.spatial.dwell import (
    LONG_DWELL_FIELDS,
    STATIONARY_VEHICLE_FIELDS,
    DwellRule,
    LongDwellRecord,
    StationaryRule,
    StationaryVehicleRecord,
    TemporalRuleEngine,
    load_temporal_config,
)
from vision_analytics.spatial.zone import ZoneEngine, load_zone_config
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
STAGE10_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage10" / "stage10_wrong_way_benchmark.csv"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage11"
BENCHMARK_PATH = ANALYTICS_DIR / "stage11_temporal_benchmark.csv"
LONG_DWELL_PATH = ANALYTICS_DIR / "long_dwell_detections.csv"
STATIONARY_PATH = ANALYTICS_DIR / "stationary_vehicle_detections.csv"

BENCHMARK_FIELDS = (
    "video_id", "source_id", "model", "device", "imgsz", "confidence_threshold",
    "tracker", "input_width", "input_height", "source_fps", "expected_frame_count",
    "frames_processed", "elapsed_seconds", "processing_fps", "stage10_processing_fps",
    "fps_ratio_vs_stage10", "dwell_rule_count", "stationary_rule_count",
    "long_dwell_count", "stationary_vehicle_count", "output_path", "status",
    "validation_message",
)


def load_runtime_sources() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def load_stage10() -> dict[str, dict[str, float]]:
    with STAGE10_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "width": float(row["input_width"]),
                "height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def draw_rule_zones(
    frame: object,
    zone_engine: ZoneEngine,
    dwell_rules: tuple[DwellRule, ...],
    stationary_rules: tuple[StationaryRule, ...],
) -> None:
    dwell_by_zone = {rule.zone_id: rule for rule in dwell_rules}
    stationary_by_zone = {rule.zone_id: rule for rule in stationary_rules}
    overlay = frame.copy()
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        if zone.zone_id in stationary_by_zone:
            color = (0, 165, 255)
        elif zone.zone_id in dwell_by_zone:
            color = (255, 255, 0)
        else:
            color = (160, 160, 160)
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        dwell_rule = dwell_by_zone.get(zone.zone_id)
        stationary_rule = stationary_by_zone.get(zone.zone_id)
        if stationary_rule:
            color = (0, 165, 255)
            suffix = f" STATIONARY {stationary_rule.duration_seconds:g}s"
        elif dwell_rule:
            color = (255, 255, 0)
            suffix = f" DWELL {dwell_rule.threshold_seconds:g}s"
        else:
            color = (160, 160, 160)
            suffix = ""
        cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
        x, y = polygon[0]
        label = zone.zone_id + suffix
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def write_csv(path: Path, fields: tuple[str, ...], records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)


def write_summary(
    path: Path,
    *,
    benchmark: dict[str, object],
    dwell_records: list[LongDwellRecord],
    stationary_records: list[StationaryVehicleRecord],
    dwell_rules: tuple[DwellRule, ...],
    stationary_rules: tuple[StationaryRule, ...],
    maximum_missing_seconds: float,
) -> None:
    dwell_counts = Counter((item.zone_id, item.class_name) for item in dwell_records)
    stationary_counts = Counter((item.zone_id, item.class_name) for item in stationary_records)
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "long_dwell_count": len(dwell_records),
        "stationary_vehicle_count": len(stationary_records),
        "long_dwell_counts": [
            {"zone_id": zone_id, "class_name": class_name, "count": count}
            for (zone_id, class_name), count in sorted(dwell_counts.items())
        ],
        "stationary_vehicle_counts": [
            {"zone_id": zone_id, "class_name": class_name, "count": count}
            for (zone_id, class_name), count in sorted(stationary_counts.items())
        ],
        "dwell_rules": [
            {"zone_id": rule.zone_id, "threshold_seconds": rule.threshold_seconds,
             "applicable_classes": sorted(rule.applicable_classes)}
            for rule in dwell_rules
        ],
        "stationary_rules": [
            {"zone_id": rule.zone_id, "duration_seconds": rule.duration_seconds,
             "movement_threshold_normalized": rule.movement_threshold,
             "applicable_classes": sorted(rule.applicable_classes)}
            for rule in stationary_rules
        ],
        "maximum_missing_seconds": maximum_missing_seconds,
        "diagnostic_warning": "Durations are observed Track time and movement is normalized image-space displacement; neither is Ground Truth physical behaviour.",
        "status": benchmark["status"],
        "validation_message": benchmark["validation_message"],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 11 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1

    zones_by_source = load_zone_config(SCENE_CONFIG)
    dwell_by_source, stationary_by_source, maximum_missing = load_temporal_config(SCENE_CONFIG)
    stage10 = load_stage10()
    benchmark_rows: list[dict[str, object]] = []
    all_dwell: list[LongDwellRecord] = []
    all_stationary: list[StationaryVehicleRecord] = []

    for source in load_runtime_sources():
        source_id = source["source_id"]
        if source_id not in zones_by_source or source_id not in stage10:
            print(f"Missing Stage 11 config or metadata for {source_id}", file=sys.stderr)
            return 1
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage11") / f"{video_id}_stage11.mp4"
        dwell_csv = ANALYTICS_DIR / f"{video_id}_long_dwell_detections.csv"
        stationary_csv = ANALYTICS_DIR / f"{video_id}_stationary_vehicle_detections.csv"
        summary_json = ANALYTICS_DIR / f"{video_id}_summary.json"
        dwell_rules = dwell_by_source[source_id]
        stationary_rules = stationary_by_source[source_id]

        tracker = StatefulByteTracker(
            MODEL_PATH, device=DEVICE, imgsz=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD, tracker_config=TRACKER_CONFIG,
        )
        trajectory = TrajectoryEngine(
            max_history_length=MAX_HISTORY_LENGTH,
            minimum_displacement=MINIMUM_TRAJECTORY_DISPLACEMENT_PIXELS,
        )
        width = round(stage10[source_id]["width"])
        height = round(stage10[source_id]["height"])
        zone_engine = ZoneEngine(zones_by_source[source_id], frame_width=width, frame_height=height)
        temporal = TemporalRuleEngine(
            dwell_rules, stationary_rules, frame_width=width, frame_height=height,
            maximum_missing_seconds=maximum_missing,
        )

        def update_and_draw(frame: object, frame_index: int, timestamp_seconds: float, source_fps: float) -> None:
            tracked = tracker.track_frame(
                frame, video_id=video_id, source_id=source_id,
                frame_index=frame_index, timestamp_seconds=timestamp_seconds,
            )
            trajectories = trajectory.update(tracked)
            zone_observations = zone_engine.update(tracked)
            new_dwell, new_stationary = temporal.update(zone_observations, trajectories)
            draw_rule_zones(frame, zone_engine, dwell_rules, stationary_rules)
            draw_trajectory_trails(frame, tracked, trajectory)
            draw_tracks(frame, tracked)
            add_overlay(frame, video_id=video_id, frame_index=frame_index, source_fps=source_fps)
            cv2.putText(frame, f"LONG_DWELL: {len(temporal.long_dwell_records)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"STATIONARY_VEHICLE: {len(temporal.stationary_vehicle_records)}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            if new_dwell or new_stationary:
                cv2.putText(frame, f"NEW TEMPORAL RULE: {len(new_dwell) + len(new_stationary)}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        print(
            f"Running temporal rules for {source_id} "
            f"({len(dwell_rules)} dwell, {len(stationary_rules)} stationary)...", flush=True,
        )
        benchmark = process_video(
            PROJECT_ROOT / input_relative, PROJECT_ROOT / output_relative,
            video_id=video_id, source_id=source_id, frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_csv(dwell_csv, LONG_DWELL_FIELDS, temporal.long_dwell_records)
        write_csv(stationary_csv, STATIONARY_VEHICLE_FIELDS, temporal.stationary_vehicle_records)
        write_summary(
            summary_json, benchmark=benchmark,
            dwell_records=temporal.long_dwell_records,
            stationary_records=temporal.stationary_vehicle_records,
            dwell_rules=dwell_rules, stationary_rules=stationary_rules,
            maximum_missing_seconds=maximum_missing,
        )
        all_dwell.extend(temporal.long_dwell_records)
        all_stationary.extend(temporal.stationary_vehicle_records)

        prior_fps = stage10[source_id]["processing_fps"]
        fps = float(benchmark["processing_fps"])
        benchmark_rows.append({
            "video_id": video_id, "source_id": source_id, "model": MODEL_NAME,
            "device": DEVICE, "imgsz": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "tracker": TRACKER_CONFIG,
            "input_width": benchmark["input_width"], "input_height": benchmark["input_height"],
            "source_fps": benchmark["source_fps"],
            "expected_frame_count": benchmark["expected_frame_count"],
            "frames_processed": benchmark["frames_processed"],
            "elapsed_seconds": benchmark["elapsed_seconds"], "processing_fps": fps,
            "stage10_processing_fps": prior_fps,
            "fps_ratio_vs_stage10": round(fps / prior_fps, 4) if prior_fps > 0 else 0.0,
            "dwell_rule_count": len(dwell_rules),
            "stationary_rule_count": len(stationary_rules),
            "long_dwell_count": len(temporal.long_dwell_records),
            "stationary_vehicle_count": len(temporal.stationary_vehicle_records),
            "output_path": output_relative.as_posix(), "status": benchmark["status"],
            "validation_message": benchmark["validation_message"],
        })
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(temporal.long_dwell_records)} dwell, "
            f"{len(temporal.stationary_vehicle_records)} stationary", flush=True,
        )

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(benchmark_rows)
    write_csv(LONG_DWELL_PATH, LONG_DWELL_FIELDS, all_dwell)
    write_csv(STATIONARY_PATH, STATIONARY_VEHICLE_FIELDS, all_stationary)
    failures = sum(row["status"] == "FAIL" for row in benchmark_rows)
    warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_PATH}")
    print(f"Wrote {len(all_dwell)} rows to {LONG_DWELL_PATH}")
    print(f"Wrote {len(all_stationary)} rows to {STATIONARY_PATH}")
    print(f"Summary: {len(benchmark_rows) - failures - warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
