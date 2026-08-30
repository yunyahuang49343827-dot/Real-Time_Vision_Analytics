#!/usr/bin/env python3
"""Run Stage 10 config-driven wrong-way monitoring on four demo videos."""

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

from vision_analytics.spatial.direction import (
    WRONG_WAY_FIELDS,
    DirectionRule,
    WrongWayEngine,
    WrongWayRecord,
    load_direction_config,
)
from vision_analytics.spatial.zone import ZoneEngine, load_zone_config
from vision_analytics.tracking.schema import TrackRecord
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
STAGE9_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage9" / "stage9_zone_benchmark.csv"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage10"
BENCHMARK_PATH = ANALYTICS_DIR / "stage10_wrong_way_benchmark.csv"
CONSOLIDATED_DETECTIONS_PATH = ANALYTICS_DIR / "wrong_way_detections.csv"

BENCHMARK_FIELDS = (
    "video_id", "source_id", "model", "device", "imgsz", "confidence_threshold",
    "tracker", "input_width", "input_height", "source_fps", "expected_frame_count",
    "frames_processed", "elapsed_seconds", "processing_fps", "stage9_processing_fps",
    "fps_ratio_vs_stage9", "direction_rule_count", "confirmed_wrong_way_count",
    "output_path", "status", "validation_message",
)


def load_runtime_sources() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def load_stage9() -> dict[str, dict[str, float]]:
    with STAGE9_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "width": float(row["input_width"]),
                "height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def draw_zones(
    frame: object,
    zone_engine: ZoneEngine,
    rules: tuple[DirectionRule, ...],
) -> None:
    rules_by_zone = {rule.zone_id: rule for rule in rules}
    overlay = frame.copy()
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        color = (0, 165, 255) if zone.zone_id in rules_by_zone else (255, 255, 0)
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.09, frame, 0.91, 0, frame)
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        rule = rules_by_zone.get(zone.zone_id)
        color = (0, 165, 255) if rule else (255, 255, 0)
        cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
        x, y = polygon[0]
        label = zone.zone_id
        if rule:
            label += " allowed:" + "/".join(sorted(rule.allowed_directions))
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_confirmed_tracks(
    frame: object,
    records: list[TrackRecord],
    engine: WrongWayEngine,
    rules: tuple[DirectionRule, ...],
) -> None:
    rule_zone_ids = tuple(rule.zone_id for rule in rules)
    for record in records:
        if not any(engine.is_confirmed(record.video_id, zone_id, record.track_id) for zone_id in rule_zone_ids):
            continue
        x1, y1, x2, y2 = map(
            round,
            (record.bbox.x1, record.bbox.y1, record.bbox.x2, record.bbox.y2),
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, "WRONG-WAY CONFIRMED", (x1, max(30, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, "WRONG-WAY CONFIRMED", (x1, max(30, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)


def write_outputs(
    records: list[WrongWayRecord],
    *,
    csv_path: Path,
    summary_path: Path,
    benchmark: dict[str, object],
    rules: tuple[DirectionRule, ...],
    consecutive_observations: int,
    minimum_net_displacement: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WRONG_WAY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.to_row() for item in records)
    grouped = Counter((item.zone_id, item.class_name, item.observed_direction) for item in records)
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "confirmed_wrong_way_count": len(records),
        "confirmed_counts": [
            {"zone_id": zone_id, "class_name": class_name, "observed_direction": direction, "count": count}
            for (zone_id, class_name, direction), count in sorted(grouped.items())
        ],
        "rules": [
            {
                "zone_id": rule.zone_id,
                "allowed_directions": sorted(rule.allowed_directions),
                "applicable_classes": sorted(rule.applicable_classes),
            }
            for rule in rules
        ],
        "consecutive_observations_required": consecutive_observations,
        "minimum_net_displacement_pixels": minimum_net_displacement,
        "diagnostic_warning": "Wrong-way detections are rule-based image-space observations, not Ground Truth or a formal accuracy measurement.",
        "status": benchmark["status"],
        "validation_message": benchmark["validation_message"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 10 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1

    zones_by_source = load_zone_config(SCENE_CONFIG)
    rules_by_source, consecutive, minimum_displacement = load_direction_config(SCENE_CONFIG)
    stage9 = load_stage9()
    benchmark_rows: list[dict[str, object]] = []
    all_records: list[WrongWayRecord] = []

    for source in load_runtime_sources():
        source_id = source["source_id"]
        if source_id not in zones_by_source or source_id not in rules_by_source or source_id not in stage9:
            print(f"Missing Stage 10 config or metadata for {source_id}", file=sys.stderr)
            return 1
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage10") / f"{video_id}_stage10.mp4"
        detection_csv = ANALYTICS_DIR / f"{video_id}_wrong_way_detections.csv"
        summary_json = ANALYTICS_DIR / f"{video_id}_summary.json"
        rules = rules_by_source[source_id]
        tracker = StatefulByteTracker(
            MODEL_PATH,
            device=DEVICE,
            imgsz=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            tracker_config=TRACKER_CONFIG,
        )
        trajectory = TrajectoryEngine(
            max_history_length=MAX_HISTORY_LENGTH,
            minimum_displacement=MINIMUM_TRAJECTORY_DISPLACEMENT_PIXELS,
        )
        zone_engine = ZoneEngine(
            zones_by_source[source_id],
            frame_width=round(stage9[source_id]["width"]),
            frame_height=round(stage9[source_id]["height"]),
        )
        direction_engine = WrongWayEngine(
            rules,
            consecutive_observations=consecutive,
            minimum_net_displacement=minimum_displacement,
        )

        def update_and_draw(frame: object, frame_index: int, timestamp_seconds: float, source_fps: float) -> None:
            tracked = tracker.track_frame(
                frame,
                video_id=video_id,
                source_id=source_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            trajectories = trajectory.update(tracked)
            zone_observations = zone_engine.update(tracked)
            new_records = direction_engine.update(trajectories, zone_observations)
            draw_zones(frame, zone_engine, rules)
            draw_trajectory_trails(frame, tracked, trajectory)
            draw_tracks(frame, tracked)
            draw_confirmed_tracks(frame, tracked, direction_engine, rules)
            add_overlay(frame, video_id=video_id, frame_index=frame_index, source_fps=source_fps)
            if new_records:
                cv2.putText(frame, f"NEW WRONG-WAY: {len(new_records)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"confirmed wrong-way: {len(direction_engine.records)}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        print(f"Monitoring direction for {source_id} ({len(rules)} configured rule(s))...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative,
            PROJECT_ROOT / output_relative,
            video_id=video_id,
            source_id=source_id,
            frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_outputs(
            direction_engine.records,
            csv_path=detection_csv,
            summary_path=summary_json,
            benchmark=benchmark,
            rules=rules,
            consecutive_observations=consecutive,
            minimum_net_displacement=minimum_displacement,
        )
        all_records.extend(direction_engine.records)
        prior_fps = stage9[source_id]["processing_fps"]
        fps = float(benchmark["processing_fps"])
        benchmark_rows.append({
            "video_id": video_id,
            "source_id": source_id,
            "model": MODEL_NAME,
            "device": DEVICE,
            "imgsz": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "tracker": TRACKER_CONFIG,
            "input_width": benchmark["input_width"],
            "input_height": benchmark["input_height"],
            "source_fps": benchmark["source_fps"],
            "expected_frame_count": benchmark["expected_frame_count"],
            "frames_processed": benchmark["frames_processed"],
            "elapsed_seconds": benchmark["elapsed_seconds"],
            "processing_fps": fps,
            "stage9_processing_fps": prior_fps,
            "fps_ratio_vs_stage9": round(fps / prior_fps, 4) if prior_fps > 0 else 0.0,
            "direction_rule_count": len(rules),
            "confirmed_wrong_way_count": len(direction_engine.records),
            "output_path": output_relative.as_posix(),
            "status": benchmark["status"],
            "validation_message": benchmark["validation_message"],
        })
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(direction_engine.records)} confirmed wrong-way",
            flush=True,
        )

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(benchmark_rows)
    with CONSOLIDATED_DETECTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WRONG_WAY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.to_row() for item in all_records)
    failures = sum(row["status"] == "FAIL" for row in benchmark_rows)
    warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_PATH}")
    print(f"Wrote {len(all_records)} rows to {CONSOLIDATED_DETECTIONS_PATH}")
    print(f"Summary: {len(benchmark_rows) - failures - warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
