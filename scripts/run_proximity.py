#!/usr/bin/env python3
"""Run Stage 12 normalized image-space person–vehicle proximity warnings."""

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

from vision_analytics.spatial.proximity import (
    PROXIMITY_FIELDS,
    ProximityEngine,
    ProximityRecord,
    ProximityRule,
    load_proximity_config,
    normalized_bbox_distance,
)
from vision_analytics.spatial.zone import ZoneEngine, load_zone_config
from vision_analytics.tracking.schema import TrackRecord
from vision_analytics.tracking.tracker import StatefulByteTracker, draw_tracks
from vision_analytics.video.pipeline import add_overlay, process_video

MODEL_NAME = "yolo26n.pt"
MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / MODEL_NAME
DEVICE = "mps"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
TRACKER_CONFIG = "bytetrack.yaml"
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.csv"
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes.yaml"
STAGE11_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage11" / "stage11_temporal_benchmark.csv"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage12"
BENCHMARK_PATH = ANALYTICS_DIR / "stage12_proximity_benchmark.csv"
DETECTIONS_PATH = ANALYTICS_DIR / "proximity_detections.csv"

BENCHMARK_FIELDS = (
    "video_id", "source_id", "model", "device", "imgsz", "confidence_threshold",
    "tracker", "input_width", "input_height", "source_fps", "expected_frame_count",
    "frames_processed", "elapsed_seconds", "processing_fps", "stage11_processing_fps",
    "fps_ratio_vs_stage11", "configured_rule_count", "enabled_rule_count",
    "proximity_warning_count", "rider_pair_exclusion_count", "pair_comparisons",
    "output_path", "status", "validation_message",
)


def load_runtime_sources() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def load_stage11() -> dict[str, dict[str, float]]:
    with STAGE11_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "width": float(row["input_width"]),
                "height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def draw_proximity_zones(
    frame: object,
    zone_engine: ZoneEngine,
    rules: tuple[ProximityRule, ...],
) -> None:
    rules_by_zone = {rule.zone_id: rule for rule in rules}
    overlay = frame.copy()
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        rule = rules_by_zone.get(zone.zone_id)
        color = (255, 0, 255) if rule is not None and rule.enabled else (128, 128, 128)
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        rule = rules_by_zone.get(zone.zone_id)
        color = (255, 0, 255) if rule is not None and rule.enabled else (128, 128, 128)
        cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
        x, y = polygon[0]
        if rule is None:
            suffix = ""
        elif rule.enabled:
            suffix = f" PROX {rule.trigger_threshold:.3f}/{rule.release_threshold:.3f}"
        else:
            suffix = " PROX disabled"
        label = zone.zone_id + suffix
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (int(x) + 6, max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_active_pairs(
    frame: object,
    tracks: list[TrackRecord],
    engine: ProximityEngine,
) -> None:
    by_id = {(item.video_id, item.track_id): item for item in tracks}
    for video_id, _zone_id, person_id, vehicle_id in engine.active_pair_keys():
        person = by_id.get((video_id, person_id))
        vehicle = by_id.get((video_id, vehicle_id))
        if person is None or vehicle is None:
            continue
        person_center = (round(person.center_x), round(person.center_y))
        vehicle_center = (round(vehicle.center_x), round(vehicle.center_y))
        cv2.line(frame, person_center, vehicle_center, (0, 0, 255), 3, cv2.LINE_AA)
        distance = normalized_bbox_distance(
            person.bbox, vehicle.bbox,
            frame_width=engine.frame_width, frame_height=engine.frame_height,
        )
        midpoint = (
            round((person.center_x + vehicle.center_x) / 2),
            round((person.center_y + vehicle.center_y) / 2),
        )
        label = f"PROX {distance:.4f}"
        cv2.putText(frame, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def write_csv(path: Path, records: list[ProximityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROXIMITY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.to_row() for item in records)


def write_summary(
    path: Path,
    *,
    benchmark: dict[str, object],
    records: list[ProximityRecord],
    rules: tuple[ProximityRule, ...],
    engine: ProximityEngine,
) -> None:
    grouped = Counter((item.zone_id, item.vehicle_class) for item in records)
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "proximity_warning_count": len(records),
        "warning_counts": [
            {"zone_id": zone_id, "vehicle_class": vehicle_class, "count": count}
            for (zone_id, vehicle_class), count in sorted(grouped.items())
        ],
        "rules": [
            {
                "zone_id": rule.zone_id,
                "enabled": rule.enabled,
                "vehicle_classes": sorted(rule.vehicle_classes),
                "trigger_distance_normalized": rule.trigger_threshold,
                "release_distance_normalized": rule.release_threshold,
                "minimum_consecutive_observations": rule.minimum_consecutive_observations,
                "rider_overlap_exclusion_ratio": rule.rider_overlap_exclusion_ratio,
            }
            for rule in rules
        ],
        "distinct_rider_pair_exclusions": len(engine.excluded_rider_pairs),
        "class_and_zone_filtered_pair_comparisons": engine.pair_comparisons,
        "diagnostic_warning": "Warnings represent normalized image-space bbox proximity only; they are not physical distance, collision probability, near-miss, or TTC estimates.",
        "status": benchmark["status"],
        "validation_message": benchmark["validation_message"],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 12 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1

    zones_by_source = load_zone_config(SCENE_CONFIG)
    rules_by_source = load_proximity_config(SCENE_CONFIG)
    stage11 = load_stage11()
    benchmark_rows: list[dict[str, object]] = []
    all_records: list[ProximityRecord] = []

    for source in load_runtime_sources():
        source_id = source["source_id"]
        if source_id not in zones_by_source or source_id not in rules_by_source or source_id not in stage11:
            print(f"Missing Stage 12 config or metadata for {source_id}", file=sys.stderr)
            return 1
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage12") / f"{video_id}_stage12.mp4"
        detections_csv = ANALYTICS_DIR / f"{video_id}_proximity_detections.csv"
        summary_json = ANALYTICS_DIR / f"{video_id}_summary.json"
        rules = rules_by_source[source_id]
        width = round(stage11[source_id]["width"])
        height = round(stage11[source_id]["height"])
        tracker = StatefulByteTracker(
            MODEL_PATH, device=DEVICE, imgsz=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD, tracker_config=TRACKER_CONFIG,
        )
        zone_engine = ZoneEngine(zones_by_source[source_id], frame_width=width, frame_height=height)
        proximity = ProximityEngine(rules, frame_width=width, frame_height=height)

        def update_and_draw(frame: object, frame_index: int, timestamp_seconds: float, source_fps: float) -> None:
            tracked = tracker.track_frame(
                frame, video_id=video_id, source_id=source_id,
                frame_index=frame_index, timestamp_seconds=timestamp_seconds,
            )
            zone_observations = zone_engine.update(tracked)
            new_records = proximity.update(tracked, zone_observations)
            draw_proximity_zones(frame, zone_engine, rules)
            draw_tracks(frame, tracked)
            draw_active_pairs(frame, tracked, proximity)
            add_overlay(frame, video_id=video_id, frame_index=frame_index, source_fps=source_fps)
            cv2.putText(frame, f"PROXIMITY WARNINGS: {len(proximity.records)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            if new_records:
                cv2.putText(frame, f"NEW PROXIMITY: {len(new_records)}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        enabled_count = sum(rule.enabled for rule in rules)
        print(
            f"Running proximity for {source_id} "
            f"({enabled_count}/{len(rules)} rule(s) enabled)...", flush=True,
        )
        benchmark = process_video(
            PROJECT_ROOT / input_relative, PROJECT_ROOT / output_relative,
            video_id=video_id, source_id=source_id, frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_csv(detections_csv, proximity.records)
        write_summary(
            summary_json, benchmark=benchmark, records=proximity.records,
            rules=rules, engine=proximity,
        )
        all_records.extend(proximity.records)

        prior_fps = stage11[source_id]["processing_fps"]
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
            "stage11_processing_fps": prior_fps,
            "fps_ratio_vs_stage11": round(fps / prior_fps, 4) if prior_fps > 0 else 0.0,
            "configured_rule_count": len(rules), "enabled_rule_count": enabled_count,
            "proximity_warning_count": len(proximity.records),
            "rider_pair_exclusion_count": len(proximity.excluded_rider_pairs),
            "pair_comparisons": proximity.pair_comparisons,
            "output_path": output_relative.as_posix(), "status": benchmark["status"],
            "validation_message": benchmark["validation_message"],
        })
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(proximity.records)} warnings, "
            f"{len(proximity.excluded_rider_pairs)} rider-pair exclusions", flush=True,
        )

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(benchmark_rows)
    write_csv(DETECTIONS_PATH, all_records)
    failures = sum(row["status"] == "FAIL" for row in benchmark_rows)
    warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_PATH}")
    print(f"Wrote {len(all_records)} rows to {DETECTIONS_PATH}")
    print(f"Summary: {len(benchmark_rows) - failures - warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
