#!/usr/bin/env python3
"""Run Stage 13 upstream rules and normalize them into unified events."""

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

from vision_analytics.events.engine import EventEngine, EventPolicy, load_event_policy
from vision_analytics.events.schema import EVENT_FIELDS, EventRecord
from vision_analytics.spatial.direction import WrongWayEngine, load_direction_config
from vision_analytics.spatial.dwell import TemporalRuleEngine, load_temporal_config
from vision_analytics.spatial.line_crossing import LineCrossingEngine, load_scene_config
from vision_analytics.spatial.proximity import ProximityEngine, load_proximity_config
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
STAGE12_BENCHMARK = PROJECT_ROOT / "outputs" / "analytics" / "stage12" / "stage12_proximity_benchmark.csv"
EVENT_DIR = PROJECT_ROOT / "outputs" / "events" / "stage13"
BENCHMARK_PATH = EVENT_DIR / "stage13_event_benchmark.csv"
EVENTS_PATH = EVENT_DIR / "events.csv"

BENCHMARK_FIELDS = (
    "video_id", "source_id", "model", "device", "imgsz", "confidence_threshold",
    "tracker", "input_width", "input_height", "source_fps", "expected_frame_count",
    "frames_processed", "elapsed_seconds", "processing_fps", "stage12_processing_fps",
    "fps_ratio_vs_stage12", "event_count", "info_count", "warning_count",
    "critical_count", "review_required_count", "output_path", "status",
    "validation_message",
)


def load_runtime_sources() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def load_stage12() -> dict[str, dict[str, float]]:
    with STAGE12_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "width": float(row["input_width"]),
                "height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def draw_spatial_config(
    frame: object,
    *,
    line_engine: LineCrossingEngine,
    zone_engine: ZoneEngine,
    policy: EventPolicy,
) -> None:
    overlay = frame.copy()
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        color = (0, 0, 255) if zone.zone_id in policy.intrusion_rules else (255, 128, 0)
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)
    for zone in zone_engine.zones:
        polygon = np.rint(zone_engine.polygon(zone.zone_id)).astype(np.int32)
        restricted = zone.zone_id in policy.intrusion_rules
        color = (0, 0, 255) if restricted else (255, 128, 0)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = polygon[0]
        suffix = " RESTRICTED_PERSON" if restricted else ""
        cv2.putText(frame, zone.zone_id + suffix, (int(x) + 5, max(24, int(y) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, zone.zone_id + suffix, (int(x) + 5, max(24, int(y) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    for line in line_engine.lines:
        start, end = line_engine.pixel_line(line.line_id)
        start_pixel = (round(start[0]), round(start[1]))
        end_pixel = (round(end[0]), round(end[1]))
        cv2.line(frame, start_pixel, end_pixel, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(frame, start_pixel, end_pixel, (0, 255, 255), 2, cv2.LINE_AA)


def draw_event_overlay(
    frame: object,
    *,
    records: list[EventRecord],
    new_events: list[EventRecord],
) -> None:
    severity_counts = Counter(item.severity for item in records)
    label = (
        f"EVENTS {len(records)}  INFO {severity_counts['INFO']}  "
        f"WARNING {severity_counts['WARNING']}  CRITICAL {severity_counts['CRITICAL']}"
    )
    cv2.putText(frame, label, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    colors = {"INFO": (0, 255, 255), "WARNING": (0, 165, 255), "CRITICAL": (0, 0, 255)}
    for index, event in enumerate(new_events[:3]):
        text = f"NEW {event.severity}: {event.event_type} {event.event_id}"
        position = (20, 138 + index * 30)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.62, colors[event.severity], 2, cv2.LINE_AA)


def write_event_csv(path: Path, records: list[EventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.to_row() for item in records)


def write_summary(
    path: Path,
    *,
    benchmark: dict[str, object],
    records: list[EventRecord],
) -> None:
    grouped = Counter((item.event_type, item.severity) for item in records)
    status_counts = Counter(item.status for item in records)
    summary = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "event_count": len(records),
        "event_type_severity_counts": [
            {"event_type": event_type, "severity": severity, "count": count}
            for (event_type, severity), count in sorted(grouped.items())
        ],
        "status_counts": dict(sorted(status_counts.items())),
        "rule_generated_warning": "Event counts are system rule outputs, not Ground Truth incidents or verified offences.",
        "status": benchmark["status"],
        "validation_message": benchmark["validation_message"],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 13 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_file():
        print(f"Model weights not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    lines_by_source, maximum_frame_gap, minimum_movement = load_scene_config(SCENE_CONFIG)
    zones_by_source = load_zone_config(SCENE_CONFIG)
    direction_by_source, direction_consecutive, direction_displacement = load_direction_config(SCENE_CONFIG)
    dwell_by_source, stationary_by_source, maximum_missing = load_temporal_config(SCENE_CONFIG)
    proximity_by_source = load_proximity_config(SCENE_CONFIG)
    policy = load_event_policy(SCENE_CONFIG)
    stage12 = load_stage12()
    benchmark_rows: list[dict[str, object]] = []
    all_events: list[EventRecord] = []

    for source in load_runtime_sources():
        source_id = source["source_id"]
        required = (
            lines_by_source, zones_by_source, direction_by_source,
            dwell_by_source, stationary_by_source, proximity_by_source, stage12,
        )
        if any(source_id not in item for item in required):
            print(f"Missing Stage 13 config or metadata for {source_id}", file=sys.stderr)
            return 1
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage13") / f"{video_id}_stage13.mp4"
        event_csv = EVENT_DIR / f"{video_id}_events.csv"
        summary_json = EVENT_DIR / f"{video_id}_summary.json"
        width = round(stage12[source_id]["width"])
        height = round(stage12[source_id]["height"])

        tracker = StatefulByteTracker(
            MODEL_PATH, device=DEVICE, imgsz=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD, tracker_config=TRACKER_CONFIG,
        )
        trajectory = TrajectoryEngine(
            max_history_length=MAX_HISTORY_LENGTH,
            minimum_displacement=MINIMUM_TRAJECTORY_DISPLACEMENT_PIXELS,
        )
        line_engine = LineCrossingEngine(
            lines_by_source[source_id], frame_width=width, frame_height=height,
            maximum_frame_gap=maximum_frame_gap,
            minimum_movement_pixels=minimum_movement,
        )
        zone_engine = ZoneEngine(zones_by_source[source_id], frame_width=width, frame_height=height)
        direction_engine = WrongWayEngine(
            direction_by_source[source_id],
            consecutive_observations=direction_consecutive,
            minimum_net_displacement=direction_displacement,
        )
        temporal_engine = TemporalRuleEngine(
            dwell_by_source[source_id], stationary_by_source[source_id],
            frame_width=width, frame_height=height,
            maximum_missing_seconds=maximum_missing,
        )
        proximity_engine = ProximityEngine(
            proximity_by_source[source_id], frame_width=width, frame_height=height,
        )
        event_engine = EventEngine(policy)

        def update_and_draw(frame: object, frame_index: int, timestamp_seconds: float, source_fps: float) -> None:
            tracked = tracker.track_frame(
                frame, video_id=video_id, source_id=source_id,
                frame_index=frame_index, timestamp_seconds=timestamp_seconds,
            )
            trajectories = trajectory.update(tracked)
            new_crossings = line_engine.update(trajectories)
            zone_observations = zone_engine.update(tracked)
            new_wrong_way = direction_engine.update(trajectories, zone_observations)
            new_dwell, new_stationary = temporal_engine.update(zone_observations, trajectories)
            new_proximity = proximity_engine.update(tracked, zone_observations)

            new_events: list[EventRecord] = []
            new_events.extend(event_engine.normalize_line_crossings(new_crossings))
            new_events.extend(event_engine.normalize_zone_transitions(zone_observations))
            new_events.extend(event_engine.normalize_wrong_way(new_wrong_way))
            new_events.extend(event_engine.normalize_long_dwell(new_dwell, frame_index=frame_index))
            new_events.extend(event_engine.normalize_stationary_vehicles(
                new_stationary,
                frame_index=frame_index,
                duration_thresholds={
                    rule.zone_id: rule.duration_seconds
                    for rule in stationary_by_source[source_id]
                },
            ))
            new_events.extend(event_engine.normalize_proximity(new_proximity))

            draw_spatial_config(frame, line_engine=line_engine, zone_engine=zone_engine, policy=policy)
            draw_trajectory_trails(frame, tracked, trajectory)
            draw_tracks(frame, tracked)
            add_overlay(frame, video_id=video_id, frame_index=frame_index, source_fps=source_fps)
            draw_event_overlay(frame, records=event_engine.records, new_events=new_events)

        print(f"Running unified event engine for {source_id}...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative, PROJECT_ROOT / output_relative,
            video_id=video_id, source_id=source_id, frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        write_event_csv(event_csv, event_engine.records)
        write_summary(summary_json, benchmark=benchmark, records=event_engine.records)
        all_events.extend(event_engine.records)

        severity_counts = Counter(item.severity for item in event_engine.records)
        prior_fps = stage12[source_id]["processing_fps"]
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
            "stage12_processing_fps": prior_fps,
            "fps_ratio_vs_stage12": round(fps / prior_fps, 4) if prior_fps > 0 else 0.0,
            "event_count": len(event_engine.records),
            "info_count": severity_counts["INFO"],
            "warning_count": severity_counts["WARNING"],
            "critical_count": severity_counts["CRITICAL"],
            "review_required_count": sum(item.status == "REVIEW_REQUIRED" for item in event_engine.records),
            "output_path": output_relative.as_posix(), "status": benchmark["status"],
            "validation_message": benchmark["validation_message"],
        })
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(event_engine.records)} events",
            flush=True,
        )

    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(benchmark_rows)
    write_event_csv(EVENTS_PATH, all_events)
    failures = sum(row["status"] == "FAIL" for row in benchmark_rows)
    warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_PATH}")
    print(f"Wrote {len(all_events)} rows to {EVENTS_PATH}")
    print(f"Summary: {len(benchmark_rows) - failures - warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
