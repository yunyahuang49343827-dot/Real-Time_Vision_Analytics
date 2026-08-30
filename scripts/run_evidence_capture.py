#!/usr/bin/env python3
"""Run Stage 14 events and capture eligible current-frame JPG evidence."""

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

from run_event_engine import draw_event_overlay, draw_spatial_config, load_runtime_sources
from vision_analytics.events.engine import EventEngine, load_event_policy
from vision_analytics.events.evidence import EvidenceCapture, load_evidence_policy
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
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes.yaml"
STAGE13_DIR = PROJECT_ROOT / "outputs" / "events" / "stage13"
STAGE13_EVENTS = STAGE13_DIR / "events.csv"
STAGE13_BENCHMARK = STAGE13_DIR / "stage13_event_benchmark.csv"
EVENT_DIR = PROJECT_ROOT / "outputs" / "events" / "stage14"
EVENTS_PATH = EVENT_DIR / "events.csv"
BENCHMARK_PATH = EVENT_DIR / "stage14_evidence_benchmark.csv"
EVIDENCE_RELATIVE_DIR = Path("outputs/evidence/stage14")
EVIDENCE_DIR = PROJECT_ROOT / EVIDENCE_RELATIVE_DIR
EVIDENCE_MANIFEST = EVIDENCE_DIR / "evidence_manifest.csv"

BENCHMARK_FIELDS = (
    "video_id", "source_id", "model", "device", "imgsz", "confidence_threshold",
    "tracker", "input_width", "input_height", "source_fps", "expected_frame_count",
    "frames_processed", "elapsed_seconds", "processing_fps", "stage13_processing_fps",
    "fps_ratio_vs_stage13", "event_count", "evidence_count", "output_path", "status",
    "validation_message",
)


def load_stage13_benchmark() -> dict[str, dict[str, float]]:
    with STAGE13_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["source_id"]: {
                "processing_fps": float(row["processing_fps"]),
                "width": float(row["input_width"]),
                "height": float(row["input_height"]),
            }
            for row in csv.DictReader(handle)
        }


def load_stage13_event_ids() -> set[str]:
    with STAGE13_EVENTS.open(newline="", encoding="utf-8") as handle:
        return {row["event_id"] for row in csv.DictReader(handle)}


def write_event_csv(path: Path, records: list[EventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)


def write_summary(
    path: Path,
    *,
    benchmark: dict[str, object],
    records: list[EventRecord],
) -> None:
    event_counts = Counter((record.event_type, record.severity) for record in records)
    capture_counts = Counter(
        record.event_type for record in records if record.evidence_path
    )
    payload = {
        "video_id": benchmark["video_id"],
        "source_id": benchmark["source_id"],
        "frames_processed": benchmark["frames_processed"],
        "event_count": len(records),
        "evidence_count": sum(capture_counts.values()),
        "event_type_severity_counts": [
            {"event_type": event_type, "severity": severity, "count": count}
            for (event_type, severity), count in sorted(event_counts.items())
        ],
        "evidence_counts": dict(sorted(capture_counts.items())),
        "evidence_warning": "Snapshots are review artifacts, not Ground Truth evidence or verified incidents.",
        "status": benchmark["status"],
        "validation_message": benchmark["validation_message"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        print("Stage 14 requires Apple MPS; CPU fallback is disabled", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_file():
        print(f"Model weights not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    if not STAGE13_EVENTS.is_file() or not STAGE13_BENCHMARK.is_file():
        print("Stage 13 artifacts are required", file=sys.stderr)
        return 1

    lines_by_source, maximum_frame_gap, minimum_movement = load_scene_config(SCENE_CONFIG)
    zones_by_source = load_zone_config(SCENE_CONFIG)
    direction_by_source, direction_consecutive, direction_displacement = load_direction_config(SCENE_CONFIG)
    dwell_by_source, stationary_by_source, maximum_missing = load_temporal_config(SCENE_CONFIG)
    proximity_by_source = load_proximity_config(SCENE_CONFIG)
    event_policy = load_event_policy(SCENE_CONFIG)
    evidence_policy = load_evidence_policy(SCENE_CONFIG)
    stage13 = load_stage13_benchmark()
    stage13_event_ids = load_stage13_event_ids()
    evidence_capture = EvidenceCapture(evidence_policy, EVIDENCE_DIR, EVIDENCE_RELATIVE_DIR)
    benchmark_rows: list[dict[str, object]] = []
    all_events: list[EventRecord] = []

    for source in load_runtime_sources():
        source_id = source["source_id"]
        required = (
            lines_by_source, zones_by_source, direction_by_source, dwell_by_source,
            stationary_by_source, proximity_by_source, stage13,
        )
        if any(source_id not in mapping for mapping in required):
            print(f"Missing Stage 14 config or metadata for {source_id}", file=sys.stderr)
            return 1

        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage14") / f"{video_id}_stage14.mp4"
        event_csv = EVENT_DIR / f"{video_id}_events.csv"
        summary_json = EVENT_DIR / f"{video_id}_summary.json"
        width = round(stage13[source_id]["width"])
        height = round(stage13[source_id]["height"])

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
            maximum_frame_gap=maximum_frame_gap, minimum_movement_pixels=minimum_movement,
        )
        zone_engine = ZoneEngine(zones_by_source[source_id], frame_width=width, frame_height=height)
        direction_engine = WrongWayEngine(
            direction_by_source[source_id], consecutive_observations=direction_consecutive,
            minimum_net_displacement=direction_displacement,
        )
        temporal_engine = TemporalRuleEngine(
            dwell_by_source[source_id], stationary_by_source[source_id],
            frame_width=width, frame_height=height, maximum_missing_seconds=maximum_missing,
        )
        proximity_engine = ProximityEngine(
            proximity_by_source[source_id], frame_width=width, frame_height=height,
        )
        event_engine = EventEngine(event_policy)
        video_events: list[EventRecord] = []
        evidence_before = len(evidence_capture.manifest_records)

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

            draw_spatial_config(frame, line_engine=line_engine, zone_engine=zone_engine, policy=event_policy)
            draw_trajectory_trails(frame, tracked, trajectory)
            draw_tracks(frame, tracked)
            add_overlay(frame, video_id=video_id, frame_index=frame_index, source_fps=source_fps)
            draw_event_overlay(frame, records=event_engine.records, new_events=new_events)
            video_events.extend(evidence_capture.capture_events(frame, new_events, tracked))

        print(f"Running evidence capture for {source_id}...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative, PROJECT_ROOT / output_relative,
            video_id=video_id, source_id=source_id, frame_processor=update_and_draw,
        )
        benchmark["output_path"] = output_relative.as_posix()
        video_evidence_count = len(evidence_capture.manifest_records) - evidence_before
        write_event_csv(event_csv, video_events)
        write_summary(summary_json, benchmark=benchmark, records=video_events)
        all_events.extend(video_events)

        prior_fps = stage13[source_id]["processing_fps"]
        current_fps = float(benchmark["processing_fps"])
        benchmark_rows.append({
            "video_id": video_id, "source_id": source_id, "model": MODEL_NAME,
            "device": DEVICE, "imgsz": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "tracker": TRACKER_CONFIG,
            "input_width": benchmark["input_width"], "input_height": benchmark["input_height"],
            "source_fps": benchmark["source_fps"],
            "expected_frame_count": benchmark["expected_frame_count"],
            "frames_processed": benchmark["frames_processed"],
            "elapsed_seconds": benchmark["elapsed_seconds"], "processing_fps": current_fps,
            "stage13_processing_fps": prior_fps,
            "fps_ratio_vs_stage13": round(current_fps / prior_fps, 4) if prior_fps > 0 else 0.0,
            "event_count": len(video_events), "evidence_count": video_evidence_count,
            "output_path": output_relative.as_posix(), "status": benchmark["status"],
            "validation_message": benchmark["validation_message"],
        })
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames, "
            f"{benchmark['processing_fps']} FPS, {len(video_events)} events, "
            f"{video_evidence_count} snapshots",
            flush=True,
        )

    stage14_event_ids = {event.event_id for event in all_events}
    if len(stage14_event_ids) != len(all_events):
        print("Duplicate Stage 14 event_id detected", file=sys.stderr)
        return 1
    if stage14_event_ids != stage13_event_ids:
        print("Stage 14 event traceability differs from Stage 13 event IDs", file=sys.stderr)
        return 1

    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(benchmark_rows)
    write_event_csv(EVENTS_PATH, all_events)
    evidence_capture.write_manifest(EVIDENCE_MANIFEST)

    failures = sum(row["status"] == "FAIL" for row in benchmark_rows)
    warnings = sum(row["status"] == "WARNING" for row in benchmark_rows)
    print(f"Wrote {len(all_events)} rows to {EVENTS_PATH}")
    print(f"Wrote {len(evidence_capture.manifest_records)} rows to {EVIDENCE_MANIFEST}")
    print(f"Summary: {len(benchmark_rows) - failures - warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
