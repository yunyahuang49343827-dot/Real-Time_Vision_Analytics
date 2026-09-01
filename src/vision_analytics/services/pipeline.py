"""Callable orchestration over existing Stage 3–15 analytics modules."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd

from vision_analytics.analytics.traffic import (
    ZONE_COLUMNS,
    build_crossing_analytics,
    build_event_summary,
    classify_density,
    load_traffic_analytics_config,
)
from vision_analytics.api.config import ApiConfig
from vision_analytics.events.engine import EventEngine, load_event_policy
from vision_analytics.events.evidence import EvidenceCapture, load_evidence_policy
from vision_analytics.events.schema import EVENT_FIELDS, EventRecord
from vision_analytics.spatial.direction import WrongWayEngine, load_direction_config
from vision_analytics.spatial.dwell import TemporalRuleEngine, load_temporal_config
from vision_analytics.spatial.line_crossing import (
    CROSSING_FIELDS,
    LineCrossingEngine,
    load_scene_config,
)
from vision_analytics.spatial.proximity import ProximityEngine, load_proximity_config
from vision_analytics.spatial.zone import ZoneEngine, load_zone_config
from vision_analytics.tracking.tracker import StatefulByteTracker
from vision_analytics.tracking.trajectory import TrajectoryEngine
from vision_analytics.video.metadata import profile_video
from vision_analytics.video.pipeline import add_overlay, process_video
from vision_analytics.services.video_delivery import transcode_browser_video
from vision_analytics.services.visualization import SupervisionVisualizer, VisualizationSettings

ProgressCallback = Callable[[float, int | None, int | None], None]


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


class ExistingAnalyticsPipeline:
    """Compose existing detection/tracking/rule engines for one uploaded video."""

    def __init__(self, config: ApiConfig) -> None:
        self.config = config

    def run(self, input_path: Path, output_directory: Path, job_id: str,
            progress_callback: ProgressCallback, *, analysis_mode: str = "standard",
            source_id: str | None = None) -> dict[str, object]:
        output_directory.mkdir(parents=True, exist_ok=True)
        source_id = source_id or self.config.source_for_analysis_mode(analysis_mode)
        metadata = profile_video(input_path, video_id=job_id, source_id=source_id)
        if metadata["validation_status"] == "FAIL":
            raise ValueError(f"input validation failed: {metadata['validation_message']}")
        width, height = int(metadata["width"]), int(metadata["height"])

        lines, maximum_gap, minimum_movement = load_scene_config(self.config.scene_config)
        zones = load_zone_config(self.config.scene_config)
        directions, direction_consecutive, direction_displacement = load_direction_config(self.config.scene_config)
        dwell, stationary, maximum_missing = load_temporal_config(self.config.scene_config)
        proximity = load_proximity_config(self.config.scene_config)
        required = (lines, zones, directions, dwell, stationary, proximity)
        if any(source_id not in mapping for mapping in required):
            raise ValueError(f"scene source_id is not fully configured: {source_id}")

        runtime_profile = self.config.runtime_profile_for(source_id)
        tracker = StatefulByteTracker(
            self.config.runtime_model, device=self.config.device, imgsz=runtime_profile.imgsz,
            confidence_threshold=runtime_profile.confidence_threshold,
            tracker_config=self.config.tracker,
        )
        trajectory = TrajectoryEngine(max_history_length=30, minimum_displacement=5.0)
        line_engine = LineCrossingEngine(
            lines[source_id], frame_width=width, frame_height=height,
            maximum_frame_gap=maximum_gap, minimum_movement_pixels=minimum_movement,
        )
        zone_engine = ZoneEngine(zones[source_id], frame_width=width, frame_height=height)
        direction_engine = WrongWayEngine(
            directions[source_id], consecutive_observations=direction_consecutive,
            minimum_net_displacement=direction_displacement,
        )
        temporal_engine = TemporalRuleEngine(
            dwell[source_id], stationary[source_id], frame_width=width, frame_height=height,
            maximum_missing_seconds=maximum_missing,
        )
        proximity_engine = ProximityEngine(proximity[source_id], frame_width=width, frame_height=height)
        event_engine = EventEngine(load_event_policy(self.config.scene_config))
        evidence_dir = output_directory / "evidence"
        evidence = EvidenceCapture(load_evidence_policy(self.config.scene_config), evidence_dir, Path("evidence"))
        captured_events: list[EventRecord] = []
        visualizer = SupervisionVisualizer(VisualizationSettings(
            heatmap_classes=self.config.heatmap_classes,
        ))
        heatmap_raw_video = output_directory / "heatmap_raw.mp4"
        heatmap_writer = cv2.VideoWriter(
            str(heatmap_raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
            float(metadata["fps"]), (width, height),
        )
        if not heatmap_writer.isOpened():
            raise RuntimeError("heatmap VideoWriter could not be opened")

        def update(frame: object, frame_index: int, timestamp: float, fps: float) -> None:
            tracks = tracker.track_frame(
                frame, video_id=job_id, source_id=source_id,
                frame_index=frame_index, timestamp_seconds=timestamp,
            )
            trajectories = trajectory.update(tracks)
            crossings = line_engine.update(trajectories)
            zone_observations = zone_engine.update(tracks)
            wrong_way = direction_engine.update(trajectories, zone_observations)
            long_dwell, stationary_events = temporal_engine.update(zone_observations, trajectories)
            proximity_events = proximity_engine.update(tracks, zone_observations)
            new_events: list[EventRecord] = []
            new_events.extend(event_engine.normalize_line_crossings(crossings))
            new_events.extend(event_engine.normalize_zone_transitions(zone_observations))
            new_events.extend(event_engine.normalize_wrong_way(wrong_way))
            new_events.extend(event_engine.normalize_long_dwell(long_dwell, frame_index=frame_index))
            new_events.extend(event_engine.normalize_stationary_vehicles(
                stationary_events, frame_index=frame_index,
                duration_thresholds={rule.zone_id: rule.duration_seconds for rule in stationary[source_id]},
            ))
            new_events.extend(event_engine.normalize_proximity(proximity_events))
            heatmap_frame = visualizer.render_heatmap(
                frame, tracks, lines=line_engine.lines, zones=zone_engine.zones, events=new_events,
            )
            add_overlay(
                heatmap_frame, video_id=job_id, frame_index=frame_index, source_fps=fps,
            )
            heatmap_writer.write(heatmap_frame)
            tracking_frame = visualizer.render_tracking(
                frame, tracks, lines=line_engine.lines, zones=zone_engine.zones, events=new_events,
            )
            add_overlay(
                tracking_frame, video_id=job_id, frame_index=frame_index, source_fps=fps,
            )
            frame[:] = tracking_frame
            captured_events.extend(evidence.capture_events(tracking_frame, new_events, tracks))

        tracking_raw_video = output_directory / "tracking_raw.mp4"
        try:
            benchmark = process_video(
                input_path, tracking_raw_video, video_id=job_id, source_id=source_id,
                frame_processor=update,
                progress_callback=lambda done, total: progress_callback(
                    done / total if total else 0.0, done, total,
                ),
            )
        finally:
            heatmap_writer.release()
        if benchmark["status"] == "FAIL":
            raise RuntimeError(str(benchmark["validation_message"]))
        if not heatmap_raw_video.is_file() or heatmap_raw_video.stat().st_size == 0:
            raise RuntimeError("heatmap visualization artifact is empty")
        tracking_delivery = transcode_browser_video(
            tracking_raw_video, output_directory / "tracking_browser.mp4",
            job_directory=output_directory,
        )
        heatmap_delivery = transcode_browser_video(
            heatmap_raw_video, output_directory / "heatmap_browser.mp4",
            job_directory=output_directory,
        )
        delivery_warnings = []
        for label, delivery in (("tracking", tracking_delivery), ("heatmap", heatmap_delivery)):
            if delivery.warning_code and delivery.warning_message:
                delivery_warnings.append({
                    "code": delivery.warning_code,
                    "message": f"{label} visualization: {delivery.warning_message}",
                })

        events_path = output_directory / "events.csv"
        _write_csv(events_path, EVENT_FIELDS, [record.to_row() for record in captured_events])
        crossings_path = output_directory / "crossings.csv"
        _write_csv(crossings_path, CROSSING_FIELDS, [record.to_row() for record in line_engine.records])
        evidence_manifest = output_directory / "evidence_manifest.csv"
        evidence.write_manifest(evidence_manifest)

        zone_rows = []
        for zone in zone_engine.zones:
            transitions = [item for item in zone_engine.transitions if item.zone_id == zone.zone_id]
            zone_rows.append({
                "video_id": job_id, "source_id": source_id, "zone_id": zone.zone_id,
                "peak_observed_occupancy": zone_engine.peak_occupancy[zone.zone_id],
                "current_observed_occupancy": zone_engine.current_occupancy[zone.zone_id],
                "tracks_observed_inside": len(zone_engine.tracks_observed_inside[zone.zone_id]),
                "zone_entry_count": sum(item.transition == "ENTER" for item in transitions),
                "zone_exit_count": sum(item.transition == "EXIT" for item in transitions),
            })
        zones_frame = pd.DataFrame(zone_rows, columns=ZONE_COLUMNS)
        crossings_frame = pd.DataFrame(
            [record.to_row() for record in line_engine.records], columns=CROSSING_FIELDS,
        )
        analytics_config = load_traffic_analytics_config(self.config.scene_config)
        summary, classes, direction, over_time = build_crossing_analytics(
            crossings_frame, config=analytics_config, zones=zones_frame,
        )
        traffic_summary_path = output_directory / "traffic_summary.csv"
        summary.to_csv(traffic_summary_path, index=False)
        class_path = output_directory / "class_distribution.csv"
        direction_path = output_directory / "direction_distribution.csv"
        over_time_path = output_directory / "traffic_over_time.csv"
        event_summary_path = output_directory / "event_summary.csv"
        classes.to_csv(class_path, index=False)
        direction.to_csv(direction_path, index=False)
        over_time.to_csv(over_time_path, index=False)
        event_frame = pd.DataFrame([record.to_row() for record in captured_events], columns=EVENT_FIELDS)
        build_event_summary(event_frame, {job_id: float(metadata["duration_seconds"])}).to_csv(
            event_summary_path, index=False,
        )

        if summary.empty:
            threshold = analytics_config.density_thresholds[source_id]
            traffic = {
                "total_line_crossing_count": 0, "person_crossing_count": 0,
                "motorized_vehicle_crossing_count": 0, "bicycle_crossing_count": 0,
                "peak_interval_start_seconds": None, "peak_interval_end_seconds": None,
                "peak_interval_count": 0,
                "zone_peak_occupancy": max(zone_engine.peak_occupancy.values(), default=0),
                "density": classify_density(max(zone_engine.peak_occupancy.values(), default=0), threshold),
                "reconciliation_status": "PASS",
            }
        else:
            traffic = summary.iloc[0].to_dict()
        grouped = Counter((item.event_type, item.severity, item.status) for item in captured_events)
        event_summary = [
            {"event_type": key[0], "severity": key[1], "status": key[2], "count": count}
            for key, count in sorted(grouped.items())
        ]
        metadata_path = output_directory / "video_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "job_id": job_id,
            "video_metadata": {
                key: metadata[key] for key in (
                    "filename", "width", "height", "fps", "frame_count",
                    "duration_seconds", "codec", "validation_status",
                )
            },
            "traffic_analytics": traffic,
            "event_summary": event_summary,
            "artifacts": {
                "processed_video": (
                    _relative(tracking_delivery.browser_path, output_directory)
                    if tracking_delivery.browser_path else None
                ),
                "processed_raw_video": _relative(tracking_raw_video, output_directory),
                "processed_browser_video": (
                    _relative(tracking_delivery.browser_path, output_directory)
                    if tracking_delivery.browser_path else None
                ),
                "tracking_raw_video": _relative(tracking_raw_video, output_directory),
                "tracking_browser_video": (
                    _relative(tracking_delivery.browser_path, output_directory)
                    if tracking_delivery.browser_path else None
                ),
                "heatmap_raw_video": _relative(heatmap_raw_video, output_directory),
                "heatmap_browser_video": (
                    _relative(heatmap_delivery.browser_path, output_directory)
                    if heatmap_delivery.browser_path else None
                ),
                "events_csv": _relative(events_path, output_directory),
                "evidence_manifest": _relative(evidence_manifest, output_directory),
                "traffic_summary_csv": _relative(traffic_summary_path, output_directory),
                "class_distribution_csv": _relative(class_path, output_directory),
                "direction_distribution_csv": _relative(direction_path, output_directory),
                "traffic_over_time_csv": _relative(over_time_path, output_directory),
                "event_summary_csv": _relative(event_summary_path, output_directory),
            },
            "warnings": delivery_warnings,
        }


class OpenCVPipelineSmokeRunner:
    """Small injectable integration runner exercising the existing video pipeline."""

    def run(self, input_path: Path, output_directory: Path, job_id: str,
            progress_callback: ProgressCallback, *, analysis_mode: str = "standard",
            source_id: str | None = None) -> dict[str, object]:
        metadata = profile_video(input_path, video_id=job_id, source_id="smoke")
        output = output_directory / "tracking_raw.mp4"
        benchmark = process_video(
            input_path, output, video_id=job_id, source_id="smoke",
            progress_callback=lambda done, total: progress_callback(
                done / total if total else 0.0, done, total,
            ),
        )
        if benchmark["status"] == "FAIL":
            raise RuntimeError(str(benchmark["validation_message"]))
        browser = output_directory / "tracking_browser.mp4"
        delivery = transcode_browser_video(output, browser, job_directory=output_directory)
        heatmap_raw = output_directory / "heatmap_raw.mp4"
        heatmap_raw.write_bytes(output.read_bytes())
        heatmap_delivery = transcode_browser_video(
            heatmap_raw, output_directory / "heatmap_browser.mp4", job_directory=output_directory,
        )
        events = output_directory / "events.csv"; _write_csv(events, EVENT_FIELDS, [])
        evidence_manifest = output_directory / "evidence_manifest.csv"
        evidence_manifest.write_text(
            "event_id,video_id,frame_index,timestamp_seconds,event_type,severity,status,evidence_path,file_size_bytes\n",
            encoding="utf-8",
        )
        traffic = output_directory / "traffic_summary.csv"
        traffic.write_text("total_line_crossing_count\n0\n", encoding="utf-8")
        return {
            "job_id": job_id,
            "video_metadata": {key: metadata[key] for key in (
                "filename", "width", "height", "fps", "frame_count", "duration_seconds",
                "codec", "validation_status",
            )},
            "traffic_analytics": {"total_line_crossing_count": 0},
            "event_summary": [],
            "artifacts": {
                "processed_video": delivery.browser_path.name if delivery.browser_path else None,
                "processed_raw_video": output.name,
                "processed_browser_video": delivery.browser_path.name if delivery.browser_path else None,
                "tracking_raw_video": output.name,
                "tracking_browser_video": delivery.browser_path.name if delivery.browser_path else None,
                "heatmap_raw_video": heatmap_raw.name,
                "heatmap_browser_video": (
                    heatmap_delivery.browser_path.name if heatmap_delivery.browser_path else None
                ),
                "events_csv": events.name,
                "evidence_manifest": evidence_manifest.name, "traffic_summary_csv": traffic.name,
            },
            "warnings": ([{
                "code": delivery.warning_code, "message": delivery.warning_message,
            }] if delivery.warning_code and delivery.warning_message else []) + ([{
                "code": heatmap_delivery.warning_code, "message": heatmap_delivery.warning_message,
            }] if heatmap_delivery.warning_code and heatmap_delivery.warning_message else []),
        }
