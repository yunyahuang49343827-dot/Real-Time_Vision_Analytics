"""Normalize upstream rule records into traceable, policy-driven events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from vision_analytics.spatial.direction import WrongWayRecord
from vision_analytics.spatial.dwell import LongDwellRecord, StationaryVehicleRecord
from vision_analytics.spatial.line_crossing import CrossingRecord
from vision_analytics.spatial.proximity import ProximityRecord
from vision_analytics.spatial.zone import ZoneObservation

from .schema import EVENT_TYPES, SEVERITIES, STATUSES, EventRecord


@dataclass(frozen=True, slots=True)
class IntrusionRule:
    zone_id: str
    severity: str

    def __post_init__(self) -> None:
        if not self.zone_id or self.severity not in SEVERITIES:
            raise ValueError("intrusion rule requires zone_id and valid severity")


@dataclass(frozen=True, slots=True)
class EventPolicy:
    severity: Mapping[str, str]
    status: Mapping[str, str]
    intrusion_rules: Mapping[str, IntrusionRule]

    def __post_init__(self) -> None:
        if set(self.severity) != set(EVENT_TYPES) or set(self.status) != set(EVENT_TYPES):
            raise ValueError("event policy must configure every event type")
        if any(value not in SEVERITIES for value in self.severity.values()):
            raise ValueError("event policy contains invalid severity")
        if any(value not in STATUSES for value in self.status.values()):
            raise ValueError("event policy contains invalid status")

    def severity_for(self, event_type: str, zone_id: str | None = None) -> str:
        if event_type == "PEDESTRIAN_INTRUSION" and zone_id in self.intrusion_rules:
            return self.intrusion_rules[zone_id].severity
        return self.severity[event_type]


class EventEngine:
    """Create unique events without changing any upstream rule semantics."""

    def __init__(self, policy: EventPolicy) -> None:
        self.policy = policy
        self.records: list[EventRecord] = []
        self._seen: set[tuple[object, ...]] = set()
        self._sequence_by_video: dict[str, int] = {}

    def normalize_line_crossings(
        self, records: Sequence[CrossingRecord]
    ) -> list[EventRecord]:
        return [
            event
            for item in records
            if (event := self._emit(
                event_type="LINE_CROSSING", video_id=item.video_id,
                source_id=item.source_id, frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds, track_id=item.track_id,
                class_name=item.class_name, line_id=item.line_id,
                rule_source="spatial.line_crossing",
                rule_value=item.crossing_direction,
                threshold="finite_segment_intersection",
            )) is not None
        ]

    def normalize_zone_transitions(
        self, records: Sequence[ZoneObservation]
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        for item in records:
            if item.transition not in {"ENTER", "EXIT"}:
                continue
            event_type = "ZONE_ENTRY" if item.transition == "ENTER" else "ZONE_EXIT"
            event = self._emit(
                event_type=event_type, video_id=item.video_id, source_id=item.source_id,
                frame_index=item.frame_index, timestamp_seconds=item.timestamp_seconds,
                track_id=item.track_id, class_name=item.class_name, zone_id=item.zone_id,
                rule_source="spatial.zone", rule_value=item.transition,
                threshold="bbox_center_in_polygon",
            )
            if event is not None:
                events.append(event)
            if (
                item.transition == "ENTER"
                and item.class_name == "person"
                and item.zone_id in self.policy.intrusion_rules
            ):
                intrusion = self._emit(
                    event_type="PEDESTRIAN_INTRUSION", video_id=item.video_id,
                    source_id=item.source_id, frame_index=item.frame_index,
                    timestamp_seconds=item.timestamp_seconds, track_id=item.track_id,
                    class_name=item.class_name, zone_id=item.zone_id,
                    rule_source="events.pedestrian_intrusion",
                    rule_value="person_zone_enter",
                    threshold="restricted_for_person=true",
                )
                if intrusion is not None:
                    events.append(intrusion)
        return events

    def normalize_wrong_way(
        self, records: Sequence[WrongWayRecord]
    ) -> list[EventRecord]:
        return [
            event
            for item in records
            if (event := self._emit(
                event_type="WRONG_WAY", video_id=item.video_id,
                source_id=item.source_id, frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds, track_id=item.track_id,
                class_name=item.class_name, zone_id=item.zone_id,
                rule_source="spatial.direction", rule_value=item.observed_direction,
                threshold=(
                    f"allowed={'|'.join(item.allowed_directions)};"
                    f"consecutive={item.consecutive_violation_count}"
                ),
            )) is not None
        ]

    def normalize_long_dwell(
        self, records: Sequence[LongDwellRecord], *, frame_index: int
    ) -> list[EventRecord]:
        return [
            event
            for item in records
            if (event := self._emit(
                event_type="LONG_DWELL", video_id=item.video_id,
                source_id=item.source_id, frame_index=frame_index,
                timestamp_seconds=item.trigger_timestamp, track_id=item.track_id,
                class_name=item.class_name, zone_id=item.zone_id,
                rule_source="spatial.dwell",
                rule_value=f"observed_seconds={item.observed_dwell_seconds:.6f}",
                threshold=f"seconds>={item.threshold_seconds:g}",
            )) is not None
        ]

    def normalize_stationary_vehicles(
        self,
        records: Sequence[StationaryVehicleRecord],
        *,
        frame_index: int,
        duration_thresholds: Mapping[str, float] | None = None,
    ) -> list[EventRecord]:
        return [
            event
            for item in records
            if (event := self._emit(
                event_type="STATIONARY_VEHICLE", video_id=item.video_id,
                source_id=item.source_id, frame_index=frame_index,
                timestamp_seconds=item.trigger_timestamp, track_id=item.track_id,
                class_name=item.class_name, zone_id=item.zone_id,
                rule_source="spatial.dwell",
                rule_value=f"normalized_displacement={item.normalized_displacement:.9f}",
                threshold=(
                    f"movement<={item.movement_threshold:g};"
                    f"duration>={(duration_thresholds or {}).get(item.zone_id, item.stationary_duration_seconds):g}s"
                ),
            )) is not None
        ]

    def normalize_proximity(
        self, records: Sequence[ProximityRecord]
    ) -> list[EventRecord]:
        return [
            event
            for item in records
            if (event := self._emit(
                event_type="PROXIMITY_WARNING", video_id=item.video_id,
                source_id=item.source_id, frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds,
                track_id=item.person_track_id,
                secondary_track_id=item.vehicle_track_id,
                class_name="person", secondary_class_name=item.vehicle_class,
                zone_id=item.zone_id, rule_source="spatial.proximity",
                rule_value=f"normalized_distance={item.normalized_distance:.9f}",
                threshold=f"trigger<={item.trigger_threshold:g}",
            )) is not None
        ]

    def _emit(
        self,
        *,
        event_type: str,
        video_id: str,
        source_id: str,
        frame_index: int,
        timestamp_seconds: float,
        track_id: int | None = None,
        secondary_track_id: int | None = None,
        class_name: str | None = None,
        secondary_class_name: str | None = None,
        zone_id: str | None = None,
        line_id: str | None = None,
        rule_source: str,
        rule_value: str,
        threshold: str,
    ) -> EventRecord | None:
        key = (
            event_type, video_id, frame_index, track_id, secondary_track_id,
            zone_id, line_id, rule_source, rule_value,
        )
        if key in self._seen:
            return None
        self._seen.add(key)
        sequence = self._sequence_by_video.get(video_id, 0) + 1
        self._sequence_by_video[video_id] = sequence
        event = EventRecord(
            event_id=f"{video_id}-EVT-{sequence:06d}", video_id=video_id,
            source_id=source_id, event_type=event_type, frame_index=frame_index,
            timestamp_seconds=timestamp_seconds, track_id=track_id,
            secondary_track_id=secondary_track_id, class_name=class_name,
            secondary_class_name=secondary_class_name, zone_id=zone_id,
            line_id=line_id, severity=self.policy.severity_for(event_type, zone_id),
            status=self.policy.status[event_type], rule_source=rule_source,
            rule_value=rule_value, threshold=threshold,
        )
        self.records.append(event)
        return event


def load_event_policy(path: Path) -> EventPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_policy = data.get("event_policy") if isinstance(data, Mapping) else None
    scenes = data.get("scenes") if isinstance(data, Mapping) else None
    if not isinstance(raw_policy, Mapping) or not isinstance(scenes, Mapping):
        raise ValueError("event config requires event_policy and scenes")
    severity = {str(key): str(value) for key, value in raw_policy.get("severity", {}).items()}
    status = {str(key): str(value) for key, value in raw_policy.get("status", {}).items()}
    intrusions: dict[str, IntrusionRule] = {}
    for scene in scenes.values():
        if not isinstance(scene, Mapping):
            raise ValueError("scene config must be a mapping")
        for zone in scene.get("zones", []):
            if not zone.get("restricted_for_person", False):
                continue
            zone_id = str(zone.get("zone_id", ""))
            if zone_id in intrusions:
                raise ValueError("restricted zone_id values must be globally unique")
            intrusions[zone_id] = IntrusionRule(
                zone_id=zone_id,
                severity=str(zone.get("pedestrian_intrusion_severity", "")),
            )
    return EventPolicy(severity=severity, status=status, intrusion_rules=intrusions)
