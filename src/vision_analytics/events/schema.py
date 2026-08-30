"""Unified schema for rule-generated vision analytics events."""

from __future__ import annotations

import math
from dataclasses import dataclass

EVENT_TYPES = frozenset({
    "LINE_CROSSING",
    "ZONE_ENTRY",
    "ZONE_EXIT",
    "WRONG_WAY",
    "LONG_DWELL",
    "STATIONARY_VEHICLE",
    "PEDESTRIAN_INTRUSION",
    "PROXIMITY_WARNING",
})
SEVERITIES = frozenset({"INFO", "WARNING", "CRITICAL"})
STATUSES = frozenset({"DETECTED", "REVIEW_REQUIRED", "CONFIRMED"})

EVENT_FIELDS = (
    "event_id", "video_id", "source_id", "event_type", "frame_index",
    "timestamp_seconds", "track_id", "secondary_track_id", "class_name",
    "secondary_class_name", "zone_id", "line_id", "severity", "status",
    "rule_source", "rule_value", "threshold",
)


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: str
    video_id: str
    source_id: str
    event_type: str
    frame_index: int
    timestamp_seconds: float
    track_id: int | None = None
    secondary_track_id: int | None = None
    class_name: str | None = None
    secondary_class_name: str | None = None
    zone_id: str | None = None
    line_id: str | None = None
    severity: str = "INFO"
    status: str = "DETECTED"
    rule_source: str = ""
    rule_value: str = ""
    threshold: str = ""

    def __post_init__(self) -> None:
        if not self.event_id or not self.video_id or not self.source_id:
            raise ValueError("event identifiers are required")
        if self.event_type not in EVENT_TYPES:
            raise ValueError("unsupported event_type")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.timestamp_seconds < 0 or not math.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("track_id must be non-negative when present")
        if self.secondary_track_id is not None and self.secondary_track_id < 0:
            raise ValueError("secondary_track_id must be non-negative when present")
        if self.severity not in SEVERITIES:
            raise ValueError("unsupported severity")
        if self.status not in STATUSES:
            raise ValueError("unsupported status")
        if not self.rule_source:
            raise ValueError("rule_source is required for traceability")

    def to_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "source_id": self.source_id,
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "track_id": "" if self.track_id is None else self.track_id,
            "secondary_track_id": "" if self.secondary_track_id is None else self.secondary_track_id,
            "class_name": self.class_name or "",
            "secondary_class_name": self.secondary_class_name or "",
            "zone_id": self.zone_id or "",
            "line_id": self.line_id or "",
            "severity": self.severity,
            "status": self.status,
            "rule_source": self.rule_source,
            "rule_value": self.rule_value,
            "threshold": self.threshold,
        }
