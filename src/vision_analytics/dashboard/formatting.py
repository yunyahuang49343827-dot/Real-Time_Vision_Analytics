"""Pure display formatting and governed interpretation wording."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SUPPORTED_STATUSES = frozenset({"CREATED", "PROCESSING", "COMPLETED", "FAILED"})
PROXIMITY_WORDING = "Image-space proximity warning — review candidate only."
WRONG_WAY_REVIEW_WORDING = "Wrong-way rule candidate — human review required."
COUNTING_WORDING = "Track-based line-crossing count; not a complete traffic census."


def normalize_progress(value: object) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    return round(min(1.0, max(0.0, number)) * 100)


def format_timestamp(seconds: object) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value) or value < 0:
        return "—"
    minutes, remainder = divmod(value, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def status_label(status: str) -> str:
    if status not in SUPPORTED_STATUSES:
        return "UNKNOWN"
    return {
        "CREATED": "Created — waiting for a worker",
        "PROCESSING": "Processing video analytics",
        "COMPLETED": "Analysis completed",
        "FAILED": "Analysis failed",
    }[status]


def event_interpretation(event_type: str, status: str) -> str:
    if event_type == "PROXIMITY_WARNING":
        return PROXIMITY_WORDING
    if event_type == "WRONG_WAY" and status == "REVIEW_REQUIRED":
        return WRONG_WAY_REVIEW_WORDING
    return "Rule-generated system event; review context before operational use."


def event_table_rows(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [{
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "severity": event.get("severity"),
        "status": event.get("status"),
        "frame_id": event.get("frame_index"),
        "timestamp": format_timestamp(event.get("timestamp_seconds")),
        "primary_track_id": event.get("track_id"),
        "interpretation": event_interpretation(
            str(event.get("event_type", "")), str(event.get("status", "")),
        ),
    } for event in events]
