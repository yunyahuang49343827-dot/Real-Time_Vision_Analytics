"""Data structures for per-frame ByteTrack observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_analytics.detection.schema import BoundingBox

TRACK_FIELDS = (
    "video_id",
    "source_id",
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
)

ALLOWED_TRACK_CLASSES = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
)


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """One tracked-object observation in one video frame."""

    video_id: str
    source_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if not self.video_id or not self.source_id:
            raise ValueError("video_id and source_id are required")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.timestamp_seconds < 0 or not math.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if self.class_id < 0 or self.class_name not in ALLOWED_TRACK_CLASSES:
            raise ValueError("track class metadata is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def center_x(self) -> float:
        return (self.bbox.x1 + self.bbox.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox.y1 + self.bbox.y2) / 2.0

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "x1": round(self.bbox.x1, 3),
            "y1": round(self.bbox.y1, 3),
            "x2": round(self.bbox.x2, 3),
            "y2": round(self.bbox.y2, 3),
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
        }
