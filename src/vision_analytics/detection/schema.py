"""Data structures for pretrained detection occurrences."""

from __future__ import annotations

import math
from dataclasses import dataclass

DETECTION_FIELDS = (
    "video_id",
    "source_id",
    "frame_index",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding-box maximums must not be below minimums")


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    video_id: str
    source_id: str
    frame_index: int
    timestamp_seconds: float
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.timestamp_seconds < 0 or not math.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self.class_id < 0 or not self.class_name:
            raise ValueError("class metadata is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "x1": round(self.bbox.x1, 3),
            "y1": round(self.bbox.y1, 3),
            "x2": round(self.bbox.x2, 3),
            "y2": round(self.bbox.y2, 3),
        }
