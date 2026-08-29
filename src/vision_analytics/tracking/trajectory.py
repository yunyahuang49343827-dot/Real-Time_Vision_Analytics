"""Bounded image-space trajectory history and movement features."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from vision_analytics.detection.detector import CLASS_COLORS

from .schema import ALLOWED_TRACK_CLASSES, TrackRecord

TRAJECTORY_FIELDS = (
    "video_id",
    "source_id",
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "class_name",
    "center_x",
    "center_y",
    "prev_center_x",
    "prev_center_y",
    "delta_x",
    "delta_y",
    "step_displacement",
    "net_displacement",
    "direction",
    "history_length",
    "frame_gap",
)

DIRECTIONS = frozenset(
    {
        "UP",
        "UP_RIGHT",
        "RIGHT",
        "DOWN_RIGHT",
        "DOWN",
        "DOWN_LEFT",
        "LEFT",
        "UP_LEFT",
        "STATIONARY",
    }
)


@dataclass(frozen=True, slots=True)
class TrajectoryObservation:
    """Image-space movement features for one tracked-object observation."""

    video_id: str
    source_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_name: str
    center_x: float
    center_y: float
    prev_center_x: float
    prev_center_y: float
    delta_x: float
    delta_y: float
    step_displacement: float
    net_displacement: float
    direction: str
    history_length: int
    frame_gap: int

    def __post_init__(self) -> None:
        numeric = (
            self.timestamp_seconds,
            self.center_x,
            self.center_y,
            self.prev_center_x,
            self.prev_center_y,
            self.delta_x,
            self.delta_y,
            self.step_displacement,
            self.net_displacement,
        )
        if not self.video_id or not self.source_id:
            raise ValueError("video_id and source_id are required")
        if self.frame_index < 0 or self.track_id < 0:
            raise ValueError("frame_index and track_id must be non-negative")
        if self.class_name not in ALLOWED_TRACK_CLASSES:
            raise ValueError("trajectory class is not allowed")
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("trajectory numeric values must be finite")
        if self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if self.step_displacement < 0 or self.net_displacement < 0:
            raise ValueError("displacements must be non-negative")
        if self.direction not in DIRECTIONS:
            raise ValueError("unsupported direction")
        if self.history_length <= 0 or self.frame_gap < 0:
            raise ValueError("history_length must be positive and frame_gap non-negative")

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "track_id": self.track_id,
            "class_name": self.class_name,
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
            "prev_center_x": round(self.prev_center_x, 3),
            "prev_center_y": round(self.prev_center_y, 3),
            "delta_x": round(self.delta_x, 3),
            "delta_y": round(self.delta_y, 3),
            "step_displacement": round(self.step_displacement, 3),
            "net_displacement": round(self.net_displacement, 3),
            "direction": self.direction,
            "history_length": self.history_length,
            "frame_gap": self.frame_gap,
        }


@dataclass(frozen=True, slots=True)
class _HistoryPoint:
    frame_index: int
    center_x: float
    center_y: float


def classify_direction(
    delta_x: float,
    delta_y: float,
    *,
    minimum_displacement: float = 5.0,
) -> str:
    """Classify an image-space vector into eight directions or stationary.

    Image coordinates increase to the right and downward, so negative delta_y is
    UP. The threshold applies to vector magnitude in pixels.
    """
    if minimum_displacement < 0:
        raise ValueError("minimum_displacement must be non-negative")
    displacement = math.hypot(delta_x, delta_y)
    if displacement < minimum_displacement:
        return "STATIONARY"
    angle = math.degrees(math.atan2(-delta_y, delta_x))
    sector = int(round(angle / 45.0)) % 8
    return (
        "RIGHT",
        "UP_RIGHT",
        "UP",
        "UP_LEFT",
        "LEFT",
        "DOWN_LEFT",
        "DOWN",
        "DOWN_RIGHT",
    )[sector]


class TrajectoryEngine:
    """Maintain bounded recent histories separated by video and Track ID."""

    def __init__(
        self,
        *,
        max_history_length: int = 30,
        minimum_displacement: float = 5.0,
    ) -> None:
        if max_history_length < 2:
            raise ValueError("max_history_length must be at least 2")
        if minimum_displacement < 0:
            raise ValueError("minimum_displacement must be non-negative")
        self.max_history_length = max_history_length
        self.minimum_displacement = minimum_displacement
        self._histories: dict[tuple[str, int], deque[_HistoryPoint]] = defaultdict(
            lambda: deque(maxlen=self.max_history_length)
        )

    def update(self, records: Sequence[TrackRecord]) -> list[TrajectoryObservation]:
        """Update histories for one frame and return one observation per track."""
        observations: list[TrajectoryObservation] = []
        for record in records:
            key = (record.video_id, record.track_id)
            history = self._histories[key]
            if history and record.frame_index <= history[-1].frame_index:
                raise ValueError("track observations must advance in frame order")

            previous = history[-1] if history else None
            point = _HistoryPoint(record.frame_index, record.center_x, record.center_y)
            history.append(point)
            origin = history[0]

            prev_center_x = previous.center_x if previous else point.center_x
            prev_center_y = previous.center_y if previous else point.center_y
            delta_x = point.center_x - prev_center_x
            delta_y = point.center_y - prev_center_y
            net_delta_x = point.center_x - origin.center_x
            net_delta_y = point.center_y - origin.center_y
            observations.append(
                TrajectoryObservation(
                    video_id=record.video_id,
                    source_id=record.source_id,
                    frame_index=record.frame_index,
                    timestamp_seconds=record.timestamp_seconds,
                    track_id=record.track_id,
                    class_name=record.class_name,
                    center_x=point.center_x,
                    center_y=point.center_y,
                    prev_center_x=prev_center_x,
                    prev_center_y=prev_center_y,
                    delta_x=delta_x,
                    delta_y=delta_y,
                    step_displacement=math.hypot(delta_x, delta_y),
                    net_displacement=math.hypot(net_delta_x, net_delta_y),
                    direction=classify_direction(
                        net_delta_x,
                        net_delta_y,
                        minimum_displacement=self.minimum_displacement,
                    ),
                    history_length=len(history),
                    frame_gap=(point.frame_index - previous.frame_index) if previous else 0,
                )
            )
        return observations

    def trail_points(self, video_id: str, track_id: int) -> tuple[tuple[int, int], ...]:
        """Return the current bounded trail as rounded OpenCV coordinates."""
        return tuple(
            (round(point.center_x), round(point.center_y))
            for point in self._histories.get((video_id, track_id), ())
        )


def draw_trajectory_trails(
    frame: object,
    records: Sequence[TrackRecord],
    engine: TrajectoryEngine,
) -> None:
    """Draw only each active track's bounded recent center-point trail."""
    for record in records:
        points = engine.trail_points(record.video_id, record.track_id)
        color = CLASS_COLORS[record.class_name]
        if len(points) >= 2:
            cv2.polylines(
                frame,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        if points:
            cv2.circle(frame, points[-1], 3, color, -1, cv2.LINE_AA)
