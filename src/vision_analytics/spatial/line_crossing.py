"""Finite virtual-line crossing and Track-ID-based deduplication."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from vision_analytics.tracking.schema import ALLOWED_TRACK_CLASSES
from vision_analytics.tracking.trajectory import TrajectoryObservation

CROSSING_FIELDS = (
    "video_id",
    "source_id",
    "line_id",
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "class_name",
    "crossing_direction",
    "center_x",
    "center_y",
)

CROSSING_DIRECTIONS = frozenset({"A_TO_B", "B_TO_A"})


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (self.x, self.y)):
            raise ValueError("normalized coordinates must be finite and within 0.0–1.0")

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[float, float]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return self.x * (frame_width - 1), self.y * (frame_height - 1)


@dataclass(frozen=True, slots=True)
class CountingLine:
    line_id: str
    start: NormalizedPoint
    end: NormalizedPoint

    def __post_init__(self) -> None:
        if not self.line_id:
            raise ValueError("line_id is required")
        if self.start == self.end:
            raise ValueError("counting line endpoints must be distinct")

    def pixel_endpoints(
        self, frame_width: int, frame_height: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            self.start.to_pixels(frame_width, frame_height),
            self.end.to_pixels(frame_width, frame_height),
        )


@dataclass(frozen=True, slots=True)
class CrossingRecord:
    video_id: str
    source_id: str
    line_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_name: str
    crossing_direction: str
    center_x: float
    center_y: float

    def __post_init__(self) -> None:
        numeric = (self.timestamp_seconds, self.center_x, self.center_y)
        if not self.video_id or not self.source_id or not self.line_id:
            raise ValueError("crossing identifiers are required")
        if self.frame_index < 0 or self.track_id < 0:
            raise ValueError("frame_index and track_id must be non-negative")
        if self.timestamp_seconds < 0 or not all(math.isfinite(value) for value in numeric):
            raise ValueError("crossing coordinates and timestamp must be finite")
        if self.class_name not in ALLOWED_TRACK_CLASSES:
            raise ValueError("crossing class is not allowed")
        if self.crossing_direction not in CROSSING_DIRECTIONS:
            raise ValueError("unsupported crossing direction")

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "line_id": self.line_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "track_id": self.track_id,
            "class_name": self.class_name,
            "crossing_direction": self.crossing_direction,
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
        }


def _cross(
    origin: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    return (end[0] - origin[0]) * (point[1] - origin[1]) - (
        end[1] - origin[1]
    ) * (point[0] - origin[0])


def finite_segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    """Return whether two finite, non-collinear segments intersect."""
    first_a = _cross(first_start, first_end, second_start)
    first_b = _cross(first_start, first_end, second_end)
    second_a = _cross(second_start, second_end, first_start)
    second_b = _cross(second_start, second_end, first_end)
    return (
        first_a * first_b <= epsilon
        and second_a * second_b <= epsilon
        and not (
            abs(first_a) <= epsilon
            and abs(first_b) <= epsilon
            and abs(second_a) <= epsilon
            and abs(second_b) <= epsilon
        )
    )


def crossing_direction(
    previous: tuple[float, float],
    current: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    *,
    epsilon: float = 1e-9,
) -> str | None:
    """Classify strict positive-side to negative-side crossing or its inverse."""
    previous_side = _cross(line_start, line_end, previous)
    current_side = _cross(line_start, line_end, current)
    if previous_side > epsilon and current_side < -epsilon:
        return "A_TO_B"
    if previous_side < -epsilon and current_side > epsilon:
        return "B_TO_A"
    return None


class LineCrossingEngine:
    """Detect and deduplicate finite line crossings for trajectory observations."""

    def __init__(
        self,
        lines: Sequence[CountingLine],
        *,
        frame_width: int,
        frame_height: int,
        maximum_frame_gap: int = 5,
        minimum_movement_pixels: float = 3.0,
    ) -> None:
        if not lines:
            raise ValueError("at least one counting line is required")
        if len({line.line_id for line in lines}) != len(lines):
            raise ValueError("line_id values must be unique")
        if maximum_frame_gap <= 0:
            raise ValueError("maximum_frame_gap must be positive")
        if minimum_movement_pixels < 0:
            raise ValueError("minimum_movement_pixels must be non-negative")
        self.lines = tuple(lines)
        self.maximum_frame_gap = maximum_frame_gap
        self.minimum_movement_pixels = minimum_movement_pixels
        self._pixel_lines = {
            line.line_id: line.pixel_endpoints(frame_width, frame_height)
            for line in self.lines
        }
        self._counted_keys: set[tuple[str, str, int]] = set()
        self.records: list[CrossingRecord] = []

    def update(
        self, observations: Sequence[TrajectoryObservation]
    ) -> list[CrossingRecord]:
        new_records: list[CrossingRecord] = []
        for observation in observations:
            if (
                observation.history_length < 2
                or observation.frame_gap <= 0
                or observation.frame_gap > self.maximum_frame_gap
                or observation.step_displacement < self.minimum_movement_pixels
            ):
                continue
            previous = (observation.prev_center_x, observation.prev_center_y)
            current = (observation.center_x, observation.center_y)
            for line in self.lines:
                key = (observation.video_id, line.line_id, observation.track_id)
                if key in self._counted_keys:
                    continue
                line_start, line_end = self._pixel_lines[line.line_id]
                direction = crossing_direction(previous, current, line_start, line_end)
                if direction is None or not finite_segments_intersect(
                    previous, current, line_start, line_end
                ):
                    continue
                record = CrossingRecord(
                    video_id=observation.video_id,
                    source_id=observation.source_id,
                    line_id=line.line_id,
                    frame_index=observation.frame_index,
                    timestamp_seconds=observation.timestamp_seconds,
                    track_id=observation.track_id,
                    class_name=observation.class_name,
                    crossing_direction=direction,
                    center_x=observation.center_x,
                    center_y=observation.center_y,
                )
                self._counted_keys.add(key)
                self.records.append(record)
                new_records.append(record)
        return new_records

    def pixel_line(
        self, line_id: str
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._pixel_lines[line_id]

    def count_for_line(self, line_id: str) -> int:
        return sum(record.line_id == line_id for record in self.records)


def _parse_point(value: object, field_name: str) -> NormalizedPoint:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-value list")
    return NormalizedPoint(float(value[0]), float(value[1]))


def load_scene_config(
    path: Path,
) -> tuple[dict[str, tuple[CountingLine, ...]], int, float]:
    """Load and validate normalized scene lines and simple crossing tolerances."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("scene config must be a mapping")
    defaults = data.get("defaults", {})
    scenes = data.get("scenes")
    if not isinstance(defaults, Mapping) or not isinstance(scenes, Mapping):
        raise ValueError("scene config requires defaults and scenes mappings")
    maximum_frame_gap = int(defaults.get("maximum_frame_gap", 5))
    minimum_movement_pixels = float(defaults.get("minimum_movement_pixels", 3.0))
    parsed: dict[str, tuple[CountingLine, ...]] = {}
    for source_id, scene in scenes.items():
        if not isinstance(scene, Mapping) or not isinstance(scene.get("lines"), list):
            raise ValueError(f"scene {source_id} requires a lines list")
        lines_list = []
        for raw_line in scene["lines"]:
            if not isinstance(raw_line, Mapping):
                raise ValueError(f"scene {source_id} line entries must be mappings")
            lines_list.append(
                CountingLine(
                    line_id=str(raw_line.get("line_id", "")),
                    start=_parse_point(raw_line.get("start"), "start"),
                    end=_parse_point(raw_line.get("end"), "end"),
                )
            )
        lines = tuple(lines_list)
        if not lines:
            raise ValueError(f"scene {source_id} requires at least one valid line")
        if len({line.line_id for line in lines}) != len(lines):
            raise ValueError(f"scene {source_id} contains duplicate line_id values")
        parsed[str(source_id)] = lines
    if maximum_frame_gap <= 0 or minimum_movement_pixels < 0:
        raise ValueError("crossing tolerances are invalid")
    return parsed, maximum_frame_gap, minimum_movement_pixels
