"""Normalized polygon zones and video-scoped Track state transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from vision_analytics.tracking.schema import ALLOWED_TRACK_CLASSES, TrackRecord

ZONE_TRANSITION_FIELDS = (
    "video_id",
    "source_id",
    "zone_id",
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "class_name",
    "transition",
    "center_x",
    "center_y",
)

ZONE_STATES = frozenset({"OUTSIDE", "ENTER", "INSIDE", "EXIT"})


@dataclass(frozen=True, slots=True)
class NormalizedZone:
    zone_id: str
    zone_type: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.zone_id or not self.zone_type:
            raise ValueError("zone_id and type are required")
        if len(self.points) < 3 or len(set(self.points)) < 3:
            raise ValueError("polygon requires at least three distinct points")
        for point in self.points:
            if len(point) != 2 or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0 for value in point
            ):
                raise ValueError("normalized zone points must be within 0.0–1.0")
        area = 0.5 * abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(
                    self.points, self.points[1:] + self.points[:1], strict=True
                )
            )
        )
        if area <= 1e-12:
            raise ValueError("polygon area must be positive")

    def pixel_polygon(self, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        return np.asarray(
            [(x * (width - 1), y * (height - 1)) for x, y in self.points],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class ZoneObservation:
    video_id: str
    source_id: str
    zone_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_name: str
    transition: str
    center_x: float
    center_y: float

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "zone_id": self.zone_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "track_id": self.track_id,
            "class_name": self.class_name,
            "transition": self.transition,
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
        }


class ZoneEngine:
    """Maintain polygon membership state without synthesizing missing-track exits."""

    def __init__(
        self, zones: Sequence[NormalizedZone], *, frame_width: int, frame_height: int
    ) -> None:
        if not zones or len({zone.zone_id for zone in zones}) != len(zones):
            raise ValueError("zones are required and zone_id values must be unique")
        self.zones = tuple(zones)
        self._polygons = {
            zone.zone_id: zone.pixel_polygon(frame_width, frame_height) for zone in zones
        }
        self._inside: dict[tuple[str, str, int], bool] = {}
        self.transitions: list[ZoneObservation] = []
        self.tracks_observed_inside: dict[str, set[tuple[str, int]]] = {
            zone.zone_id: set() for zone in zones
        }
        self.current_occupancy = {zone.zone_id: 0 for zone in zones}
        self.peak_occupancy = {zone.zone_id: 0 for zone in zones}

    def update(self, records: Sequence[TrackRecord]) -> list[ZoneObservation]:
        observations: list[ZoneObservation] = []
        occupancy = {zone.zone_id: 0 for zone in self.zones}
        for record in records:
            for zone in self.zones:
                polygon = self._polygons[zone.zone_id]
                inside = cv2.pointPolygonTest(
                    polygon, (float(record.center_x), float(record.center_y)), False
                ) >= 0
                key = (record.video_id, zone.zone_id, record.track_id)
                previous = self._inside.get(key)
                if previous is None:
                    state = "INSIDE" if inside else "OUTSIDE"
                elif previous and inside:
                    state = "INSIDE"
                elif previous and not inside:
                    state = "EXIT"
                elif not previous and inside:
                    state = "ENTER"
                else:
                    state = "OUTSIDE"
                self._inside[key] = inside
                if inside:
                    occupancy[zone.zone_id] += 1
                    self.tracks_observed_inside[zone.zone_id].add(
                        (record.video_id, record.track_id)
                    )
                observation = ZoneObservation(
                    video_id=record.video_id,
                    source_id=record.source_id,
                    zone_id=zone.zone_id,
                    frame_index=record.frame_index,
                    timestamp_seconds=record.timestamp_seconds,
                    track_id=record.track_id,
                    class_name=record.class_name,
                    transition=state,
                    center_x=record.center_x,
                    center_y=record.center_y,
                )
                observations.append(observation)
                if state in {"ENTER", "EXIT"}:
                    self.transitions.append(observation)
        self.current_occupancy = occupancy
        for zone_id, count in occupancy.items():
            self.peak_occupancy[zone_id] = max(self.peak_occupancy[zone_id], count)
        return observations

    def polygon(self, zone_id: str) -> np.ndarray:
        return self._polygons[zone_id]


def load_zone_config(path: Path) -> dict[str, tuple[NormalizedZone, ...]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("scenes"), Mapping):
        raise ValueError("zone config requires a scenes mapping")
    parsed: dict[str, tuple[NormalizedZone, ...]] = {}
    for source_id, scene in data["scenes"].items():
        if not isinstance(scene, Mapping) or not isinstance(scene.get("zones"), list):
            raise ValueError(f"scene {source_id} requires a zones list")
        zones = []
        for raw in scene["zones"]:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("points"), list):
                raise ValueError(f"scene {source_id} has an invalid zone")
            zones.append(
                NormalizedZone(
                    zone_id=str(raw.get("zone_id", "")),
                    zone_type=str(raw.get("type", "")),
                    points=tuple(tuple(map(float, point)) for point in raw["points"]),
                )
            )
        if not zones or len({zone.zone_id for zone in zones}) != len(zones):
            raise ValueError(f"scene {source_id} has missing or duplicate zones")
        parsed[str(source_id)] = tuple(zones)
    return parsed
