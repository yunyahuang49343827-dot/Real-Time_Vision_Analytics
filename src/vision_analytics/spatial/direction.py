"""Config-driven consecutive direction monitoring inside polygon zones."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from vision_analytics.tracking.schema import ALLOWED_TRACK_CLASSES
from vision_analytics.tracking.trajectory import DIRECTIONS, TrajectoryObservation

from .zone import ZoneObservation

WRONG_WAY_FIELDS = (
    "video_id", "source_id", "zone_id", "frame_index", "timestamp_seconds",
    "track_id", "class_name", "observed_direction", "allowed_directions",
    "consecutive_violation_count", "net_displacement",
)


@dataclass(frozen=True, slots=True)
class DirectionRule:
    zone_id: str
    allowed_directions: frozenset[str]
    applicable_classes: frozenset[str]

    def __post_init__(self) -> None:
        moving = DIRECTIONS - {"STATIONARY"}
        if not self.zone_id or not self.allowed_directions or not self.applicable_classes:
            raise ValueError("direction rule fields must not be empty")
        if not self.allowed_directions <= moving:
            raise ValueError("allowed_directions contains unsupported values")
        if not self.applicable_classes <= ALLOWED_TRACK_CLASSES:
            raise ValueError("applicable_classes contains unsupported values")


@dataclass(frozen=True, slots=True)
class WrongWayRecord:
    video_id: str
    source_id: str
    zone_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_name: str
    observed_direction: str
    allowed_directions: tuple[str, ...]
    consecutive_violation_count: int
    net_displacement: float

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id, "source_id": self.source_id,
            "zone_id": self.zone_id, "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6), "track_id": self.track_id,
            "class_name": self.class_name, "observed_direction": self.observed_direction,
            "allowed_directions": "|".join(self.allowed_directions),
            "consecutive_violation_count": self.consecutive_violation_count,
            "net_displacement": round(self.net_displacement, 3),
        }


class WrongWayEngine:
    """Confirm persistent disallowed movement once per video/zone/Track."""

    def __init__(
        self, rules: Sequence[DirectionRule], *, consecutive_observations: int = 8,
        minimum_net_displacement: float = 20.0,
    ) -> None:
        if len({rule.zone_id for rule in rules}) != len(rules):
            raise ValueError("direction rule zone_id values must be unique")
        if consecutive_observations <= 0 or minimum_net_displacement < 0:
            raise ValueError("wrong-way thresholds are invalid")
        self.rules = {rule.zone_id: rule for rule in rules}
        self.consecutive_observations = consecutive_observations
        self.minimum_net_displacement = minimum_net_displacement
        self._streaks: dict[tuple[str, str, int], int] = {}
        self._confirmed: set[tuple[str, str, int]] = set()
        self.records: list[WrongWayRecord] = []

    def is_confirmed(self, video_id: str, zone_id: str, track_id: int) -> bool:
        """Return whether this video/zone/Track key has already been confirmed."""
        return (video_id, zone_id, track_id) in self._confirmed

    def update(
        self, trajectories: Sequence[TrajectoryObservation],
        zone_observations: Sequence[ZoneObservation],
    ) -> list[WrongWayRecord]:
        trajectory_by_track = {(item.video_id, item.track_id): item for item in trajectories}
        new_records: list[WrongWayRecord] = []
        for zone_item in zone_observations:
            rule = self.rules.get(zone_item.zone_id)
            if rule is None:
                continue
            key = (zone_item.video_id, zone_item.zone_id, zone_item.track_id)
            trajectory = trajectory_by_track.get((zone_item.video_id, zone_item.track_id))
            inside = zone_item.transition in {"ENTER", "INSIDE"}
            violates = (
                inside
                and trajectory is not None
                and trajectory.frame_gap == 1
                and zone_item.class_name in rule.applicable_classes
                and trajectory.direction != "STATIONARY"
                and trajectory.net_displacement >= self.minimum_net_displacement
                and trajectory.direction not in rule.allowed_directions
            )
            if not violates:
                self._streaks[key] = 0
                continue
            count = self._streaks.get(key, 0) + 1
            self._streaks[key] = count
            if count < self.consecutive_observations or key in self._confirmed:
                continue
            record = WrongWayRecord(
                video_id=zone_item.video_id, source_id=zone_item.source_id,
                zone_id=zone_item.zone_id, frame_index=zone_item.frame_index,
                timestamp_seconds=zone_item.timestamp_seconds, track_id=zone_item.track_id,
                class_name=zone_item.class_name, observed_direction=trajectory.direction,
                allowed_directions=tuple(sorted(rule.allowed_directions)),
                consecutive_violation_count=count, net_displacement=trajectory.net_displacement,
            )
            self._confirmed.add(key); self.records.append(record); new_records.append(record)
        return new_records


def load_direction_config(path: Path) -> tuple[dict[str, tuple[DirectionRule, ...]], int, float]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("scenes"), Mapping):
        raise ValueError("direction config requires scenes")
    defaults = data.get("defaults", {})
    consecutive = int(defaults.get("wrong_way_consecutive_observations", 8))
    displacement = float(defaults.get("wrong_way_minimum_net_displacement_pixels", 20.0))
    parsed = {}
    for source_id, scene in data["scenes"].items():
        rules = []
        for zone in scene.get("zones", []):
            if "allowed_directions" not in zone:
                continue
            rules.append(DirectionRule(
                zone_id=str(zone.get("zone_id", "")),
                allowed_directions=frozenset(map(str, zone.get("allowed_directions", []))),
                applicable_classes=frozenset(map(str, zone.get("applicable_classes", []))),
            ))
        parsed[str(source_id)] = tuple(rules)
    if consecutive <= 0 or displacement < 0:
        raise ValueError("wrong-way defaults are invalid")
    return parsed, consecutive, displacement
