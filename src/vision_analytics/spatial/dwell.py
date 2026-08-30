"""Observed dwell episodes and normalized image-space stationary rules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from vision_analytics.tracking.schema import ALLOWED_TRACK_CLASSES
from vision_analytics.tracking.trajectory import TrajectoryObservation

from .zone import ZoneObservation

VEHICLE_CLASSES = frozenset({"bicycle", "car", "motorcycle", "bus", "truck"})

LONG_DWELL_FIELDS = (
    "video_id", "source_id", "zone_id", "track_id", "class_name",
    "first_inside_timestamp", "trigger_timestamp", "observed_dwell_seconds",
    "threshold_seconds",
)

STATIONARY_VEHICLE_FIELDS = (
    "video_id", "source_id", "zone_id", "track_id", "class_name",
    "stationary_start_timestamp", "trigger_timestamp", "stationary_duration_seconds",
    "normalized_displacement", "movement_threshold",
)


@dataclass(frozen=True, slots=True)
class DwellRule:
    zone_id: str
    threshold_seconds: float
    applicable_classes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.zone_id or not math.isfinite(self.threshold_seconds) or self.threshold_seconds <= 0:
            raise ValueError("dwell rule requires a zone and positive threshold")
        if not self.applicable_classes or not self.applicable_classes <= ALLOWED_TRACK_CLASSES:
            raise ValueError("dwell applicable_classes are invalid")


@dataclass(frozen=True, slots=True)
class StationaryRule:
    zone_id: str
    duration_seconds: float
    movement_threshold: float
    applicable_classes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.zone_id or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("stationary rule requires a zone and positive duration")
        if (
            not math.isfinite(self.movement_threshold)
            or self.movement_threshold < 0
            or self.movement_threshold > 1
        ):
            raise ValueError("stationary movement threshold must be normalized to 0.0–1.0")
        if not self.applicable_classes or not self.applicable_classes <= VEHICLE_CLASSES:
            raise ValueError("stationary applicable_classes must contain only vehicle classes")


@dataclass(frozen=True, slots=True)
class LongDwellRecord:
    video_id: str
    source_id: str
    zone_id: str
    track_id: int
    class_name: str
    first_inside_timestamp: float
    trigger_timestamp: float
    observed_dwell_seconds: float
    threshold_seconds: float

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "zone_id": self.zone_id,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_inside_timestamp": round(self.first_inside_timestamp, 6),
            "trigger_timestamp": round(self.trigger_timestamp, 6),
            "observed_dwell_seconds": round(self.observed_dwell_seconds, 6),
            "threshold_seconds": self.threshold_seconds,
        }


@dataclass(frozen=True, slots=True)
class StationaryVehicleRecord:
    video_id: str
    source_id: str
    zone_id: str
    track_id: int
    class_name: str
    stationary_start_timestamp: float
    trigger_timestamp: float
    stationary_duration_seconds: float
    normalized_displacement: float
    movement_threshold: float

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "zone_id": self.zone_id,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "stationary_start_timestamp": round(self.stationary_start_timestamp, 6),
            "trigger_timestamp": round(self.trigger_timestamp, 6),
            "stationary_duration_seconds": round(self.stationary_duration_seconds, 6),
            "normalized_displacement": round(self.normalized_displacement, 9),
            "movement_threshold": self.movement_threshold,
        }


@dataclass(slots=True)
class _Episode:
    start_timestamp: float
    last_seen_timestamp: float
    triggered: bool = False


class TemporalRuleEngine:
    """Maintain separate observed dwell and stationary episodes per Track/zone."""

    def __init__(
        self,
        dwell_rules: Sequence[DwellRule],
        stationary_rules: Sequence[StationaryRule],
        *,
        frame_width: int,
        frame_height: int,
        maximum_missing_seconds: float = 1.0,
    ) -> None:
        if len({rule.zone_id for rule in dwell_rules}) != len(dwell_rules):
            raise ValueError("dwell rule zone_id values must be unique")
        if len({rule.zone_id for rule in stationary_rules}) != len(stationary_rules):
            raise ValueError("stationary rule zone_id values must be unique")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not math.isfinite(maximum_missing_seconds) or maximum_missing_seconds < 0:
            raise ValueError("maximum_missing_seconds must be finite and non-negative")
        self.dwell_rules = {rule.zone_id: rule for rule in dwell_rules}
        self.stationary_rules = {rule.zone_id: rule for rule in stationary_rules}
        self.maximum_missing_seconds = maximum_missing_seconds
        self.frame_diagonal = math.hypot(frame_width, frame_height)
        self._dwell_episodes: dict[tuple[str, str, int], _Episode] = {}
        self._stationary_episodes: dict[tuple[str, str, int], _Episode] = {}
        self.long_dwell_records: list[LongDwellRecord] = []
        self.stationary_vehicle_records: list[StationaryVehicleRecord] = []

    def update(
        self,
        zone_observations: Sequence[ZoneObservation],
        trajectories: Sequence[TrajectoryObservation],
    ) -> tuple[list[LongDwellRecord], list[StationaryVehicleRecord]]:
        trajectory_by_track = {
            (item.video_id, item.track_id): item for item in trajectories
        }
        new_dwell: list[LongDwellRecord] = []
        new_stationary: list[StationaryVehicleRecord] = []
        for item in zone_observations:
            key = (item.video_id, item.zone_id, item.track_id)
            dwell_rule = self.dwell_rules.get(item.zone_id)
            if dwell_rule is not None:
                record = self._update_dwell(key, item, dwell_rule)
                if record is not None:
                    self.long_dwell_records.append(record)
                    new_dwell.append(record)

            stationary_rule = self.stationary_rules.get(item.zone_id)
            if stationary_rule is not None:
                trajectory = trajectory_by_track.get((item.video_id, item.track_id))
                record = self._update_stationary(key, item, trajectory, stationary_rule)
                if record is not None:
                    self.stationary_vehicle_records.append(record)
                    new_stationary.append(record)
        return new_dwell, new_stationary

    def _update_dwell(
        self,
        key: tuple[str, str, int],
        item: ZoneObservation,
        rule: DwellRule,
    ) -> LongDwellRecord | None:
        inside = item.transition in {"ENTER", "INSIDE"}
        if not inside or item.class_name not in rule.applicable_classes:
            if item.transition == "EXIT" or item.class_name not in rule.applicable_classes:
                self._dwell_episodes.pop(key, None)
            return None

        episode = self._dwell_episodes.get(key)
        if (
            episode is None
            or item.transition == "ENTER"
            or item.timestamp_seconds - episode.last_seen_timestamp > self.maximum_missing_seconds
        ):
            episode = _Episode(item.timestamp_seconds, item.timestamp_seconds)
            self._dwell_episodes[key] = episode
        else:
            episode.last_seen_timestamp = item.timestamp_seconds

        observed_seconds = item.timestamp_seconds - episode.start_timestamp
        if observed_seconds < rule.threshold_seconds or episode.triggered:
            return None
        episode.triggered = True
        return LongDwellRecord(
            video_id=item.video_id,
            source_id=item.source_id,
            zone_id=item.zone_id,
            track_id=item.track_id,
            class_name=item.class_name,
            first_inside_timestamp=episode.start_timestamp,
            trigger_timestamp=item.timestamp_seconds,
            observed_dwell_seconds=observed_seconds,
            threshold_seconds=rule.threshold_seconds,
        )

    def _update_stationary(
        self,
        key: tuple[str, str, int],
        item: ZoneObservation,
        trajectory: TrajectoryObservation | None,
        rule: StationaryRule,
    ) -> StationaryVehicleRecord | None:
        inside = item.transition in {"ENTER", "INSIDE"}
        valid = (
            inside
            and item.class_name in rule.applicable_classes
            and trajectory is not None
            and trajectory.frame_gap <= 1
        )
        normalized_displacement = (
            trajectory.net_displacement / self.frame_diagonal if trajectory is not None else math.inf
        )
        low_movement = valid and normalized_displacement <= rule.movement_threshold
        episode = self._stationary_episodes.get(key)
        excessive_gap = (
            episode is not None
            and item.timestamp_seconds - episode.last_seen_timestamp > self.maximum_missing_seconds
        )
        if not low_movement or excessive_gap:
            self._stationary_episodes.pop(key, None)
            if not low_movement:
                return None
            episode = None

        if episode is None:
            episode = _Episode(item.timestamp_seconds, item.timestamp_seconds)
            self._stationary_episodes[key] = episode
        else:
            episode.last_seen_timestamp = item.timestamp_seconds

        stationary_seconds = item.timestamp_seconds - episode.start_timestamp
        if stationary_seconds < rule.duration_seconds or episode.triggered:
            return None
        episode.triggered = True
        return StationaryVehicleRecord(
            video_id=item.video_id,
            source_id=item.source_id,
            zone_id=item.zone_id,
            track_id=item.track_id,
            class_name=item.class_name,
            stationary_start_timestamp=episode.start_timestamp,
            trigger_timestamp=item.timestamp_seconds,
            stationary_duration_seconds=stationary_seconds,
            normalized_displacement=normalized_displacement,
            movement_threshold=rule.movement_threshold,
        )


def load_temporal_config(
    path: Path,
) -> tuple[dict[str, tuple[DwellRule, ...]], dict[str, tuple[StationaryRule, ...]], float]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("scenes"), Mapping):
        raise ValueError("temporal config requires scenes")
    defaults = data.get("defaults", {})
    maximum_missing = float(defaults.get("temporal_maximum_missing_seconds", 1.0))
    if not math.isfinite(maximum_missing) or maximum_missing < 0:
        raise ValueError("temporal maximum missing seconds is invalid")

    dwell_by_source: dict[str, tuple[DwellRule, ...]] = {}
    stationary_by_source: dict[str, tuple[StationaryRule, ...]] = {}
    for source_id, scene in data["scenes"].items():
        if not isinstance(scene, Mapping) or not isinstance(scene.get("zones"), list):
            raise ValueError(f"scene {source_id} requires zones")
        dwell_rules = []
        stationary_rules = []
        for zone in scene["zones"]:
            if "dwell_threshold_seconds" in zone:
                dwell_rules.append(DwellRule(
                    zone_id=str(zone.get("zone_id", "")),
                    threshold_seconds=float(zone["dwell_threshold_seconds"]),
                    applicable_classes=frozenset(map(str, zone.get("dwell_applicable_classes", []))),
                ))
            if "stationary_duration_seconds" in zone:
                if zone.get("type") != "stationary_monitoring":
                    raise ValueError("stationary rule requires a stationary_monitoring zone")
                stationary_rules.append(StationaryRule(
                    zone_id=str(zone.get("zone_id", "")),
                    duration_seconds=float(zone["stationary_duration_seconds"]),
                    movement_threshold=float(zone.get("stationary_movement_threshold_normalized", -1)),
                    applicable_classes=frozenset(map(str, zone.get("stationary_applicable_classes", []))),
                ))
        dwell_by_source[str(source_id)] = tuple(dwell_rules)
        stationary_by_source[str(source_id)] = tuple(stationary_rules)
    return dwell_by_source, stationary_by_source, maximum_missing
