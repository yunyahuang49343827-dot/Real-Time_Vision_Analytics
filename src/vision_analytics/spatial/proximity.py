"""Zone-filtered normalized image-space person–vehicle proximity heuristic."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from vision_analytics.detection.schema import BoundingBox
from vision_analytics.tracking.schema import TrackRecord

from .zone import ZoneObservation

VEHICLE_CLASSES = frozenset({"bicycle", "car", "motorcycle", "bus", "truck"})
RIDER_VEHICLE_CLASSES = frozenset({"bicycle", "motorcycle"})

PROXIMITY_FIELDS = (
    "video_id", "source_id", "zone_id", "frame_index", "timestamp_seconds",
    "person_track_id", "vehicle_track_id", "vehicle_class", "normalized_distance",
    "trigger_threshold", "consecutive_observations",
)


def bbox_distance(first: BoundingBox, second: BoundingBox) -> float:
    """Return minimum Euclidean gap between two axis-aligned bboxes in pixels."""
    gap_x = max(first.x1 - second.x2, second.x1 - first.x2, 0.0)
    gap_y = max(first.y1 - second.y2, second.y1 - first.y2, 0.0)
    return math.hypot(gap_x, gap_y)


def normalized_bbox_distance(
    first: BoundingBox,
    second: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    return bbox_distance(first, second) / math.hypot(frame_width, frame_height)


def rider_self_vehicle_overlap(
    person_bbox: BoundingBox,
    vehicle_bbox: BoundingBox,
    *,
    vehicle_class: str,
    overlap_threshold: float,
) -> bool:
    """Heuristically exclude an overlapping person and their likely two-wheeler."""
    if vehicle_class not in RIDER_VEHICLE_CLASSES:
        return False
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be within 0.0–1.0")
    intersection_width = max(
        0.0, min(person_bbox.x2, vehicle_bbox.x2) - max(person_bbox.x1, vehicle_bbox.x1)
    )
    intersection_height = max(
        0.0, min(person_bbox.y2, vehicle_bbox.y2) - max(person_bbox.y1, vehicle_bbox.y1)
    )
    intersection_area = intersection_width * intersection_height
    person_area = (person_bbox.x2 - person_bbox.x1) * (person_bbox.y2 - person_bbox.y1)
    vehicle_area = (vehicle_bbox.x2 - vehicle_bbox.x1) * (vehicle_bbox.y2 - vehicle_bbox.y1)
    smaller_area = min(person_area, vehicle_area)
    overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0.0
    person_bottom_center = ((person_bbox.x1 + person_bbox.x2) / 2.0, person_bbox.y2)
    bottom_center_contained = (
        vehicle_bbox.x1 <= person_bottom_center[0] <= vehicle_bbox.x2
        and vehicle_bbox.y1 <= person_bottom_center[1] <= vehicle_bbox.y2
    )
    return overlap_ratio >= overlap_threshold or bottom_center_contained


@dataclass(frozen=True, slots=True)
class ProximityRule:
    zone_id: str
    enabled: bool
    vehicle_classes: frozenset[str] = frozenset()
    trigger_threshold: float = 0.0
    release_threshold: float = 0.0
    minimum_consecutive_observations: int = 1
    rider_overlap_exclusion_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("proximity rule requires zone_id")
        if not self.enabled:
            return
        if not self.vehicle_classes or not self.vehicle_classes <= VEHICLE_CLASSES:
            raise ValueError("proximity vehicle_classes are invalid")
        if not (
            math.isfinite(self.trigger_threshold)
            and math.isfinite(self.release_threshold)
            and 0.0 <= self.trigger_threshold < self.release_threshold <= 1.0
        ):
            raise ValueError("proximity requires trigger < release normalized thresholds")
        if self.minimum_consecutive_observations <= 0:
            raise ValueError("minimum consecutive observations must be positive")
        if not 0.0 <= self.rider_overlap_exclusion_ratio <= 1.0:
            raise ValueError("rider overlap ratio must be within 0.0–1.0")


@dataclass(frozen=True, slots=True)
class ProximityRecord:
    video_id: str
    source_id: str
    zone_id: str
    frame_index: int
    timestamp_seconds: float
    person_track_id: int
    vehicle_track_id: int
    vehicle_class: str
    normalized_distance: float
    trigger_threshold: float
    consecutive_observations: int

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "zone_id": self.zone_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "person_track_id": self.person_track_id,
            "vehicle_track_id": self.vehicle_track_id,
            "vehicle_class": self.vehicle_class,
            "normalized_distance": round(self.normalized_distance, 9),
            "trigger_threshold": self.trigger_threshold,
            "consecutive_observations": self.consecutive_observations,
        }


@dataclass(slots=True)
class _PairState:
    consecutive: int = 0
    active: bool = False


class ProximityEngine:
    """Apply per-zone hysteresis to class-filtered person–vehicle pairs."""

    def __init__(
        self,
        rules: Sequence[ProximityRule],
        *,
        frame_width: int,
        frame_height: int,
    ) -> None:
        if len({rule.zone_id for rule in rules}) != len(rules):
            raise ValueError("proximity rule zone_id values must be unique")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.rules = {rule.zone_id: rule for rule in rules}
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._states: dict[tuple[str, str, int, int], _PairState] = {}
        self.records: list[ProximityRecord] = []
        self.excluded_rider_pairs: set[tuple[str, str, int, int]] = set()
        self.pair_comparisons = 0

    def update(
        self,
        tracks: Sequence[TrackRecord],
        zone_observations: Sequence[ZoneObservation],
    ) -> list[ProximityRecord]:
        tracks_by_id = {(item.video_id, item.track_id): item for item in tracks}
        inside_by_zone: dict[str, list[TrackRecord]] = defaultdict(list)
        for item in zone_observations:
            rule = self.rules.get(item.zone_id)
            if rule is None or not rule.enabled or item.transition not in {"ENTER", "INSIDE"}:
                continue
            track = tracks_by_id.get((item.video_id, item.track_id))
            if track is not None and (
                track.class_name == "person" or track.class_name in rule.vehicle_classes
            ):
                inside_by_zone[item.zone_id].append(track)

        observed_keys: set[tuple[str, str, int, int]] = set()
        new_records: list[ProximityRecord] = []
        for zone_id, filtered_tracks in inside_by_zone.items():
            rule = self.rules[zone_id]
            people = [item for item in filtered_tracks if item.class_name == "person"]
            vehicles = [item for item in filtered_tracks if item.class_name in rule.vehicle_classes]
            for person in people:
                for vehicle in vehicles:
                    key = (person.video_id, zone_id, person.track_id, vehicle.track_id)
                    observed_keys.add(key)
                    state = self._states.setdefault(key, _PairState())
                    if rider_self_vehicle_overlap(
                        person.bbox,
                        vehicle.bbox,
                        vehicle_class=vehicle.class_name,
                        overlap_threshold=rule.rider_overlap_exclusion_ratio,
                    ):
                        state.consecutive = 0
                        state.active = False
                        self.excluded_rider_pairs.add(key)
                        continue

                    self.pair_comparisons += 1
                    distance = normalized_bbox_distance(
                        person.bbox,
                        vehicle.bbox,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                    )
                    if state.active:
                        if distance > rule.release_threshold:
                            state.active = False
                            state.consecutive = 0
                        continue
                    if distance > rule.trigger_threshold:
                        state.consecutive = 0
                        continue
                    state.consecutive += 1
                    if state.consecutive < rule.minimum_consecutive_observations:
                        continue
                    state.active = True
                    record = ProximityRecord(
                        video_id=person.video_id,
                        source_id=person.source_id,
                        zone_id=zone_id,
                        frame_index=person.frame_index,
                        timestamp_seconds=person.timestamp_seconds,
                        person_track_id=person.track_id,
                        vehicle_track_id=vehicle.track_id,
                        vehicle_class=vehicle.class_name,
                        normalized_distance=distance,
                        trigger_threshold=rule.trigger_threshold,
                        consecutive_observations=state.consecutive,
                    )
                    self.records.append(record)
                    new_records.append(record)

        for key, state in self._states.items():
            if key not in observed_keys:
                state.consecutive = 0
                state.active = False
        return new_records

    def active_pair_keys(self) -> tuple[tuple[str, str, int, int], ...]:
        return tuple(key for key, state in self._states.items() if state.active)


def load_proximity_config(path: Path) -> dict[str, tuple[ProximityRule, ...]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("scenes"), Mapping):
        raise ValueError("proximity config requires scenes")
    parsed: dict[str, tuple[ProximityRule, ...]] = {}
    for source_id, scene in data["scenes"].items():
        if not isinstance(scene, Mapping) or not isinstance(scene.get("zones"), list):
            raise ValueError(f"scene {source_id} requires zones")
        rules = []
        for zone in scene["zones"]:
            raw = zone.get("proximity")
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                raise ValueError("proximity config must be a mapping")
            enabled = bool(raw.get("enabled", False))
            rules.append(ProximityRule(
                zone_id=str(zone.get("zone_id", "")),
                enabled=enabled,
                vehicle_classes=frozenset(map(str, raw.get("vehicle_classes", []))),
                trigger_threshold=float(raw.get("trigger_distance_normalized", 0.0)),
                release_threshold=float(raw.get("release_distance_normalized", 0.0)),
                minimum_consecutive_observations=int(raw.get("minimum_consecutive_observations", 1)),
                rider_overlap_exclusion_ratio=float(raw.get("rider_overlap_exclusion_ratio", 0.25)),
            ))
        parsed[str(source_id)] = tuple(rules)
    return parsed
