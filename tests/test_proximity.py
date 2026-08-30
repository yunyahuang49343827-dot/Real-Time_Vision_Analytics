from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.schema import BoundingBox
from vision_analytics.spatial.proximity import (
    ProximityEngine,
    ProximityRule,
    bbox_distance,
    normalized_bbox_distance,
    rider_self_vehicle_overlap,
)
from vision_analytics.spatial.zone import ZoneObservation
from vision_analytics.tracking.schema import TrackRecord


def rule(*, enabled: bool = True, consecutive: int = 2) -> ProximityRule:
    return ProximityRule(
        zone_id="zone", enabled=enabled,
        vehicle_classes=frozenset({"bicycle", "car", "motorcycle", "bus", "truck"}),
        trigger_threshold=0.1, release_threshold=0.2,
        minimum_consecutive_observations=consecutive,
        rider_overlap_exclusion_ratio=0.25,
    )


def track(
    track_id: int,
    class_name: str,
    bbox: BoundingBox,
    *,
    frame: int = 0,
) -> TrackRecord:
    class_ids = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
    return TrackRecord(
        video_id="video", source_id="source", frame_index=frame,
        timestamp_seconds=frame / 10, track_id=track_id,
        class_id=class_ids[class_name], class_name=class_name,
        confidence=0.9, bbox=bbox,
    )


def zone_observation(item: TrackRecord, *, transition: str = "INSIDE") -> ZoneObservation:
    return ZoneObservation(
        video_id=item.video_id, source_id=item.source_id, zone_id="zone",
        frame_index=item.frame_index, timestamp_seconds=item.timestamp_seconds,
        track_id=item.track_id, class_name=item.class_name, transition=transition,
        center_x=item.center_x, center_y=item.center_y,
    )


def update(engine: ProximityEngine, items: list[TrackRecord], *, transition: str = "INSIDE") -> list[object]:
    return engine.update(items, [zone_observation(item, transition=transition) for item in items])


PERSON_BOX = BoundingBox(0, 0, 10, 10)
CLOSE_CAR_BOX = BoundingBox(15, 0, 25, 10)
MID_CAR_BOX = BoundingBox(25, 0, 35, 10)
FAR_CAR_BOX = BoundingBox(35, 0, 45, 10)


def test_bbox_gap_distance_and_overlap() -> None:
    assert bbox_distance(BoundingBox(0, 0, 10, 10), BoundingBox(13, 14, 20, 20)) == 5
    assert bbox_distance(BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)) == 0


def test_normalized_distance_uses_frame_diagonal() -> None:
    assert normalized_bbox_distance(
        PERSON_BOX, CLOSE_CAR_BOX, frame_width=60, frame_height=80
    ) == 0.05


def test_valid_person_vehicle_pair_requires_consecutive_observations() -> None:
    engine = ProximityEngine([rule()], frame_width=60, frame_height=80)
    assert update(engine, [track(1, "person", PERSON_BOX), track(2, "car", CLOSE_CAR_BOX)]) == []
    records = update(engine, [track(1, "person", PERSON_BOX, frame=1), track(2, "car", CLOSE_CAR_BOX, frame=1)])
    assert len(records) == 1
    assert records[0].person_track_id == 1
    assert records[0].vehicle_track_id == 2


def test_person_person_and_vehicle_vehicle_are_ignored() -> None:
    engine = ProximityEngine([rule(consecutive=1)], frame_width=60, frame_height=80)
    assert update(engine, [track(1, "person", PERSON_BOX), track(2, "person", CLOSE_CAR_BOX)]) == []
    assert update(engine, [track(3, "car", PERSON_BOX), track(4, "bus", CLOSE_CAR_BOX)]) == []


def test_outside_zone_and_disabled_rule_are_ignored() -> None:
    items = [track(1, "person", PERSON_BOX), track(2, "car", CLOSE_CAR_BOX)]
    enabled = ProximityEngine([rule(consecutive=1)], frame_width=60, frame_height=80)
    assert update(enabled, items, transition="OUTSIDE") == []
    disabled = ProximityEngine([rule(enabled=False, consecutive=1)], frame_width=60, frame_height=80)
    assert update(disabled, items) == []


def test_hysteresis_and_episode_dedup() -> None:
    engine = ProximityEngine([rule(consecutive=1)], frame_width=60, frame_height=80)
    assert len(update(engine, [track(1, "person", PERSON_BOX), track(2, "car", CLOSE_CAR_BOX)])) == 1
    assert update(engine, [track(1, "person", PERSON_BOX, frame=1), track(2, "car", MID_CAR_BOX, frame=1)]) == []
    assert update(engine, [track(1, "person", PERSON_BOX, frame=2), track(2, "car", CLOSE_CAR_BOX, frame=2)]) == []
    assert update(engine, [track(1, "person", PERSON_BOX, frame=3), track(2, "car", FAR_CAR_BOX, frame=3)]) == []
    assert len(update(engine, [track(1, "person", PERSON_BOX, frame=4), track(2, "car", CLOSE_CAR_BOX, frame=4)])) == 1


def test_pair_state_is_isolated() -> None:
    engine = ProximityEngine([rule()], frame_width=60, frame_height=80)
    first = [track(1, "person", PERSON_BOX), track(2, "car", CLOSE_CAR_BOX), track(3, "bus", CLOSE_CAR_BOX)]
    assert update(engine, first) == []
    second = [track(1, "person", PERSON_BOX, frame=1), track(2, "car", CLOSE_CAR_BOX, frame=1)]
    records = update(engine, second)
    assert [(item.person_track_id, item.vehicle_track_id) for item in records] == [(1, 2)]


def test_rider_self_vehicle_overlap_exclusion() -> None:
    motorcycle_box = BoundingBox(5, 5, 15, 15)
    assert rider_self_vehicle_overlap(
        PERSON_BOX, motorcycle_box, vehicle_class="motorcycle", overlap_threshold=0.25
    )
    assert not rider_self_vehicle_overlap(
        PERSON_BOX, motorcycle_box, vehicle_class="car", overlap_threshold=0.25
    )
    assert rider_self_vehicle_overlap(
        PERSON_BOX, BoundingBox(4, 9, 6, 12),
        vehicle_class="motorcycle", overlap_threshold=0.9,
    )
    engine = ProximityEngine([rule(consecutive=1)], frame_width=60, frame_height=80)
    assert update(engine, [track(1, "person", PERSON_BOX), track(2, "motorcycle", motorcycle_box)]) == []
    assert len(engine.excluded_rider_pairs) == 1


def test_missing_observation_resets_consecutive_streak() -> None:
    engine = ProximityEngine([rule()], frame_width=60, frame_height=80)
    pair = [track(1, "person", PERSON_BOX), track(2, "car", CLOSE_CAR_BOX)]
    assert update(engine, pair) == []
    assert engine.update([], []) == []
    assert update(engine, [track(1, "person", PERSON_BOX, frame=2), track(2, "car", CLOSE_CAR_BOX, frame=2)]) == []


def test_invalid_hysteresis_thresholds() -> None:
    with pytest.raises(ValueError, match="trigger < release"):
        ProximityRule(
            "zone", True, frozenset({"car"}), trigger_threshold=0.2,
            release_threshold=0.1,
        )
