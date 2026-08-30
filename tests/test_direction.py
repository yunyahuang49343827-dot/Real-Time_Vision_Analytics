from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.spatial.direction import DirectionRule, WrongWayEngine
from vision_analytics.spatial.zone import ZoneObservation
from vision_analytics.tracking.trajectory import TrajectoryObservation


def rule(
    zone_id: str = "lane",
    *,
    allowed: tuple[str, ...] = ("DOWN", "DOWN_LEFT", "DOWN_RIGHT"),
    classes: tuple[str, ...] = ("car",),
) -> DirectionRule:
    return DirectionRule(zone_id, frozenset(allowed), frozenset(classes))


def trajectory(
    frame: int,
    *,
    track_id: int = 1,
    direction: str = "UP",
    displacement: float = 30.0,
    frame_gap: int = 1,
    class_name: str = "car",
) -> TrajectoryObservation:
    return TrajectoryObservation(
        video_id="video",
        source_id="source",
        frame_index=frame,
        timestamp_seconds=frame / 10,
        track_id=track_id,
        class_name=class_name,
        center_x=50,
        center_y=50,
        prev_center_x=50,
        prev_center_y=51,
        delta_x=0,
        delta_y=-1,
        step_displacement=1,
        net_displacement=displacement,
        direction=direction,
        history_length=max(2, frame + 1),
        frame_gap=frame_gap,
    )


def zone_observation(
    frame: int,
    *,
    track_id: int = 1,
    zone_id: str = "lane",
    transition: str = "INSIDE",
    class_name: str = "car",
) -> ZoneObservation:
    return ZoneObservation(
        video_id="video",
        source_id="source",
        zone_id=zone_id,
        frame_index=frame,
        timestamp_seconds=frame / 10,
        track_id=track_id,
        class_name=class_name,
        transition=transition,
        center_x=50,
        center_y=50,
    )


def update(
    engine: WrongWayEngine,
    frame: int,
    **kwargs: object,
) -> list[object]:
    track_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in {"track_id", "direction", "displacement", "frame_gap", "class_name"}
    }
    zone_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in {"track_id", "zone_id", "transition", "class_name"}
    }
    return engine.update(
        [trajectory(frame, **track_kwargs)],
        [zone_observation(frame, **zone_kwargs)],
    )


def test_allowed_and_diagonal_directions_do_not_violate() -> None:
    engine = WrongWayEngine([rule()], consecutive_observations=2, minimum_net_displacement=20)
    assert update(engine, 1, direction="DOWN") == []
    assert update(engine, 2, direction="DOWN_RIGHT") == []
    assert engine.records == []


def test_opposite_direction_requires_consecutive_threshold() -> None:
    engine = WrongWayEngine([rule()], consecutive_observations=3, minimum_net_displacement=20)
    assert update(engine, 1) == []
    assert update(engine, 2) == []
    records = update(engine, 3)
    assert len(records) == 1
    assert records[0].observed_direction == "UP"
    assert records[0].consecutive_violation_count == 3


def test_stationary_and_insufficient_displacement_do_not_violate() -> None:
    engine = WrongWayEngine([rule()], consecutive_observations=1, minimum_net_displacement=20)
    assert update(engine, 1, direction="STATIONARY") == []
    assert update(engine, 2, displacement=19.9) == []


def test_confirmation_is_deduplicated() -> None:
    engine = WrongWayEngine([rule()], consecutive_observations=1, minimum_net_displacement=20)
    assert len(update(engine, 1)) == 1
    assert update(engine, 2) == []
    assert len(engine.records) == 1
    assert engine.is_confirmed("video", "lane", 1)


def test_track_and_zone_state_are_isolated() -> None:
    engine = WrongWayEngine(
        [rule("lane"), rule("other")],
        consecutive_observations=2,
        minimum_net_displacement=20,
    )
    assert update(engine, 1, track_id=1, zone_id="lane") == []
    assert update(engine, 1, track_id=2, zone_id="lane") == []
    assert update(engine, 1, track_id=3, zone_id="other") == []
    assert len(update(engine, 2, track_id=1, zone_id="lane")) == 1
    assert len(update(engine, 2, track_id=3, zone_id="other")) == 1
    assert not engine.is_confirmed("video", "lane", 2)


def test_applicable_class_and_outside_zone_are_required() -> None:
    engine = WrongWayEngine([rule(classes=("car",))], consecutive_observations=1, minimum_net_displacement=20)
    assert update(engine, 1, class_name="person") == []
    assert update(engine, 2, transition="OUTSIDE") == []
    assert update(engine, 3, transition="EXIT") == []


def test_initial_inside_still_requires_full_threshold() -> None:
    engine = WrongWayEngine([rule()], consecutive_observations=2, minimum_net_displacement=20)
    assert update(engine, 1, transition="INSIDE") == []
    assert len(update(engine, 2, transition="INSIDE")) == 1


def test_missing_observation_and_frame_gap_break_streak() -> None:
    engine = WrongWayEngine([rule()], consecutive_observations=2, minimum_net_displacement=20)
    assert update(engine, 1) == []
    assert engine.update([], []) == []
    assert update(engine, 3, frame_gap=2) == []
    assert update(engine, 4) == []
    assert len(update(engine, 5)) == 1
