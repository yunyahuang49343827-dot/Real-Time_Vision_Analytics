from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.spatial.dwell import (
    DwellRule,
    StationaryRule,
    TemporalRuleEngine,
)
from vision_analytics.spatial.zone import ZoneObservation
from vision_analytics.tracking.trajectory import TrajectoryObservation


def zone(
    timestamp: float,
    *,
    transition: str = "INSIDE",
    zone_id: str = "zone",
    track_id: int = 1,
    class_name: str = "car",
) -> ZoneObservation:
    return ZoneObservation(
        video_id="video", source_id="source", zone_id=zone_id,
        frame_index=round(timestamp * 10), timestamp_seconds=timestamp,
        track_id=track_id, class_name=class_name, transition=transition,
        center_x=50, center_y=50,
    )


def trajectory(
    timestamp: float,
    *,
    displacement: float = 1.0,
    frame_gap: int = 1,
    track_id: int = 1,
    class_name: str = "car",
) -> TrajectoryObservation:
    return TrajectoryObservation(
        video_id="video", source_id="source", frame_index=round(timestamp * 10),
        timestamp_seconds=timestamp, track_id=track_id, class_name=class_name,
        center_x=50, center_y=50, prev_center_x=50, prev_center_y=50,
        delta_x=0, delta_y=0, step_displacement=0,
        net_displacement=displacement, direction="STATIONARY",
        history_length=2, frame_gap=frame_gap,
    )


def engine(
    *,
    dwell: bool = True,
    stationary: bool = False,
    maximum_missing: float = 1.0,
) -> TemporalRuleEngine:
    dwell_rules = [DwellRule("zone", 2.0, frozenset({"car", "person"}))] if dwell else []
    stationary_rules = [StationaryRule("zone", 2.0, 0.02, frozenset({"car"}))] if stationary else []
    return TemporalRuleEngine(
        dwell_rules, stationary_rules, frame_width=60, frame_height=80,
        maximum_missing_seconds=maximum_missing,
    )


def test_dwell_initial_inside_starts_observed_episode() -> None:
    temporal = engine()
    assert temporal.update([zone(5)], [])[0] == []
    assert temporal.update([zone(6)], [])[0] == []
    records, _ = temporal.update([zone(7)], [])
    assert records[0].first_inside_timestamp == 5
    assert records[0].observed_dwell_seconds == 2


def test_dwell_enter_inside_threshold_and_episode_dedup() -> None:
    temporal = engine()
    temporal.update([zone(0, transition="ENTER")], [])
    assert temporal.update([zone(1)], [])[0] == []
    assert len(temporal.update([zone(2)], [])[0]) == 1
    assert temporal.update([zone(3)], [])[0] == []


def test_dwell_exit_reset_and_reentry_creates_new_episode() -> None:
    temporal = engine()
    temporal.update([zone(0, transition="ENTER")], [])
    temporal.update([zone(1)], [])
    assert len(temporal.update([zone(2)], [])[0]) == 1
    temporal.update([zone(3, transition="EXIT")], [])
    temporal.update([zone(4, transition="ENTER")], [])
    temporal.update([zone(5)], [])
    records, _ = temporal.update([zone(6)], [])
    assert len(records) == 1
    assert records[0].first_inside_timestamp == 4
    assert len(temporal.long_dwell_records) == 2


def test_dwell_temporary_missing_continues_but_excessive_gap_resets() -> None:
    temporary = engine(maximum_missing=1.0)
    temporary.update([zone(0)], [])
    temporary.update([], [])
    temporary.update([zone(0.8)], [])
    temporary.update([zone(1.6)], [])
    records, _ = temporary.update([zone(2.0)], [])
    assert records[0].first_inside_timestamp == 0

    excessive = engine(maximum_missing=1.0)
    excessive.update([zone(0)], [])
    assert excessive.update([zone(2.0)], [])[0] == []
    excessive.update([zone(3.0)], [])
    records, _ = excessive.update([zone(4.0)], [])
    assert records[0].first_inside_timestamp == 2.0


def test_stationary_moving_vehicle_resets_episode() -> None:
    temporal = engine(dwell=False, stationary=True)
    temporal.update([zone(0)], [trajectory(0)])
    temporal.update([zone(1)], [trajectory(1, displacement=5)])
    assert temporal.update([zone(2)], [trajectory(2)])[1] == []
    assert temporal.update([zone(4)], [trajectory(4)])[1] == []


def test_stationary_duration_threshold_and_episode_dedup() -> None:
    temporal = engine(dwell=False, stationary=True)
    temporal.update([zone(0)], [trajectory(0)])
    assert temporal.update([zone(1)], [trajectory(1)])[1] == []
    records = temporal.update([zone(2)], [trajectory(2)])[1]
    assert len(records) == 1
    assert records[0].stationary_duration_seconds == 2
    assert temporal.update([zone(3)], [trajectory(3)])[1] == []


def test_stationary_displacement_is_normalized_by_frame_diagonal() -> None:
    temporal = engine(dwell=False, stationary=True)
    temporal.update([zone(0)], [trajectory(0, displacement=1.5)])
    temporal.update([zone(1)], [trajectory(1, displacement=1.5)])
    record = temporal.update([zone(2)], [trajectory(2, displacement=1.5)])[1][0]
    assert record.normalized_displacement == 0.015


def test_stationary_class_scope_disabled_rule_and_outside_zone() -> None:
    temporal = engine(dwell=False, stationary=True)
    assert temporal.update([zone(0, class_name="person")], [trajectory(0, class_name="person")])[1] == []
    assert temporal.update([zone(1, transition="OUTSIDE")], [trajectory(1)])[1] == []
    disabled = engine(dwell=False, stationary=False)
    assert disabled.update([zone(0)], [trajectory(0)])[1] == []


def test_stationary_movement_or_exit_allows_new_episode() -> None:
    temporal = engine(dwell=False, stationary=True)
    temporal.update([zone(0)], [trajectory(0)])
    temporal.update([zone(1)], [trajectory(1)])
    assert len(temporal.update([zone(2)], [trajectory(2)])[1]) == 1
    temporal.update([zone(3, transition="EXIT")], [trajectory(3)])
    temporal.update([zone(4, transition="ENTER")], [trajectory(4)])
    temporal.update([zone(5)], [trajectory(5)])
    assert len(temporal.update([zone(6)], [trajectory(6)])[1]) == 1
    assert len(temporal.stationary_vehicle_records) == 2
