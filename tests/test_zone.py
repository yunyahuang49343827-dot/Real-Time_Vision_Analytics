from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.schema import BoundingBox
from vision_analytics.spatial.zone import NormalizedZone, ZoneEngine
from vision_analytics.tracking.schema import TrackRecord


def zone(zone_id: str = "zone", offset: float = 0.0) -> NormalizedZone:
    return NormalizedZone(
        zone_id=zone_id,
        zone_type="test",
        points=((0.2 + offset, 0.2), (0.8 + offset, 0.2), (0.8 + offset, 0.8), (0.2 + offset, 0.8)),
    )


def track(
    x: float,
    y: float,
    *,
    track_id: int = 1,
    frame: int = 0,
    class_name: str = "car",
) -> TrackRecord:
    class_id = 0 if class_name == "person" else 2
    return TrackRecord(
        video_id="video",
        source_id="source",
        frame_index=frame,
        timestamp_seconds=frame / 10,
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(x - 1, y - 1, x + 1, y + 1),
    )


def engine(*zones: NormalizedZone) -> ZoneEngine:
    return ZoneEngine(zones or (zone(),), frame_width=101, frame_height=101)


def test_inside_outside_and_boundary_is_inside() -> None:
    states = engine().update([track(50, 50), track(5, 5, track_id=2), track(20, 50, track_id=3)])
    assert [state.transition for state in states] == ["INSIDE", "OUTSIDE", "INSIDE"]


def test_initial_inside_does_not_emit_enter() -> None:
    zone_engine = engine()
    assert zone_engine.update([track(50, 50)])[0].transition == "INSIDE"
    assert zone_engine.transitions == []


def test_outside_to_inside_is_enter() -> None:
    zone_engine = engine()
    zone_engine.update([track(5, 5)])
    assert zone_engine.update([track(50, 50, frame=1)])[0].transition == "ENTER"


def test_inside_to_inside_is_inside() -> None:
    zone_engine = engine()
    zone_engine.update([track(40, 40)])
    assert zone_engine.update([track(50, 50, frame=1)])[0].transition == "INSIDE"


def test_inside_to_outside_is_exit() -> None:
    zone_engine = engine()
    zone_engine.update([track(50, 50)])
    assert zone_engine.update([track(5, 5, frame=1)])[0].transition == "EXIT"


def test_missing_observation_does_not_auto_exit() -> None:
    zone_engine = engine()
    zone_engine.update([track(50, 50)])
    assert zone_engine.update([]) == []
    assert zone_engine.transitions == []
    assert zone_engine.current_occupancy["zone"] == 0
    assert zone_engine.update([track(5, 5, frame=2)])[0].transition == "EXIT"


def test_different_track_states_are_isolated() -> None:
    zone_engine = engine()
    zone_engine.update([track(5, 5, track_id=1), track(50, 50, track_id=2)])
    states = zone_engine.update([track(50, 50, track_id=1, frame=1), track(50, 50, track_id=2, frame=1)])
    assert [state.transition for state in states] == ["ENTER", "INSIDE"]


def test_different_zone_states_are_isolated() -> None:
    left = NormalizedZone("left", "test", ((0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)))
    right = NormalizedZone("right", "test", ((0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)))
    states = engine(left, right).update([track(20, 50)])
    assert {state.zone_id: state.transition for state in states} == {"left": "INSIDE", "right": "OUTSIDE"}


def test_class_change_does_not_reset_state() -> None:
    zone_engine = engine()
    zone_engine.update([track(5, 5, class_name="car")])
    state = zone_engine.update([track(50, 50, frame=1, class_name="person")])[0]
    assert state.transition == "ENTER"


def test_normalized_polygon_conversion() -> None:
    polygon = zone().pixel_polygon(101, 201)
    assert np.allclose(polygon, [[20, 40], [80, 40], [80, 160], [20, 160]])


def test_invalid_polygon_and_duplicate_zone_id() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        NormalizedZone("bad", "test", ((0, 0), (1, 1), (0, 0)))
    with pytest.raises(ValueError, match="positive"):
        NormalizedZone("bad", "test", ((0, 0), (0.5, 0.5), (1, 1)))
    with pytest.raises(ValueError, match="unique"):
        engine(zone("same"), zone("same"))
