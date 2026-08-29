from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.schema import BoundingBox
from vision_analytics.tracking.schema import TrackRecord
from vision_analytics.tracking.trajectory import (
    TRAJECTORY_FIELDS,
    TrajectoryEngine,
    classify_direction,
)


def track(
    *,
    track_id: int = 1,
    frame_index: int = 0,
    center_x: float = 10.0,
    center_y: float = 10.0,
    video_id: str = "video",
) -> TrackRecord:
    return TrackRecord(
        video_id=video_id,
        source_id="source",
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10.0,
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(center_x - 2, center_y - 2, center_x + 2, center_y + 2),
    )


def test_history_accumulates_and_displacements_are_image_space_pixels() -> None:
    engine = TrajectoryEngine(max_history_length=30, minimum_displacement=1.0)
    first = engine.update([track(frame_index=1, center_x=10, center_y=20)])[0]
    second = engine.update([track(frame_index=2, center_x=13, center_y=24)])[0]

    assert first.history_length == 1
    assert first.direction == "STATIONARY"
    assert second.prev_center_x == 10
    assert second.prev_center_y == 20
    assert second.delta_x == 3
    assert second.delta_y == 4
    assert second.step_displacement == pytest.approx(5)
    assert second.net_displacement == pytest.approx(5)
    assert second.history_length == 2
    assert tuple(second.to_row()) == TRAJECTORY_FIELDS


def test_different_track_and_video_histories_do_not_mix() -> None:
    engine = TrajectoryEngine(minimum_displacement=1.0)
    engine.update([track(track_id=1), track(track_id=2)])
    by_track = {
        observation.track_id: observation
        for observation in engine.update(
            [
                track(track_id=1, frame_index=1, center_x=20),
                track(track_id=2, frame_index=1, center_x=5),
            ]
        )
    }
    other_video = engine.update(
        [track(track_id=1, video_id="other", frame_index=1, center_x=100)]
    )[0]

    assert by_track[1].delta_x == 10
    assert by_track[2].delta_x == -5
    assert other_video.history_length == 1
    assert other_video.delta_x == 0


def test_history_is_bounded_to_configured_maximum() -> None:
    engine = TrajectoryEngine(max_history_length=3, minimum_displacement=0.0)
    final = None
    for frame_index in range(6):
        final = engine.update(
            [track(frame_index=frame_index, center_x=float(frame_index * 10))]
        )[0]

    assert final is not None
    assert final.history_length == 3
    assert final.net_displacement == pytest.approx(20)
    assert engine.trail_points("video", 1) == ((30, 10), (40, 10), (50, 10))


@pytest.mark.parametrize(
    ("delta_x", "delta_y", "expected"),
    [
        (0, -10, "UP"),
        (10, -10, "UP_RIGHT"),
        (10, 0, "RIGHT"),
        (10, 10, "DOWN_RIGHT"),
        (0, 10, "DOWN"),
        (-10, 10, "DOWN_LEFT"),
        (-10, 0, "LEFT"),
        (-10, -10, "UP_LEFT"),
    ],
)
def test_eight_direction_classification(
    delta_x: float, delta_y: float, expected: str
) -> None:
    assert classify_direction(delta_x, delta_y) == expected


def test_small_movement_is_stationary() -> None:
    assert classify_direction(2.0, 2.0, minimum_displacement=5.0) == "STATIONARY"


def test_frame_gap_uses_previous_observed_frame() -> None:
    engine = TrajectoryEngine()
    engine.update([track(frame_index=4)])
    observation = engine.update([track(frame_index=9, center_x=20)])[0]
    assert observation.frame_gap == 5


def test_empty_input_returns_empty_output() -> None:
    assert TrajectoryEngine().update([]) == []
