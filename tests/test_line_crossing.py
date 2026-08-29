from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.spatial.line_crossing import (
    CROSSING_FIELDS,
    CountingLine,
    LineCrossingEngine,
    NormalizedPoint,
    finite_segments_intersect,
)
from vision_analytics.tracking.trajectory import TrajectoryObservation


def line(
    line_id: str = "line",
    start: tuple[float, float] = (0.0, 0.5),
    end: tuple[float, float] = (1.0, 0.5),
) -> CountingLine:
    return CountingLine(
        line_id=line_id,
        start=NormalizedPoint(*start),
        end=NormalizedPoint(*end),
    )


def observation(
    *,
    track_id: int = 1,
    previous: tuple[float, float] = (50.0, 60.0),
    current: tuple[float, float] = (50.0, 40.0),
    frame_index: int = 2,
    frame_gap: int = 1,
) -> TrajectoryObservation:
    delta_x = current[0] - previous[0]
    delta_y = current[1] - previous[1]
    displacement = (delta_x**2 + delta_y**2) ** 0.5
    return TrajectoryObservation(
        video_id="video",
        source_id="source",
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10,
        track_id=track_id,
        class_name="car",
        center_x=current[0],
        center_y=current[1],
        prev_center_x=previous[0],
        prev_center_y=previous[1],
        delta_x=delta_x,
        delta_y=delta_y,
        step_displacement=displacement,
        net_displacement=displacement,
        direction="UP" if delta_y < 0 else "DOWN",
        history_length=2,
        frame_gap=frame_gap,
    )


def engine(*lines: CountingLine, maximum_frame_gap: int = 5) -> LineCrossingEngine:
    return LineCrossingEngine(
        lines or (line(),),
        frame_width=101,
        frame_height=101,
        maximum_frame_gap=maximum_frame_gap,
        minimum_movement_pixels=3.0,
    )


def test_a_to_b_crossing() -> None:
    records = engine().update([observation()])
    assert len(records) == 1
    assert records[0].crossing_direction == "A_TO_B"
    assert tuple(records[0].to_row()) == CROSSING_FIELDS


def test_b_to_a_crossing() -> None:
    records = engine().update(
        [observation(previous=(50.0, 40.0), current=(50.0, 60.0))]
    )
    assert len(records) == 1
    assert records[0].crossing_direction == "B_TO_A"


def test_no_crossing_when_centers_remain_on_same_side() -> None:
    assert engine().update(
        [observation(previous=(50.0, 70.0), current=(50.0, 60.0))]
    ) == []


def test_finite_line_segment_rejects_infinite_extension_crossing() -> None:
    short_line = line(start=(0.4, 0.5), end=(0.6, 0.5))
    assert engine(short_line).update(
        [observation(previous=(80.0, 60.0), current=(80.0, 40.0))]
    ) == []
    assert not finite_segments_intersect(
        (80.0, 60.0), (80.0, 40.0), (40.0, 50.0), (60.0, 50.0)
    )


def test_diagonal_counting_line_is_supported() -> None:
    diagonal = line(start=(0.2, 0.2), end=(0.8, 0.8))
    records = engine(diagonal).update(
        [observation(previous=(50.0, 20.0), current=(50.0, 80.0))]
    )
    assert len(records) == 1
    assert records[0].crossing_direction == "B_TO_A"


def test_duplicate_prevention_for_same_video_line_and_track() -> None:
    crossing_engine = engine()
    first = crossing_engine.update([observation(frame_index=2)])
    second = crossing_engine.update(
        [
            observation(
                previous=(50.0, 40.0),
                current=(50.0, 60.0),
                frame_index=3,
            )
        ]
    )
    assert len(first) == 1
    assert second == []
    assert crossing_engine.count_for_line("line") == 1


def test_different_track_ids_count_independently() -> None:
    crossing_engine = engine()
    records = crossing_engine.update(
        [observation(track_id=1), observation(track_id=2)]
    )
    assert {record.track_id for record in records} == {1, 2}


def test_same_track_counts_independently_on_different_lines() -> None:
    crossing_engine = engine(
        line("upper", start=(0.0, 0.4), end=(1.0, 0.4)),
        line("lower", start=(0.0, 0.6), end=(1.0, 0.6)),
    )
    records = crossing_engine.update(
        [observation(previous=(50.0, 70.0), current=(50.0, 30.0))]
    )
    assert {record.line_id for record in records} == {"upper", "lower"}


def test_excessive_frame_gap_is_rejected() -> None:
    assert engine(maximum_frame_gap=5).update([observation(frame_gap=6)]) == []


def test_normalized_coordinate_conversion() -> None:
    point = NormalizedPoint(0.5, 0.25)
    assert point.to_pixels(201, 101) == pytest.approx((100.0, 25.0))


def test_invalid_line_config() -> None:
    with pytest.raises(ValueError, match="0.0–1.0"):
        NormalizedPoint(1.1, 0.5)
    with pytest.raises(ValueError, match="distinct"):
        line(start=(0.5, 0.5), end=(0.5, 0.5))
    with pytest.raises(ValueError, match="unique"):
        engine(line("duplicate"), line("duplicate"))


def test_minimum_movement_rejects_jitter() -> None:
    crossing_engine = LineCrossingEngine(
        [line()],
        frame_width=101,
        frame_height=101,
        minimum_movement_pixels=3.0,
    )
    assert crossing_engine.update(
        [observation(previous=(50.0, 51.0), current=(50.0, 49.0))]
    ) == []


def test_empty_input_returns_empty_output() -> None:
    assert engine().update([]) == []
