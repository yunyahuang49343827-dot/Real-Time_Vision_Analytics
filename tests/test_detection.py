from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.detector import filter_detection_candidates
from vision_analytics.detection.schema import (
    DETECTION_FIELDS,
    BoundingBox,
    DetectionRecord,
)

CLASS_NAMES = {0: "person", 2: "car", 16: "dog"}


def test_target_class_filter_excludes_non_target_class() -> None:
    filtered = filter_detection_candidates(
        [[1, 2, 30, 40], [5, 6, 50, 60]],
        [0, 16],
        [0.9, 0.95],
        CLASS_NAMES,
        confidence_threshold=0.25,
    )

    assert [candidate[1] for candidate in filtered] == ["person"]


def test_confidence_filter_includes_threshold_and_excludes_below_it() -> None:
    filtered = filter_detection_candidates(
        [[1, 2, 30, 40], [5, 6, 50, 60]],
        [2, 2],
        [0.249, 0.25],
        CLASS_NAMES,
        confidence_threshold=0.25,
    )

    assert len(filtered) == 1
    assert filtered[0][2] == pytest.approx(0.25)


def test_empty_detection_input_returns_empty_list() -> None:
    assert (
        filter_detection_candidates(
            [], [], [], CLASS_NAMES, confidence_threshold=0.25
        )
        == []
    )


def test_detection_schema_flattens_bbox_to_required_fields() -> None:
    record = DetectionRecord(
        video_id="video",
        source_id="source",
        frame_index=3,
        timestamp_seconds=0.1,
        class_id=2,
        class_name="car",
        confidence=0.875,
        bbox=BoundingBox(1.0, 2.0, 30.0, 40.0),
    )

    row = record.to_row()
    assert tuple(row) == DETECTION_FIELDS
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (
        1.0,
        2.0,
        30.0,
        40.0,
    )


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValueError, match="maximums"):
        BoundingBox(10.0, 2.0, 1.0, 40.0)
