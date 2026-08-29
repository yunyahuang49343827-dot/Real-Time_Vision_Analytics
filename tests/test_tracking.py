from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.schema import BoundingBox
from vision_analytics.tracking.schema import (
    ALLOWED_TRACK_CLASSES,
    TRACK_FIELDS,
    TrackRecord,
)
from vision_analytics.tracking.tracker import build_track_records


class SyntheticBoxes:
    """Small Results-like object accepted by Ultralytics BYTETracker."""

    def __init__(self, xywh: object, conf: object, classes: object) -> None:
        self.xywh = np.asarray(xywh, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=float).reshape(-1)
        self.cls = np.asarray(classes, dtype=float).reshape(-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, key: object) -> SyntheticBoxes:
        return SyntheticBoxes(self.xywh[key], self.conf[key], self.cls[key])


def sample_record() -> TrackRecord:
    return TrackRecord(
        video_id="video",
        source_id="source",
        frame_index=3,
        timestamp_seconds=0.1,
        track_id=7,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(10.0, 20.0, 30.0, 60.0),
    )


def test_track_schema_and_bbox_center() -> None:
    record = sample_record()
    assert record.center_x == pytest.approx(20.0)
    assert record.center_y == pytest.approx(40.0)
    assert tuple(record.to_row()) == TRACK_FIELDS


def test_allowed_classes_are_exact_stage6_taxonomy() -> None:
    assert ALLOWED_TRACK_CLASSES == {
        "person", "bicycle", "car", "motorcycle", "bus", "truck"
    }
    with pytest.raises(ValueError, match="class metadata"):
        TrackRecord(
            video_id="video",
            source_id="source",
            frame_index=0,
            timestamp_seconds=0.0,
            track_id=1,
            class_id=99,
            class_name="dog",
            confidence=0.5,
            bbox=BoundingBox(0.0, 0.0, 1.0, 1.0),
        )


def test_empty_track_result() -> None:
    assert build_track_records(
        [],
        [],
        [],
        [],
        {2: "car"},
        video_id="video",
        source_id="source",
        frame_index=0,
        timestamp_seconds=0.0,
    ) == []


def test_track_serialization() -> None:
    row = sample_record().to_row()
    assert row["track_id"] == 7
    assert row["class_name"] == "car"
    assert row["center_x"] == 20.0
    assert row["center_y"] == 40.0


def test_bytetrack_sequential_frames_keep_persistent_id() -> None:
    config = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker = BYTETracker(config)
    observed_ids = []
    for center_x in (100.0, 102.0, 104.0):
        output = tracker.update(
            SyntheticBoxes([[center_x, 100.0, 40.0, 30.0]], [0.9], [2])
        )
        assert output.shape == (1, 8)
        observed_ids.append(int(output[0, 4]))
    assert len(set(observed_ids)) == 1
