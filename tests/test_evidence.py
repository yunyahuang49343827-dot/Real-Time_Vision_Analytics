"""Tests for Stage 14 event evidence snapshots."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.schema import BoundingBox
from vision_analytics.events.evidence import EvidenceCapture, EvidencePolicy
from vision_analytics.events.schema import EventRecord
from vision_analytics.tracking.schema import TrackRecord


def make_policy() -> EvidencePolicy:
    return EvidencePolicy(
        capture_event_types=frozenset({
            "WRONG_WAY", "LONG_DWELL", "STATIONARY_VEHICLE",
            "PEDESTRIAN_INTRUSION", "PROXIMITY_WARNING",
        }),
        capture_severities=frozenset({"WARNING", "CRITICAL"}),
        capture_statuses=frozenset({"REVIEW_REQUIRED"}),
    )


def make_event(
    event_id: str = "demo-EVT-000001",
    *,
    event_type: str = "PROXIMITY_WARNING",
    severity: str = "WARNING",
    status: str = "REVIEW_REQUIRED",
) -> EventRecord:
    return EventRecord(
        event_id=event_id, video_id="demo", source_id="source", event_type=event_type,
        frame_index=12, timestamp_seconds=0.4, track_id=3,
        secondary_track_id=8 if event_type == "PROXIMITY_WARNING" else None,
        class_name="person", secondary_class_name="car", zone_id="mixed_zone",
        severity=severity, status=status, rule_source="test.rule", rule_value="0.01",
        threshold="0.012",
    )


def make_track(track_id: int, class_name: str, x_offset: float) -> TrackRecord:
    return TrackRecord(
        video_id="demo", source_id="source", frame_index=12, timestamp_seconds=0.4,
        track_id=track_id, class_id=0 if class_name == "person" else 2,
        class_name=class_name, confidence=0.8,
        bbox=BoundingBox(x_offset, 20.0, x_offset + 30.0, 80.0),
    )


def make_capture(tmp_path: Path) -> EvidenceCapture:
    return EvidenceCapture(make_policy(), tmp_path / "evidence", Path("outputs/evidence/stage14"))


def test_capture_policy_filters_event_type_and_info() -> None:
    policy = make_policy()
    assert policy.should_capture(make_event())
    assert policy.should_capture(make_event(event_type="WRONG_WAY", severity="CRITICAL"))
    assert not policy.should_capture(make_event(
        event_type="LINE_CROSSING", severity="INFO", status="DETECTED",
    ))


def test_review_required_can_capture_when_severity_is_info() -> None:
    policy = make_policy()
    assert policy.should_capture(make_event(severity="INFO", status="REVIEW_REQUIRED"))


def test_unlisted_warning_does_not_capture() -> None:
    policy = make_policy()
    assert not policy.should_capture(make_event(
        event_type="ZONE_ENTRY", severity="WARNING", status="REVIEW_REQUIRED",
    ))


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        EvidencePolicy(frozenset({"UNKNOWN"}), frozenset(), frozenset())
    with pytest.raises(ValueError):
        EvidencePolicy(frozenset(), frozenset(), frozenset(), jpeg_quality=101)


def test_deterministic_filename_and_safe_event_id() -> None:
    assert EvidenceCapture.filename("demo-EVT-000001") == "demo-EVT-000001.jpg"
    with pytest.raises(ValueError):
        EvidenceCapture.filename("../escape")


def test_capture_writes_readable_image_and_path(tmp_path: Path) -> None:
    capture = make_capture(tmp_path)
    frame = np.full((120, 180, 3), 80, dtype=np.uint8)
    event = make_event()
    result = capture.capture_events(
        frame, [event], [make_track(3, "person", 10), make_track(8, "car", 90)],
    )[0]
    expected = tmp_path / "evidence" / f"{event.event_id}.jpg"
    assert result.evidence_path == f"outputs/evidence/stage14/{event.event_id}.jpg"
    assert expected.is_file() and expected.stat().st_size > 0
    assert cv2.imread(str(expected)) is not None


def test_one_event_produces_one_snapshot_and_manifest_row(tmp_path: Path) -> None:
    capture = make_capture(tmp_path)
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    event = make_event()
    first = capture.capture(frame, event)
    second = capture.capture(frame, event)
    assert first == second
    assert len(capture.manifest_records) == 1
    assert len(list((tmp_path / "evidence").glob("*.jpg"))) == 1


def test_info_event_remains_without_evidence(tmp_path: Path) -> None:
    capture = make_capture(tmp_path)
    event = make_event(event_type="ZONE_EXIT", severity="INFO", status="DETECTED")
    result = capture.capture(np.zeros((50, 50, 3), dtype=np.uint8), event)
    assert result.evidence_path is None
    assert capture.manifest_records == []


def test_manifest_csv(tmp_path: Path) -> None:
    capture = make_capture(tmp_path)
    event = make_event()
    capture.capture(np.zeros((80, 100, 3), dtype=np.uint8), event)
    manifest = tmp_path / "manifest.csv"
    capture.write_manifest(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["event_id"] == event.event_id
    assert int(rows[0]["file_size_bytes"]) > 0


def test_empty_events_and_empty_manifest(tmp_path: Path) -> None:
    capture = make_capture(tmp_path)
    assert capture.capture_events(np.zeros((20, 20, 3), dtype=np.uint8), []) == []
    manifest = tmp_path / "empty.csv"
    capture.write_manifest(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
