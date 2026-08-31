from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.evaluation.system import (
    aggregate_event_review,
    aggregate_tracking_review,
    crossing_confusion_metrics,
    sha256_file,
    validate_evidence_trace,
    validate_runtime_model,
    validate_system_manifest,
)


def test_system_manifest_validity_and_scene_coverage():
    rows = [
        {"scene": scene, "video_id": f"v{i}", "start_frame": 1, "end_frame": 4,
         "evaluation_purpose": "sample"}
        for i, scene in enumerate(("Highway", "Taipei", "Urban", "Aerial"))
    ]
    validate_system_manifest(pd.DataFrame(rows), {f"v{i}": 10 for i in range(4)})
    with pytest.raises(ValueError):
        validate_system_manifest(pd.DataFrame(rows[:-1]), {f"v{i}": 10 for i in range(4)})


def test_crossing_confusion_and_precision_recall():
    review = pd.DataFrame({
        "reference_id": ["a", "b", "c", "d", "e"],
        "review_category": ["CORRECT", "CORRECT", "MISSED", "FALSE", "DUPLICATE"],
    })
    result = crossing_confusion_metrics(review)
    assert result["true_positive"] == 2
    assert result["false_positive"] == 2
    assert result["false_negative"] == 1
    assert result["sample_precision"] == pytest.approx(0.5)
    assert result["sample_recall"] == pytest.approx(2 / 3)


def test_tracking_review_aggregation_keeps_fragmentation_distinct_from_switch():
    review = pd.DataFrame({
        "physical_object_id": ["p1", "p2"], "scene": ["Highway", "Taipei"],
        "video_id": ["v1", "v2"], "class_name": ["car", "person"],
        "start_frame": [0, 5], "end_frame": [10, 20],
        "track_ids_observed": ["1", "2|3"], "fragmentation_count": [0, 1],
        "id_switch_observed": [False, False],
    })
    result = aggregate_tracking_review(review)
    assert result["fragmented_objects"] == 1
    assert result["fragmentation_count"] == 1
    assert result["id_switch_objects"] == 0


def test_controlled_and_natural_event_sources_are_distinct():
    review = pd.DataFrame({
        "review_id": ["n", "c"], "event_type": ["WRONG_WAY", "WRONG_WAY"],
        "review_result": ["FALSE_EVENT", "TRUE_EVENT"],
        "source_type": ["NATURAL", "CONTROLLED_SYNTHETIC"],
    })
    result = aggregate_event_review(review)
    assert result["natural_reviewed"] == 1
    assert result["controlled_reviewed"] == 1
    review.loc[1, "source_type"] = "NATURAL_SYNTHETIC"
    with pytest.raises(ValueError):
        aggregate_event_review(review)


def test_rejected_candidate_cannot_be_runtime(tmp_path: Path):
    pretrained = tmp_path / "pretrained.pt"
    candidate = tmp_path / "candidate.pt"
    pretrained.write_bytes(b"pretrained")
    candidate.write_bytes(b"candidate")
    result = validate_runtime_model(pretrained, sha256_file(pretrained), candidate)
    assert result["rejected_candidate_used"] == 0
    with pytest.raises(ValueError, match="rejected"):
        validate_runtime_model(candidate, sha256_file(candidate), candidate)


def test_runtime_hash_provenance_is_enforced(tmp_path: Path):
    runtime = tmp_path / "runtime.pt"
    rejected = tmp_path / "rejected.pt"
    runtime.write_bytes(b"runtime")
    rejected.write_bytes(b"rejected")
    with pytest.raises(ValueError, match="hash"):
        validate_runtime_model(runtime, "0" * 64, rejected)


def test_evidence_trace_validation(tmp_path: Path):
    image = tmp_path / "E1.jpg"
    assert cv2.imwrite(str(image), np.zeros((16, 16, 3), dtype=np.uint8))
    events = pd.DataFrame([{
        "event_id": "E1", "frame_index": 12, "track_id": 7,
        "evidence_path": image.relative_to(tmp_path).as_posix(),
    }])
    evidence = pd.DataFrame([{
        "event_id": "E1", "frame_index": 12,
        "evidence_path": image.relative_to(tmp_path).as_posix(),
    }])
    review = pd.DataFrame([{"event_id": "E1", "expected_frame": 12, "expected_track_id": 7}])
    result, metrics = validate_evidence_trace(review, events, evidence, tmp_path)
    assert metrics == {"reviewed": 1, "passed": 1, "failed": 0}
    assert result.iloc[0]["trace_status"] == "PASS"


def test_invalid_review_categories_are_rejected():
    with pytest.raises(ValueError, match="invalid crossing"):
        crossing_confusion_metrics(pd.DataFrame({
            "reference_id": ["a"], "review_category": ["MAYBE"]
        }))
