from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.error_analysis import (
    REVIEW_FIELDS,
    compute_confidence_statistics,
    select_review_samples,
    validate_review_row,
)


def detection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [1, 1, 4, 5, 8, 8, 9],
            "class_name": ["car", "person", "car", "truck", "motorcycle", "car", "bicycle"],
            "confidence": [0.9, 0.3, 0.5, 0.7, 0.4, 0.8, 0.25],
            "x1": [0, 1, 2, 3, 4, 5, 6],
            "y1": [0, 1, 2, 3, 4, 5, 6],
            "x2": [10, 11, 12, 13, 14, 15, 16],
            "y2": [10, 11, 12, 13, 14, 15, 16],
        }
    )


def test_confidence_statistics_quantiles_and_counts() -> None:
    result = compute_confidence_statistics(detection_frame())
    cars = result[result["class_name"] == "car"].iloc[0]

    assert cars["occurrence_count"] == 3
    assert cars["mean"] == pytest.approx((0.9 + 0.5 + 0.8) / 3)
    assert cars["median"] == pytest.approx(0.8)
    assert cars["p10"] <= cars["p25"] <= cars["p75"] <= cars["p90"]


def test_empty_confidence_statistics_has_stable_schema() -> None:
    result = compute_confidence_statistics(pd.DataFrame())
    assert list(result.columns) == [
        "class_name", "occurrence_count", "mean", "median", "p10", "p25", "p75", "p90"
    ]


def test_sampling_is_unique_and_mixes_uniform_with_targeted() -> None:
    samples = select_review_samples(
        detection_frame(),
        video_id="video",
        source_id="pexels_37258214",
        frame_count=12,
        source_fps=2.0,
        uniform_count=3,
        targeted_count=3,
    )

    assert len(samples) == 6
    assert len({sample.frame_index for sample in samples}) == 6
    assert {sample.sample_type for sample in samples} == {"UNIFORM", "TARGETED"}
    assert all(sample.timestamp_seconds == pytest.approx(sample.frame_index / 2.0) for sample in samples)


def test_review_schema_accepts_false_negative_without_predicted_class() -> None:
    row = {field: "" for field in REVIEW_FIELDS}
    row.update(
        {
            "video_id": "video",
            "frame_index": "3",
            "timestamp_seconds": "0.1",
            "sample_type": "TARGETED",
            "review_result": "FALSE_NEGATIVE",
            "error_category": "Small Object",
        }
    )
    validate_review_row(row)


def test_review_schema_rejects_invalid_result() -> None:
    row = {field: "" for field in REVIEW_FIELDS}
    row["review_result"] = "MAYBE"
    with pytest.raises(ValueError, match="invalid review_result"):
        validate_review_row(row)
