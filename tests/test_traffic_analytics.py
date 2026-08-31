"""Tests for Stage 15 structured traffic analytics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.analytics.traffic import (
    CLASS_DISTRIBUTION_COLUMNS,
    CROSSING_COLUMNS,
    DIRECTION_DISTRIBUTION_COLUMNS,
    EVENT_SUMMARY_COLUMNS,
    SUMMARY_COLUMNS,
    TIME_COLUMNS,
    ZONE_COLUMNS,
    DensityThreshold,
    TrafficAnalyticsConfig,
    build_crossing_analytics,
    build_event_summary,
    classify_density,
    reconcile_event_line_crossings,
    reconcile_crossing_totals,
)


def config() -> TrafficAnalyticsConfig:
    return TrafficAnalyticsConfig(
        time_bucket_seconds=10,
        density_thresholds={"source": DensityThreshold(2, 5)},
    )


def crossings() -> pd.DataFrame:
    rows = [
        ("video", "source", "line", 1, 1.0, 1, "car", "A_TO_B", 1.0, 1.0),
        ("video", "source", "line", 2, 2.0, 2, "person", "B_TO_A", 2.0, 2.0),
        ("video", "source", "line", 3, 11.0, 3, "bicycle", "A_TO_B", 3.0, 3.0),
        ("video", "source", "line", 4, 12.0, 4, "truck", "A_TO_B", 4.0, 4.0),
        ("video", "source", "line", 5, 13.0, 5, "motorcycle", "B_TO_A", 5.0, 5.0),
    ]
    return pd.DataFrame(rows, columns=CROSSING_COLUMNS)


def zones(peak: int = 4) -> pd.DataFrame:
    return pd.DataFrame([(
        "video", "source", "zone", peak, 1, 5, 3, 2,
    )], columns=ZONE_COLUMNS)


def analytics(peak: int = 4):
    return build_crossing_analytics(crossings(), config=config(), zones=zones(peak))


def test_total_and_category_crossing_counts() -> None:
    summary, _, _, _ = analytics()
    row = summary.iloc[0]
    assert row["total_line_crossing_count"] == 5
    assert row["person_crossing_count"] == 1
    assert row["motorized_vehicle_crossing_count"] == 3
    assert row["bicycle_crossing_count"] == 1


def test_class_distribution_and_percentage() -> None:
    _, classes, _, _ = analytics()
    assert classes["crossing_count"].sum() == 5
    assert classes["percentage"].sum() == pytest.approx(100.0)
    assert classes.set_index("class_name").loc["car", "percentage"] == 20.0


def test_direction_distribution_and_percentage() -> None:
    _, _, directions, _ = analytics()
    by_direction = directions.set_index("crossing_direction")
    assert by_direction.loc["A_TO_B", "crossing_count"] == 3
    assert by_direction.loc["B_TO_A", "crossing_count"] == 2
    assert directions["percentage"].sum() == pytest.approx(100.0)


def test_time_bucket_and_peak_interval() -> None:
    summary, _, _, timeline = analytics()
    assert timeline["interval_start_seconds"].tolist() == [0, 10]
    assert timeline["total_crossing_count"].tolist() == [2, 3]
    assert summary.iloc[0]["peak_interval_start_seconds"] == 10
    assert summary.iloc[0]["peak_interval_count"] == 3


def test_density_low_medium_high() -> None:
    threshold = DensityThreshold(2, 5)
    assert classify_density(2, threshold) == "LOW"
    assert classify_density(3, threshold) == "MEDIUM"
    assert classify_density(5, threshold) == "MEDIUM"
    assert classify_density(6, threshold) == "HIGH"


def test_summary_uses_zone_peak_density() -> None:
    summary, _, _, _ = analytics(peak=6)
    assert summary.iloc[0]["zone_peak_occupancy"] == 6
    assert summary.iloc[0]["density"] == "HIGH"


def test_event_type_severity_status_summary() -> None:
    events = pd.DataFrame([
        ("video", "source", "WRONG_WAY", "CRITICAL", "REVIEW_REQUIRED"),
        ("video", "source", "LINE_CROSSING", "INFO", "DETECTED"),
        ("video", "source", "LINE_CROSSING", "INFO", "DETECTED"),
    ], columns=["video_id", "source_id", "event_type", "severity", "status"])
    result = build_event_summary(events, {"video": 60.0})
    assert set(result["metric"]) == {"EVENT_TYPE", "SEVERITY", "STATUS"}
    line = result[(result["metric"] == "EVENT_TYPE") & (result["category"] == "LINE_CROSSING")].iloc[0]
    assert line["event_count"] == 2
    assert line["percentage"] == pytest.approx(66.666667)
    assert line["events_per_minute"] == 2.0


def test_empty_data() -> None:
    summary, classes, directions, timeline = build_crossing_analytics(
        pd.DataFrame(columns=CROSSING_COLUMNS), config=config(), zones=zones(),
    )
    assert list(summary.columns) == list(SUMMARY_COLUMNS) and summary.empty
    assert list(classes.columns) == list(CLASS_DISTRIBUTION_COLUMNS) and classes.empty
    assert list(directions.columns) == list(DIRECTION_DISTRIBUTION_COLUMNS) and directions.empty
    assert list(timeline.columns) == list(TIME_COLUMNS) and timeline.empty
    empty_events = build_event_summary(pd.DataFrame(), {})
    assert list(empty_events.columns) == list(EVENT_SUMMARY_COLUMNS) and empty_events.empty


def test_reconciliation_failure_for_class_counts() -> None:
    summary, classes, directions, _ = analytics()
    broken = classes.copy()
    broken.loc[0, "crossing_count"] += 1
    with pytest.raises(ValueError, match="class crossing counts"):
        reconcile_crossing_totals(summary, broken, directions)


def test_reconciliation_failure_for_direction_counts() -> None:
    summary, classes, directions, _ = analytics()
    broken = directions.copy()
    broken.loc[0, "crossing_count"] -= 1
    with pytest.raises(ValueError, match="direction crossing counts"):
        reconcile_crossing_totals(summary, classes, broken)


def test_cross_stage_line_crossing_reconciliation_failure() -> None:
    summary, _, _, _ = analytics()
    events = pd.DataFrame([
        ("video", "source", "LINE_CROSSING"),
    ], columns=["video_id", "source_id", "event_type"])
    with pytest.raises(ValueError, match="Stage 14 LINE_CROSSING"):
        reconcile_event_line_crossings(summary, events)
