"""Pandas-based aggregation of existing structured traffic outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

MOTORIZED_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})
CROSSING_COLUMNS = (
    "video_id", "source_id", "line_id", "frame_index", "timestamp_seconds",
    "track_id", "class_name", "crossing_direction", "center_x", "center_y",
)
SUMMARY_COLUMNS = (
    "video_id", "source_id", "total_line_crossing_count", "person_crossing_count",
    "motorized_vehicle_crossing_count", "bicycle_crossing_count",
    "peak_interval_start_seconds", "peak_interval_end_seconds", "peak_interval_count",
    "zone_peak_occupancy", "density", "occupancy_source", "reconciliation_status",
)
CLASS_DISTRIBUTION_COLUMNS = (
    "video_id", "source_id", "class_name", "crossing_count", "percentage",
)
DIRECTION_DISTRIBUTION_COLUMNS = (
    "video_id", "source_id", "crossing_direction", "crossing_count", "percentage",
)
TIME_COLUMNS = (
    "video_id", "source_id", "interval_start_seconds", "interval_end_seconds",
    "total_crossing_count", "person_crossing_count",
    "motorized_vehicle_crossing_count", "bicycle_crossing_count",
)
EVENT_SUMMARY_COLUMNS = (
    "video_id", "source_id", "metric", "category", "event_count", "percentage",
    "events_per_minute",
)
ZONE_COLUMNS = (
    "video_id", "source_id", "zone_id", "peak_observed_occupancy",
    "current_observed_occupancy", "tracks_observed_inside", "zone_entry_count",
    "zone_exit_count",
)


@dataclass(frozen=True, slots=True)
class DensityThreshold:
    low_max_occupancy: int
    medium_max_occupancy: int

    def __post_init__(self) -> None:
        if self.low_max_occupancy < 0:
            raise ValueError("low_max_occupancy must be non-negative")
        if self.medium_max_occupancy <= self.low_max_occupancy:
            raise ValueError("medium_max_occupancy must exceed low_max_occupancy")


@dataclass(frozen=True, slots=True)
class TrafficAnalyticsConfig:
    time_bucket_seconds: int
    density_thresholds: Mapping[str, DensityThreshold]

    def __post_init__(self) -> None:
        if self.time_bucket_seconds <= 0:
            raise ValueError("time_bucket_seconds must be positive")
        if not self.density_thresholds:
            raise ValueError("at least one density threshold is required")


def load_traffic_analytics_config(path: Path) -> TrafficAnalyticsConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = payload.get("traffic_analytics")
    if not isinstance(raw, Mapping):
        raise ValueError("traffic_analytics config is required")
    raw_thresholds = raw.get("density_thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("traffic_analytics.density_thresholds is required")
    thresholds = {
        str(source_id): DensityThreshold(
            low_max_occupancy=int(values["low_max_occupancy"]),
            medium_max_occupancy=int(values["medium_max_occupancy"]),
        )
        for source_id, values in raw_thresholds.items()
    }
    return TrafficAnalyticsConfig(
        time_bucket_seconds=int(raw.get("time_bucket_seconds", 10)),
        density_thresholds=thresholds,
    )


def classify_density(peak_occupancy: int, threshold: DensityThreshold) -> str:
    """Classify observed image-space occupancy using a transparent heuristic."""
    if peak_occupancy < 0:
        raise ValueError("peak_occupancy must be non-negative")
    if peak_occupancy <= threshold.low_max_occupancy:
        return "LOW"
    if peak_occupancy <= threshold.medium_max_occupancy:
        return "MEDIUM"
    return "HIGH"


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def build_crossing_analytics(
    crossings: pd.DataFrame,
    *,
    config: TrafficAnalyticsConfig,
    zones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build traffic summary and crossing distributions from Stage 8 records."""
    if crossings.empty:
        return (
            _empty(SUMMARY_COLUMNS), _empty(CLASS_DISTRIBUTION_COLUMNS),
            _empty(DIRECTION_DISTRIBUTION_COLUMNS), _empty(TIME_COLUMNS),
        )
    _require_columns(crossings, CROSSING_COLUMNS, "crossings")
    _require_columns(zones, ZONE_COLUMNS, "zones")
    work = crossings.copy()
    work["timestamp_seconds"] = pd.to_numeric(work["timestamp_seconds"], errors="raise")
    if (work["timestamp_seconds"] < 0).any():
        raise ValueError("crossing timestamps must be non-negative")

    keys = ["video_id", "source_id"]
    totals = work.groupby(keys, sort=True).size().rename("total_line_crossing_count")

    class_distribution = (
        work.groupby(keys + ["class_name"], sort=True).size()
        .rename("crossing_count").reset_index()
    )
    class_distribution["percentage"] = (
        class_distribution["crossing_count"]
        / class_distribution.set_index(keys).index.map(totals)
        * 100.0
    ).round(6)

    direction_distribution = (
        work.groupby(keys + ["crossing_direction"], sort=True).size()
        .rename("crossing_count").reset_index()
    )
    direction_distribution["percentage"] = (
        direction_distribution["crossing_count"]
        / direction_distribution.set_index(keys).index.map(totals)
        * 100.0
    ).round(6)

    bucket = config.time_bucket_seconds
    work["interval_start_seconds"] = (work["timestamp_seconds"] // bucket * bucket).astype(int)
    work["interval_end_seconds"] = work["interval_start_seconds"] + bucket
    work["is_person"] = work["class_name"].eq("person").astype(int)
    work["is_motorized"] = work["class_name"].isin(MOTORIZED_CLASSES).astype(int)
    work["is_bicycle"] = work["class_name"].eq("bicycle").astype(int)
    traffic_over_time = (
        work.groupby(keys + ["interval_start_seconds", "interval_end_seconds"], sort=True)
        .agg(
            total_crossing_count=("track_id", "size"),
            person_crossing_count=("is_person", "sum"),
            motorized_vehicle_crossing_count=("is_motorized", "sum"),
            bicycle_crossing_count=("is_bicycle", "sum"),
        )
        .reset_index()
    )

    zone_peaks = zones.groupby(keys, sort=True)["peak_observed_occupancy"].max()
    summary_rows: list[dict[str, object]] = []
    for (video_id, source_id), group in work.groupby(keys, sort=True):
        if source_id not in config.density_thresholds:
            raise ValueError(f"missing density threshold for {source_id}")
        zone_key = (video_id, source_id)
        if zone_key not in zone_peaks.index:
            raise ValueError(f"missing zone occupancy summary for {video_id}")
        peak_occupancy = int(zone_peaks.loc[zone_key])
        timed = traffic_over_time[
            traffic_over_time["video_id"].eq(video_id)
            & traffic_over_time["source_id"].eq(source_id)
        ].sort_values(["total_crossing_count", "interval_start_seconds"], ascending=[False, True])
        peak_interval = timed.iloc[0]
        summary_rows.append({
            "video_id": video_id,
            "source_id": source_id,
            "total_line_crossing_count": len(group),
            "person_crossing_count": int(group["class_name"].eq("person").sum()),
            "motorized_vehicle_crossing_count": int(group["class_name"].isin(MOTORIZED_CLASSES).sum()),
            "bicycle_crossing_count": int(group["class_name"].eq("bicycle").sum()),
            "peak_interval_start_seconds": int(peak_interval["interval_start_seconds"]),
            "peak_interval_end_seconds": int(peak_interval["interval_end_seconds"]),
            "peak_interval_count": int(peak_interval["total_crossing_count"]),
            "zone_peak_occupancy": peak_occupancy,
            "density": classify_density(peak_occupancy, config.density_thresholds[source_id]),
            "occupancy_source": "stage9_peak_observed_occupancy_summary",
            "reconciliation_status": "PENDING",
        })
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    reconcile_crossing_totals(summary, class_distribution, direction_distribution)
    summary["reconciliation_status"] = "PASS"
    return summary, class_distribution, direction_distribution, traffic_over_time


def reconcile_crossing_totals(
    summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    direction_distribution: pd.DataFrame,
) -> None:
    """Raise instead of silently accepting inconsistent crossing aggregates."""
    if summary.empty:
        return
    keys = ["video_id", "source_id"]
    expected = summary.set_index(keys)["total_line_crossing_count"].astype(int)
    class_totals = class_distribution.groupby(keys)["crossing_count"].sum().astype(int)
    direction_totals = direction_distribution.groupby(keys)["crossing_count"].sum().astype(int)
    if not expected.equals(class_totals.reindex(expected.index)):
        raise ValueError("class crossing counts do not reconcile with total line crossings")
    if not expected.equals(direction_totals.reindex(expected.index)):
        raise ValueError("direction crossing counts do not reconcile with total line crossings")


def reconcile_event_line_crossings(summary: pd.DataFrame, events: pd.DataFrame) -> None:
    """Ensure Stage 14 normalized LINE_CROSSING events still match Stage 8 records."""
    if summary.empty:
        return
    _require_columns(events, ("video_id", "source_id", "event_type"), "events")
    keys = ["video_id", "source_id"]
    expected = summary.set_index(keys)["total_line_crossing_count"].astype(int)
    actual = (
        events[events["event_type"].eq("LINE_CROSSING")]
        .groupby(keys).size().astype(int)
    )
    if not expected.equals(actual.reindex(expected.index)):
        raise ValueError("Stage 14 LINE_CROSSING events do not reconcile with Stage 8 crossings")


def build_event_summary(
    events: pd.DataFrame,
    durations_seconds: Mapping[str, float],
) -> pd.DataFrame:
    """Summarize Stage 14 events separately by type, severity, and status."""
    if events.empty:
        return _empty(EVENT_SUMMARY_COLUMNS)
    required = ("video_id", "source_id", "event_type", "severity", "status")
    _require_columns(events, required, "events")
    rows: list[dict[str, object]] = []
    keys = ["video_id", "source_id"]
    for (video_id, source_id), group in events.groupby(keys, sort=True):
        duration = float(durations_seconds.get(video_id, 0.0))
        if duration <= 0:
            raise ValueError(f"missing positive duration for {video_id}")
        total = len(group)
        for metric, column in (
            ("EVENT_TYPE", "event_type"), ("SEVERITY", "severity"), ("STATUS", "status")
        ):
            for category, count in group[column].value_counts().sort_index().items():
                rows.append({
                    "video_id": video_id,
                    "source_id": source_id,
                    "metric": metric,
                    "category": category,
                    "event_count": int(count),
                    "percentage": round(float(count) / total * 100.0, 6),
                    "events_per_minute": round(float(count) / duration * 60.0, 6),
                })
    return pd.DataFrame(rows, columns=EVENT_SUMMARY_COLUMNS)
