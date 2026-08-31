#!/usr/bin/env python3
"""Build Stage 15 traffic analytics from existing structured artifacts only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.analytics.traffic import (
    ZONE_COLUMNS,
    build_crossing_analytics,
    build_event_summary,
    load_traffic_analytics_config,
    reconcile_event_line_crossings,
)

SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes.yaml"
STAGE8_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage8"
STAGE9_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage9"
STAGE14_EVENTS = PROJECT_ROOT / "outputs" / "events" / "stage14" / "events.csv"
STAGE14_BENCHMARK = (
    PROJECT_ROOT / "outputs" / "events" / "stage14" / "stage14_evidence_benchmark.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analytics" / "stage15"


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def load_zone_summaries(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for zone_id, values in payload.get("zone_diagnostics", {}).items():
            rows.append({
                "video_id": payload["video_id"],
                "source_id": payload["source_id"],
                "zone_id": zone_id,
                "peak_observed_occupancy": int(values["peak_observed_occupancy"]),
                "current_observed_occupancy": int(values["current_observed_occupancy"]),
                "tracks_observed_inside": int(values["tracks_observed_inside"]),
                "zone_entry_count": int(values["zone_entry_count"]),
                "zone_exit_count": int(values["zone_exit_count"]),
            })
    return pd.DataFrame(rows, columns=ZONE_COLUMNS)


def load_durations() -> dict[str, float]:
    with STAGE14_BENCHMARK.open(newline="", encoding="utf-8") as handle:
        return {
            row["video_id"]: float(row["frames_processed"]) / float(row["source_fps"])
            for row in csv.DictReader(handle)
        }


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def write_video_summaries(
    summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    direction_distribution: pd.DataFrame,
    traffic_over_time: pd.DataFrame,
    event_summary: pd.DataFrame,
    zones: pd.DataFrame,
    *,
    time_bucket_seconds: int,
) -> None:
    for _, summary_row in summary.iterrows():
        video_id = str(summary_row["video_id"])
        selector = lambda frame: frame[frame["video_id"].eq(video_id)]
        payload = {
            "video_id": video_id,
            "source_id": summary_row["source_id"],
            "traffic_summary": summary_row.to_dict(),
            "class_distribution": records(selector(class_distribution)),
            "direction_distribution": records(selector(direction_distribution)),
            "traffic_over_time": records(selector(traffic_over_time)),
            "zone_occupancy_summary": records(selector(zones)),
            "event_summary": records(selector(event_summary)),
            "time_bucket_seconds": time_bucket_seconds,
            "density_warning": (
                "LOW/MEDIUM/HIGH is a project heuristic based on Stage 9 peak observed "
                "image-space occupancy, not an official congestion standard."
            ),
            "occupancy_limitation": (
                "Stage 9 provides summary peak/current occupancy only; no complete occupancy "
                "time series is claimed or generated."
            ),
            "event_warning": "Event frequencies are rule-generated system outputs, not verified incidents.",
            "status": "PASS",
        }
        (OUTPUT_DIR / f"{video_id}_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> int:
    required = (STAGE8_DIR, STAGE9_DIR, STAGE14_EVENTS, STAGE14_BENCHMARK)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"Missing required structured artifacts: {missing}", file=sys.stderr)
        return 1

    crossing_paths = sorted(STAGE8_DIR.glob("*_crossings.csv"))
    zone_paths = sorted(STAGE9_DIR.glob("*_summary.json"))
    if len(crossing_paths) != 4 or len(zone_paths) != 4:
        print("Expected exactly four Stage 8 crossing files and four Stage 9 summaries", file=sys.stderr)
        return 1

    config = load_traffic_analytics_config(SCENE_CONFIG)
    crossings = load_csvs(crossing_paths)
    zones = load_zone_summaries(zone_paths)
    events = pd.read_csv(STAGE14_EVENTS)
    durations = load_durations()

    summary, class_distribution, direction_distribution, traffic_over_time = (
        build_crossing_analytics(crossings, config=config, zones=zones)
    )
    event_summary = build_event_summary(events, durations)
    reconcile_event_line_crossings(summary, events)
    if len(summary) != 4:
        print(f"Expected four traffic summaries, got {len(summary)}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "traffic_summary.csv", index=False)
    class_distribution.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    direction_distribution.to_csv(OUTPUT_DIR / "direction_distribution.csv", index=False)
    traffic_over_time.to_csv(OUTPUT_DIR / "traffic_over_time.csv", index=False)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)
    write_video_summaries(
        summary, class_distribution, direction_distribution, traffic_over_time,
        event_summary, zones, time_bucket_seconds=config.time_bucket_seconds,
    )

    print(f"Read {len(crossings)} Stage 8 line-crossing records")
    print(f"Read {len(zones)} Stage 9 zone summary records")
    print(f"Read {len(events)} Stage 14 event records")
    for row in summary.to_dict(orient="records"):
        print(
            f"  {row['source_id']}: crossings={row['total_line_crossing_count']}, "
            f"peak_interval={row['peak_interval_start_seconds']}-"
            f"{row['peak_interval_end_seconds']}s ({row['peak_interval_count']}), "
            f"peak_occupancy={row['zone_peak_occupancy']} density={row['density']} "
            f"reconciliation={row['reconciliation_status']}"
        )
    print(f"Wrote Stage 15 analytics to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
