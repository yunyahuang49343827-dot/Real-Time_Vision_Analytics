#!/usr/bin/env python3
"""Run the Stage 3 OpenCV-only pipeline for all runtime videos."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.video.pipeline import BENCHMARK_FIELDS, process_video

SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "stage3"
BENCHMARK_PATH = (
    PROJECT_ROOT / "outputs" / "analytics" / "stage3_video_benchmark.csv"
)


def load_runtime_video_sources(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def main() -> int:
    sources = load_runtime_video_sources(SOURCE_MANIFEST)
    if not sources:
        print(f"No runtime videos found in {SOURCE_MANIFEST}", file=sys.stderr)
        return 1

    benchmarks: list[dict[str, object]] = []
    for source in sources:
        input_relative = Path(source["local_path"])
        video_id = input_relative.stem
        output_relative = Path("outputs/videos/stage3") / f"{video_id}_stage3.mp4"
        print(f"Processing {source['source_id']} ({input_relative.name})...", flush=True)
        benchmark = process_video(
            PROJECT_ROOT / input_relative,
            PROJECT_ROOT / output_relative,
            video_id=video_id,
            source_id=source["source_id"],
        )
        benchmark["output_path"] = output_relative.as_posix()
        benchmarks.append(benchmark)
        print(
            f"  {benchmark['status']}: {benchmark['frames_processed']} frames in "
            f"{benchmark['elapsed_seconds']}s ({benchmark['processing_fps']} processing FPS)",
            flush=True,
        )

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=BENCHMARK_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(benchmarks)

    failures = sum(row["status"] == "FAIL" for row in benchmarks)
    warnings = sum(row["status"] == "WARNING" for row in benchmarks)
    print(f"Wrote {len(benchmarks)} rows to {BENCHMARK_PATH}")
    print(
        f"Summary: {len(benchmarks) - failures - warnings} PASS, "
        f"{warnings} WARNING, {failures} FAIL"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
