#!/usr/bin/env python3
"""Profile Stage 1 runtime videos using metadata plus one-frame validation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.video.metadata import VIDEO_METADATA_FIELDS, profile_video

SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "video_metadata.csv"


def load_runtime_video_sources(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            row
            for row in rows
            if row["asset_type"] == "video" and row["role"] == "runtime_demo"
        ]


def main() -> int:
    sources = load_runtime_video_sources(SOURCE_MANIFEST)
    if not sources:
        print(f"No runtime video sources found in {SOURCE_MANIFEST}", file=sys.stderr)
        return 1

    profiles: list[dict[str, object]] = []
    for source in sources:
        relative_path = Path(source["local_path"])
        profile = profile_video(
            PROJECT_ROOT / relative_path,
            video_id=relative_path.stem,
            source_id=source["source_id"],
        )
        profiles.append(profile)
        print(
            f"{profile['source_id']}: {profile['validation_status']} - "
            f"{profile['width']}x{profile['height']} @ {profile['fps']} fps, "
            f"{profile['frame_count']} frames"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=VIDEO_METADATA_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(profiles)

    failures = sum(profile["validation_status"] == "FAIL" for profile in profiles)
    warnings = sum(profile["validation_status"] == "WARNING" for profile in profiles)
    print(f"Wrote {len(profiles)} rows to {OUTPUT_PATH}")
    print(f"Validation summary: {len(profiles) - failures - warnings} PASS, {warnings} WARNING, {failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
