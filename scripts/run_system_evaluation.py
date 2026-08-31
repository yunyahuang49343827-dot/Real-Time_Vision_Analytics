#!/usr/bin/env python3
"""Build Stage 19 sampled system-evaluation artifacts without rerunning CV inference."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.evaluation.system import (  # noqa: E402
    aggregate_event_review,
    aggregate_tracking_review,
    crossing_confusion_metrics,
    sha256_file,
    validate_evidence_trace,
    validate_runtime_model,
    validate_system_manifest,
)

SOURCE_DIR = PROJECT_ROOT / "data/manifests/stage19"
OUTPUT_DIR = PROJECT_ROOT / "outputs/evaluation/stage19"


def main() -> int:
    config_path = PROJECT_ROOT / "configs/system_evaluation_stage19.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(PROJECT_ROOT / "data/interim/video_metadata.csv")
    frame_counts = dict(zip(metadata["video_id"], metadata["frame_count"]))

    system_manifest = pd.read_csv(SOURCE_DIR / "system_evaluation_manifest.csv")
    tracking = pd.read_csv(SOURCE_DIR / "tracking_review.csv")
    crossings = pd.read_csv(SOURCE_DIR / "crossing_review.csv")
    events_review = pd.read_csv(SOURCE_DIR / "event_review.csv")
    evidence_review = pd.read_csv(SOURCE_DIR / "evidence_review.csv")
    controlled = pd.read_csv(SOURCE_DIR / "controlled_validation_manifest.csv")

    validate_system_manifest(system_manifest, frame_counts)
    governance = validate_runtime_model(
        PROJECT_ROOT / config["runtime_model"],
        str(config["runtime_model_sha256"]),
        PROJECT_ROOT / config["rejected_candidate"],
    )
    if bool(config.get("stage18_locked_test_used")):
        raise ValueError("Stage 18 LOCKED_TEST must not be used for Stage 19 system evaluation")
    if set(controlled["source_type"]) != {"CONTROLLED_SYNTHETIC"}:
        raise ValueError("controlled cases must be explicitly labeled CONTROLLED_SYNTHETIC")

    evidence_trace, evidence_metrics = validate_evidence_trace(
        evidence_review,
        pd.read_csv(PROJECT_ROOT / "outputs/events/stage14/events.csv"),
        pd.read_csv(PROJECT_ROOT / "outputs/evidence/stage14/evidence_manifest.csv"),
        PROJECT_ROOT,
    )
    if evidence_metrics["failed"]:
        raise ValueError("evidence trace validation failed")

    metrics = {
        "stage": 19,
        "evaluation_protocol": "sampled manual system review",
        "runtime_governance": governance,
        "stage18_locked_test_used": 0,
        "evaluation_ranges": len(system_manifest),
        "scenes_covered": sorted(system_manifest["scene"].unique()),
        "tracking": aggregate_tracking_review(tracking),
        "crossing": crossing_confusion_metrics(crossings),
        "events": aggregate_event_review(events_review),
        "controlled_validation": {
            "cases": len(controlled), "passed": int((controlled["expected_result"] == controlled["observed_result"]).sum()),
            "source_label": "CONTROLLED_SYNTHETIC",
        },
        "evidence_integrity": evidence_metrics,
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "stage14_events_sha256": sha256_file(PROJECT_ROOT / "outputs/events/stage14/events.csv"),
            "stage14_evidence_manifest_sha256": sha256_file(PROJECT_ROOT / "outputs/evidence/stage14/evidence_manifest.csv"),
        },
        "limitations": [
            "Tracking continuity uses manually sampled physical objects; MOTA, HOTA, and IDF1 are not reported.",
            "Crossing precision/recall applies only to selected reviewed references, not the full videos.",
            "Proximity review concerns normalized image-space usefulness, not collision probability or physical distance.",
            "Controlled positive paths are synthetic system validation and are not natural Taiwan incidents.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "system_evaluation_manifest.csv", "tracking_review.csv", "crossing_review.csv",
        "event_review.csv", "controlled_validation_manifest.csv",
    ):
        shutil.copy2(SOURCE_DIR / name, OUTPUT_DIR / name)
    evidence_trace.to_csv(OUTPUT_DIR / "evidence_integrity_review.csv", index=False)
    (OUTPUT_DIR / "system_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "evaluation_manifest.json").write_text(json.dumps({
        "runtime_model": config["runtime_model"],
        "runtime_model_sha256": governance["runtime_model_sha256"],
        "rejected_candidate": config["rejected_candidate"],
        "rejected_candidate_used": 0,
        "stage18_locked_test_used": 0,
        "system_evaluation_manifest_sha256": sha256_file(SOURCE_DIR / "system_evaluation_manifest.csv"),
        "review_files": {
            name: sha256_file(SOURCE_DIR / name) for name in (
                "tracking_review.csv", "crossing_review.csv", "event_review.csv",
                "controlled_validation_manifest.csv", "evidence_review.csv",
            )
        },
    }, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "error_propagation_summary.md").write_text(_error_summary(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


def _error_summary(metrics: dict) -> str:
    crossing = metrics["crossing"]
    tracking = metrics["tracking"]
    return f"""# Stage 19 End-to-End Error Propagation Summary

## Scope and governance

This evaluation covers six short frame ranges from the four runtime videos. Runtime provenance is fixed to `models/pretrained/yolo26n.pt`; the rejected Stage 17 candidate and the Stage 18 detection LOCKED_TEST were not used. Manual statistics apply only to the documented sample.

## Sample observations

- Tracking: {tracking['physical_objects_reviewed']} physical objects reviewed, {tracking['fragmented_objects']} fragmentation candidates ({tracking['fragmentation_count']} fragment breaks), and {tracking['id_switch_objects']} verified ID-switch objects. Fragmentation means one physical object was reacquired under a new ID; an ID switch means an existing ID transferred to another physical object. They are not interchangeable.
- Line crossing: {crossing['category_counts']['CORRECT']} correct, {crossing['category_counts']['MISSED']} missed, {crossing['category_counts']['FALSE']} false, and {crossing['category_counts']['DUPLICATE']} duplicate in selected references. Sample precision and recall are descriptive only.
- Wrong-way: all three natural candidates were false operational alarms caused by perspective-sensitive image-space direction, not verified wrong-way incidents.
- Long dwell: highway candidates captured real observed zone residence, but largely reflected slow moving traffic and broad-zone perspective; aerial candidates were especially sensitive to broad ROI and small-object gaps.
- Proximity: several warnings were useful review candidates, while rider/self-motorcycle overlap, occlusion, and perspective created false or ambiguous warnings. This remains normalized image-space proximity, not physical risk.

## Propagation chains

- Detection miss → missing Track → missed crossing or missed/shortened event episode. This was most visible for aerial small objects.
- Tracking fragmentation → potential duplicate crossing, broken dwell continuity, or a repeated person–vehicle pair episode. No duplicate was confirmed in the selected crossing references, but two fragmentation candidates show the mechanism remains possible.
- Bounding-box jitter and perspective → direction instability → wrong-way false candidate. All three natural wrong-way candidates followed this chain.
- Person taxonomy and small-object miss → reduced person/motorcycle track coverage → proximity and intrusion limitations. A rider may also appear as both person and motorcycle, complicating operational interpretation.
- Class instability (car/bus/truck/commercial van) can change per-class analytics without changing identity; it is distinct from track fragmentation.

## Evidence integrity

All {metrics['evidence_integrity']['reviewed']} sampled evidence records passed event ID, expected frame, primary Track, path, filename, non-empty file, and OpenCV readability checks. Evidence is a review artifact, not Ground Truth.

## Controlled validation

The positive paths for WRONG_WAY, STATIONARY_VEHICLE, and PEDESTRIAN_INTRUSION were recorded as `CONTROLLED_SYNTHETIC`. These cases validate system paths only and are not real Taiwan incidents.
"""


if __name__ == "__main__":
    raise SystemExit(main())
