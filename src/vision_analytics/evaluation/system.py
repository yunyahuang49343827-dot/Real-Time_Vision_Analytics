"""Deterministic helpers for sampled tracking and event-system evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import cv2
import pandas as pd

SCENES = frozenset({"Highway", "Taipei", "Urban", "Aerial"})
CROSSING_CATEGORIES = frozenset({"CORRECT", "MISSED", "FALSE", "DUPLICATE"})
EVENT_CATEGORIES = frozenset({
    "TRUE_EVENT", "FALSE_EVENT", "USEFUL_REVIEW_CANDIDATE", "AMBIGUOUS"
})
SOURCE_TYPES = frozenset({"NATURAL", "CONTROLLED_SYNTHETIC"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_system_manifest(manifest: pd.DataFrame, frame_counts: Mapping[str, int]) -> None:
    required = {"scene", "video_id", "start_frame", "end_frame", "evaluation_purpose"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"system evaluation manifest missing columns: {sorted(missing)}")
    if set(manifest["scene"]) != SCENES:
        raise ValueError("system evaluation set must cover Highway, Taipei, Urban, and Aerial")
    for row in manifest.to_dict("records"):
        video_id = str(row["video_id"])
        if video_id not in frame_counts:
            raise ValueError(f"unknown video_id: {video_id}")
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if start < 0 or end < start or end >= int(frame_counts[video_id]):
            raise ValueError(f"invalid evaluation frame range for {video_id}: {start}-{end}")


def aggregate_tracking_review(review: pd.DataFrame) -> dict[str, object]:
    required = {
        "physical_object_id", "scene", "video_id", "class_name", "start_frame",
        "end_frame", "track_ids_observed", "fragmentation_count", "id_switch_observed",
    }
    if required - set(review.columns):
        raise ValueError("tracking review schema is incomplete")
    if review["physical_object_id"].duplicated().any():
        raise ValueError("physical_object_id must be unique")
    fragments = pd.to_numeric(review["fragmentation_count"], errors="raise")
    if (fragments < 0).any():
        raise ValueError("fragmentation_count cannot be negative")
    switches = review["id_switch_observed"].astype(str).str.lower().map({"true": True, "false": False})
    if switches.isna().any():
        raise ValueError("id_switch_observed must be true or false")
    reviewed = len(review)
    return {
        "physical_objects_reviewed": reviewed,
        "unfragmented_objects": int((fragments == 0).sum()),
        "fragmented_objects": int((fragments > 0).sum()),
        "fragmentation_count": int(fragments.sum()),
        "id_switch_objects": int(switches.sum()),
        "sampled_unfragmented_rate": float((fragments == 0).sum() / reviewed) if reviewed else None,
        "scope": "manually sampled physical objects only; not MOTA/HOTA/IDF1",
    }


def crossing_confusion_metrics(review: pd.DataFrame) -> dict[str, object]:
    if "review_category" not in review or "reference_id" not in review:
        raise ValueError("crossing review schema is incomplete")
    if review["reference_id"].duplicated().any():
        raise ValueError("crossing reference_id must be unique")
    categories = set(review["review_category"])
    if not categories <= CROSSING_CATEGORIES:
        raise ValueError(f"invalid crossing categories: {sorted(categories - CROSSING_CATEGORIES)}")
    counts = review["review_category"].value_counts().reindex(sorted(CROSSING_CATEGORIES), fill_value=0)
    tp, fn = int(counts["CORRECT"]), int(counts["MISSED"])
    fp = int(counts["FALSE"] + counts["DUPLICATE"])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "category_counts": {name: int(counts[name]) for name in sorted(CROSSING_CATEGORIES)},
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "sample_precision": precision, "sample_recall": recall,
        "scope": "selected clips only; not full-dataset counting accuracy",
    }


def aggregate_event_review(review: pd.DataFrame) -> dict[str, object]:
    required = {"review_id", "event_type", "review_result", "source_type"}
    if required - set(review.columns):
        raise ValueError("event review schema is incomplete")
    if review["review_id"].duplicated().any():
        raise ValueError("event review_id must be unique")
    invalid_results = set(review["review_result"]) - EVENT_CATEGORIES
    invalid_sources = set(review["source_type"]) - SOURCE_TYPES
    if invalid_results or invalid_sources:
        raise ValueError("event review contains invalid result or source type")
    grouped = review.groupby(["source_type", "event_type", "review_result"]).size()
    return {
        "total_reviewed": len(review),
        "natural_reviewed": int((review["source_type"] == "NATURAL").sum()),
        "controlled_reviewed": int((review["source_type"] == "CONTROLLED_SYNTHETIC").sum()),
        "breakdown": [
            {"source_type": a, "event_type": b, "review_result": c, "count": int(n)}
            for (a, b, c), n in grouped.items()
        ],
    }


def validate_runtime_model(runtime_path: Path, expected_sha256: str, rejected_path: Path) -> dict[str, object]:
    runtime = Path(runtime_path).resolve()
    rejected = Path(rejected_path).resolve()
    if runtime == rejected:
        raise ValueError("rejected Stage 17 candidate cannot be used by Stage 19")
    actual_hash = sha256_file(runtime)
    if actual_hash != expected_sha256:
        raise ValueError("runtime model hash does not match governed pretrained model")
    return {
        "runtime_model": str(runtime_path), "runtime_model_sha256": actual_hash,
        "rejected_candidate_used": 0, "governance_status": "PASS",
    }


def validate_evidence_trace(
    review: pd.DataFrame, events: pd.DataFrame, evidence_manifest: pd.DataFrame, project_root: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    event_index = events.set_index("event_id", drop=False)
    evidence_index = evidence_manifest.set_index("event_id", drop=False)
    rows = []
    for row in review.to_dict("records"):
        event_id = str(row["event_id"])
        valid = event_id in event_index.index and event_id in evidence_index.index
        message = ""
        if valid:
            event = event_index.loc[event_id]
            evidence = evidence_index.loc[event_id]
            path = project_root / str(evidence["evidence_path"])
            checks = {
                "frame_match": int(event["frame_index"]) == int(evidence["frame_index"]) == int(row["expected_frame"]),
                "track_match": int(event["track_id"]) == int(row["expected_track_id"]),
                "path_match": str(event["evidence_path"]) == str(evidence["evidence_path"]),
                "filename_match": path.name == f"{event_id}.jpg",
                "file_exists": path.is_file() and path.stat().st_size > 0,
                "image_readable": cv2.imread(str(path)) is not None,
            }
            valid = all(checks.values())
            message = ";".join(name for name, ok in checks.items() if not ok)
        else:
            message = "event_or_manifest_row_missing"
        rows.append({**row, "trace_status": "PASS" if valid else "FAIL", "validation_message": message})
    result = pd.DataFrame(rows)
    counts = {"reviewed": len(result), "passed": int((result["trace_status"] == "PASS").sum())}
    counts["failed"] = counts["reviewed"] - counts["passed"]
    return result, counts
