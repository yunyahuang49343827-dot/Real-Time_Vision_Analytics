"""Governed helpers for targeted annotation-frame candidate preparation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2


ALLOWED_COVERAGE_TAGS = frozenset({
    "PERSON_PRESENT", "SMALL_PERSON", "MULTIPLE_PERSON", "PERSON_MOTORCYCLE",
    "PERSON_BICYCLE", "PERSON_CAR", "DENSE_TRAFFIC", "OCCLUSION",
    "INTERSECTION", "CROSSWALK", "AERIAL_SMALL_OBJECT", "HARD_NEGATIVE",
})
ALLOWED_METHODS = frozenset({
    "MODEL_ASSISTED_POSITIVE", "SYSTEMATIC_TEMPORAL", "DIFFICULT_HEURISTIC",
    "HARD_NEGATIVE_CONTEXT",
})


def frame_in_ranges(frame_id: int, ranges: Sequence[tuple[int, int]]) -> bool:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    return any(start <= frame_id <= end for start, end in ranges)


def timestamp_seconds(frame_id: int, fps: float) -> float:
    if frame_id < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("frame_id and fps must be valid")
    return frame_id / fps


def deterministic_candidate_id(scene: str, frame_id: int) -> str:
    normalized = scene.strip().lower().replace(" ", "_")
    if not normalized or frame_id < 0:
        raise ValueError("scene and frame_id must be valid")
    return f"{normalized}_frame_{frame_id:06d}"


def extract_frame(video_path: Path, frame_id: int):
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"cannot decode frame {frame_id} from {video_path}")
    return frame


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def balanced_temporal_select(
    rows: Sequence[Mapping[str, object]], *, target: int, minimum_gap_frames: int,
    method_fractions: Mapping[str, float],
) -> list[Mapping[str, object]]:
    """Greedily balance sampling methods while enforcing one scene-wide frame gap."""
    if target <= 0 or minimum_gap_frames <= 0:
        raise ValueError("target and minimum_gap_frames must be positive")
    if set(method_fractions) != ALLOWED_METHODS or not math.isclose(
        sum(method_fractions.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("method fractions must cover allowed methods and sum to one")
    unique = {int(row["frame_id"]): row for row in rows}
    selected: list[Mapping[str, object]] = []
    selected_frames: list[int] = []

    def add(row: Mapping[str, object]) -> bool:
        frame_id = int(row["frame_id"])
        if any(abs(frame_id - other) < minimum_gap_frames for other in selected_frames):
            return False
        selected.append(row); selected_frames.append(frame_id)
        return True

    by_method: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in unique.values():
        method = str(row["sampling_method"])
        if method not in ALLOWED_METHODS:
            raise ValueError(f"unsupported sampling method: {method}")
        by_method[method].append(row)
    for method in sorted(ALLOWED_METHODS):
        quota = round(target * method_fractions[method])
        ranked = sorted(by_method[method], key=lambda row: (-float(row["priority_score"]), int(row["frame_id"])))
        accepted = 0
        for row in ranked:
            if add(row):
                accepted += 1
            if accepted >= quota:
                break
    ranked_all = sorted(unique.values(), key=lambda row: (-float(row["priority_score"]), int(row["frame_id"])))
    for row in ranked_all:
        if len(selected) >= target:
            break
        if int(row["frame_id"]) not in selected_frames:
            add(row)
    return sorted(selected, key=lambda row: int(row["frame_id"]))


def choose_duplicate_representatives(
    rows: Sequence[Mapping[str, object]],
) -> tuple[set[str], dict[str, str]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["duplicate_group_id"])].append(row)
    selected: set[str] = set()
    reasons: dict[str, str] = {}
    for group_id, members in groups.items():
        ranked = sorted(members, key=lambda row: (
            -float(row["priority_score"]), int(row["frame_id"]), str(row["candidate_id"])
        ))
        keep = str(ranked[0]["candidate_id"]); selected.add(keep)
        reasons[keep] = f"DUPLICATE_GROUP_REPRESENTATIVE:{group_id}"
        for row in ranked[1:]:
            reasons[str(row["candidate_id"])] = f"DUPLICATE_REDUCED:{group_id}"
    return selected, reasons


def conservative_duplicate_groups(
    rows: Sequence[Mapping[str, object]], *, threshold: int,
    max_frame_gap_by_video: Mapping[str, int],
) -> list[dict[str, object]]:
    """Group exact matches globally and dHash matches only within a short window."""
    if threshold < 0 or threshold > 64:
        raise ValueError("threshold must be in 0..64")
    ordered = sorted(rows, key=lambda row: (
        str(row["video_id"]), int(row["frame_id"]), str(row["candidate_id"])
    ))
    if len({str(row["candidate_id"]) for row in ordered}) != len(ordered):
        raise ValueError("candidate_id must be unique")
    groups: list[list[Mapping[str, object]]] = []
    exact_group: dict[str, int] = {}
    leaders_by_video: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row in ordered:
        sha = str(row["image_sha256"]); video_id = str(row["video_id"])
        frame_id = int(row["frame_id"]); dhash = str(row["dhash"])
        if sha in exact_group:
            groups[exact_group[sha]].append(row)
            continue
        matched: int | None = None
        max_gap = int(max_frame_gap_by_video[video_id])
        for leader_frame, group_index, leader_hash in reversed(leaders_by_video[video_id]):
            if frame_id - leader_frame > max_gap:
                break
            if (int(dhash, 16) ^ int(leader_hash, 16)).bit_count() <= threshold:
                matched = group_index
                break
        if matched is None:
            matched = len(groups)
            groups.append([])
            leaders_by_video[video_id].append((frame_id, matched, dhash))
        groups[matched].append(row)
        exact_group[sha] = matched
    result: list[dict[str, object]] = []
    for index, members in enumerate(groups, start=1):
        sha_counts: dict[str, int] = defaultdict(int)
        for row in members:
            sha_counts[str(row["image_sha256"])] += 1
        for row in members:
            exact_size = sha_counts[str(row["image_sha256"])]
            result.append({
                "candidate_id": str(row["candidate_id"]),
                "duplicate_group_id": f"V2_FRAME_DUP_{index:05d}",
                "group_size": len(members),
                "exact_duplicate_group_size": exact_size,
                "grouping_basis": (
                    "EXACT" if exact_size > 1 else "NEAR" if len(members) > 1 else "SINGLETON"
                ),
            })
    return result


def validate_candidate_manifest(
    rows: Sequence[Mapping[str, object]], *, excluded_ranges: Mapping[str, Sequence[tuple[int, int]]],
) -> None:
    if not rows:
        raise ValueError("candidate manifest is empty")
    ids = [str(row["candidate_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id must be unique")
    for row in rows:
        video_id, frame_id = str(row["video_id"]), int(row["frame_id"])
        overlap = frame_in_ranges(frame_id, excluded_ranges.get(video_id, ()))
        if overlap or bool(row.get("stage19_overlap")):
            raise ValueError(f"Stage 19 overlap detected: {video_id}:{frame_id}")
        tags = {tag for tag in str(row.get("coverage_tags", "")).split(";") if tag}
        if not tags.issubset(ALLOWED_COVERAGE_TAGS):
            raise ValueError(f"invalid coverage tags: {sorted(tags - ALLOWED_COVERAGE_TAGS)}")
        if str(row.get("sampling_method")) not in ALLOWED_METHODS:
            raise ValueError("invalid sampling method")
        if str(row.get("annotation_status")) != "NOT_ANNOTATED":
            raise ValueError("annotation_status must remain NOT_ANNOTATED")
        for field in ("image_sha256", "dhash", "duplicate_group_id", "image_path"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"required field is empty: {field}")
