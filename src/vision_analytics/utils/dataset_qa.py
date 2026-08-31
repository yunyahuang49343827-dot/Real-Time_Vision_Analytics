"""Dataset QA, duplicate grouping, and split-governance primitives."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2

SPLITS = ("TRAIN", "VAL", "LOCKED_TEST")


@dataclass(frozen=True, slots=True)
class Annotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area_normalized(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class AnnotationIssue:
    issue_type: str
    severity: str
    line_number: int | None
    message: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(image: object) -> str:
    """Return a transparent 64-bit difference hash for a decoded image."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("decoded image is required")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(first: str, second: str) -> int:
    if len(first) != 16 or len(second) != 16:
        raise ValueError("dHash values must be 16 hexadecimal characters")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def parse_yolo_annotation(
    text: str,
    *,
    class_count: int,
) -> tuple[list[Annotation], list[AnnotationIssue]]:
    annotations: list[Annotation] = []
    issues: list[AnnotationIssue] = []
    if not text.strip():
        issues.append(AnnotationIssue(
            "EMPTY_ANNOTATION", "WARNING", None,
            "Empty label retained; source semantics for negative images are unconfirmed.",
        ))
        return annotations, issues

    seen: set[tuple[object, ...]] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            issues.append(AnnotationIssue("BLANK_ROW", "WARNING", line_number, "Blank row ignored."))
            continue
        columns = line.split()
        if len(columns) != 5:
            issues.append(AnnotationIssue(
                "INVALID_ROW_FORMAT", "ERROR", line_number,
                f"Expected 5 columns, found {len(columns)}.",
            ))
            continue
        try:
            class_id = int(columns[0])
            values = tuple(float(value) for value in columns[1:])
        except ValueError:
            issues.append(AnnotationIssue(
                "INVALID_ROW_FORMAT", "ERROR", line_number,
                "Class ID must be an integer and coordinates must be numeric.",
            ))
            continue
        if class_id < 0 or class_id >= class_count:
            issues.append(AnnotationIssue(
                "INVALID_CLASS_ID", "ERROR", line_number,
                f"Class ID {class_id} is outside [0, {class_count - 1}].",
            ))
            continue
        if not all(math.isfinite(value) for value in values):
            issues.append(AnnotationIssue(
                "INVALID_COORDINATES", "ERROR", line_number, "Coordinates must be finite.",
            ))
            continue
        x_center, y_center, width, height = values
        if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
            issues.append(AnnotationIssue(
                "INVALID_COORDINATES", "ERROR", line_number,
                "Normalized x/y center must be in [0, 1].",
            ))
            continue
        if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
            issues.append(AnnotationIssue(
                "INVALID_BOX_SIZE", "ERROR", line_number,
                "Normalized width/height must be in (0, 1].",
            ))
            continue
        if (
            x_center - width / 2 < -1e-6 or x_center + width / 2 > 1.0 + 1e-6
            or y_center - height / 2 < -1e-6 or y_center + height / 2 > 1.0 + 1e-6
        ):
            issues.append(AnnotationIssue(
                "BOX_OUTSIDE_IMAGE", "ERROR", line_number,
                "Normalized box extent exceeds image bounds.",
            ))
            continue
        key = (class_id,) + tuple(round(value, 9) for value in values)
        if key in seen:
            issues.append(AnnotationIssue(
                "DUPLICATE_ANNOTATION", "ERROR", line_number,
                "Exact duplicate annotation row.",
            ))
            continue
        seen.add(key)
        annotations.append(Annotation(class_id, x_center, y_center, width, height))
    return annotations, issues


def map_source_class(source_class: str, mapping: Mapping[str, str]) -> str:
    if source_class not in mapping:
        raise ValueError(f"missing application taxonomy mapping for {source_class}")
    return mapping[source_class]


def find_unmatched_assets(
    image_keys: Iterable[str], label_keys: Iterable[str],
) -> tuple[set[str], set[str]]:
    images, labels = set(image_keys), set(label_keys)
    return images - labels, labels - images


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            keep, merge = sorted((first_root, second_root))
            self.parent[merge] = keep


def build_duplicate_groups(
    records: Sequence[Mapping[str, object]],
    *,
    near_duplicate_threshold: int,
) -> list[dict[str, object]]:
    """Group exact, near, and explicit source groups without claiming scene truth."""
    if near_duplicate_threshold < 0 or near_duplicate_threshold > 64:
        raise ValueError("near_duplicate_threshold must be between 0 and 64")
    image_ids = [str(record["image_id"]) for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_id must be unique")
    disjoint = _DisjointSet(image_ids)
    exact_members: dict[str, list[str]] = {}
    source_members: dict[str, list[str]] = {}
    for record in records:
        image_id = str(record["image_id"])
        exact_members.setdefault(str(record["sha256"]), []).append(image_id)
        source_group = str(record.get("source_group_id") or "")
        if source_group:
            source_members.setdefault(source_group, []).append(image_id)
    for members in list(exact_members.values()) + list(source_members.values()):
        for image_id in members[1:]:
            disjoint.union(members[0], image_id)

    for index, first in enumerate(records):
        for second in records[index + 1:]:
            if hamming_distance(str(first["phash"]), str(second["phash"])) <= near_duplicate_threshold:
                disjoint.union(str(first["image_id"]), str(second["image_id"]))

    roots = {image_id: disjoint.find(image_id) for image_id in image_ids}
    ordered_roots = {root: f"GROUP_{index:05d}" for index, root in enumerate(sorted(set(roots.values())), start=1)}
    group_sizes: dict[str, int] = {}
    for root in roots.values():
        group_sizes[root] = group_sizes.get(root, 0) + 1
    results: list[dict[str, object]] = []
    for record in records:
        image_id = str(record["image_id"])
        sha = str(record["sha256"])
        root = roots[image_id]
        results.append({
            "image_id": image_id,
            "group_id": ordered_roots[root],
            "sha256": sha,
            "phash": str(record["phash"]),
            "exact_duplicate_count": len(exact_members[sha]),
            "group_size": group_sizes[root],
            "source_group_id": str(record.get("source_group_id") or ""),
            "grouping_basis": "EXACT_OR_NEAR_OR_SOURCE" if group_sizes[root] > 1 else "SINGLETON",
        })
    return results


def deterministic_group_split(
    records: Sequence[Mapping[str, object]],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to deterministic splits, prioritizing group integrity."""
    if set(ratios) != set(SPLITS) or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("ratios must define TRAIN/VAL/LOCKED_TEST and sum to 1")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("split ratios must be positive")
    group_sizes: dict[str, int] = {}
    for record in records:
        group_id = str(record["group_id"])
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
    if len(group_sizes) < 3:
        raise ValueError("at least three groups are required to create all splits")

    rng = random.Random(seed)
    random_keys = {group_id: rng.random() for group_id in sorted(group_sizes)}
    groups = sorted(group_sizes, key=lambda group_id: (-group_sizes[group_id], random_keys[group_id], group_id))
    total = sum(group_sizes.values())
    target = {split: ratios[split] * total for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}
    for index, group_id in enumerate(groups):
        if index < len(SPLITS):
            split = SPLITS[index]
        else:
            split = max(SPLITS, key=lambda name: (target[name] - counts[name], -SPLITS.index(name)))
        assignments[group_id] = split
        counts[split] += group_sizes[group_id]
    if set(assignments.values()) != set(SPLITS):
        raise ValueError("TRAIN, VAL, and LOCKED_TEST must all exist")
    return assignments


def validate_split_manifest(records: Sequence[Mapping[str, object]]) -> None:
    included = [record for record in records if str(record.get("split")) != "EXCLUDED"]
    image_ids = [str(record["image_id"]) for record in included]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("an image appears more than once in the split manifest")
    if {str(record["split"]) for record in included} != set(SPLITS):
        raise ValueError("TRAIN, VAL, and LOCKED_TEST must all exist")
    group_splits: dict[str, set[str]] = {}
    for record in included:
        group_splits.setdefault(str(record["group_id"]), set()).add(str(record["split"]))
    leaking = [group_id for group_id, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"groups cross splits: {leaking[:5]}")
