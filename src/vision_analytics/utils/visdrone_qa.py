"""VisDrone2019-DET parsing, taxonomy, coverage, and leakage helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SOURCE_CLASSES = {
    0: "ignored_regions", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}
APPLICATION_MAPPING = {
    "pedestrian": "person", "people": "person", "bicycle": "bicycle",
    "car": "car", "truck": "truck", "bus": "bus", "motor": "motorcycle",
}
UNSUPPORTED_CLASSES = frozenset({"van", "tricycle", "awning-tricycle"})


@dataclass(frozen=True, slots=True)
class VisDroneAnnotation:
    left: float
    top: float
    width: float
    height: float
    score: int
    class_id: int
    truncation: int
    occlusion: int

    @property
    def source_class(self) -> str:
        return SOURCE_CLASSES[self.class_id]

    def normalized(self, image_width: int, image_height: int) -> dict[str, float]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        return {
            "x_center": (self.left + self.width / 2) / image_width,
            "y_center": (self.top + self.height / 2) / image_height,
            "width_normalized": self.width / image_width,
            "height_normalized": self.height / image_height,
            "area_normalized": self.width * self.height / (image_width * image_height),
        }


@dataclass(frozen=True, slots=True)
class VisDroneIssue:
    issue_type: str
    severity: str
    line_number: int | None
    message: str


def parse_visdrone_annotation(
    text: str, *, image_width: int, image_height: int,
) -> tuple[list[VisDroneAnnotation], list[VisDroneIssue]]:
    annotations: list[VisDroneAnnotation] = []
    issues: list[VisDroneIssue] = []
    seen: set[tuple[float | int, ...]] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().rstrip(",")
        if not line:
            continue
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 8:
            issues.append(VisDroneIssue("MALFORMED_ROW", "ERROR", line_number,
                                        f"Expected 8 columns, found {len(columns)}."))
            continue
        try:
            left, top, width, height = (float(value) for value in columns[:4])
            score, class_id, truncation, occlusion = (int(value) for value in columns[4:])
        except ValueError:
            issues.append(VisDroneIssue("MALFORMED_ROW", "ERROR", line_number,
                                        "Coordinates must be numeric and metadata integer-valued."))
            continue
        values = (left, top, width, height)
        if not all(math.isfinite(value) for value in values):
            issues.append(VisDroneIssue("INVALID_BBOX", "ERROR", line_number,
                                        "BBox coordinates must be finite."))
            continue
        if class_id not in SOURCE_CLASSES:
            issues.append(VisDroneIssue("INVALID_CLASS_ID", "ERROR", line_number,
                                        f"Unknown source class ID {class_id}."))
            continue
        if score not in {0, 1}:
            issues.append(VisDroneIssue("INVALID_SCORE", "ERROR", line_number,
                                        f"Ground-truth score must be 0 or 1, found {score}."))
            continue
        if width <= 0 or height <= 0:
            issues.append(VisDroneIssue("INVALID_BBOX_SIZE", "ERROR", line_number,
                                        "BBox width and height must be positive."))
            continue
        if left < 0 or top < 0 or left + width > image_width or top + height > image_height:
            issues.append(VisDroneIssue("BBOX_OUTSIDE_IMAGE", "ERROR", line_number,
                                        "BBox extent is outside image bounds."))
            continue
        if truncation not in {0, 1}:
            issues.append(VisDroneIssue("INVALID_TRUNCATION", "ERROR", line_number,
                                        f"Unsupported truncation value {truncation}."))
            continue
        if occlusion not in {0, 1, 2}:
            issues.append(VisDroneIssue("INVALID_OCCLUSION", "ERROR", line_number,
                                        f"Unsupported occlusion value {occlusion}."))
            continue
        key = (left, top, width, height, score, class_id, truncation, occlusion)
        if key in seen:
            issues.append(VisDroneIssue("DUPLICATE_ANNOTATION", "WARNING", line_number,
                                        "Exact duplicate annotation row retained in raw only."))
            continue
        seen.add(key)
        annotations.append(VisDroneAnnotation(*key))
    return annotations, issues


def map_visdrone_class(source_class: str) -> tuple[str | None, str]:
    if source_class in APPLICATION_MAPPING:
        return APPLICATION_MAPPING[source_class], "MAPPED"
    if source_class in UNSUPPORTED_CLASSES:
        return None, "EXCLUDED_FROM_V2_TARGET"
    if source_class in {"ignored_regions", "others"}:
        return None, "IGNORED_OR_OTHER"
    raise ValueError(f"unknown VisDrone source class: {source_class}")


def size_bin(area_normalized: float) -> str:
    if not math.isfinite(area_normalized) or area_normalized <= 0:
        raise ValueError("normalized area must be positive and finite")
    if area_normalized < 0.01:
        return "SMALL_LT_0.01"
    if area_normalized < 0.09:
        return "MEDIUM_0.01_TO_LT_0.09"
    return "LARGE_GE_0.09"


def sequence_group_id(partition: str, image_stem: str) -> str:
    parts = image_stem.split("_")
    if len(parts) < 4 or not parts[0].isdigit():
        raise ValueError(f"cannot derive VisDrone sequence group from {image_stem}")
    return f"{partition}:{parts[0]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def near_duplicate_groups(
    records: Sequence[Mapping[str, object]], *, threshold: int = 6,
) -> list[dict[str, object]]:
    """Find exact/near image groups with lossless 7-band candidate blocking."""
    if threshold < 0 or threshold > 6:
        raise ValueError("this blocking implementation supports threshold 0..6")
    ids = [str(row["image_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("image_id must be unique")
    dsu = _DisjointSet(ids)
    exact: dict[str, str] = {}
    exact_sizes: dict[str, int] = {}
    buckets: dict[tuple[int, str], list[int]] = {}
    values = [int(str(row["phash"]), 16) for row in records]
    widths = [10, 9, 9, 9, 9, 9, 9]
    shifts, consumed = [], 64
    for width in widths:
        consumed -= width
        shifts.append(consumed)
    for index, row in enumerate(records):
        image_id, sha = str(row["image_id"]), str(row["sha256"])
        exact_sizes[sha] = exact_sizes.get(sha, 0) + 1
        if sha in exact:
            dsu.union(image_id, exact[sha])
        exact.setdefault(sha, image_id)
        for band, (width, shift) in enumerate(zip(widths, shifts, strict=True)):
            key = (band, f"{(values[index] >> shift) & ((1 << width) - 1):x}")
            buckets.setdefault(key, []).append(index)
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        for position, first in enumerate(members):
            for second in members[position + 1:]:
                pair = (min(first, second), max(first, second))
                if pair in checked:
                    continue
                checked.add(pair)
                if _hamming(str(records[first]["phash"]), str(records[second]["phash"])) <= threshold:
                    dsu.union(ids[first], ids[second])
    roots = {image_id: dsu.find(image_id) for image_id in ids}
    sizes: dict[str, int] = {}
    for root in roots.values():
        sizes[root] = sizes.get(root, 0) + 1
    names = {root: f"VD_DUP_{i:05d}" for i, root in enumerate(sorted(sizes), 1)}
    return [
        {"image_id": image_id, "duplicate_group_id": names[roots[image_id]],
         "group_size": sizes[roots[image_id]],
         "exact_duplicate_group_size": exact_sizes[str(row["sha256"])],
         "grouping_basis": (
             "EXACT" if exact_sizes[str(row["sha256"])] > 1 and sizes[roots[image_id]] == exact_sizes[str(row["sha256"])]
             else "EXACT_AND_NEAR" if exact_sizes[str(row["sha256"])] > 1
             else "NEAR" if sizes[roots[image_id]] > 1 else "SINGLETON"
         ),
         "is_duplicate": sizes[roots[image_id]] > 1}
        for image_id, row in zip(ids, records, strict=True)
    ]


def cross_dataset_overlap(
    candidate: Sequence[Mapping[str, object]], reference: Sequence[Mapping[str, object]], *, threshold: int = 6,
) -> dict[str, object]:
    ref_sha = {str(row["sha256"]) for row in reference}
    exact = [str(row["image_id"]) for row in candidate if str(row["sha256"]) in ref_sha]
    near_pairs = []
    for row in candidate:
        for other in reference:
            if _hamming(str(row["phash"]), str(other["phash"])) <= threshold:
                near_pairs.append((str(row["image_id"]), str(other["image_id"])))
    return {"exact_overlap_count": len(exact), "near_overlap_count": len(near_pairs),
            "exact_image_ids": exact, "near_pairs": near_pairs}


def aggregate_coverage(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate mapped target coverage by class while retaining size/difficulty dimensions."""
    groups: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in records:
        if str(row.get("mapping_disposition")) != "MAPPED" or bool(row.get("is_ignored")):
            continue
        key = (str(row["application_class"]), str(row["size_bin"]), int(row["occlusion"]))
        group = groups.setdefault(key, {"image_ids": set(), "bbox_count": 0})
        group["image_ids"].add(str(row["image_id"]))
        group["bbox_count"] = int(group["bbox_count"]) + 1
    return [
        {"application_class": class_name, "size_bin": bin_name, "occlusion": occlusion,
         "image_count": len(group["image_ids"]), "bbox_count": group["bbox_count"]}
        for (class_name, bin_name, occlusion), group in sorted(groups.items())
    ]
