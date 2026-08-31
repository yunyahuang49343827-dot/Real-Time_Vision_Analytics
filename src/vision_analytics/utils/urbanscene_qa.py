"""UrbanScene image-level inventory, taxonomy, and leakage helpers.

UrbanScene's published record does not define an object-annotation schema.  This
module deliberately audits annotation availability instead of guessing a parser
from the dataset title or converting folder labels into bounding boxes.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SOURCE_CATEGORY_MAPPING = {
    "Pedestrians": ("person", "IMAGE_LEVEL_SEMANTIC_NOT_BBOX"),
    "Pedestrian": ("person", "IMAGE_LEVEL_SEMANTIC_NOT_BBOX"),
    "Motorbikes": ("motorcycle", "IMAGE_LEVEL_SEMANTIC_NOT_BBOX"),
    "Motorbike": ("motorcycle", "IMAGE_LEVEL_SEMANTIC_NOT_BBOX"),
    "Cyclists": ("bicycle", "AMBIGUOUS_RIDER_CYCLE_IMAGE_LEVEL_ONLY"),
    "Cyclist": ("bicycle", "AMBIGUOUS_RIDER_CYCLE_IMAGE_LEVEL_ONLY"),
    "Motorbikes_&_Cyclist": (None, "BROAD_MIXED_IMAGE_LEVEL_CATEGORY"),
    "Motorbikes & Cyclist": (None, "BROAD_MIXED_IMAGE_LEVEL_CATEGORY"),
    "Motorbikes & Cyclists": (None, "BROAD_MIXED_IMAGE_LEVEL_CATEGORY"),
    "Traffic": (None, "BROAD_TRAFFIC_IMAGE_LEVEL_CATEGORY"),
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
ANNOTATION_SUFFIXES = frozenset({".txt", ".xml", ".json", ".csv"})


def map_source_category(source_category: str) -> tuple[str | None, str]:
    """Map only documented image-level semantics; never imply bbox labels."""
    try:
        return SOURCE_CATEGORY_MAPPING[source_category]
    except KeyError as error:
        raise ValueError(f"unknown UrbanScene source category: {source_category}") from error


def discover_annotation_artifacts(root: Path) -> dict[str, object]:
    """Audit possible annotation files without assuming any unpublished schema."""
    root = Path(root)
    images = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES
        and path.name.lower() not in {"readme.txt", "license.txt"}
    ]
    image_stems = defaultdict(int)
    for path in images:
        image_stems[path.stem] += 1
    paired = [path for path in candidates if image_stems[path.stem] > 0]
    if not candidates:
        status = "IMAGE_LEVEL_CATEGORY_ONLY"
    elif not paired:
        status = "UNVERIFIED_NON_PAIRED_METADATA"
    else:
        status = "UNVERIFIED_ANNOTATION_CANDIDATES"
    return {
        "status": status,
        "image_count": len(images),
        "candidate_annotation_count": len(candidates),
        "paired_annotation_count": len(paired),
        "candidate_paths": [path.relative_to(root).as_posix() for path in sorted(candidates)],
        "paired_paths": [path.relative_to(root).as_posix() for path in sorted(paired)],
    }


def detection_suitability(annotation_audit: Mapping[str, object]) -> dict[str, str]:
    """Apply the V2-2B hard gate without interpreting image folders as labels."""
    if annotation_audit.get("status") == "IMAGE_LEVEL_CATEGORY_ONLY":
        return {
            "suitability_status": "NOT_SUITABLE_FOR_YOLO_OBJECT_DETECTION",
            "training_pool_decision": "REJECT_FOR_SUPERVISED_YOLO_TRAINING",
            "acceptance_decision": "REJECT",
        }
    return {
        "suitability_status": "REQUIRES_ANNOTATION_SCHEMA_REVIEW",
        "training_pool_decision": "QUARANTINED_NOT_APPROVED_FOR_TRAINING",
        "acceptance_decision": "REJECT",
    }


def validate_bbox(
    left: float, top: float, width: float, height: float,
    *, image_width: int, image_height: int,
) -> bool:
    """Validate generic pixel bbox geometry without claiming a source parser."""
    values = (left, top, width, height)
    if image_width <= 0 or image_height <= 0 or not all(math.isfinite(value) for value in values):
        return False
    return width > 0 and height > 0 and left >= 0 and top >= 0 \
        and left + width <= image_width and top + height <= image_height


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


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def cross_dataset_overlap_blocked(
    candidate: Sequence[Mapping[str, object]],
    reference: Sequence[Mapping[str, object]],
    *, threshold: int = 6,
) -> dict[str, object]:
    """Exact and dHash near-overlap using lossless threshold+1 band blocking."""
    if threshold < 0 or threshold > 6:
        raise ValueError("blocked overlap supports a dHash threshold from 0 to 6")
    ref_sha: dict[str, list[str]] = defaultdict(list)
    for row in reference:
        ref_sha[str(row["sha256"])].append(str(row["image_id"]))
    exact_pairs = [
        (str(row["image_id"]), reference_id)
        for row in candidate for reference_id in ref_sha.get(str(row["sha256"]), [])
    ]

    band_count = threshold + 1
    base, extra = divmod(64, band_count)
    widths = [base + (1 if index < extra else 0) for index in range(band_count)]
    shifts, consumed = [], 64
    for width in widths:
        consumed -= width
        shifts.append(consumed)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    reference_values = [int(str(row["phash"]), 16) for row in reference]
    for index, value in enumerate(reference_values):
        for band, (width, shift) in enumerate(zip(widths, shifts, strict=True)):
            buckets[(band, (value >> shift) & ((1 << width) - 1))].append(index)
    near_pairs: list[tuple[str, str]] = []
    for row in candidate:
        value = int(str(row["phash"]), 16)
        possible: set[int] = set()
        for band, (width, shift) in enumerate(zip(widths, shifts, strict=True)):
            possible.update(buckets.get((band, (value >> shift) & ((1 << width) - 1)), []))
        for index in sorted(possible):
            other = reference[index]
            if _hamming(str(row["phash"]), str(other["phash"])) <= threshold:
                near_pairs.append((str(row["image_id"]), str(other["image_id"])))
    return {
        "exact_overlap_count": len(exact_pairs),
        "near_overlap_count": len(near_pairs),
        "exact_pairs": exact_pairs,
        "near_pairs": near_pairs,
        "near_method": "dHash-64 lossless band blocking",
        "near_hamming_threshold": threshold,
    }


def aggregate_image_coverage(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in records:
        groups[(str(row["source_category"]), str(row["time_of_day"]),
                str(row["mapping_disposition"]))] += 1
    return [
        {
            "source_category": category,
            "time_of_day": time_of_day,
            "mapping_disposition": disposition,
            "image_count": count,
            "bbox_count": "UNAVAILABLE_NO_OBJECT_ANNOTATIONS",
        }
        for (category, time_of_day, disposition), count in sorted(groups.items())
    ]


def duplicate_groups(
    records: Sequence[Mapping[str, object]], *, threshold: int = 6,
) -> list[dict[str, object]]:
    """Group exact/near duplicates while reusing the project's audited dHash logic."""
    from vision_analytics.utils.visdrone_qa import near_duplicate_groups

    rows = near_duplicate_groups(records, threshold=threshold)
    for row in rows:
        row["duplicate_group_id"] = str(row["duplicate_group_id"]).replace(
            "VD_DUP_", "URBAN_DUP_"
        )
    return rows
