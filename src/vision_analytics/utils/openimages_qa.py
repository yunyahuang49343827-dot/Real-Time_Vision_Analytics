"""Open Images V7 targeted-subset parsing and governance helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TARGET_DISPLAY_TO_APPLICATION = {
    "Person": "person",
    "Bicycle": "bicycle",
    "Car": "car",
    "Motorcycle": "motorcycle",
    "Bus": "bus",
    "Truck": "truck",
}
TRAFFIC_CLASSES = frozenset({"bicycle", "car", "motorcycle", "bus", "truck"})
BOX_COLUMNS = (
    "ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
    "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside",
)


@dataclass(frozen=True, slots=True)
class OpenImagesBox:
    image_id: str
    source_class_id: str
    application_class: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    is_occluded: int
    is_truncated: int
    is_group_of: int
    is_depiction: int
    is_inside: int

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


def resolve_target_classes(
    class_rows: Iterable[tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Resolve exact official display names to MID and application class."""
    by_name = {display_name: source_id for source_id, display_name in class_rows}
    missing = sorted(set(TARGET_DISPLAY_TO_APPLICATION) - set(by_name))
    if missing:
        raise ValueError(f"official class descriptions missing target names: {missing}")
    if len({by_name[name] for name in TARGET_DISPLAY_TO_APPLICATION}) != 6:
        raise ValueError("target class IDs must be unique")
    return {
        by_name[name]: (name, application_class)
        for name, application_class in TARGET_DISPLAY_TO_APPLICATION.items()
    }


def parse_box_row(
    row: Mapping[str, object], class_mapping: Mapping[str, tuple[str, str]],
) -> OpenImagesBox:
    missing = [column for column in BOX_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"missing bbox columns: {missing}")
    source_id = str(row["LabelName"])
    if source_id not in class_mapping:
        raise ValueError(f"unmapped source class ID: {source_id}")
    values = tuple(float(row[name]) for name in ("XMin", "XMax", "YMin", "YMax"))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    xmin, xmax, ymin, ymax = values
    if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
        raise ValueError("bbox coordinates must define a positive box inside normalized bounds")
    attributes = tuple(int(row[name]) for name in (
        "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"
    ))
    if any(value not in {-1, 0, 1} for value in attributes):
        raise ValueError("bbox attributes must be -1, 0, or 1")
    return OpenImagesBox(
        image_id=str(row["ImageID"]), source_class_id=source_id,
        application_class=class_mapping[source_id][1],
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        is_occluded=attributes[0], is_truncated=attributes[1],
        is_group_of=attributes[2], is_depiction=attributes[3], is_inside=attributes[4],
    )


def size_bin(area: float) -> str:
    if not math.isfinite(area) or area <= 0:
        raise ValueError("normalized bbox area must be positive and finite")
    if area < 0.01:
        return "SMALL_LT_0.01"
    if area < 0.09:
        return "MEDIUM_0.01_TO_LT_0.09"
    return "LARGE_GE_0.09"


def box_candidate_status(box: OpenImagesBox) -> str:
    if box.is_depiction == 1:
        return "EXCLUDED_DEPICTION"
    if box.is_group_of == 1:
        return "EXCLUDED_GROUP_OF"
    return "ELIGIBLE"


def context_tags(application_classes: Iterable[str]) -> tuple[str, ...]:
    classes = set(application_classes)
    if "person" not in classes:
        return ()
    traffic = sorted(classes & TRAFFIC_CLASSES)
    tags = [f"PERSON_{name.upper()}" for name in traffic]
    if len(traffic) > 1:
        tags.append("PERSON_MULTI_TRAFFIC")
    if not traffic:
        tags.append("PERSON_ONLY")
    return tuple(tags)


def license_status(row: Mapping[str, object]) -> tuple[str, str]:
    """Require per-image CC BY URL and attribution traceability."""
    required = ("OriginalURL", "OriginalLandingURL", "License", "AuthorProfileURL", "Author")
    missing = [name for name in required if not str(row.get(name, "")).strip()]
    if missing:
        return "REQUIRES_REVIEW", f"LICENSE_UNVERIFIED missing {','.join(missing)}"
    license_url = str(row["License"]).rstrip("/").lower()
    if license_url == "https://creativecommons.org/licenses/by/2.0":
        return "REQUIRES_REVIEW", (
            "Per-image metadata lists CC BY 2.0 with source and author attribution, but "
            "Open Images explicitly requires users to verify each image license."
        )
    if license_url.startswith("https://creativecommons.org/licenses/"):
        return "REQUIRES_REVIEW", f"Unexpected Creative Commons license: {row['License']}"
    return "REJECTED", f"Unsupported image license: {row['License']}"


def aggregate_candidate(boxes: Sequence[OpenImagesBox]) -> dict[str, object]:
    if not boxes:
        raise ValueError("candidate requires at least one box")
    if len({box.image_id for box in boxes}) != 1:
        raise ValueError("candidate boxes must belong to one image")
    eligible = [box for box in boxes if box_candidate_status(box) == "ELIGIBLE"]
    classes = sorted({box.application_class for box in eligible})
    tags = context_tags(classes)
    person = [box for box in eligible if box.application_class == "person"]
    traffic = [box for box in eligible if box.application_class in TRAFFIC_CLASSES]
    return {
        "image_id": boxes[0].image_id,
        "source_class_ids": ";".join(sorted({box.source_class_id for box in boxes})),
        "application_classes": ";".join(classes),
        "person_box_count": len(person),
        "traffic_box_count": len(traffic),
        "context_tags": ";".join(tags),
        "small_person_count": sum(box.area < 0.01 for box in person),
        "occluded_person_count": sum(box.is_occluded == 1 for box in person),
        "truncated_person_count": sum(box.is_truncated == 1 for box in person),
        "small_occluded_person_count": sum(box.area < 0.01 and box.is_occluded == 1 for box in person),
        "small_truncated_person_count": sum(box.area < 0.01 and box.is_truncated == 1 for box in person),
        "group_of_box_count": sum(box.is_group_of == 1 for box in boxes),
        "depiction_box_count": sum(box.is_depiction == 1 for box in boxes),
        "candidate_status": "ELIGIBLE" if person else "EXCLUDED_NO_ELIGIBLE_PERSON",
    }


def deterministic_pilot_selection(
    rows: Sequence[Mapping[str, object]], *, limit: int, seed: int,
) -> list[str]:
    """Select traffic contexts first, then hard person cases, deterministically."""
    import random

    if limit <= 0:
        raise ValueError("pilot limit must be positive")
    eligible = [row for row in rows if row.get("candidate_status") == "ELIGIBLE"
                and row.get("license_status") in {"VERIFIED", "REQUIRES_REVIEW"}]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    scores = []
    for row in eligible:
        tags = str(row.get("context_tags", ""))
        rare = sum(tag in tags for tag in (
            "PERSON_MOTORCYCLE", "PERSON_BICYCLE", "PERSON_BUS", "PERSON_TRUCK"
        ))
        traffic = "PERSON_ONLY" not in tags
        hard = int(row.get("small_person_count", 0)) + int(row.get("occluded_person_count", 0)) \
            + int(row.get("truncated_person_count", 0))
        scores.append((rare, int(traffic), min(hard, 5), row))
    scores.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [str(item[3]["image_id"]) for item in scores[:limit]]


def duplicate_exact(
    pilot: Sequence[Mapping[str, object]], reference_sets: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reference_name, reference in reference_sets.items():
        index: dict[str, list[str]] = defaultdict(list)
        for item in reference:
            index[str(item["sha256"])].append(str(item["image_id"]))
        for item in pilot:
            matches = index.get(str(item["sha256"]), [])
            rows.append({
                "pilot_image_id": str(item["image_id"]), "reference_dataset": reference_name,
                "exact_overlap": bool(matches), "reference_image_ids": ";".join(matches),
            })
    return rows


def validate_pilot_manifest(
    rows: Sequence[Mapping[str, object]], *, expected_count: int,
) -> None:
    """Validate deterministic pilot identity and governance fields before download/use."""
    if len(rows) != expected_count:
        raise ValueError(f"pilot count mismatch: expected {expected_count}, got {len(rows)}")
    image_ids = [str(row.get("image_id", "")).strip() for row in rows]
    if any(not image_id for image_id in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("pilot image_id values must be non-empty and unique")
    for row in rows:
        if row.get("license_status") not in {"VERIFIED", "REQUIRES_REVIEW"}:
            raise ValueError("pilot contains rejected or unclassified license status")
        if not str(row.get("source_split", "")).strip():
            raise ValueError("pilot source_split is required")
        if not str(row.get("local_path", "")).strip():
            raise ValueError("pilot local_path is required")


def validate_stage18_usage(value: str) -> None:
    """Prevent the sealed V1 test set from becoming V2 selection data."""
    if value != "OVERLAP_CHECK_ONLY":
        raise ValueError("Stage 18 Locked Test may only be used for overlap checking")
