"""Governance helpers for the controlled V2 detection training pool."""

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


APPLICATION_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
OPENIMAGES_SOURCE_MAPPING = {
    "Person": "person", "Bicycle": "bicycle", "Car": "car",
    "Motorcycle": "motorcycle", "Bus": "bus", "Truck": "truck",
}
TAIWAN_SOURCE_MAPPING = {
    "human": "person", "bicycle": "bicycle", "car": "car",
    "motorbike": "motorcycle", "bus": "bus", "truck": "truck",
}
ALLOWED_LICENSE = "https://creativecommons.org/licenses/by/2.0/"
ALLOWED_DOMAIN = frozenset({"TRAFFIC_RELEVANT", "PARTIALLY_RELEVANT"})
ALLOWED_ANNOTATION = frozenset({"ACCEPTABLE", "ACCEPTABLE_WITH_NOTE"})
REQUIRED_RARE_TAGS = frozenset({
    "PERSON_MOTORCYCLE", "PERSON_BICYCLE", "PERSON_BUS", "PERSON_TRUCK",
    "PERSON_MULTI_TRAFFIC",
})
NON_TRAFFIC_TITLE = re.compile(
    r"\b(race|racing|rally|motorsport|motor ?show|car ?show|auto ?show|museum|showroom|"
    r"stunt|freestyle|velodrome|drag|miniature|toy|exhibition|championship|autocross|"
    r"motocross|speedway|track day|concept car|classic car|bike restoration|air show)\b", re.I,
)
TRAFFIC_TITLE = re.compile(
    r"\b(street|traffic|road|city|urban|intersection|bus station|bus stop|taxi|market|"
    r"scooter|police|parking|commute|highway|airport|crosswalk|downtown|transit)\b", re.I,
)


def parse_flickr_license_evidence(page_html: str) -> dict[str, str]:
    """Extract current image-level license evidence from Flickr JSON-LD."""
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        page_html, flags=re.IGNORECASE | re.DOTALL,
    )
    for raw in scripts:
        try:
            payload = json.loads(html.unescape(raw).strip())
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if value.get("@type") == "ImageObject" and value.get("license"):
                    creator = value.get("creator", {})
                    if isinstance(creator, list):
                        creator = creator[0] if creator else {}
                    return {
                        "observed_license_url": str(value.get("license", "")),
                        "acquire_license_page": str(value.get("acquireLicensePage", "")),
                        "content_url": str(value.get("contentUrl", "")),
                        "creator_name": str(creator.get("name", "")) if isinstance(creator, dict) else "",
                    }
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return {"observed_license_url": "", "acquire_license_page": "",
            "content_url": "", "creator_name": ""}


def license_review_status(
    metadata: Mapping[str, object], evidence: Mapping[str, object], *, http_status: int,
) -> tuple[str, str]:
    required = ("image_id", "license_reference", "original_url", "author", "license_url")
    missing = [key for key in required if not str(metadata.get(key, "")).strip()]
    if missing:
        return "MISSING_LICENSE_METADATA", f"missing metadata: {','.join(missing)}"
    if http_status != 200:
        return "REQUIRES_REVIEW", f"source landing page returned HTTP {http_status}"
    expected = str(metadata["license_url"])
    observed = str(evidence.get("observed_license_url", ""))
    if expected != ALLOWED_LICENSE or observed != ALLOWED_LICENSE:
        return "REJECTED", f"expected/observed license mismatch: {expected!r} vs {observed!r}"
    landing = str(metadata["license_reference"]).rstrip("/")
    acquire = str(evidence.get("acquire_license_page", "")).rstrip("/")
    if acquire != landing:
        return "REQUIRES_REVIEW", "JSON-LD acquireLicensePage does not match source landing page"
    photo_match = re.search(r"/photos/[^/]+/(\d+)", landing)
    if not photo_match or photo_match.group(1) not in str(evidence.get("content_url", "")):
        return "REQUIRES_REVIEW", "current licensed content URL cannot be tied to landing photo ID"
    return "LICENSE_APPROVED", "Current Flickr ImageObject JSON-LD confirms CC BY 2.0 and photo identity"


def openimages_training_gate(row: Mapping[str, object]) -> tuple[bool, str]:
    if row.get("license_review_status") != "LICENSE_APPROVED":
        return False, f"LICENSE:{row.get('license_review_status', 'MISSING')}"
    domain = str(row.get("domain_relevance", ""))
    if domain not in ALLOWED_DOMAIN:
        return False, f"DOMAIN:{domain or 'MISSING'}"
    if domain == "PARTIALLY_RELEVANT":
        tags = set(str(row.get("context_tags", "")).split(";"))
        if not tags.intersection(REQUIRED_RARE_TAGS) or not str(row.get("domain_review_notes", "")).strip():
            return False, "DOMAIN:PARTIAL_WITHOUT_RARE_COVERAGE_JUSTIFICATION"
    annotation = str(row.get("annotation_review_status", ""))
    if annotation not in ALLOWED_ANNOTATION:
        return False, f"ANNOTATION:{annotation or 'MISSING'}"
    if str(row.get("candidate_status", "")) not in {
        "QUARANTINED_LICENSE_REVIEW", "ELIGIBLE", "APPROVED_FOR_V2_TRAINING"
    }:
        return False, f"SOURCE_CANDIDATE:{row.get('candidate_status', 'MISSING')}"
    return True, "ALL_GATES_PASS"


def auditable_domain_review(row: Mapping[str, object]) -> tuple[str, str, str]:
    """Conservative, title/context rule calibrated against contact-sheet review."""
    if row.get("license_review_status") != "LICENSE_APPROVED":
        return "AMBIGUOUS", "LICENSE_BLOCKED_NOT_PROMOTED", "RULE_ASSISTED"
    title = str(row.get("title", ""))
    tags = set(str(row.get("context_tags", "")).split(";"))
    if NON_TRAFFIC_TITLE.search(title):
        return "NON_TRAFFIC", "Explicit non-road event/display/recreation title keyword", "RULE_ASSISTED"
    if TRAFFIC_TITLE.search(title):
        return "TRAFFIC_RELEVANT", "Explicit road/transit/urban title keyword", "RULE_ASSISTED"
    rare = sorted(tags.intersection(REQUIRED_RARE_TAGS))
    if rare:
        return ("PARTIALLY_RELEVANT",
                f"Rare coverage retained for {','.join(rare)}; no explicit non-traffic keyword",
                "RULE_ASSISTED")
    return "AMBIGUOUS", "Generic Person+Car co-occurrence lacks auditable road semantics", "RULE_ASSISTED"


def auditable_annotation_review(row: Mapping[str, object]) -> tuple[str, str, str]:
    if row.get("license_review_status") != "LICENSE_APPROVED":
        return "REJECTED_AMBIGUOUS", "License-blocked image not promoted for annotation review", "RULE_ASSISTED"
    if not bool(row.get("image_readable")):
        return "REJECTED_AMBIGUOUS", "Image is not readable", "AUTOMATED_QA"
    total = sum(int(row.get(f"{name}_boxes", 0)) for name in APPLICATION_CLASSES)
    if int(row.get("person_boxes", 0)) <= 0 or total <= 0:
        return "REJECTED_INCOMPLETE", "No eligible Person/target annotations remain", "AUTOMATED_QA"
    if int(row.get("person_boxes", 0)) >= 20 or total >= 35:
        return ("REJECTED_AMBIGUOUS",
                "Dense scene requires exhaustive completeness review beyond contact-sheet resolution",
                "RULE_ASSISTED")
    return ("ACCEPTABLE_WITH_NOTE",
            "BBox geometry/mapping passed; rule-assisted contact-sheet review is not exhaustive completeness proof",
            "RULE_ASSISTED")


def map_application_class(dataset_source: str, source_class: str) -> str:
    mapping = {
        "taiwan_cctv_v3": TAIWAN_SOURCE_MAPPING,
        "openimages_v7": OPENIMAGES_SOURCE_MAPPING,
    }.get(dataset_source)
    if mapping is None or source_class not in mapping:
        raise ValueError(f"unsupported source taxonomy: {dataset_source}:{source_class}")
    return mapping[source_class]


def annotation_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(
        [dict(sorted((str(key), value) for key, value in row.items())) for row in rows],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deterministic_source_group_split(
    records: Sequence[Mapping[str, object]], *, val_fraction: float, seed: int,
) -> dict[str, str]:
    """Assign intact groups within each source to TRAIN/VAL deterministically."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    group_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        group_rows[str(row["group_id"])].append(row)
    for group_id, members in group_rows.items():
        if len({str(row["dataset_source"]) for row in members}) != 1:
            raise ValueError(f"cross-source duplicate group requires rejection: {group_id}")
    by_source: dict[str, list[str]] = defaultdict(list)
    for group_id, members in group_rows.items():
        by_source[str(members[0]["dataset_source"])].append(group_id)
    assignments: dict[str, str] = {}
    for source, groups in sorted(by_source.items()):
        if len(groups) < 2:
            raise ValueError(f"source {source} needs at least two groups")
        rng = random.Random(f"{seed}:{source}")
        ranked = sorted(groups)
        rng.shuffle(ranked)
        target = max(1, min(len(ranked) - 1, round(len(ranked) * val_fraction)))
        val_groups = set(ranked[:target])
        assignments.update({group_id: "VAL" if group_id in val_groups else "TRAIN"
                            for group_id in groups})
    return assignments


def validate_training_manifest(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("training manifest is empty")
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("image_id must be unique")
    if {str(row["split"]) for row in rows} != {"TRAIN", "VAL"}:
        raise ValueError("manifest must contain exactly TRAIN and VAL")
    group_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
        source_splits[str(row["dataset_source"])].add(str(row["split"]))
        for field in ("image_sha256", "annotation_sha256", "source_image_id"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"required manifest field is empty: {field}")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("duplicate/source group crosses TRAIN and VAL")
    if any(splits != {"TRAIN", "VAL"} for splits in source_splits.values()):
        raise ValueError("each dataset source must contribute to TRAIN and VAL")


def select_review_candidates(
    rows: Iterable[Mapping[str, object]], *, limit: int, seed: int,
) -> list[str]:
    """Balance rare traffic contexts and hard Person cases before visual review."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = []
    rng = random.Random(seed)
    for row in rows:
        tags = set(str(row.get("context_tags", "")).split(";"))
        if not tags.intersection(REQUIRED_RARE_TAGS | {"PERSON_CAR"}):
            continue
        title = str(row.get("title", ""))
        score = (
            4 * int("PERSON_MOTORCYCLE" in tags)
            + 4 * int("PERSON_BICYCLE" in tags)
            + 3 * int("PERSON_MULTI_TRAFFIC" in tags)
            + 2 * int(bool(tags.intersection({"PERSON_BUS", "PERSON_TRUCK"})))
            + min(int(row.get("small_person_count", 0)), 4)
            + min(int(row.get("occluded_person_count", 0)), 3)
            + 2 * bool(TRAFFIC_TITLE.search(title))
            - 8 * bool(NON_TRAFFIC_TITLE.search(title))
        )
        ranked.append((score, rng.random(), str(row["image_id"])))
    ranked.sort(reverse=True)
    return [image_id for _, _, image_id in ranked[:limit]]
