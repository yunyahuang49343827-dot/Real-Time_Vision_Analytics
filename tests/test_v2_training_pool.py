from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.utils.v2_training_pool import (  # noqa: E402
    annotation_sha256,
    auditable_annotation_review,
    auditable_domain_review,
    deterministic_source_group_split,
    license_review_status,
    map_application_class,
    openimages_training_gate,
    parse_flickr_license_evidence,
    select_review_candidates,
    validate_training_manifest,
)


def test_flickr_license_evidence_and_acceptance_gate() -> None:
    page = '''<script type="application/ld+json">{"@type":"ImageObject",
      "license":"https://creativecommons.org/licenses/by/2.0/",
      "acquireLicensePage":"https://www.flickr.com/photos/user/123",
      "contentUrl":"https://live.staticflickr.com/x/123_ab.jpg",
      "creator":{"name":"Creator"}}</script>'''
    evidence = parse_flickr_license_evidence(page)
    metadata = {"image_id": "oi", "license_reference": "https://www.flickr.com/photos/user/123",
                "original_url": "https://image", "author": "Creator",
                "license_url": "https://creativecommons.org/licenses/by/2.0/"}
    assert license_review_status(metadata, evidence, http_status=200)[0] == "LICENSE_APPROVED"
    assert license_review_status(metadata, evidence, http_status=404)[0] == "REQUIRES_REVIEW"


def test_license_metadata_is_not_automatic_approval() -> None:
    metadata = {"image_id": "oi", "license_reference": "https://landing",
                "original_url": "https://image", "author": "Creator",
                "license_url": "https://creativecommons.org/licenses/by/2.0/"}
    assert license_review_status(metadata, {}, http_status=200)[0] == "REJECTED"


def test_domain_annotation_and_approved_rejected_separation() -> None:
    base = {"license_review_status": "LICENSE_APPROVED", "domain_relevance": "TRAFFIC_RELEVANT",
            "annotation_review_status": "ACCEPTABLE", "candidate_status": "QUARANTINED_LICENSE_REVIEW",
            "context_tags": "PERSON_MOTORCYCLE", "domain_review_notes": "road scooter context"}
    assert openimages_training_gate(base) == (True, "ALL_GATES_PASS")
    assert openimages_training_gate({**base, "domain_relevance": "NON_TRAFFIC"})[0] is False
    assert openimages_training_gate({**base, "annotation_review_status": "REJECTED_INCOMPLETE"})[0] is False
    assert openimages_training_gate({**base, "license_review_status": "REQUIRES_REVIEW"})[0] is False
    partial = {**base, "domain_relevance": "PARTIALLY_RELEVANT", "domain_review_notes": ""}
    assert openimages_training_gate(partial)[0] is False


def test_application_taxonomy_mapping_preserves_source_distinction() -> None:
    assert map_application_class("taiwan_cctv_v3", "human") == "person"
    assert map_application_class("taiwan_cctv_v3", "motorbike") == "motorcycle"
    assert map_application_class("openimages_v7", "Person") == "person"
    with pytest.raises(ValueError):
        map_application_class("openimages_v7", "human")


def test_annotation_hash_is_deterministic() -> None:
    rows = [{"class": "person", "x": .1}, {"class": "car", "x": .2}]
    assert annotation_sha256(rows) == annotation_sha256(rows)
    assert annotation_sha256(rows) != annotation_sha256(rows[:1])


def test_deterministic_source_aware_group_split_and_integrity() -> None:
    rows = [
        {"image_id": f"{source}-{i}", "source_image_id": str(i), "dataset_source": source,
         "group_id": f"{source}-g{i}", "image_sha256": f"sha-{source}-{i}",
         "annotation_sha256": f"ann-{source}-{i}"}
        for source in ("taiwan_cctv_v3", "openimages_v7") for i in range(5)
    ]
    first = deterministic_source_group_split(rows, val_fraction=.2, seed=2301)
    second = deterministic_source_group_split(rows, val_fraction=.2, seed=2301)
    assert first == second
    manifest = [{**row, "split": first[row["group_id"]]} for row in rows]
    validate_training_manifest(manifest)
    assert {row["dataset_source"] for row in manifest if row["split"] == "VAL"} == {
        "taiwan_cctv_v3", "openimages_v7"
    }


def test_group_leakage_and_source_balance_are_rejected() -> None:
    rows = [
        {"image_id": "a", "source_image_id": "a", "dataset_source": "one", "group_id": "g",
         "split": "TRAIN", "image_sha256": "a", "annotation_sha256": "a"},
        {"image_id": "b", "source_image_id": "b", "dataset_source": "one", "group_id": "g",
         "split": "VAL", "image_sha256": "b", "annotation_sha256": "b"},
    ]
    with pytest.raises(ValueError):
        validate_training_manifest(rows)


def test_review_selection_is_deterministic_and_prefers_rare_context() -> None:
    rows = [
        {"image_id": "car", "context_tags": "PERSON_CAR", "title": "car portrait",
         "small_person_count": 0, "occluded_person_count": 0},
        {"image_id": "motor", "context_tags": "PERSON_MOTORCYCLE", "title": "urban scooter street",
         "small_person_count": 1, "occluded_person_count": 1},
    ]
    assert select_review_candidates(rows, limit=1, seed=1) == ["motor"]
    assert select_review_candidates(rows, limit=1, seed=1) == ["motor"]


def test_rule_assisted_review_is_conservative_and_auditable() -> None:
    base = {"license_review_status": "LICENSE_APPROVED", "title": "Urban scooter street",
            "context_tags": "PERSON_MOTORCYCLE", "image_readable": True,
            "person_boxes": 2, "motorcycle_boxes": 1, "bicycle_boxes": 0,
            "car_boxes": 0, "bus_boxes": 0, "truck_boxes": 0}
    assert auditable_domain_review(base)[0] == "TRAFFIC_RELEVANT"
    assert auditable_domain_review({**base, "title": "Motorcycle racing championship"})[0] == "NON_TRAFFIC"
    assert auditable_domain_review({**base, "title": "Untitled"})[0] == "PARTIALLY_RELEVANT"
    assert auditable_domain_review({**base, "title": "Untitled", "context_tags": "PERSON_CAR"})[0] == "AMBIGUOUS"
    assert auditable_annotation_review(base)[0] == "ACCEPTABLE_WITH_NOTE"
    assert auditable_annotation_review({**base, "person_boxes": 20})[0] == "REJECTED_AMBIGUOUS"


def test_stage18_overlap_policy_and_raw_immutability_are_configured() -> None:
    text = (PROJECT_ROOT / "configs/v2_training_pool.yaml").read_text(encoding="utf-8")
    assert "OVERLAP_CHECK_ONLY" in text
    assert "raw_immutable: true" in text
    assert "final_v2_holdout_created: false" in text
