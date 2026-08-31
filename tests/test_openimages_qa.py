from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.utils.openimages_qa import (  # noqa: E402
    aggregate_candidate,
    box_candidate_status,
    context_tags,
    deterministic_pilot_selection,
    duplicate_exact,
    license_status,
    parse_box_row,
    resolve_target_classes,
    size_bin,
    validate_pilot_manifest,
    validate_stage18_usage,
)

CLASS_ROWS = [
    ("person-mid", "Person"), ("bike-mid", "Bicycle"), ("car-mid", "Car"),
    ("motor-mid", "Motorcycle"), ("bus-mid", "Bus"), ("truck-mid", "Truck"),
]


def row(**updates):
    value = {"ImageID": "image", "Source": "xclick", "LabelName": "person-mid",
             "Confidence": 1, "XMin": .1, "XMax": .2, "YMin": .2, "YMax": .4,
             "IsOccluded": 0, "IsTruncated": 0, "IsGroupOf": 0,
             "IsDepiction": 0, "IsInside": 0}
    value.update(updates)
    return value


def test_official_class_id_resolution_and_application_mapping() -> None:
    resolved = resolve_target_classes(CLASS_ROWS)
    assert resolved["person-mid"] == ("Person", "person")
    assert resolved["motor-mid"] == ("Motorcycle", "motorcycle")
    with pytest.raises(ValueError):
        resolve_target_classes(CLASS_ROWS[:-1])


def test_bbox_parser_preserves_attributes_and_calculates_small_area() -> None:
    box = parse_box_row(row(IsOccluded=1, IsTruncated=1), resolve_target_classes(CLASS_ROWS))
    assert box.area == pytest.approx(.02)
    assert box.is_occluded == box.is_truncated == 1
    assert size_bin(.009) == "SMALL_LT_0.01"
    assert size_bin(.01) == "MEDIUM_0.01_TO_LT_0.09"
    assert size_bin(.09) == "LARGE_GE_0.09"


@pytest.mark.parametrize("updates", [
    {"XMin": -.1}, {"XMin": .5, "XMax": .4}, {"YMax": 1.1}, {"IsOccluded": 2},
])
def test_invalid_bbox_or_attribute_is_rejected(updates) -> None:
    with pytest.raises(ValueError):
        parse_box_row(row(**updates), resolve_target_classes(CLASS_ROWS))


def test_context_tags_include_multi_traffic() -> None:
    assert context_tags(["person"]) == ("PERSON_ONLY",)
    assert context_tags(["person", "car"]) == ("PERSON_CAR",)
    assert context_tags(["person", "motorcycle", "bicycle"]) == (
        "PERSON_BICYCLE", "PERSON_MOTORCYCLE", "PERSON_MULTI_TRAFFIC"
    )
    assert context_tags(["car"]) == ()


def test_depiction_and_group_of_are_filtered_but_occlusion_is_retained() -> None:
    mapping = resolve_target_classes(CLASS_ROWS)
    assert box_candidate_status(parse_box_row(row(IsDepiction=1), mapping)) == "EXCLUDED_DEPICTION"
    assert box_candidate_status(parse_box_row(row(IsGroupOf=1), mapping)) == "EXCLUDED_GROUP_OF"
    assert box_candidate_status(parse_box_row(row(IsOccluded=1), mapping)) == "ELIGIBLE"


def test_license_status_requires_per_image_traceability() -> None:
    metadata = {"OriginalURL": "https://image", "OriginalLandingURL": "https://landing",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "AuthorProfileURL": "https://author", "Author": "Creator"}
    assert license_status(metadata)[0] == "REQUIRES_REVIEW"
    metadata["Author"] = ""
    assert license_status(metadata)[0] == "REQUIRES_REVIEW"
    metadata["Author"] = "Creator"; metadata["License"] = "https://example.com/proprietary"
    assert license_status(metadata)[0] == "REJECTED"


def test_candidate_aggregation_counts_person_and_traffic() -> None:
    mapping = resolve_target_classes(CLASS_ROWS)
    boxes = [parse_box_row(row(XMax=.15, YMax=.25, IsOccluded=1), mapping),
             parse_box_row(row(LabelName="car-mid"), mapping),
             parse_box_row(row(LabelName="person-mid", IsGroupOf=1), mapping)]
    candidate = aggregate_candidate(boxes)
    assert candidate["person_box_count"] == 1
    assert candidate["traffic_box_count"] == 1
    assert candidate["context_tags"] == "PERSON_CAR"
    assert candidate["group_of_box_count"] == 1


def test_pilot_selection_is_deterministic_and_prefers_traffic() -> None:
    rows = [
        {"image_id": "only", "candidate_status": "ELIGIBLE", "license_status": "REQUIRES_REVIEW",
         "context_tags": "PERSON_ONLY", "small_person_count": 5,
         "occluded_person_count": 0, "truncated_person_count": 0},
        {"image_id": "motor", "candidate_status": "ELIGIBLE", "license_status": "REQUIRES_REVIEW",
         "context_tags": "PERSON_MOTORCYCLE", "small_person_count": 0,
         "occluded_person_count": 0, "truncated_person_count": 0},
    ]
    assert deterministic_pilot_selection(rows, limit=1, seed=1) == ["motor"]
    assert deterministic_pilot_selection(rows, limit=1, seed=1) == ["motor"]


def test_duplicate_check_and_stage18_reference_is_overlap_only() -> None:
    pilot = [{"image_id": "p", "sha256": "same"}]
    references = {"stage18_locked_test_OVERLAP_CHECK_ONLY": [
        {"image_id": "locked", "sha256": "same"}
    ]}
    result = duplicate_exact(pilot, references)
    assert result[0]["exact_overlap"] is True
    assert "OVERLAP_CHECK_ONLY" in result[0]["reference_dataset"]


def test_pilot_manifest_integrity() -> None:
    rows = [{"image_id": "one", "source_split": "validation",
             "license_status": "REQUIRES_REVIEW", "local_path": "raw/one.jpg"}]
    validate_pilot_manifest(rows, expected_count=1)
    with pytest.raises(ValueError):
        validate_pilot_manifest(rows * 2, expected_count=2)
    with pytest.raises(ValueError):
        validate_pilot_manifest([{**rows[0], "license_status": "REJECTED"}], expected_count=1)


def test_stage18_usage_guard() -> None:
    validate_stage18_usage("OVERLAP_CHECK_ONLY")
    with pytest.raises(ValueError):
        validate_stage18_usage("MODEL_SELECTION")
