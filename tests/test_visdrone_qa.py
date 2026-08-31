from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.utils.visdrone_qa import (  # noqa: E402
    aggregate_coverage,
    cross_dataset_overlap,
    map_visdrone_class,
    near_duplicate_groups,
    parse_visdrone_annotation,
    sequence_group_id,
    size_bin,
)


def test_visdrone_parser_preserves_source_fields_and_normalizes() -> None:
    rows, issues = parse_visdrone_annotation(
        "10,20,30,40,1,1,1,2\n", image_width=100, image_height=200
    )
    assert issues == []
    assert rows[0].class_id == 1 and rows[0].source_class == "pedestrian"
    assert rows[0].truncation == 1 and rows[0].occlusion == 2
    normalized = rows[0].normalized(100, 200)
    assert normalized["x_center"] == pytest.approx(0.25)
    assert normalized["area_normalized"] == pytest.approx(0.06)


@pytest.mark.parametrize("row,issue", [
    ("1,2,0,4,1,1,0,0", "INVALID_BBOX_SIZE"),
    ("90,2,20,4,1,1,0,0", "BBOX_OUTSIDE_IMAGE"),
    ("1,2,3,4,1,99,0,0", "INVALID_CLASS_ID"),
    ("1,2,3", "MALFORMED_ROW"),
])
def test_invalid_annotations_are_reported(row: str, issue: str) -> None:
    parsed, issues = parse_visdrone_annotation(row, image_width=100, image_height=100)
    assert parsed == []
    assert issues[0].issue_type == issue


def test_duplicate_annotation_is_not_duplicated_in_parsed_representation() -> None:
    row = "1,2,3,4,1,4,0,0"
    parsed, issues = parse_visdrone_annotation(f"{row}\n{row}\n", image_width=100, image_height=100)
    assert len(parsed) == 1
    assert issues[0].issue_type == "DUPLICATE_ANNOTATION"


def test_class_mapping_and_unsupported_classes() -> None:
    assert map_visdrone_class("pedestrian") == ("person", "MAPPED")
    assert map_visdrone_class("people") == ("person", "MAPPED")
    assert map_visdrone_class("motor") == ("motorcycle", "MAPPED")
    for source in ("van", "tricycle", "awning-tricycle"):
        assert map_visdrone_class(source) == (None, "EXCLUDED_FROM_V2_TARGET")
    assert map_visdrone_class("ignored_regions") == (None, "IGNORED_OR_OTHER")


def test_sequence_grouping_is_partition_scoped() -> None:
    assert sequence_group_id("train", "0000271_01401_d_0000380") == "train:0000271"
    assert sequence_group_id("val", "0000271_01401_d_0000380") == "val:0000271"
    with pytest.raises(ValueError):
        sequence_group_id("train", "unexpected")


def test_size_bins_match_v1_taiwan_definitions() -> None:
    assert size_bin(0.009) == "SMALL_LT_0.01"
    assert size_bin(0.01) == "MEDIUM_0.01_TO_LT_0.09"
    assert size_bin(0.09) == "LARGE_GE_0.09"


def test_near_and_exact_duplicate_grouping() -> None:
    rows = [
        {"image_id": "a", "sha256": "same", "phash": "0000000000000000"},
        {"image_id": "b", "sha256": "same", "phash": "0000000000000001"},
        {"image_id": "c", "sha256": "other", "phash": "ffffffffffffffff"},
    ]
    groups = {row["image_id"]: row for row in near_duplicate_groups(rows, threshold=1)}
    assert groups["a"]["duplicate_group_id"] == groups["b"]["duplicate_group_id"]
    assert groups["a"]["group_size"] == 2
    assert groups["c"]["group_size"] == 1


def test_v1_locked_test_overlap_check() -> None:
    candidate = [{"image_id": "new", "sha256": "new-sha", "phash": "0000000000000000"}]
    reference = [{"image_id": "locked", "sha256": "old-sha", "phash": "ffffffffffffffff"}]
    clean = cross_dataset_overlap(candidate, reference, threshold=6)
    assert clean["exact_overlap_count"] == clean["near_overlap_count"] == 0
    reference[0]["sha256"] = "new-sha"
    overlap = cross_dataset_overlap(candidate, reference, threshold=6)
    assert overlap["exact_overlap_count"] == 1


def test_coverage_aggregation_filters_unsupported_and_ignored_rows() -> None:
    rows = [
        {"image_id": "a", "application_class": "person", "mapping_disposition": "MAPPED",
         "is_ignored": False, "size_bin": "SMALL_LT_0.01", "occlusion": 1},
        {"image_id": "b", "application_class": "person", "mapping_disposition": "MAPPED",
         "is_ignored": False, "size_bin": "SMALL_LT_0.01", "occlusion": 1},
        {"image_id": "c", "application_class": "", "mapping_disposition": "EXCLUDED_FROM_V2_TARGET",
         "is_ignored": False, "size_bin": "SMALL_LT_0.01", "occlusion": 0},
    ]
    assert aggregate_coverage(rows) == [{
        "application_class": "person", "size_bin": "SMALL_LT_0.01", "occlusion": 1,
        "image_count": 2, "bbox_count": 2,
    }]


def test_raw_annotation_text_is_not_modified_by_parser() -> None:
    raw = "10,20,30,40,1,10,0,1,\n"
    original = raw.encode()
    parse_visdrone_annotation(raw, image_width=100, image_height=100)
    assert raw.encode() == original
