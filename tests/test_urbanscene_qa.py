from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.utils.urbanscene_qa import (  # noqa: E402
    aggregate_image_coverage,
    cross_dataset_overlap_blocked,
    detection_suitability,
    discover_annotation_artifacts,
    duplicate_groups,
    map_source_category,
    validate_bbox,
)


def test_inventory_annotation_audit_detects_image_only_tree(tmp_path: Path) -> None:
    (tmp_path / "Morning" / "Pedestrians").mkdir(parents=True)
    (tmp_path / "Morning" / "Pedestrians" / "frame.jpg").write_bytes(b"image placeholder")
    audit = discover_annotation_artifacts(tmp_path)
    assert audit["image_count"] == 1
    assert audit["candidate_annotation_count"] == 0
    assert audit["status"] == "IMAGE_LEVEL_CATEGORY_ONLY"


def test_annotation_audit_does_not_parse_unpublished_schema(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    (tmp_path / "labels" / "a.txt").write_text("1 2 3 4", encoding="utf-8")
    audit = discover_annotation_artifacts(tmp_path)
    assert audit["status"] == "UNVERIFIED_ANNOTATION_CANDIDATES"
    assert audit["paired_annotation_count"] == 1


def test_image_only_dataset_fails_object_detection_acceptance_gate(tmp_path: Path) -> None:
    (tmp_path / "x.jpg").write_bytes(b"x")
    result = detection_suitability(discover_annotation_artifacts(tmp_path))
    assert result == {
        "suitability_status": "NOT_SUITABLE_FOR_YOLO_OBJECT_DETECTION",
        "training_pool_decision": "REJECT_FOR_SUPERVISED_YOLO_TRAINING",
        "acceptance_decision": "REJECT",
    }


@pytest.mark.parametrize("bbox", [
    (0, 0, 0, 10), (-1, 0, 10, 10), (95, 0, 10, 10),
    (0, 95, 10, 10), (0, 0, float("nan"), 10),
])
def test_invalid_bbox_geometry_is_rejected(bbox: tuple[float, ...]) -> None:
    assert not validate_bbox(*bbox, image_width=100, image_height=100)
    assert validate_bbox(0, 0, 100, 100, image_width=100, image_height=100)


def test_taxonomy_mapping_keeps_broad_traffic_unmapped() -> None:
    assert map_source_category("Pedestrians") == ("person", "IMAGE_LEVEL_SEMANTIC_NOT_BBOX")
    assert map_source_category("Traffic") == (None, "BROAD_TRAFFIC_IMAGE_LEVEL_CATEGORY")
    assert map_source_category("Motorbikes & Cyclists") == (
        None, "BROAD_MIXED_IMAGE_LEVEL_CATEGORY"
    )
    with pytest.raises(ValueError):
        map_source_category("car")


def test_source_taxonomy_is_preserved_in_coverage() -> None:
    rows = [{"source_category": "Traffic", "time_of_day": "Morning",
             "mapping_disposition": "BROAD_TRAFFIC_IMAGE_LEVEL_CATEGORY"}]
    result = aggregate_image_coverage(rows)
    assert result[0]["source_category"] == "Traffic"
    assert result[0]["bbox_count"] == "UNAVAILABLE_NO_OBJECT_ANNOTATIONS"


def test_exact_and_near_overlap_with_locked_reference() -> None:
    candidate = [
        {"image_id": "urban-a", "sha256": "same", "phash": "0000000000000000"},
        {"image_id": "urban-b", "sha256": "different", "phash": "ffffffffffffffff"},
    ]
    reference = [
        {"image_id": "locked-a", "sha256": "same", "phash": "0000000000000001"},
    ]
    overlap = cross_dataset_overlap_blocked(candidate, reference, threshold=1)
    assert overlap["exact_overlap_count"] == 1
    assert overlap["near_overlap_count"] == 1


def test_internal_exact_and_near_duplicate_grouping() -> None:
    rows = [
        {"image_id": "a", "sha256": "x", "phash": "0000000000000000"},
        {"image_id": "b", "sha256": "y", "phash": "0000000000000001"},
        {"image_id": "c", "sha256": "z", "phash": "ffffffffffffffff"},
    ]
    grouped = {row["image_id"]: row for row in duplicate_groups(rows, threshold=1)}
    assert grouped["a"]["duplicate_group_id"].startswith("URBAN_DUP_")
    assert grouped["a"]["duplicate_group_id"] == grouped["b"]["duplicate_group_id"]
    assert grouped["c"]["group_size"] == 1


def test_overlap_blocking_reports_clean_sets() -> None:
    candidate = [{"image_id": "u", "sha256": "u", "phash": "0000000000000000"}]
    reference = [{"image_id": "v1", "sha256": "v1", "phash": "ffffffffffffffff"}]
    result = cross_dataset_overlap_blocked(candidate, reference, threshold=6)
    assert result["exact_overlap_count"] == result["near_overlap_count"] == 0
    assert result["near_hamming_threshold"] == 6
