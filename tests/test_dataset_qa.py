"""Tests for Stage 16 dataset QA and split governance."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.utils.dataset_qa import (
    build_duplicate_groups,
    deterministic_group_split,
    find_unmatched_assets,
    hamming_distance,
    image_dhash,
    map_source_class,
    parse_yolo_annotation,
    sha256_file,
    validate_split_manifest,
)


def record(image_id: str, sha: str, phash: str, source_group: str = "") -> dict[str, object]:
    return {
        "image_id": image_id, "sha256": sha, "phash": phash,
        "source_group_id": source_group,
    }


def test_valid_yolo_annotation() -> None:
    annotations, issues = parse_yolo_annotation("0 0.5 0.5 0.2 0.4\n", class_count=2)
    assert len(annotations) == 1
    assert annotations[0].area_normalized == pytest.approx(0.08)
    assert issues == []


@pytest.mark.parametrize("text,issue_type", [
    ("0 0.5 0.5 0.2", "INVALID_ROW_FORMAT"),
    ("class 0.5 0.5 0.2 0.2", "INVALID_ROW_FORMAT"),
    ("0 1.2 0.5 0.2 0.2", "INVALID_COORDINATES"),
    ("0 0.5 0.5 0.0 0.2", "INVALID_BOX_SIZE"),
    ("0 0.95 0.5 0.2 0.2", "BOX_OUTSIDE_IMAGE"),
    ("4 0.5 0.5 0.2 0.2", "INVALID_CLASS_ID"),
])
def test_invalid_yolo_annotations(text: str, issue_type: str) -> None:
    annotations, issues = parse_yolo_annotation(text, class_count=2)
    assert annotations == []
    assert issues[0].issue_type == issue_type
    assert issues[0].severity == "ERROR"


def test_duplicate_and_empty_annotations() -> None:
    row = "0 0.5 0.5 0.2 0.2"
    annotations, issues = parse_yolo_annotation(f"{row}\n{row}\n", class_count=1)
    assert len(annotations) == 1
    assert any(issue.issue_type == "DUPLICATE_ANNOTATION" for issue in issues)
    empty, empty_issues = parse_yolo_annotation("", class_count=1)
    assert empty == []
    assert empty_issues[0].issue_type == "EMPTY_ANNOTATION"
    assert empty_issues[0].severity == "WARNING"


def test_missing_image_and_label_detection() -> None:
    images_without_labels, labels_without_images = find_unmatched_assets(
        {"matched", "image_only"}, {"matched", "label_only"},
    )
    assert images_without_labels == {"image_only"}
    assert labels_without_images == {"label_only"}


def test_source_to_application_mapping_without_raw_mutation(tmp_path: Path) -> None:
    raw_label = tmp_path / "label.txt"
    raw_label.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    before = sha256_file(raw_label)
    assert map_source_class("human", {"human": "person", "motorbike": "motorcycle"}) == "person"
    assert map_source_class("motorbike", {"human": "person", "motorbike": "motorcycle"}) == "motorcycle"
    assert sha256_file(raw_label) == before
    assert raw_label.read_text(encoding="utf-8").startswith("5 ")


def test_exact_duplicate_grouping() -> None:
    grouped = build_duplicate_groups([
        record("a", "same", "0000000000000000"),
        record("b", "same", "ffffffffffffffff"),
        record("c", "other", "aaaaaaaaaaaaaaaa"),
    ], near_duplicate_threshold=0)
    by_id = {item["image_id"]: item for item in grouped}
    assert by_id["a"]["group_id"] == by_id["b"]["group_id"]
    assert by_id["a"]["exact_duplicate_count"] == 2


def test_near_duplicate_grouping() -> None:
    assert hamming_distance("0000000000000000", "0000000000000003") == 2
    grouped = build_duplicate_groups([
        record("a", "sha-a", "0000000000000000"),
        record("b", "sha-b", "0000000000000003"),
        record("c", "sha-c", "ffffffffffffffff"),
    ], near_duplicate_threshold=2)
    by_id = {item["image_id"]: item for item in grouped}
    assert by_id["a"]["group_id"] == by_id["b"]["group_id"]
    assert by_id["a"]["group_id"] != by_id["c"]["group_id"]


def test_image_dhash_is_deterministic() -> None:
    image = np.tile(np.arange(18, dtype=np.uint8), (16, 1))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    assert image_dhash(image) == image_dhash(image.copy())
    assert len(image_dhash(image)) == 16


def test_deterministic_split_and_locked_test() -> None:
    groups = [{"image_id": f"img-{index}", "group_id": f"g-{index // 2}"} for index in range(30)]
    ratios = {"TRAIN": 0.7, "VAL": 0.15, "LOCKED_TEST": 0.15}
    first = deterministic_group_split(groups, ratios=ratios, seed=1601)
    second = deterministic_group_split(groups, ratios=ratios, seed=1601)
    assert first == second
    assert "LOCKED_TEST" in set(first.values())


def test_group_integrity_and_image_uniqueness() -> None:
    manifest = [
        {"image_id": "a", "group_id": "g1", "split": "TRAIN"},
        {"image_id": "b", "group_id": "g1", "split": "TRAIN"},
        {"image_id": "c", "group_id": "g2", "split": "VAL"},
        {"image_id": "d", "group_id": "g3", "split": "LOCKED_TEST"},
    ]
    validate_split_manifest(manifest)
    with pytest.raises(ValueError, match="groups cross splits"):
        validate_split_manifest(manifest + [{"image_id": "e", "group_id": "g1", "split": "VAL"}])
    with pytest.raises(ValueError, match="more than once"):
        validate_split_manifest(manifest + [{"image_id": "a", "group_id": "g4", "split": "TRAIN"}])


def test_locked_test_required() -> None:
    manifest = [
        {"image_id": "a", "group_id": "g1", "split": "TRAIN"},
        {"image_id": "b", "group_id": "g2", "split": "VAL"},
    ]
    with pytest.raises(ValueError, match="must all exist"):
        validate_split_manifest(manifest)
