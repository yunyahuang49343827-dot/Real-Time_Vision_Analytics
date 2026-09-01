from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.video.targeted_sampling import (  # noqa: E402
    ALLOWED_METHODS,
    balanced_temporal_select,
    choose_duplicate_representatives,
    conservative_duplicate_groups,
    deterministic_candidate_id,
    extract_frame,
    frame_in_ranges,
    sha256_bytes,
    timestamp_seconds,
    validate_candidate_manifest,
)
from vision_analytics.utils.dataset_qa import sha256_file  # noqa: E402


def test_stage19_frame_exclusion_is_inclusive() -> None:
    ranges = [(10, 20), (30, 35)]
    assert frame_in_ranges(10, ranges) and frame_in_ranges(20, ranges)
    assert not frame_in_ranges(9, ranges) and not frame_in_ranges(21, ranges)


def test_timestamp_and_candidate_id_are_deterministic() -> None:
    assert timestamp_seconds(75, 50) == pytest.approx(1.5)
    assert deterministic_candidate_id("Taipei", 1234) == "taipei_frame_001234"
    with pytest.raises(ValueError):
        timestamp_seconds(1, 0)


def test_frame_extraction_from_synthetic_video(tmp_path: Path) -> None:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for value in (0, 80, 160):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    frame = extract_frame(path, 1)
    assert frame.shape == (24, 32, 3)
    assert frame.mean() == pytest.approx(80, abs=5)


def test_balanced_selection_enforces_minimum_temporal_separation() -> None:
    methods = sorted(ALLOWED_METHODS)
    rows = [{"frame_id": frame, "sampling_method": methods[frame % 4],
             "priority_score": 100 - frame} for frame in range(40)]
    fractions = {method: .25 for method in methods}
    selected = balanced_temporal_select(rows, target=8, minimum_gap_frames=3,
                                        method_fractions=fractions)
    frames = [int(row["frame_id"]) for row in selected]
    assert len(frames) == 8
    assert min(b - a for a, b in zip(frames, frames[1:])) >= 3


def test_sha256_and_duplicate_representative_selection() -> None:
    assert sha256_bytes(b"frame") == sha256_bytes(b"frame")
    rows = [
        {"candidate_id": "a", "duplicate_group_id": "g", "priority_score": 5, "frame_id": 10},
        {"candidate_id": "b", "duplicate_group_id": "g", "priority_score": 8, "frame_id": 20},
        {"candidate_id": "c", "duplicate_group_id": "h", "priority_score": 1, "frame_id": 30},
    ]
    selected, reasons = choose_duplicate_representatives(rows)
    assert selected == {"b", "c"}
    assert reasons["a"].startswith("DUPLICATE_REDUCED")


def test_conservative_duplicates_do_not_chain_across_time() -> None:
    rows = [
        {"candidate_id": "a", "video_id": "v", "frame_id": 0,
         "image_sha256": "sha-a", "dhash": "0000000000000000"},
        {"candidate_id": "b", "video_id": "v", "frame_id": 2,
         "image_sha256": "sha-b", "dhash": "0000000000000001"},
        {"candidate_id": "c", "video_id": "v", "frame_id": 100,
         "image_sha256": "sha-c", "dhash": "0000000000000000"},
    ]
    grouped = conservative_duplicate_groups(
        rows, threshold=1, max_frame_gap_by_video={"v": 5},
    )
    by_id = {row["candidate_id"]: row for row in grouped}
    assert by_id["a"]["duplicate_group_id"] == by_id["b"]["duplicate_group_id"]
    assert by_id["c"]["duplicate_group_id"] != by_id["a"]["duplicate_group_id"]


def test_manifest_integrity_and_annotation_status() -> None:
    row = {
        "candidate_id": "taipei_frame_000001", "video_id": "v", "frame_id": 1,
        "image_sha256": "sha", "dhash": "hash", "duplicate_group_id": "g",
        "image_path": "image.jpg", "coverage_tags": "PERSON_PRESENT;SMALL_PERSON",
        "sampling_method": "MODEL_ASSISTED_POSITIVE", "stage19_overlap": False,
        "annotation_status": "NOT_ANNOTATED",
    }
    validate_candidate_manifest([row], excluded_ranges={"v": [(10, 20)]})
    with pytest.raises(ValueError):
        validate_candidate_manifest([{**row, "frame_id": 10}], excluded_ranges={"v": [(10, 20)]})
    with pytest.raises(ValueError):
        validate_candidate_manifest([{**row, "annotation_status": "ANNOTATED"}],
                                    excluded_ranges={"v": []})
    with pytest.raises(ValueError):
        validate_candidate_manifest([{**row, "coverage_tags": "FAKE_GT"}], excluded_ranges={"v": []})


def test_raw_video_immutability_policy_is_configured() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/v2_targeted_sampling.yaml").read_text(encoding="utf-8")
    )
    assert config["governance"]["model_prediction_is_ground_truth"] is False
    assert config["governance"]["training_performed"] is False
    assert config["governance"]["final_v2_holdout_created"] is False
    for scene in config["scenes"].values():
        video_id = scene["video_id"]
        expected = config["raw_video_sha256"][video_id]
        filename = next((PROJECT_ROOT / "data/raw/videos").glob(f"{video_id}.mp4"))
        assert sha256_file(filename) == expected
