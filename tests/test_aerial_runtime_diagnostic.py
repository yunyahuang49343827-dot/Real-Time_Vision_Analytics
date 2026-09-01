from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.evaluation.aerial_runtime import (  # noqa: E402
    APPROVED_RUNTIME_MODEL,
    aggregate_runtime_metrics,
    experiment_output_directory,
    load_diagnostic_config,
    normalized_bbox_area,
    render_comparison_report,
    validate_runtime_assets,
)

CONFIG_PATH = PROJECT_ROOT / "configs/stage21_1a_aerial_diagnostic.yaml"


def test_experiment_matrix_and_only_imgsz_conf_vary() -> None:
    config = load_diagnostic_config(CONFIG_PATH, PROJECT_ROOT)
    assert [(item.name, item.imgsz, item.conf) for item in config.experiments] == [
        ("baseline_640_025", 640, 0.25),
        ("lowconf_640_015", 640, 0.15),
        ("highres_960_025", 960, 0.25),
        ("highres_lowconf_960_015", 960, 0.15),
    ]
    assert config.device == "mps"
    assert config.tracker == "bytetrack.yaml"
    assert config.max_history_length == 30
    assert config.minimum_displacement_pixels == 5.0
    assert (config.start_frame, config.end_frame) == (280, 459)


def test_runtime_model_is_fixed_and_rejected_model_usage_is_zero() -> None:
    config = load_diagnostic_config(CONFIG_PATH, PROJECT_ROOT)
    assert config.runtime_model.relative_to(PROJECT_ROOT) == APPROVED_RUNTIME_MODEL
    governance = validate_runtime_assets(config)
    assert governance["runtime_model_sha256"] == config.expected_model_sha256
    assert governance["input_video_sha256"] == config.expected_input_sha256
    assert governance["rejected_model_used"] == 0


def test_output_directories_are_isolated_and_contained() -> None:
    config = load_diagnostic_config(CONFIG_PATH, PROJECT_ROOT)
    directories = {
        experiment_output_directory(config.output_directory, item.name)
        for item in config.experiments
    }
    assert len(directories) == 4
    assert all(path.parent == config.output_directory for path in directories)
    with pytest.raises(ValueError, match="experiment name"):
        experiment_output_directory(config.output_directory, "../escape")


def test_normalized_small_object_area() -> None:
    assert normalized_bbox_area(10, 20, 30, 60, frame_width=100, frame_height=100) == pytest.approx(0.08)
    assert normalized_bbox_area(0, 0, 5, 5, frame_width=100, frame_height=100) == pytest.approx(0.0025)
    with pytest.raises(ValueError):
        normalized_bbox_area(0, 0, 1, 1, frame_width=0, frame_height=100)


def test_runtime_metrics_aggregation() -> None:
    detections = [
        {"frame_index": 1, "class_name": "car", "confidence": 0.20, "normalized_bbox_area": 0.005},
        {"frame_index": 1, "class_name": "motorcycle", "confidence": 0.249, "normalized_bbox_area": 0.002},
        {"frame_index": 2, "class_name": "bus", "confidence": 0.80, "normalized_bbox_area": 0.03},
    ]
    tracks = [
        {"frame_index": 1, "track_id": 7, "class_name": "car"},
        {"frame_index": 3, "track_id": 7, "class_name": "car"},
        {"frame_index": 2, "track_id": 8, "class_name": "bus"},
    ]
    metrics = aggregate_runtime_metrics(
        detections,
        tracks,
        total_frames=3,
        processing_seconds=1.5,
        small_area_threshold=0.01,
        low_confidence_band=(0.15, 0.25),
    )
    assert metrics["processing_fps"] == pytest.approx(2.0)
    assert metrics["total_detection_observations"] == 3
    assert metrics["small_vehicle_detection_observations"] == 2
    assert metrics["low_confidence_015_025_observations"] == 2
    assert metrics["per_class_detection_observations"] == {"bus": 1, "car": 1, "motorcycle": 1}
    assert metrics["tracking_observations"] == 3
    assert metrics["diagnostic_unique_track_ids"] == 2
    assert metrics["frame_gap_fragmentation_candidates"] == 1


def test_comparison_report_has_governance_and_no_accuracy_claim() -> None:
    config = load_diagnostic_config(CONFIG_PATH, PROJECT_ROOT)
    summaries = [{
        "experiment": item.name,
        "imgsz": item.imgsz,
        "conf": item.conf,
        "total_detection_observations": 10,
        "small_vehicle_detection_observations": 4,
        "diagnostic_unique_track_ids": 3,
        "processing_fps": 2.5,
    } for item in config.experiments]
    report = render_comparison_report(config, summaries, config.expected_model_sha256)
    assert "No detection GT" in report
    assert "no formal accuracy claim" in report
    assert "MANUAL_VISUAL_REVIEW_REQUIRED" in report
    assert "models/pretrained/yolo26n.pt" in report
    assert "Stage 17 rejected model usage: `0`" in report
