"""Tests for Stage 18 Locked Test evaluation governance and policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.evaluation.locked_test import (
    build_comparison,
    decide_promotion,
    materialize_locked_test,
    match_predictions,
    validate_locked_isolation,
)
from vision_analytics.training.governance import sha256_file


def sample_manifest() -> pd.DataFrame:
    rows = []
    for split, count in (("TRAIN", 2), ("VAL", 1), ("LOCKED_TEST", 3), ("EXCLUDED", 1)):
        rows.extend({"image_id": f"{split}-{index}", "split": split} for index in range(count))
    return pd.DataFrame(rows)


def comparison_frames(person_delta: float = 0.0, overall_delta: float = 0.03):
    classes = ["bicycle", "bus", "car", "person", "motorcycle", "truck"]
    overall = pd.DataFrame([
        {"model": "pretrained", **{metric: 0.5 for metric in ("precision", "recall", "map50", "map50_95")}},
        {"model": "fine_tuned", **{metric: 0.5 + overall_delta for metric in ("precision", "recall", "map50", "map50_95")}},
    ])
    rows = []
    for model in ("pretrained", "fine_tuned"):
        for name in classes:
            value = 0.5
            if model == "fine_tuned":
                value += person_delta if name == "person" else 0.01
            rows.append({"model": model, "class_name": name,
                         **{metric: value for metric in ("precision", "recall", "map50", "map50_95")}})
    return overall, pd.DataFrame(rows)


def policy() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "evaluation_stage18.yaml").read_text())["promotion_policy"]


def test_only_locked_test_and_count() -> None:
    manifest = sample_manifest()
    ids = set(manifest.loc[manifest.split == "LOCKED_TEST", "image_id"])
    result = validate_locked_isolation(manifest, ids, expected_count=3)
    assert result == {"locked_test_count": 3, "train_used": 0, "val_used": 0, "excluded_used": 0}


def test_locked_leakage_rejected() -> None:
    manifest = sample_manifest()
    ids = set(manifest.loc[manifest.split == "LOCKED_TEST", "image_id"])
    ids.add("TRAIN-0")
    with pytest.raises(ValueError, match="LOCKED_TEST|leakage"):
        validate_locked_isolation(manifest, ids, expected_count=4)


def test_materialization_copies_only_locked_test(tmp_path: Path) -> None:
    rows = []
    for split in ("TRAIN", "VAL", "LOCKED_TEST", "LOCKED_TEST", "EXCLUDED"):
        image_id = f"{split}-{len(rows)}"
        image = tmp_path / "raw" / f"{image_id}.jpg"
        label = tmp_path / "raw" / f"{image_id}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(image_id.encode())
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append({
            "image_id": image_id, "split": split,
            "image_path": image.relative_to(tmp_path).as_posix(),
            "label_path": label.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(image),
        })
    materialized, isolation = materialize_locked_test(
        pd.DataFrame(rows), project_root=tmp_path,
        destination_root=tmp_path / "processed" / "locked", expected_count=2,
    )
    assert len(materialized) == 2
    assert set(materialized["image_id"]) == {"LOCKED_TEST-2", "LOCKED_TEST-3"}
    assert isolation["train_used"] == isolation["val_used"] == isolation["excluded_used"] == 0
    assert len(list((tmp_path / "processed" / "locked" / "images").iterdir())) == 2


def test_matching_and_metric_delta() -> None:
    correct = match_predictions(
        np.array([0]), np.array([0]), np.array([[0, 0, 10, 10]], dtype=np.float32),
        np.array([[0, 0, 10, 10]], dtype=np.float32), [0.5, 0.75]
    )
    assert correct.tolist() == [[True, True]]
    overall, per_class = comparison_frames()
    comparison = build_comparison(overall, per_class)
    row = comparison.query("scope == 'overall' and metric == 'map50_95'").iloc[0]
    assert row["delta"] == pytest.approx(0.03)


def test_per_class_regression_detection_and_reject_policy() -> None:
    overall, per_class = comparison_frames(person_delta=-0.2)
    comparison = build_comparison(overall, per_class)
    person = comparison.query("class_name == 'person' and metric == 'recall'").iloc[0]
    assert bool(person["regression"])
    decision = decide_promotion(comparison, policy(), integrity=True)
    assert decision["decision"] == "REJECT"
    assert any("person" in reason for reason in decision["reasons"])


def test_promotion_and_integrity_policy() -> None:
    overall, per_class = comparison_frames()
    comparison = build_comparison(overall, per_class)
    assert decide_promotion(comparison, policy(), integrity=True)["decision"] == "PROMOTE"
    assert decide_promotion(comparison, policy(), integrity=False)["decision"] == "REJECT"


def test_models_share_identical_config_and_locked_count_is_271() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "evaluation_stage18.yaml").read_text())
    assert config["protocol"]["device"] == "mps"
    assert config["protocol"]["imgsz"] == 640
    manifest = pd.read_csv(PROJECT_ROOT / config["split_manifest"], keep_default_na=False)
    assert (manifest["split"] == "LOCKED_TEST").sum() == 271


def test_model_and_manifest_hash_provenance() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "evaluation_stage18.yaml").read_text())
    stage17 = yaml.safe_load((PROJECT_ROOT / "models/finetuned/stage17/model_manifest.json").read_text())
    assert sha256_file(PROJECT_ROOT / config["split_manifest"]) == stage17["dataset_split_manifest_sha256"]
    assert sha256_file(PROJECT_ROOT / config["models"]["pretrained"]) == stage17["base_model_sha256"]
    assert sha256_file(PROJECT_ROOT / config["models"]["fine_tuned"]) == stage17["best_weights_sha256"]
