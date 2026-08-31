"""Tests for Stage 17 materialization and candidate-model governance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.training.governance import (
    MODEL_MANIFEST_REQUIRED_FIELDS,
    load_split_manifest,
    materialize_dataset,
    sha256_file,
    validate_model_manifest,
    validate_training_isolation,
    write_training_data_yaml,
)


def manifest_frame(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for split, count in (("TRAIN", 2), ("VAL", 1), ("LOCKED_TEST", 1), ("EXCLUDED", 1)):
        for index in range(count):
            stem = f"{split.lower()}-{index}"
            image = tmp_path / "raw" / f"{stem}.jpg"
            label = tmp_path / "raw" / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{stem}".encode())
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            rows.append({
                "image_id": stem,
                "image_path": image.relative_to(tmp_path).as_posix(),
                "label_path": label.relative_to(tmp_path).as_posix(),
                "split": split,
                "exclusion_reason": "invalid" if split == "EXCLUDED" else "",
                "sha256": sha256_file(image),
            })
    return pd.DataFrame(rows)


def test_training_isolation_counts_and_forbidden_splits(tmp_path: Path) -> None:
    manifest = manifest_frame(tmp_path)
    selected = {
        "TRAIN": set(manifest.loc[manifest["split"] == "TRAIN", "image_id"]),
        "VAL": set(manifest.loc[manifest["split"] == "VAL", "image_id"]),
    }
    result = validate_training_isolation(manifest, selected, expected_train=2, expected_val=1)
    assert result["locked_test_used"] == 0
    assert result["excluded_used"] == 0
    assert result["locked_test_available"] == 1
    assert result["excluded_available"] == 1


def test_locked_test_and_excluded_leakage_fail(tmp_path: Path) -> None:
    manifest = manifest_frame(tmp_path)
    train_ids = set(manifest.loc[manifest["split"] == "TRAIN", "image_id"])
    val_ids = set(manifest.loc[manifest["split"] == "VAL", "image_id"])
    train_ids.add(str(manifest.loc[manifest["split"] == "LOCKED_TEST", "image_id"].iloc[0]))
    with pytest.raises(ValueError, match="counts differ|LOCKED_TEST"):
        validate_training_isolation(manifest, {"TRAIN": train_ids, "VAL": val_ids}, expected_train=2, expected_val=1)


def test_materialization_train_val_only_and_raw_labels_unchanged(tmp_path: Path) -> None:
    manifest = manifest_frame(tmp_path)
    raw_label_hashes = {
        row.image_id: sha256_file(tmp_path / row.label_path)
        for row in manifest.itertuples()
    }
    materialized, isolation = materialize_dataset(
        manifest,
        project_root=tmp_path,
        dataset_root=tmp_path / "processed" / "dataset",
        expected_train=2,
        expected_val=1,
    )
    assert len(materialized) == 3
    assert set(materialized["split"]) == {"TRAIN", "VAL"}
    assert isolation["locked_test_used"] == isolation["excluded_used"] == 0
    assert not (tmp_path / "processed" / "dataset" / "locked_test").exists()
    for row in manifest.itertuples():
        assert sha256_file(tmp_path / row.label_path) == raw_label_hashes[row.image_id]


def test_data_yaml_has_application_names_and_no_test_path(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    names = {0: "bicycle", 1: "bus", 2: "car", 3: "person", 4: "motorcycle", 5: "truck"}
    write_training_data_yaml(path, dataset_root=tmp_path / "dataset", class_names=names)
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert payload["names"] == names
    assert payload["train"] == "train/images"
    assert payload["val"] == "val/images"
    assert "test" not in payload
    assert "locked" not in text.lower()


def test_stage17_training_config() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "training_stage17.yaml").read_text())
    assert config["expected_counts"] == {
        "TRAIN": 1266, "VAL": 271, "LOCKED_TEST_USED": 0, "EXCLUDED_USED": 0,
    }
    assert config["training"] == {
        "imgsz": 640, "epochs": 50, "patience": 10, "batch": 8,
        "device": "mps", "seed": 1601, "workers": 4,
        "deterministic": True, "cache": False,
    }


def test_model_manifest_required_fields_hashes_and_candidate_status(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    weights = tmp_path / "best.pt"
    split = tmp_path / "split.csv"
    base.write_bytes(b"base")
    weights.write_bytes(b"candidate")
    split.write_text("manifest", encoding="utf-8")
    payload = {field: "value" for field in MODEL_MANIFEST_REQUIRED_FIELDS}
    payload.update({
        "base_model_sha256": sha256_file(base),
        "dataset_split_manifest_sha256": sha256_file(split),
        "best_weights_sha256": sha256_file(weights),
        "locked_test_used": 0,
        "excluded_used": 0,
        "model_status": "CANDIDATE",
        "device": "mps",
    })
    validate_model_manifest(payload)
    assert payload["base_model_sha256"] != payload["best_weights_sha256"]
    broken = dict(payload, locked_test_used=1)
    with pytest.raises(ValueError, match="forbidden"):
        validate_model_manifest(broken)


def test_load_split_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = manifest_frame(tmp_path)
    path = tmp_path / "manifest.csv"
    pd.concat([manifest, manifest.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_split_manifest(path)
