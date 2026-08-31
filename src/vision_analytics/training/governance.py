"""Stage 17 dataset materialization and model-manifest governance."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

TRAINING_SPLITS = ("TRAIN", "VAL")
FORBIDDEN_SPLITS = ("LOCKED_TEST", "EXCLUDED")
MODEL_MANIFEST_REQUIRED_FIELDS = {
    "base_model",
    "base_model_sha256",
    "dataset_split_manifest_sha256",
    "train_count",
    "val_count",
    "locked_test_used",
    "excluded_used",
    "class_names",
    "imgsz",
    "epochs_requested",
    "epochs_completed",
    "batch",
    "device",
    "seed",
    "best_epoch",
    "best_validation_metrics",
    "best_weights_path",
    "best_weights_sha256",
    "model_status",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, keep_default_na=False)
    required = {"image_id", "image_path", "label_path", "split", "exclusion_reason"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"split manifest is missing columns: {sorted(missing)}")
    if manifest["image_id"].duplicated().any():
        raise ValueError("split manifest image_id values must be unique")
    return manifest


def validate_training_isolation(
    manifest: pd.DataFrame,
    selected_ids: Mapping[str, set[str]],
    *,
    expected_train: int,
    expected_val: int,
) -> dict[str, object]:
    if set(selected_ids) != set(TRAINING_SPLITS):
        raise ValueError("selected_ids must contain TRAIN and VAL only")
    train_ids, val_ids = selected_ids["TRAIN"], selected_ids["VAL"]
    locked_ids = set(manifest.loc[manifest["split"] == "LOCKED_TEST", "image_id"])
    excluded_ids = set(manifest.loc[manifest["split"] == "EXCLUDED", "image_id"])
    if len(train_ids) != expected_train or len(val_ids) != expected_val:
        raise ValueError(
            f"governed counts differ: TRAIN={len(train_ids)} VAL={len(val_ids)}; "
            f"expected {expected_train}/{expected_val}"
        )
    if train_ids & val_ids:
        raise ValueError("TRAIN and VAL image IDs overlap")
    used_ids = train_ids | val_ids
    locked_overlap = used_ids & locked_ids
    excluded_overlap = used_ids & excluded_ids
    if locked_overlap:
        raise ValueError(f"LOCKED_TEST leakage detected: {sorted(locked_overlap)[:5]}")
    if excluded_overlap:
        raise ValueError(f"EXCLUDED leakage detected: {sorted(excluded_overlap)[:5]}")
    expected_ids = set(manifest.loc[manifest["split"].isin(TRAINING_SPLITS), "image_id"])
    if used_ids != expected_ids:
        raise ValueError("materialized IDs do not exactly match governed TRAIN and VAL IDs")
    return {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "locked_test_available": len(locked_ids),
        "excluded_available": len(excluded_ids),
        "locked_test_used": len(locked_overlap),
        "excluded_used": len(excluded_overlap),
        "train_val_overlap": 0,
    }


def write_training_data_yaml(
    path: Path,
    *,
    dataset_root: Path,
    class_names: Mapping[int, str],
) -> None:
    if sorted(class_names) != list(range(len(class_names))):
        raise ValueError("class IDs must be contiguous and zero-based")
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {int(key): str(value) for key, value in class_names.items()},
    }
    serialized = yaml.safe_dump(payload, sort_keys=False)
    if "test:" in serialized.lower() or "locked" in serialized.lower():
        raise ValueError("training data YAML must not contain a test or locked-test path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def materialize_dataset(
    manifest: pd.DataFrame,
    *,
    project_root: Path,
    dataset_root: Path,
    expected_train: int,
    expected_val: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Copy governed TRAIN/VAL assets without altering raw files."""
    staging_root = dataset_root.with_name(f"{dataset_root.name}.staging")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    selected_ids: dict[str, set[str]] = {split: set() for split in TRAINING_SPLITS}
    rows: list[dict[str, object]] = []
    for split in TRAINING_SPLITS:
        destination_split = split.lower()
        image_dir = staging_root / destination_split / "images"
        label_dir = staging_root / destination_split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source_rows = manifest.loc[manifest["split"] == split].sort_values("image_id")
        for row in source_rows.to_dict("records"):
            image_id = str(row["image_id"])
            source_image = project_root / str(row["image_path"])
            source_label = project_root / str(row["label_path"])
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(f"missing governed source assets for {image_id}")
            safe_stem = image_id.replace("/", "__")
            destination_image = image_dir / f"{safe_stem}{source_image.suffix.lower()}"
            destination_label = label_dir / f"{safe_stem}.txt"
            shutil.copy2(source_image, destination_image)
            shutil.copy2(source_label, destination_label)
            if sha256_file(destination_image) != str(row["sha256"]):
                raise ValueError(f"materialized image hash differs for {image_id}")
            if source_label.read_bytes() != destination_label.read_bytes():
                raise ValueError(f"materialized label bytes differ for {image_id}")
            selected_ids[split].add(image_id)
            rows.append({
                "image_id": image_id,
                "split": split,
                "source_image_path": str(row["image_path"]),
                "source_label_path": str(row["label_path"]),
                "materialized_image_path": destination_image.relative_to(project_root).as_posix(),
                "materialized_label_path": destination_label.relative_to(project_root).as_posix(),
                "image_sha256": str(row["sha256"]),
            })
    isolation = validate_training_isolation(
        manifest,
        selected_ids,
        expected_train=expected_train,
        expected_val=expected_val,
    )
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    staging_root.rename(dataset_root)
    materialized = pd.DataFrame(rows)
    return materialized, isolation


def validate_model_manifest(payload: Mapping[str, object]) -> None:
    missing = MODEL_MANIFEST_REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"model manifest is missing fields: {sorted(missing)}")
    if int(payload["locked_test_used"]) != 0 or int(payload["excluded_used"]) != 0:
        raise ValueError("candidate model provenance includes forbidden samples")
    if payload["model_status"] != "CANDIDATE":
        raise ValueError("Stage 17 model_status must remain CANDIDATE")
    if str(payload["device"]) != "mps":
        raise ValueError("Stage 17 training device must be mps")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
