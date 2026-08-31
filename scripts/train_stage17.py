#!/usr/bin/env python3
"""Run one controlled YOLO26n Taiwan fine-tuning job on Apple MPS."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.training.governance import (
    load_split_manifest,
    sha256_file,
    validate_model_manifest,
    validate_training_isolation,
    write_json,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "training_stage17.yaml"


def finite(value: object) -> float | None:
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def cumulative_training_seconds(values: list[object]) -> float | None:
    """Sum Ultralytics elapsed-time segments across checkpoint resumes."""
    elapsed = [number for value in values if (number := finite(value)) is not None]
    if not elapsed:
        return None
    total = 0.0
    previous = elapsed[0]
    for current in elapsed[1:]:
        if current < previous:
            total += previous
        previous = current
    return total + previous


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    training = config["training"]
    expected = config["expected_counts"]
    manifest_path = PROJECT_ROOT / config["split_manifest"]
    data_yaml_path = PROJECT_ROOT / config["data_yaml"]
    base_model_path = PROJECT_ROOT / config["base_model"]
    output_root = PROJECT_ROOT / config["output_root"]
    finetuned_root = PROJECT_ROOT / config["finetuned_root"]
    materialized_path = output_root / "materialized_manifest.csv"
    if not all(path.is_file() for path in (manifest_path, data_yaml_path, base_model_path, materialized_path)):
        raise FileNotFoundError("run scripts/materialize_stage17_dataset.py before training")
    if str(training["device"]) != "mps":
        raise RuntimeError("Stage 17 requires device=mps")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not built and available; CPU fallback is forbidden")

    manifest = load_split_manifest(manifest_path)
    materialized = pd.read_csv(materialized_path, keep_default_na=False)
    selected = {
        split: set(materialized.loc[materialized["split"] == split, "image_id"])
        for split in ("TRAIN", "VAL")
    }
    isolation = validate_training_isolation(
        manifest,
        selected,
        expected_train=int(expected["TRAIN"]),
        expected_val=int(expected["VAL"]),
    )
    data_yaml_text = data_yaml_path.read_text(encoding="utf-8").lower()
    if "test:" in data_yaml_text or "locked" in data_yaml_text:
        raise RuntimeError("training data YAML exposes a forbidden test split")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run_dir = output_root / "ultralytics"
    existing_results = run_dir / "results.csv"
    resume_weights = run_dir / "weights" / "last.pt"
    completed_before_resume = 0
    if existing_results.is_file():
        completed_before_resume = len(pd.read_csv(existing_results))
    if 0 < completed_before_resume < int(training["epochs"]) and resume_weights.is_file():
        print(f"Resuming the governed Stage 17 run after epoch {completed_before_resume}")
        model = YOLO(str(resume_weights))
        model.train(resume=True)
    elif completed_before_resume >= int(training["epochs"]):
        print("Stage 17 training epochs already complete; rebuilding governed summaries only")
        model = YOLO(str(resume_weights))
    else:
        model = YOLO(str(base_model_path))
        model.train(
            data=str(data_yaml_path),
            imgsz=int(training["imgsz"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            batch=int(training["batch"]),
            device="mps",
            seed=int(training["seed"]),
            workers=int(training["workers"]),
            deterministic=bool(training["deterministic"]),
            cache=bool(training["cache"]),
            project=str(output_root),
            name="ultralytics",
            exist_ok=True,
            verbose=True,
            plots=True,
        )
        run_dir = Path(model.trainer.save_dir)
    continuation_elapsed_seconds = time.perf_counter() - started
    results_csv = run_dir / "results.csv"
    source_best = run_dir / "weights" / "best.pt"
    source_last = run_dir / "weights" / "last.pt"
    if not all(path.is_file() for path in (results_csv, source_best, source_last)):
        raise RuntimeError("Ultralytics training did not produce required artifacts")

    finetuned_root.mkdir(parents=True, exist_ok=True)
    best_weights = finetuned_root / "best.pt"
    last_weights = finetuned_root / "last.pt"
    shutil.copy2(source_best, best_weights)
    shutil.copy2(source_last, last_weights)

    val_model = YOLO(str(best_weights))
    val_metrics = val_model.val(
        data=str(data_yaml_path),
        split="val",
        imgsz=int(training["imgsz"]),
        batch=int(training["batch"]),
        device="mps",
        workers=int(training["workers"]),
        project=str(output_root),
        name="best_val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    overall = {
        "precision": finite(val_metrics.box.mp),
        "recall": finite(val_metrics.box.mr),
        "map50": finite(val_metrics.box.map50),
        "map50_95": finite(val_metrics.box.map),
    }
    per_class = []
    for class_id in sorted(val_metrics.names):
        per_class.append({
            "class_id": int(class_id),
            "class_name": str(val_metrics.names[class_id]),
            "precision": finite(val_metrics.box.p[class_id]),
            "recall": finite(val_metrics.box.r[class_id]),
            "map50": finite(val_metrics.box.ap50[class_id]),
            "map50_95": finite(val_metrics.box.maps[class_id]),
        })

    results = pd.read_csv(results_csv)
    metric_column = "metrics/mAP50-95(B)"
    best_index = int(results[metric_column].astype(float).idxmax())
    best_epoch = int(results.loc[best_index, "epoch"])
    loss_columns = [column for column in results.columns if "loss" in column]
    losses = {
        "best_epoch": {column.strip(): finite(results.loc[best_index, column]) for column in loss_columns},
        "last_epoch": {column.strip(): finite(results.iloc[-1][column]) for column in loss_columns},
    }
    class_names = {int(key): str(value) for key, value in config["class_names"].items()}
    model_manifest = {
        "base_model": config["base_model"],
        "base_model_sha256": sha256_file(base_model_path),
        "dataset_split_manifest_sha256": sha256_file(manifest_path),
        "train_count": isolation["train_count"],
        "val_count": isolation["val_count"],
        "locked_test_used": isolation["locked_test_used"],
        "excluded_used": isolation["excluded_used"],
        "class_names": class_names,
        "imgsz": int(training["imgsz"]),
        "epochs_requested": int(training["epochs"]),
        "epochs_completed": len(results),
        "batch": int(training["batch"]),
        "device": "mps",
        "seed": int(training["seed"]),
        "best_epoch": best_epoch,
        "best_validation_metrics": overall,
        "best_weights_path": best_weights.relative_to(PROJECT_ROOT).as_posix(),
        "best_weights_sha256": sha256_file(best_weights),
        "last_weights_path": last_weights.relative_to(PROJECT_ROOT).as_posix(),
        "last_weights_sha256": sha256_file(last_weights),
        "model_status": "CANDIDATE",
    }
    validate_model_manifest(model_manifest)
    summary = {
        "model_manifest": model_manifest,
        "per_class_validation_metrics": per_class,
        "losses": losses,
        "training_elapsed_seconds": cumulative_training_seconds(results["time"].tolist()),
        "continuation_and_final_validation_elapsed_seconds": continuation_elapsed_seconds,
        "runtime": {
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "mps_is_built": torch.backends.mps.is_built(),
            "mps_is_available": torch.backends.mps.is_available(),
        },
        "locked_test_policy": "Not read by training or validation; reserved for Stage 18.",
    }
    write_json(output_root / "training_summary.json", summary)
    write_json(output_root / "model_manifest.json", model_manifest)
    write_json(finetuned_root / "model_manifest.json", model_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
