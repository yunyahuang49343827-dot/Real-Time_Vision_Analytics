#!/usr/bin/env python3
"""Run the one-time governed Stage 18 pretrained/candidate locked evaluation."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.evaluation.locked_test import (
    build_comparison,
    canonical_predictions,
    decide_promotion,
    materialize_locked_test,
    match_predictions,
    metrics_from_stats,
    read_yolo_labels,
    write_json,
    write_locked_yaml,
)
from vision_analytics.training.governance import load_split_manifest, sha256_file

CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation_stage18.yaml"


def evaluate_model(
    model_name: str,
    weights: Path,
    materialized: pd.DataFrame,
    class_names: dict[int, str],
    protocol: dict,
    output_root: Path,
) -> tuple[dict, list[dict], pd.DataFrame, dict]:
    model = YOLO(str(weights))
    image_paths = [str(PROJECT_ROOT / path) for path in materialized["image_path"]]
    kwargs = {
        "source": image_paths,
        "stream": True,
        "imgsz": int(protocol["imgsz"]),
        "device": str(protocol["device"]),
        "batch": int(protocol["batch"]),
        "conf": float(protocol["conf"]),
        "iou": float(protocol["iou"]),
        "max_det": int(protocol["max_det"]),
        "half": bool(protocol["half"]),
        "agnostic_nms": bool(protocol["agnostic_nms"]),
        "verbose": False,
        "save": False,
    }
    stats, prediction_rows, diagnostics = [], [], []
    started = time.perf_counter()
    results = model.predict(**kwargs)
    for row, result in zip(materialized.to_dict("records"), results, strict=True):
        image = cv2.imread(str(PROJECT_ROOT / row["image_path"]))
        if image is None:
            raise RuntimeError(f"cannot read locked image {row['image_id']}")
        height, width = image.shape[:2]
        target_cls, target_boxes = read_yolo_labels(PROJECT_ROOT / row["label_path"], width, height)
        pred_boxes, pred_conf, pred_cls = canonical_predictions(result, class_names)
        correct = match_predictions(
            pred_cls, target_cls, pred_boxes, target_boxes, protocol["iou_thresholds"]
        )
        stats.append({
            "tp": correct,
            "conf": pred_conf,
            "pred_cls": pred_cls,
            "target_cls": target_cls,
            "target_img": np.unique(target_cls),
            "im_name": str(row["image_id"]),
        })
        for box, confidence, class_id in zip(pred_boxes, pred_conf, pred_cls):
            prediction_rows.append({
                "model": model_name, "image_id": row["image_id"],
                "class_id": int(class_id), "class_name": class_names[int(class_id)],
                "confidence": float(confidence), "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
            })
        matched_at_50 = correct[:, 0] if len(correct) else np.zeros(0, dtype=bool)
        for class_id in class_names:
            target_count = int((target_cls == class_id).sum())
            true_positive_count = int(((pred_cls == class_id) & matched_at_50).sum())
            diagnostics.append({
                "model": model_name, "image_id": row["image_id"],
                "class_name": class_names[class_id], "target_count": target_count,
                "tp_at_iou50": true_positive_count,
                "fn_candidate_count": max(0, target_count - true_positive_count),
            })
    elapsed = time.perf_counter() - started
    overall, per_class = metrics_from_stats(stats, class_names)
    runtime = {"elapsed_seconds": elapsed, "images": len(materialized), "images_per_second": len(materialized) / elapsed}
    pd.DataFrame(prediction_rows).to_csv(output_root / f"{model_name}_predictions.csv", index=False)
    return overall, per_class, pd.DataFrame(diagnostics), runtime


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    sealed = output_root / "evaluation_manifest.json"
    if sealed.is_file():
        raise RuntimeError("Stage 18 evaluation is already sealed; Locked Test rerun is forbidden")
    if config["protocol"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("Stage 18 requires available MPS; CPU fallback is forbidden")

    manifest_path = PROJECT_ROOT / config["split_manifest"]
    locked_root = PROJECT_ROOT / config["locked_test_root"]
    locked_yaml = PROJECT_ROOT / config["locked_test_yaml"]
    class_names = {int(key): str(value) for key, value in config["class_names"].items()}
    manifest = load_split_manifest(manifest_path)
    materialized, isolation = materialize_locked_test(
        manifest, project_root=PROJECT_ROOT, destination_root=locked_root
    )
    write_locked_yaml(locked_yaml, locked_root=locked_root, class_names=class_names)
    output_root.mkdir(parents=True, exist_ok=True)
    materialized.to_csv(output_root / "locked_test_manifest.csv", index=False)
    state = {
        "status": "IN_PROGRESS",
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    write_json(output_root / "evaluation_run_state.json", state)

    overall_rows, class_rows, diagnostic_frames, runtimes = [], [], [], {}
    model_hashes = {}
    for model_name in ("pretrained", "fine_tuned"):
        weights = PROJECT_ROOT / config["models"][model_name]
        model_hashes[model_name] = sha256_file(weights)
        overall, per_class, diagnostics, runtime = evaluate_model(
            model_name, weights, materialized, class_names, config["protocol"], output_root
        )
        overall_rows.append({"model": model_name, **overall})
        class_rows.extend({"model": model_name, **row} for row in per_class)
        diagnostic_frames.append(diagnostics)
        runtimes[model_name] = runtime

    overall_frame = pd.DataFrame(overall_rows)
    per_class_frame = pd.DataFrame(class_rows)
    comparison = build_comparison(overall_frame, per_class_frame)
    integrity = (
        isolation == {"locked_test_count": 271, "train_used": 0, "val_used": 0, "excluded_used": 0}
        and model_hashes["pretrained"] == sha256_file(PROJECT_ROOT / config["models"]["pretrained"])
        and model_hashes["fine_tuned"] == sha256_file(PROJECT_ROOT / config["models"]["fine_tuned"])
    )
    decision = decide_promotion(comparison, config["promotion_policy"], integrity)
    decision.update({"evaluation_integrity": integrity, "runtime_model_replaced": False})
    overall_frame.to_csv(output_root / "overall_metrics.csv", index=False)
    per_class_frame.to_csv(output_root / "per_class_metrics.csv", index=False)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostics.groupby(["model", "class_name"], as_index=False)[
        ["target_count", "tp_at_iou50", "fn_candidate_count"]
    ].sum().to_csv(output_root / "diagnostic_error_summary.csv", index=False)
    write_json(output_root / "promotion_decision.json", decision)
    evaluation_manifest = {
        "status": "SEALED",
        "locked_test_first_use_stage": 18,
        "locked_test_isolation": isolation,
        "locked_test_image_ids": sorted(materialized["image_id"].astype(str).tolist()),
        "stage16_split_manifest": config["split_manifest"],
        "stage16_split_manifest_sha256": sha256_file(manifest_path),
        "models": {
            name: {"path": config["models"][name], "sha256": model_hashes[name]}
            for name in ("pretrained", "fine_tuned")
        },
        "application_class_names": class_names,
        "class_mapping_policy": "Model output class names are mapped to the shared application IDs; raw labels unchanged.",
        "evaluation_protocol": config["protocol"],
        "identical_protocol_for_both_models": True,
        "runtimes": runtimes,
        "promotion_policy": config["promotion_policy"],
        "promotion_decision": decision["decision"],
        "locked_test_tuning_performed": False,
        "runtime_model_replaced": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
    }
    write_json(sealed, evaluation_manifest)
    write_json(output_root / "evaluation_run_state.json", {**state, "status": "SEALED"})
    print(json.dumps({"overall": overall_rows, "decision": decision, "manifest": evaluation_manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
