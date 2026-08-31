"""Stage 18 locked-test materialization, metrics, and promotion policy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics.utils.metrics import DetMetrics, box_iou

from vision_analytics.training.governance import sha256_file

LOCKED_SPLIT = "LOCKED_TEST"
FORBIDDEN_SPLITS = ("TRAIN", "VAL", "EXCLUDED")
METRICS = ("precision", "recall", "map50", "map50_95")


def validate_locked_isolation(
    manifest: pd.DataFrame,
    selected_ids: set[str],
    *,
    expected_count: int = 271,
) -> dict[str, int]:
    locked_ids = set(manifest.loc[manifest["split"] == LOCKED_SPLIT, "image_id"].astype(str))
    if len(selected_ids) != expected_count or selected_ids != locked_ids:
        raise ValueError("materialized IDs do not exactly match the governed LOCKED_TEST")
    result = {"locked_test_count": len(selected_ids)}
    for split in FORBIDDEN_SPLITS:
        ids = set(manifest.loc[manifest["split"] == split, "image_id"].astype(str))
        overlap = len(selected_ids & ids)
        result[f"{split.lower()}_used"] = overlap
        if overlap:
            raise ValueError(f"{split} leakage detected in LOCKED_TEST materialization")
    return result


def materialize_locked_test(
    manifest: pd.DataFrame,
    *,
    project_root: Path,
    destination_root: Path,
    expected_count: int = 271,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Copy only locked images/labels, preserving source bytes and raw IDs."""
    staging = destination_root.with_name(f"{destination_root.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    image_dir, label_dir = staging / "images", staging / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    locked = manifest.loc[manifest["split"] == LOCKED_SPLIT].sort_values("image_id")
    for row in locked.to_dict("records"):
        image_id = str(row["image_id"])
        source_image = project_root / str(row["image_path"])
        source_label = project_root / str(row["label_path"])
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"missing locked source assets for {image_id}")
        safe_stem = image_id.replace("/", "__")
        image_path = image_dir / f"{safe_stem}{source_image.suffix.lower()}"
        label_path = label_dir / f"{safe_stem}.txt"
        shutil.copy2(source_image, image_path)
        shutil.copy2(source_label, label_path)
        if sha256_file(image_path) != str(row["sha256"]):
            raise ValueError(f"locked image hash mismatch for {image_id}")
        if source_label.read_bytes() != label_path.read_bytes():
            raise ValueError(f"locked label bytes changed for {image_id}")
        selected_ids.add(image_id)
        rows.append({
            "image_id": image_id,
            "source_image_path": str(row["image_path"]),
            "source_label_path": str(row["label_path"]),
            "image_path": image_path.relative_to(project_root).as_posix(),
            "label_path": label_path.relative_to(project_root).as_posix(),
            "sha256": str(row["sha256"]),
        })
    isolation = validate_locked_isolation(manifest, selected_ids, expected_count=expected_count)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    staging.rename(destination_root)
    materialized = pd.DataFrame(rows)
    materialized["image_path"] = materialized["image_path"].str.replace(
        f"{destination_root.name}.staging/", f"{destination_root.name}/", regex=False
    )
    materialized["label_path"] = materialized["label_path"].str.replace(
        f"{destination_root.name}.staging/", f"{destination_root.name}/", regex=False
    )
    return materialized, isolation


def write_locked_yaml(path: Path, *, locked_root: Path, class_names: Mapping[int, str]) -> None:
    payload = {
        "path": str(locked_root.resolve()),
        "val": "images",
        "names": {int(key): str(value) for key, value in class_names.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    classes, boxes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, bw, bh = (float(value) for value in line.split())
        classes.append(int(class_id))
        boxes.append([
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ])
    return np.asarray(classes, dtype=np.float32), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def match_predictions(
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
    pred_boxes: np.ndarray,
    true_boxes: np.ndarray,
    iou_thresholds: Sequence[float],
) -> np.ndarray:
    """Match detections exactly as the Ultralytics greedy validator does."""
    correct = np.zeros((len(pred_classes), len(iou_thresholds)), dtype=bool)
    if not len(pred_classes) or not len(true_classes):
        return correct
    iou = box_iou(torch.from_numpy(true_boxes), torch.from_numpy(pred_boxes)).numpy()
    iou *= true_classes[:, None] == pred_classes[None, :]
    for index, threshold in enumerate(iou_thresholds):
        matches = np.asarray(np.nonzero(iou >= threshold)).T
        if len(matches):
            if len(matches) > 1:
                matches = matches[np.argsort(iou[matches[:, 0], matches[:, 1]])[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), index] = True
    return correct


def canonical_predictions(result, application_names: Mapping[int, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map either COCO or fine-tuned output IDs to the shared application taxonomy by name."""
    name_to_id = {name: class_id for class_id, name in application_names.items()}
    boxes, confidences, classes = [], [], []
    for box, confidence, model_class in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    ):
        class_name = str(result.names[int(model_class)])
        if class_name not in name_to_id:
            continue
        boxes.append(box)
        confidences.append(confidence)
        classes.append(name_to_id[class_name])
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(confidences, dtype=np.float32),
        np.asarray(classes, dtype=np.float32),
    )


def metrics_from_stats(stats: list[dict[str, np.ndarray]], class_names: Mapping[int, str]) -> tuple[dict, list[dict]]:
    metrics = DetMetrics(names=dict(class_names))
    for stat in stats:
        metrics.update_stats(stat)
    metrics.process(plot=False)
    overall_values = metrics.mean_results()
    overall = dict(zip(METRICS, (float(value) for value in overall_values)))
    per_class = []
    result_by_class = {
        int(class_id): metrics.class_result(index)
        for index, class_id in enumerate(metrics.ap_class_index)
    }
    for class_id, class_name in class_names.items():
        values = result_by_class.get(int(class_id), (0.0, 0.0, 0.0, 0.0))
        per_class.append({
            "class_id": int(class_id),
            "class_name": class_name,
            **dict(zip(METRICS, (float(value) for value in values))),
            "target_count": int(metrics.nt_per_class[int(class_id)]),
        })
    return overall, per_class


def build_comparison(
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    *,
    regression_tolerance: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        values = overall.set_index("model")[metric]
        delta = float(values["fine_tuned"] - values["pretrained"])
        rows.append({"scope": "overall", "class_name": "ALL", "metric": metric,
                     "pretrained": values["pretrained"], "fine_tuned": values["fine_tuned"],
                     "delta": delta, "regression": delta < -regression_tolerance})
    indexed = per_class.set_index(["model", "class_name"])
    for class_name in sorted(per_class["class_name"].unique()):
        for metric in METRICS:
            pretrained = float(indexed.loc[("pretrained", class_name), metric])
            fine_tuned = float(indexed.loc[("fine_tuned", class_name), metric])
            delta = fine_tuned - pretrained
            rows.append({"scope": "per_class", "class_name": class_name, "metric": metric,
                         "pretrained": pretrained, "fine_tuned": fine_tuned,
                         "delta": delta, "regression": delta < -regression_tolerance})
    return pd.DataFrame(rows)


def decide_promotion(comparison: pd.DataFrame, policy: Mapping[str, object], integrity: bool) -> dict[str, object]:
    """Apply the predeclared PROMOTE/HOLD/REJECT policy without model tuning."""
    if not integrity:
        return {"decision": "REJECT", "reasons": ["evaluation integrity failed"]}

    def delta(scope: str, class_name: str, metric: str) -> float:
        row = comparison.loc[
            (comparison["scope"] == scope)
            & (comparison["class_name"] == class_name)
            & (comparison["metric"] == metric), "delta"
        ]
        if len(row) != 1:
            raise ValueError(f"missing comparison metric: {scope}/{class_name}/{metric}")
        return float(row.iloc[0])

    rejects, holds = [], []
    overall_map = delta("overall", "ALL", "map50_95")
    if overall_map < -float(policy["reject_overall_map50_95_regression"]):
        rejects.append(f"overall mAP50-95 regression {overall_map:.4f}")
    for name in policy["critical_classes"]:
        class_map = delta("per_class", str(name), "map50_95")
        if class_map < -float(policy["reject_critical_map50_95_regression"]):
            rejects.append(f"{name} mAP50-95 regression {class_map:.4f}")
    person_recall = delta("per_class", "person", "recall")
    if person_recall < -float(policy["reject_person_recall_regression"]):
        rejects.append(f"person recall regression {person_recall:.4f}")
    if rejects:
        return {"decision": "REJECT", "reasons": rejects}

    if delta("overall", "ALL", "map50") < float(policy["minimum_overall_map50_delta"]):
        holds.append("overall mAP50 improvement below promotion minimum")
    if overall_map < float(policy["minimum_overall_map50_95_delta"]):
        holds.append("overall mAP50-95 improvement below promotion minimum")
    if delta("overall", "ALL", "recall") < -float(policy["maximum_overall_recall_regression"]):
        holds.append("overall recall regression exceeds promotion allowance")
    for name in policy["critical_classes"]:
        if delta("per_class", str(name), "map50_95") < -float(policy["maximum_critical_map50_95_regression"]):
            holds.append(f"{name} mAP50-95 regression exceeds promotion allowance")
        if delta("per_class", str(name), "recall") < -float(policy["maximum_critical_recall_regression"]):
            holds.append(f"{name} recall regression exceeds promotion allowance")
    return {"decision": "HOLD" if holds else "PROMOTE", "reasons": holds or ["all promotion gates passed"]}


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
