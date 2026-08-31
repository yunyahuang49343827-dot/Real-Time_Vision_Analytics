#!/usr/bin/env python3
"""Explain sealed Stage 18 results without rerunning inference or tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics.utils.metrics import box_iou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.evaluation.locked_test import read_yolo_labels

CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation_stage18.yaml"


def size_bin(box: np.ndarray, width: int, height: int) -> str:
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])) / (width * height)
    if area < 0.01:
        return "SMALL_LT_0.01"
    if area < 0.09:
        return "MEDIUM_0.01_TO_LT_0.09"
    return "LARGE_GE_0.09"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    if not (output_root / "evaluation_manifest.json").is_file():
        raise RuntimeError("sealed Stage 18 evaluation is required")
    class_names = {int(key): str(value) for key, value in config["class_names"].items()}
    locked = pd.read_csv(output_root / "locked_test_manifest.csv", keep_default_na=False)
    rows, confusion_rows = [], []
    preview_candidates = []
    for model_name in ("pretrained", "fine_tuned"):
        predictions = pd.read_csv(output_root / f"{model_name}_predictions.csv", keep_default_na=False)
        grouped = {key: frame for key, frame in predictions.groupby("image_id")}
        for sample in locked.to_dict("records"):
            image = cv2.imread(str(PROJECT_ROOT / sample["image_path"]))
            height, width = image.shape[:2]
            target_cls, target_boxes = read_yolo_labels(PROJECT_ROOT / sample["label_path"], width, height)
            pred = grouped.get(sample["image_id"], pd.DataFrame())
            pred_boxes = pred[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32) if len(pred) else np.empty((0, 4), np.float32)
            pred_cls = pred["class_id"].to_numpy(dtype=int) if len(pred) else np.empty(0, dtype=int)
            ious = box_iou(torch.from_numpy(target_boxes), torch.from_numpy(pred_boxes)).numpy() \
                if len(pred_boxes) and len(target_boxes) else np.zeros((len(target_boxes), len(pred_boxes)))
            for target_index, (class_id, target_box) in enumerate(zip(target_cls.astype(int), target_boxes)):
                same = np.where((pred_cls == class_id) & (ious[target_index] >= 0.5))[0]
                other = np.where((pred_cls != class_id) & (ious[target_index] >= 0.5))[0]
                predicted_class = ""
                if len(same):
                    outcome = "MATCHED_AT_IOU50"
                elif len(other):
                    best = other[np.argmax(ious[target_index, other])]
                    predicted_class = class_names[int(pred_cls[best])]
                    outcome = "CLASS_CONFUSION_CANDIDATE"
                    confusion_rows.append({
                        "model": model_name, "target_class": class_names[class_id],
                        "predicted_class": predicted_class, "count": 1,
                    })
                else:
                    outcome = "MISS_OR_LOCALIZATION_CANDIDATE"
                record = {
                    "model": model_name, "image_id": sample["image_id"],
                    "target_class": class_names[class_id], "predicted_class": predicted_class,
                    "outcome": outcome, "size_bin": size_bin(target_box, width, height),
                    "target_area_normalized": float(
                        (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]) / (width * height)
                    ),
                }
                rows.append(record)
                if model_name == "fine_tuned" and outcome != "MATCHED_AT_IOU50":
                    preview_candidates.append({**record, "target_box": target_box, "image_path": sample["image_path"]})

    details = pd.DataFrame(rows)
    details.to_csv(output_root / "diagnostic_object_outcomes.csv", index=False)
    details.groupby(["model", "target_class", "outcome", "size_bin"], as_index=False).size().rename(
        columns={"size": "object_count"}
    ).to_csv(output_root / "diagnostic_miss_analysis.csv", index=False)
    confusion = pd.DataFrame(confusion_rows, columns=["model", "target_class", "predicted_class", "count"])
    if len(confusion):
        confusion = confusion.groupby(["model", "target_class", "predicted_class"], as_index=False)["count"].sum()
    confusion.to_csv(output_root / "diagnostic_confusions.csv", index=False)

    previews = output_root / "diagnostic_previews"
    previews.mkdir(exist_ok=True)
    selected = []
    candidates = pd.DataFrame(preview_candidates)
    for class_name in ("person", "motorcycle", "bicycle", "car", "bus", "truck"):
        subset = candidates.loc[candidates["target_class"] == class_name].sort_values("target_area_normalized")
        for _, candidate in subset.head(2).iterrows():
            image = cv2.imread(str(PROJECT_ROOT / candidate["image_path"]))
            box = np.asarray(candidate["target_box"], dtype=int)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 3)
            cv2.putText(image, f"GT {class_name} | {candidate['outcome']}",
                        (max(0, box[0]), max(25, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            filename = f"{class_name}_{len(selected):02d}.jpg"
            path = previews / filename
            cv2.imwrite(str(path), image)
            selected.append({
                "image_id": candidate["image_id"], "target_class": class_name,
                "outcome": candidate["outcome"], "size_bin": candidate["size_bin"],
                "preview_path": path.relative_to(PROJECT_ROOT).as_posix(),
            })
    pd.DataFrame(selected).to_csv(output_root / "diagnostic_review_samples.csv", index=False)
    print(details.groupby(["model", "target_class", "outcome"]).size().to_string())
    print(f"Diagnostic previews: {len(selected)} (sealed predictions only; no inference rerun)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
