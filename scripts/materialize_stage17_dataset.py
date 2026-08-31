#!/usr/bin/env python3
"""Materialize only Stage 16 TRAIN and VAL assets for Stage 17."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.training.governance import (
    load_split_manifest,
    materialize_dataset,
    sha256_file,
    write_training_data_yaml,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "training_stage17.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / config["split_manifest"]
    dataset_root = PROJECT_ROOT / config["processed_root"]
    data_yaml_path = PROJECT_ROOT / config["data_yaml"]
    output_root = PROJECT_ROOT / config["output_root"]
    expected = config["expected_counts"]
    class_names = {int(key): str(value) for key, value in config["class_names"].items()}

    manifest = load_split_manifest(manifest_path)
    materialized, isolation = materialize_dataset(
        manifest,
        project_root=PROJECT_ROOT,
        dataset_root=dataset_root,
        expected_train=int(expected["TRAIN"]),
        expected_val=int(expected["VAL"]),
    )
    write_training_data_yaml(data_yaml_path, dataset_root=dataset_root, class_names=class_names)
    output_root.mkdir(parents=True, exist_ok=True)
    materialized.to_csv(output_root / "materialized_manifest.csv", index=False)
    report = {
        **isolation,
        "split_manifest_path": config["split_manifest"],
        "split_manifest_sha256": sha256_file(manifest_path),
        "data_yaml_path": config["data_yaml"],
        "class_names": class_names,
        "raw_labels_modified": False,
        "status": "PASS",
    }
    (output_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
