#!/usr/bin/env python3
"""Run Stage 16 QA and governed splitting on the raw Taiwan CCTV dataset."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.utils.dataset_qa import (
    SPLITS,
    build_duplicate_groups,
    deterministic_group_split,
    image_dhash,
    map_source_class,
    parse_yolo_annotation,
    sha256_file,
    validate_split_manifest,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_qa.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_qa" / "stage16"
PREVIEW_DIR = OUTPUT_DIR / "previews"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_source_classes(raw_root: Path, expected: list[str]) -> list[str]:
    yaml_paths = sorted(raw_root.rglob("data.yaml"))
    if len(yaml_paths) != 1:
        raise ValueError(f"expected one raw data.yaml, found {len(yaml_paths)}")
    payload = yaml.safe_load(yaml_paths[0].read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names")
    if isinstance(raw_names, dict):
        names = [str(raw_names[key]) for key in sorted(raw_names, key=lambda value: int(value))]
    elif isinstance(raw_names, list):
        names = [str(value) for value in raw_names]
    else:
        raise ValueError("raw data.yaml names must be a list or class-id mapping")
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(f"raw classes {names} do not match governed source taxonomy {expected}")
    return names


def asset_key(path: Path, raw_root: Path, directory_name: str) -> tuple[str, ...]:
    parts = path.relative_to(raw_root).parts
    if directory_name not in parts:
        raise ValueError(f"asset is not under a {directory_name} directory: {path}")
    index = parts.index(directory_name)
    return tuple(parts[:index] + parts[index + 1:-1] + (path.stem,))


def source_split_from_key(key: tuple[str, ...]) -> str:
    return key[0] if len(key) > 1 else "unspecified"


def infer_source_group(image_path: Path) -> str:
    """Use only explicit camera/sequence tokens; otherwise defer to visual hashes."""
    original = image_path.stem.split(".rf.", 1)[0]
    match = re.search(r"(?i)(?:^|[_-])(camera|cam|cctv|ch)[_-]?(\d+)(?:[_-]|$)", original)
    if match:
        return f"CAMERA:{match.group(1).lower()}{match.group(2)}"
    match = re.search(r"(?i)(?:^|[_-])(sequence|seq)[_-]?([a-z0-9]+)(?:[_-]|$)", original)
    if match:
        return f"SEQUENCE:{match.group(2).lower()}"
    return ""


def raw_tree_integrity(raw_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(raw_root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def draw_previews(
    included: list[dict[str, object]],
    annotations_by_id: dict[str, list[object]],
    source_classes: list[str],
    application_mapping: dict[str, str],
    *,
    per_class: int,
) -> list[str]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for stale_preview in PREVIEW_DIR.glob("*.jpg"):
        stale_preview.unlink()
    selected: dict[str, int] = {name: 0 for name in source_classes}
    paths: list[str] = []
    for record in sorted(included, key=lambda item: str(item["image_id"])):
        annotations = annotations_by_id[str(record["image_id"])]
        present = {source_classes[item.class_id] for item in annotations}
        wanted = [name for name in sorted(present) if selected[name] < per_class]
        if not wanted:
            continue
        image = cv2.imread(str(record["absolute_image_path"]))
        height, width = image.shape[:2]
        for annotation in annotations:
            source_name = source_classes[annotation.class_id]
            application_name = application_mapping[source_name]
            x1 = round((annotation.x_center - annotation.width / 2) * width)
            y1 = round((annotation.y_center - annotation.height / 2) * height)
            x2 = round((annotation.x_center + annotation.width / 2) * width)
            y2 = round((annotation.y_center + annotation.height / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                image, f"{source_name}->{application_name}", (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                image, f"{source_name}->{application_name}", (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA,
            )
        for name in wanted:
            filename = f"{name}_{selected[name] + 1:02d}_{record['image_id']}.jpg"
            safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
            output_path = PREVIEW_DIR / safe_filename
            if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(f"failed to write preview {output_path}")
            paths.append(output_path.relative_to(PROJECT_ROOT).as_posix())
            selected[name] += 1
        if all(count >= per_class for count in selected.values()):
            break
    return paths


def main() -> int:
    config = load_config()
    dataset = config["dataset"]
    duplicate_config = config["duplicate_governance"]
    split_config = config["split_governance"]
    preview_config = config["preview"]
    raw_root = PROJECT_ROOT / str(dataset["raw_root"])
    if not raw_root.is_dir():
        print(
            f"Raw Taiwan CCTV dataset not found at {raw_root}. Use the official Roboflow "
            "Universe v3 download flow; access may require sign-in/API key.",
            file=sys.stderr,
        )
        return 2

    expected_classes = [str(value) for value in dataset["source_classes"]]
    application_mapping = {str(key): str(value) for key, value in dataset["application_mapping"].items()}
    source_classes = load_source_classes(raw_root, expected_classes)
    if set(application_mapping) != set(source_classes):
        raise ValueError("application mapping must cover the unchanged source taxonomy")

    image_paths = sorted(path for path in raw_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = sorted(path for path in raw_root.rglob("*.txt") if "labels" in path.parts)
    image_by_key = {asset_key(path, raw_root, "images"): path for path in image_paths}
    label_by_key = {asset_key(path, raw_root, "labels"): path for path in label_paths}
    all_keys = sorted(set(image_by_key) | set(label_by_key))
    if len(image_by_key) != int(dataset["expected_image_count"]):
        raise ValueError(
            f"raw image count {len(image_by_key)} does not match governed version count "
            f"{dataset['expected_image_count']}"
        )

    inventory: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    valid_records: list[dict[str, object]] = []
    annotations_by_id: dict[str, list[object]] = {}
    class_images: dict[str, set[str]] = defaultdict(set)
    class_boxes: Counter[str] = Counter()
    bbox_areas: list[float] = []
    resolutions: Counter[tuple[int, int]] = Counter()

    for key in all_keys:
        image_path = image_by_key.get(key)
        label_path = label_by_key.get(key)
        image_id = "/".join(key)
        source_split = source_split_from_key(key)
        exclusion_reasons: list[str] = []
        width = height = annotation_count = 0
        sha256 = phash = ""
        image_readable = False
        annotations: list[object] = []

        if image_path is None:
            exclusion_reasons.append("LABEL_WITHOUT_IMAGE")
            issues.append({
                "image_id": image_id, "image_path": "", "label_path": label_path.relative_to(PROJECT_ROOT).as_posix(),
                "issue_type": "LABEL_WITHOUT_IMAGE", "severity": "ERROR", "line_number": "",
                "message": "Label exists without a matching image.",
            })
        else:
            image = cv2.imread(str(image_path))
            if image is None or image.size == 0:
                exclusion_reasons.append("CORRUPTED_IMAGE")
                issues.append({
                    "image_id": image_id, "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "label_path": "" if label_path is None else label_path.relative_to(PROJECT_ROOT).as_posix(),
                    "issue_type": "CORRUPTED_IMAGE", "severity": "ERROR", "line_number": "",
                    "message": "OpenCV could not decode the image.",
                })
            else:
                image_readable = True
                height, width = image.shape[:2]
                resolutions[(width, height)] += 1
                sha256 = sha256_file(image_path)
                phash = image_dhash(image)

        if label_path is None:
            exclusion_reasons.append("IMAGE_WITHOUT_LABEL")
            issues.append({
                "image_id": image_id,
                "image_path": "" if image_path is None else image_path.relative_to(PROJECT_ROOT).as_posix(),
                "label_path": "", "issue_type": "IMAGE_WITHOUT_LABEL", "severity": "ERROR",
                "line_number": "", "message": "Image exists without a matching label file.",
            })
        else:
            annotations, annotation_issues = parse_yolo_annotation(
                label_path.read_text(encoding="utf-8"), class_count=len(source_classes),
            )
            annotation_count = len(annotations)
            for issue in annotation_issues:
                issues.append({
                    "image_id": image_id,
                    "image_path": "" if image_path is None else image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "label_path": label_path.relative_to(PROJECT_ROOT).as_posix(),
                    "issue_type": issue.issue_type, "severity": issue.severity,
                    "line_number": "" if issue.line_number is None else issue.line_number,
                    "message": issue.message,
                })
                if issue.severity == "ERROR":
                    exclusion_reasons.append(issue.issue_type)

        record = {
            "image_id": image_id,
            "image_path": "" if image_path is None else image_path.relative_to(PROJECT_ROOT).as_posix(),
            "label_path": "" if label_path is None else label_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_split": source_split, "image_readable": image_readable,
            "width": width, "height": height, "annotation_count": annotation_count,
            "sha256": sha256, "phash": phash,
            "qa_status": "EXCLUDED" if exclusion_reasons else "PASS",
            "exclusion_reason": ";".join(sorted(set(exclusion_reasons))),
        }
        inventory.append(record)
        if not exclusion_reasons and image_path is not None and label_path is not None:
            governed_record = dict(record)
            governed_record["absolute_image_path"] = image_path
            governed_record["source_group_id"] = infer_source_group(image_path)
            valid_records.append(governed_record)
            annotations_by_id[image_id] = annotations
            for annotation in annotations:
                source_class = source_classes[annotation.class_id]
                class_images[source_class].add(image_id)
                class_boxes[source_class] += 1
                bbox_areas.append(annotation.area_normalized)

    duplicate_rows = build_duplicate_groups(
        valid_records,
        near_duplicate_threshold=int(duplicate_config["near_duplicate_hamming_threshold"]),
    )
    duplicate_by_id = {str(row["image_id"]): row for row in duplicate_rows}
    assignments = deterministic_group_split(
        duplicate_rows, ratios={str(k): float(v) for k, v in split_config["ratios"].items()},
        seed=int(split_config["seed"]),
    )
    split_manifest: list[dict[str, object]] = []
    inventory_by_id = {str(row["image_id"]): row for row in inventory}
    for record in inventory:
        image_id = str(record["image_id"])
        duplicate = duplicate_by_id.get(image_id)
        group_id = "" if duplicate is None else str(duplicate["group_id"])
        split_manifest.append({
            "image_id": image_id, "image_path": record["image_path"], "label_path": record["label_path"],
            "split": "EXCLUDED" if duplicate is None else assignments[group_id],
            "group_id": group_id, "sha256": record["sha256"], "phash": record["phash"],
            "width": record["width"], "height": record["height"],
            "annotation_count": record["annotation_count"],
            "source_split": record["source_split"], "exclusion_reason": record["exclusion_reason"],
        })
    validate_split_manifest(split_manifest)

    class_distribution = [{
        "source_class": source_class,
        "application_class": map_source_class(source_class, application_mapping),
        "image_count": len(class_images[source_class]),
        "bbox_count": class_boxes[source_class],
    } for source_class in source_classes]

    split_class_images: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_class_boxes: Counter[tuple[str, str]] = Counter()
    split_by_image = {str(row["image_id"]): str(row["split"]) for row in split_manifest}
    for image_id, annotations in annotations_by_id.items():
        split = split_by_image[image_id]
        if split not in SPLITS:
            continue
        for annotation in annotations:
            source_class = source_classes[annotation.class_id]
            split_class_images[(split, source_class)].add(image_id)
            split_class_boxes[(split, source_class)] += 1
    split_class_distribution = [{
        "split": split, "source_class": source_class,
        "application_class": map_source_class(source_class, application_mapping),
        "image_count": len(split_class_images[(split, source_class)]),
        "bbox_count": split_class_boxes[(split, source_class)],
    } for split in SPLITS for source_class in source_classes]

    previews = draw_previews(
        valid_records, annotations_by_id, source_classes, application_mapping,
        per_class=int(preview_config["images_per_class"]),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(OUTPUT_DIR / "dataset_inventory.csv", index=False)
    pd.DataFrame(class_distribution).to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    pd.DataFrame(issues, columns=(
        "image_id", "image_path", "label_path", "issue_type", "severity", "line_number", "message",
    )).to_csv(OUTPUT_DIR / "annotation_issues.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(OUTPUT_DIR / "duplicate_groups.csv", index=False)
    pd.DataFrame(split_manifest).to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)
    pd.DataFrame(split_class_distribution).to_csv(OUTPUT_DIR / "split_class_distribution.csv", index=False)

    areas = pd.Series(bbox_areas, dtype=float)
    split_counts = Counter(row["split"] for row in split_manifest)
    exact_groups = len({row["sha256"] for row in duplicate_rows if int(row["exact_duplicate_count"]) > 1})
    multi_groups = len({row["group_id"] for row in duplicate_rows if int(row["group_size"]) > 1})
    summary = {
        "dataset": {
            "source_id": dataset["source_id"], "title": dataset["title"], "owner": dataset["owner"],
            "source_url": dataset["source_url"], "version": dataset["version"],
            "access_date": dataset["access_date"],
            "license": dataset["license"], "raw_root": dataset["raw_root"],
            "archive_sha256": dataset["archive_sha256"],
            "raw_tree_sha256": raw_tree_integrity(raw_root),
            "source_classes_in_raw_id_order": source_classes,
            "application_mapping": application_mapping,
        },
        "counts": {
            "raw_images": len(image_by_key), "raw_labels": len(label_by_key),
            "inventory_records": len(inventory), "included_images": len(valid_records),
            "excluded_records": split_counts["EXCLUDED"], "bounding_boxes": len(bbox_areas),
            "annotation_issues": len(issues),
        },
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "duplicate_governance": {
            "sha256_exact_duplicate_groups": exact_groups,
            "multi_member_governance_groups": multi_groups,
            "phash_algorithm": duplicate_config["phash_algorithm"],
            "near_duplicate_hamming_threshold": duplicate_config["near_duplicate_hamming_threshold"],
            "limitation": duplicate_config["note"],
        },
        "bbox_area_normalized": {
            "count": len(areas), "min": float(areas.min()) if len(areas) else None,
            "p10": float(areas.quantile(0.10)) if len(areas) else None,
            "median": float(areas.median()) if len(areas) else None,
            "p90": float(areas.quantile(0.90)) if len(areas) else None,
            "max": float(areas.max()) if len(areas) else None,
            "small_lt_0_01": int((areas < 0.01).sum()),
            "medium_0_01_to_lt_0_09": int(((areas >= 0.01) & (areas < 0.09)).sum()),
            "large_ge_0_09": int((areas >= 0.09).sum()),
        },
        "resolution_distribution": [
            {"width": width, "height": height, "image_count": count}
            for (width, height), count in sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))
        ],
        "leakage_checks": {
            "image_split_unique": True, "group_cross_split_count": 0,
            "exact_duplicate_cross_split_count": 0, "near_duplicate_cross_split_count": 0,
            "locked_test_exists": split_counts["LOCKED_TEST"] > 0,
        },
        "locked_test_policy": split_config["locked_test_policy"],
        "preview_paths": previews,
        "taxonomy_limitation": (
            "Raw labels are unchanged. Commercial van versus car/truck/bus and occluded/small-object "
            "ambiguities require manual review; Stage 16 performs no relabeling."
        ),
        "status": "PASS" if not split_counts["EXCLUDED"] else "WARNING",
    }
    (OUTPUT_DIR / "qa_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Raw images={len(image_by_key)} labels={len(label_by_key)} boxes={len(bbox_areas)}")
    print(f"Issues={len(issues)} excluded={split_counts['EXCLUDED']}")
    print("Splits " + " ".join(f"{split}={split_counts[split]}" for split in SPLITS))
    print(f"Exact duplicate groups={exact_groups}; multi-member governance groups={multi_groups}")
    print(f"Wrote Stage 16 QA outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
