#!/usr/bin/env python3
"""Run V2-2B UrbanScene acquisition integrity and image-level QA."""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.utils.dataset_qa import image_dhash  # noqa: E402
from vision_analytics.utils.urbanscene_qa import (  # noqa: E402
    aggregate_image_coverage,
    cross_dataset_overlap_blocked,
    detection_suitability,
    discover_annotation_artifacts,
    duplicate_groups,
    map_source_category,
    raw_tree_sha256,
    sha256_file,
)

CONFIG_PATH = PROJECT_ROOT / "configs/v2_urbanscene_qa.yaml"
OUTPUT = PROJECT_ROOT / "outputs/data_qa/v2_urbanscene"
EXPECTED_ARCHIVES = {
    "Motorbikes_&_Cyclist.zip": (1168975616, "42fd6393116e9817d456a03fce5b85afb81b309b0588dcae29d0e64b0c07c109"),
    "Pedestrians.zip": (1133335639, "c6ebd3a86441d6109226dd3679c9e192d4f9ec73292365bc18b0a10949fcd33e"),
    "Traffic.zip": (1371652798, "88732a107bc2ec1c2aea39503db895bb9c1e069f7e3b87621bd5528eacb7ff8c"),
}


def _source_category(path: Path, root: Path) -> str:
    values = [part.lower().replace("_", " ") for part in path.relative_to(root).parts[:-1]]
    for value in reversed(values):
        if "pedestrian" in value:
            return "Pedestrians"
        if "motorbike" in value and "cycl" in value:
            return "Motorbikes & Cyclists"
        if "motorbike" in value:
            return "Motorbikes"
        if "cycl" in value:
            return "Cyclists"
        if "traffic" in value:
            return "Traffic"
    return "UNKNOWN"


def _time_of_day(path: Path, root: Path) -> str:
    values = [part.lower() for part in path.relative_to(root).parts[:-1]]
    aliases = {
        "MORNING": ("morning", "mrngresized"),
        "EVENING": ("evening", "everesized"),
        "NIGHT": ("night", "ngtresized"),
    }
    for label, tokens in aliases.items():
        if any(any(token in value for token in tokens) for value in values):
            return label
    return "UNSPECIFIED"


def _archive_integrity(raw_root: Path) -> list[dict[str, object]]:
    archives = {path.name: path for path in (raw_root / "packages").glob("*.zip")}
    rows = []
    for name, (expected_size, expected_sha) in EXPECTED_ARCHIVES.items():
        path = archives.get(name)
        if path is None:
            rows.append({"filename": name, "status": "MISSING", "size_bytes": 0,
                         "sha256": "", "expected_size_bytes": expected_size,
                         "expected_sha256": expected_sha, "zip_integrity": "NOT_TESTED"})
            continue
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
        actual_sha = sha256_file(path)
        valid = path.stat().st_size == expected_size and actual_sha == expected_sha and bad_member is None
        rows.append({"filename": name, "status": "PASS" if valid else "FAIL",
                     "size_bytes": path.stat().st_size, "sha256": actual_sha,
                     "expected_size_bytes": expected_size, "expected_sha256": expected_sha,
                     "zip_integrity": "PASS" if bad_member is None else "FAIL"})
    return rows


def _reference_records() -> dict[str, list[dict[str, object]]]:
    taiwan = pd.read_csv(OUTPUT.parent / "stage16" / "dataset_inventory.csv")
    split = pd.read_csv(OUTPUT.parent / "stage16" / "split_manifest.csv")
    locked_ids = set(split.loc[split["split"] == "LOCKED_TEST", "image_id"].astype(str))
    visdrone = pd.read_csv(OUTPUT.parent / "v2_visdrone" / "dataset_inventory.csv")
    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        usable = frame.loc[frame["phash"].notna() & frame["sha256"].notna(), ["image_id", "sha256", "phash"]]
        return usable.astype(str).to_dict("records")
    return {
        "v1_taiwan_all": records(taiwan),
        "stage18_locked_test": records(taiwan.loc[taiwan["image_id"].astype(str).isin(locked_ids)]),
        "v2_visdrone": records(visdrone),
    }


def _write_structure_previews(candidates: dict[str, Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for key, path in candidates.items():
        image = cv2.imread(str(path))
        if image is None:
            continue
        cv2.putText(image, f"SOURCE FOLDER SAMPLE: {key}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "NO BBOX ANNOTATION AVAILABLE", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(destination / f"{key.lower().replace(' ', '_').replace('&', 'and')}.jpg"), image)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_root = PROJECT_ROOT / config["dataset"]["raw_root"]
    extracted = raw_root / "extracted"
    if not extracted.is_dir():
        raise FileNotFoundError(f"missing extracted raw tree: {extracted}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive_rows = _archive_integrity(raw_root)
    if any(row["status"] != "PASS" for row in archive_rows):
        raise ValueError("official archive byte/hash/integrity verification failed")

    annotation_audit = discover_annotation_artifacts(extracted)
    images = sorted(path for path in extracted.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    inventory_rows, issue_rows = [], []
    preview_candidates: dict[str, Path] = {}
    for path in images:
        image_id = path.relative_to(extracted).as_posix()
        category = _source_category(path, extracted)
        time_of_day = _time_of_day(path, extracted)
        image = cv2.imread(str(path))
        readable = image is not None
        width = int(image.shape[1]) if readable else 0
        height = int(image.shape[0]) if readable else 0
        if category == "UNKNOWN":
            application_class, disposition = "", "UNKNOWN_SOURCE_FOLDER"
            issue_rows.append({"image_id": image_id, "issue_type": "UNKNOWN_SOURCE_FOLDER",
                               "severity": "WARNING", "message": "No documented category found in path."})
        else:
            application_class, disposition = map_source_category(category)
        if not readable:
            issue_rows.append({"image_id": image_id, "issue_type": "UNREADABLE_IMAGE",
                               "severity": "ERROR", "message": "OpenCV could not decode image."})
        else:
            preview_candidates.setdefault(f"{category}_{time_of_day}", path)
            if (width, height) != (768, 1024):
                issue_rows.append({
                    "image_id": image_id, "issue_type": "NON_STANDARD_RESOLUTION",
                    "severity": "WARNING",
                    "message": f"Decoded as {width}x{height}; published dominant size is 768x1024.",
                })
        inventory_rows.append({
            "image_id": image_id, "image_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "source_category": category, "time_of_day": time_of_day,
            "application_class_semantic": application_class or "",
            "mapping_disposition": disposition, "image_readable": readable,
            "width": width, "height": height,
            "annotation_path": "", "annotation_type": annotation_audit["status"],
            "object_bbox_count": "UNAVAILABLE_NO_OBJECT_ANNOTATIONS",
            "sha256": sha256_file(path), "phash": image_dhash(image) if readable else "",
            "capture_group_id": f"FOLDER:{category}:{time_of_day}",
            "capture_group_method": "FOLDER_GROUP_ONLY_NOT_SEQUENCE_ID",
        })

    if annotation_audit["status"] == "IMAGE_LEVEL_CATEGORY_ONLY":
        issue_rows.append({"image_id": "__DATASET__", "issue_type": "NO_OBJECT_LEVEL_BBOX_ANNOTATIONS",
                           "severity": "BLOCKER", "message": (
                               "No paired TXT/XML/JSON/CSV object annotations exist in the extracted raw tree."
                           )})
    else:
        issue_rows.append({"image_id": "__DATASET__", "issue_type": "UNVERIFIED_ANNOTATION_SCHEMA",
                           "severity": "BLOCKER", "message": (
                               "Possible annotation artifacts require documented owner schema before parsing."
                           )})

    inventory = pd.DataFrame(inventory_rows).sort_values("image_id")
    usable = inventory.loc[inventory["image_readable"] & inventory["phash"].ne("")]
    duplicate_frame = pd.DataFrame(duplicate_groups(
        usable[["image_id", "sha256", "phash"]].to_dict("records"),
        threshold=int(config["qa"]["near_duplicate_hamming_threshold"]),
    ))
    inventory = inventory.merge(duplicate_frame, on="image_id", how="left")
    class_distribution = pd.DataFrame(aggregate_image_coverage(inventory.to_dict("records")))
    mapping_rows = []
    for source in sorted(inventory["source_category"].unique()):
        application, disposition = map_source_category(source)
        mapping_rows.append({
            "source_category": source, "application_class_semantic": application or "",
            "mapping_disposition": disposition, "source_taxonomy_preserved": True,
            "usable_as_object_detection_label": False,
        })
    mapping = pd.DataFrame(mapping_rows)
    issues = pd.DataFrame(issue_rows, columns=("image_id", "issue_type", "severity", "message"))

    overlaps = {}
    for name, reference in _reference_records().items():
        result = cross_dataset_overlap_blocked(
            usable[["image_id", "sha256", "phash"]].to_dict("records"), reference,
            threshold=int(config["qa"]["near_duplicate_hamming_threshold"]),
        )
        overlaps[name] = {
            **{key: value for key, value in result.items() if key not in {"exact_pairs", "near_pairs"}},
            "exact_pairs": result["exact_pairs"],
            "near_pairs_sample": result["near_pairs"][:100],
            "near_pairs_sample_truncated": len(result["near_pairs"]) > 100,
        }
    multi = duplicate_frame.loc[duplicate_frame["group_size"] > 1]
    resolution = (inventory.groupby(["width", "height"], dropna=False).size()
                  .rename("image_count").reset_index().sort_values("image_count", ascending=False))
    gate = detection_suitability(annotation_audit)
    suitability = gate["suitability_status"]
    training_decision = gate["training_pool_decision"]
    summary = {
        "dataset": config["dataset"],
        "official_archive_integrity": archive_rows,
        "acquisition_method": (
            "Three official per-file Mendeley public download endpoints; each archive checked "
            "against the repository-published byte size and SHA256."
        ),
        "extracted_raw_tree_sha256": raw_tree_sha256(extracted),
        "counts": {"images": len(inventory), "readable_images": int(inventory["image_readable"].sum()),
                   "unreadable_images": int((~inventory["image_readable"]).sum()),
                   "object_bbox_annotations": 0, "annotation_issues": len(issues)},
        "annotation_audit": annotation_audit,
        "source_category_image_counts": inventory["source_category"].value_counts().sort_index().to_dict(),
        "time_of_day_image_counts": inventory["time_of_day"].value_counts().sort_index().to_dict(),
        "resolution_distribution": resolution.to_dict("records"),
        "coverage": {
            "person_bbox_count": "UNAVAILABLE_NO_OBJECT_ANNOTATIONS",
            "motorcycle_bbox_count": "UNAVAILABLE_NO_OBJECT_ANNOTATIONS",
            "bicycle_bbox_count": "UNAVAILABLE_NO_OBJECT_ANNOTATIONS",
            "car_bus_truck_bbox_counts": "UNAVAILABLE_NO_OBJECT_ANNOTATIONS",
            "small_object_statistics": "NOT_AVAILABLE_WITHOUT_BBOX_ANNOTATIONS",
            "occlusion_metadata": "NOT_AVAILABLE",
            "image_level_folder_counts_only": class_distribution.to_dict("records"),
        },
        "duplicate_governance": {
            "exact_duplicate_images": int(inventory["sha256"].duplicated(keep=False).sum()),
            "exact_duplicate_groups": int(inventory.loc[inventory["sha256"].duplicated(keep=False), "sha256"].nunique()),
            "multi_member_duplicate_groups": int(multi["duplicate_group_id"].nunique()),
            "multi_member_images": len(multi), "phash": "dHash-64",
            "near_hamming_threshold": config["qa"]["near_duplicate_hamming_threshold"],
        },
        "cross_dataset_overlap": overlaps,
        "sequence_governance": {
            "official_sequence_or_capture_id_available": False,
            "available_context": "category and Morning/Evening/Night folder only",
            "future_split_rule": "Keep exact/near-duplicate components intact; folder groups are not asserted as sequences.",
        },
        "suitability_status": suitability,
        "training_pool_decision": training_decision,
        "acceptance": {
            "decision": gate["acceptance_decision"],
            "reason": "The official raw tree has image-level folders but no verified object-level bbox annotations.",
            "license": "CC BY 4.0",
            "license_status": "VERIFIED",
        },
        "governance": {
            "training_performed": False, "final_v2_holdout_created": False,
            "v1_or_stage18_artifacts_modified": False,
            "stage18_locked_test_usage": "HASH_AND_DHASH_OVERLAP_CHECK_ONLY",
        },
    }

    inventory.to_csv(OUTPUT / "dataset_inventory.csv", index=False)
    class_distribution.to_csv(OUTPUT / "class_distribution.csv", index=False)
    mapping.to_csv(OUTPUT / "application_mapping.csv", index=False)
    issues.to_csv(OUTPUT / "annotation_issues.csv", index=False)
    duplicate_frame.to_csv(OUTPUT / "duplicate_groups.csv", index=False)
    (OUTPUT / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_structure_previews(dict(list(preview_candidates.items())[:12]), OUTPUT / "previews")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
