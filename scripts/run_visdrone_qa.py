#!/usr/bin/env python3
"""Run V2-2 VisDrone2019-DET acquisition integrity and descriptive QA."""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.utils.dataset_qa import image_dhash  # noqa: E402
from vision_analytics.utils.visdrone_qa import (  # noqa: E402
    SOURCE_CLASSES,
    cross_dataset_overlap,
    map_visdrone_class,
    near_duplicate_groups,
    parse_visdrone_annotation,
    raw_tree_sha256,
    sequence_group_id,
    sha256_file,
    size_bin,
)

CONFIG_PATH = PROJECT_ROOT / "configs/v2_visdrone_qa.yaml"
OUTPUT = PROJECT_ROOT / "outputs/data_qa/v2_visdrone"


def partition_roots(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    extracted = raw_root / "extracted"
    roots = {
        "train": extracted / "VisDrone2019-DET-train",
        "val": extracted / "VisDrone2019-DET-val",
        "test-dev": extracted,
    }
    result = {}
    for name, root in roots.items():
        image_dir, annotation_dir = root / "images", root / "annotations"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(f"missing extracted {name} images/annotations under {root}")
        result[name] = (image_dir, annotation_dir)
    return result


def archive_integrity(raw_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((raw_root / "archives").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
        rows.append({
            "filename": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "zip_integrity": "PASS" if bad_member is None else "FAIL",
            "bad_member": bad_member or "",
        })
    if len(rows) != 3 or any(row["zip_integrity"] != "PASS" for row in rows):
        raise ValueError("three valid annotated VisDrone archives are required")
    return rows


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_root = PROJECT_ROOT / config["dataset"]["raw_root"]
    archives = archive_integrity(raw_root)
    roots = partition_roots(raw_root)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "previews").mkdir(exist_ok=True)

    inventory_rows: list[dict[str, object]] = []
    parsed_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    preview_candidates: dict[str, tuple[Path, list]] = {}

    for partition, (image_dir, annotation_dir) in roots.items():
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        labels = {path.stem: path for path in annotation_dir.glob("*.txt")}
        for stem in sorted(images.keys() - labels.keys()):
            issue_rows.append(_issue(partition, stem, "IMAGE_WITHOUT_ANNOTATION", "ERROR", None,
                                     "Image has no matching raw annotation file."))
        for stem in sorted(labels.keys() - images.keys()):
            issue_rows.append(_issue(partition, stem, "ANNOTATION_WITHOUT_IMAGE", "ERROR", None,
                                     "Annotation has no matching raw image file."))
        for stem in sorted(images.keys() | labels.keys()):
            image_path, label_path = images.get(stem), labels.get(stem)
            image_id = f"{partition}/{stem}"
            if image_path is None or label_path is None:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                issue_rows.append(_issue(partition, stem, "UNREADABLE_IMAGE", "ERROR", None,
                                         "OpenCV could not decode image."))
                inventory_rows.append({
                    "image_id": image_id, "official_partition": partition,
                    "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "annotation_path": label_path.relative_to(PROJECT_ROOT).as_posix(),
                    "image_readable": False, "width": 0, "height": 0,
                    "raw_annotation_rows": 0, "valid_annotation_rows": 0,
                    "ignored_region_rows": 0, "target_annotation_rows": 0,
                    "sha256": sha256_file(image_path), "phash": "",
                    "sequence_group_id": sequence_group_id(partition, stem),
                    "sequence_group_method": "INFERRED_NOT_OFFICIAL",
                })
                continue
            height, width = image.shape[:2]
            raw_text = label_path.read_text(encoding="utf-8-sig")
            annotations, issues = parse_visdrone_annotation(raw_text, image_width=width, image_height=height)
            for issue in issues:
                issue_rows.append(_issue(partition, stem, issue.issue_type, issue.severity,
                                         issue.line_number, issue.message))
            raw_rows = sum(bool(line.strip()) for line in raw_text.splitlines())
            ignored_count = sum(item.class_id == 0 or item.score == 0 for item in annotations)
            target_count = 0
            for line_number, item in enumerate(annotations, 1):
                application_class, disposition = map_visdrone_class(item.source_class)
                normalized = item.normalized(width, height)
                bin_name = size_bin(normalized["area_normalized"])
                is_ignored = item.class_id == 0 or item.score == 0
                if disposition == "MAPPED" and not is_ignored:
                    target_count += 1
                    preview_candidates.setdefault(application_class, (image_path, annotations))
                parsed_rows.append({
                    "image_id": image_id, "official_partition": partition,
                    "source_annotation_line": line_number, "source_class_id": item.class_id,
                    "source_class": item.source_class, "application_class": application_class or "",
                    "mapping_disposition": disposition, "evaluation_score": item.score,
                    "is_ignored": is_ignored, "bbox_left": item.left, "bbox_top": item.top,
                    "bbox_width": item.width, "bbox_height": item.height,
                    **normalized, "size_bin": bin_name, "truncation": item.truncation,
                    "occlusion": item.occlusion,
                })
            inventory_rows.append({
                "image_id": image_id, "official_partition": partition,
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "annotation_path": label_path.relative_to(PROJECT_ROOT).as_posix(),
                "image_readable": True, "width": width, "height": height,
                "raw_annotation_rows": raw_rows, "valid_annotation_rows": len(annotations),
                "ignored_region_rows": ignored_count, "target_annotation_rows": target_count,
                "sha256": sha256_file(image_path), "phash": image_dhash(image),
                "sequence_group_id": sequence_group_id(partition, stem),
                "sequence_group_method": "INFERRED_NOT_OFFICIAL",
            })

    inventory = pd.DataFrame(inventory_rows).sort_values("image_id")
    parsed = pd.DataFrame(parsed_rows)
    issues = pd.DataFrame(issue_rows, columns=(
        "official_partition", "image_stem", "issue_type", "severity", "line_number", "message"
    ))
    duplicate_rows = pd.DataFrame(near_duplicate_groups(
        inventory.loc[inventory["image_readable"]].to_dict("records"),
        threshold=int(config["qa"]["near_duplicate_hamming_threshold"]),
    ))
    inventory = inventory.merge(duplicate_rows, on="image_id", how="left")

    class_distribution = _class_distribution(parsed)
    size_distribution = _size_distribution(parsed)
    occlusion_distribution = _occlusion_distribution(parsed)
    sequence_groups = inventory.groupby(
        ["official_partition", "sequence_group_id", "sequence_group_method"], as_index=False
    ).agg(image_count=("image_id", "size"), first_image_id=("image_id", "min"),
          last_image_id=("image_id", "max"))
    mapping = pd.DataFrame([
        {"source_class_id": class_id, "source_class": source_class,
         "application_class": map_visdrone_class(source_class)[0] or "",
         "disposition": map_visdrone_class(source_class)[1], "raw_class_id_preserved": True}
        for class_id, source_class in SOURCE_CLASSES.items()
    ])

    v1_inventory = pd.read_csv(PROJECT_ROOT / "outputs/data_qa/stage16/dataset_inventory.csv")
    v1_split = pd.read_csv(PROJECT_ROOT / "outputs/data_qa/stage16/split_manifest.csv")
    locked_ids = set(v1_split.loc[v1_split["split"] == "LOCKED_TEST", "image_id"].astype(str))
    reference = v1_inventory.loc[v1_inventory["image_id"].astype(str).isin(locked_ids)].to_dict("records")
    overlap = cross_dataset_overlap(
        inventory.loc[inventory["image_readable"]].to_dict("records"), reference,
        threshold=int(config["qa"]["near_duplicate_hamming_threshold"]),
    )

    _write_previews(preview_candidates, OUTPUT / "previews")
    acceptance = _acceptance(config, inventory, issues, overlap)
    summary = _coverage_summary(
        config, archives, inventory, parsed, issues, duplicate_rows, sequence_groups,
        class_distribution, size_distribution, occlusion_distribution, overlap, acceptance, raw_root,
    )

    inventory.to_csv(OUTPUT / "dataset_inventory.csv", index=False)
    parsed.to_csv(OUTPUT / "parsed_annotations.csv", index=False)
    class_distribution.to_csv(OUTPUT / "class_distribution.csv", index=False)
    mapping.to_csv(OUTPUT / "application_mapping.csv", index=False)
    issues.to_csv(OUTPUT / "annotation_issues.csv", index=False)
    size_distribution.to_csv(OUTPUT / "object_size_distribution.csv", index=False)
    occlusion_distribution.to_csv(OUTPUT / "occlusion_distribution.csv", index=False)
    duplicate_rows.to_csv(OUTPUT / "duplicate_groups.csv", index=False)
    sequence_groups.to_csv(OUTPUT / "sequence_groups.csv", index=False)
    (OUTPUT / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _issue(partition, stem, kind, severity, line, message):
    return {"official_partition": partition, "image_stem": stem, "issue_type": kind,
            "severity": severity, "line_number": line, "message": message}


def _class_distribution(parsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (class_id, source_class, app, disposition), group in parsed.groupby(
        ["source_class_id", "source_class", "application_class", "mapping_disposition"], dropna=False
    ):
        active = group.loc[~group["is_ignored"]]
        rows.append({"source_class_id": class_id, "source_class": source_class,
                     "application_class": app, "mapping_disposition": disposition,
                     "image_count": active["image_id"].nunique(), "bbox_count": len(active),
                     "ignored_bbox_count": int(group["is_ignored"].sum())})
    return pd.DataFrame(rows).sort_values("source_class_id")


def _size_distribution(parsed: pd.DataFrame) -> pd.DataFrame:
    active = parsed.loc[(parsed["mapping_disposition"] == "MAPPED") & ~parsed["is_ignored"]]
    grouped = active.groupby(["application_class", "size_bin"]).size().rename("bbox_count").reset_index()
    totals = grouped.groupby("application_class")["bbox_count"].transform("sum")
    grouped["class_percentage"] = 100 * grouped["bbox_count"] / totals
    return grouped.sort_values(["application_class", "size_bin"])


def _occlusion_distribution(parsed: pd.DataFrame) -> pd.DataFrame:
    active = parsed.loc[(parsed["mapping_disposition"] == "MAPPED") & ~parsed["is_ignored"]].copy()
    active["occlusion_label"] = active["occlusion"].map({0: "NONE", 1: "PARTIAL", 2: "HEAVY"})
    active["truncation_label"] = active["truncation"].map({0: "NONE", 1: "PARTIAL"})
    return active.groupby(
        ["application_class", "size_bin", "occlusion", "occlusion_label",
         "truncation", "truncation_label"], as_index=False
    ).size().rename(columns={"size": "bbox_count"})


def _acceptance(config, inventory, issues, overlap):
    critical = {"UNREADABLE_IMAGE", "IMAGE_WITHOUT_ANNOTATION", "ANNOTATION_WITHOUT_IMAGE",
                "INVALID_CLASS_ID", "MALFORMED_ROW"}
    critical_count = int(issues["issue_type"].isin(critical).sum()) if len(issues) else 0
    if critical_count or overlap["exact_overlap_count"] or overlap["near_overlap_count"]:
        decision = "REJECT"
    else:
        decision = str(config["governance"]["acceptance_if_qa_passes"])
    return {"decision": decision, "license_status": config["dataset"]["license_status"],
            "training_pool_status": config["dataset"]["intended_use_status"],
            "critical_qa_issue_count": critical_count,
            "reason": "Coverage candidate is acceptable for governed research QA, but training eligibility is blocked pending explicit license/usage review."}


def _coverage_summary(config, archives, inventory, parsed, issues, duplicates, sequences,
                      classes, sizes, occlusion, overlap, acceptance, raw_root):
    active = parsed.loc[(parsed["mapping_disposition"] == "MAPPED") & ~parsed["is_ignored"]]
    person_source = {}
    for source in ("pedestrian", "people"):
        subset = parsed.loc[(parsed["source_class"] == source) & ~parsed["is_ignored"]]
        person_source[source] = {"image_count": subset["image_id"].nunique(), "bbox_count": len(subset)}
    person = active.loc[active["application_class"] == "person"]
    small_occ = active.loc[(active["size_bin"] == "SMALL_LT_0.01") & (active["occlusion"] > 0)]
    return {
        "dataset": config["dataset"], "archive_integrity": archives,
        "extracted_raw_tree_sha256": raw_tree_sha256(raw_root / "extracted"),
        "counts": {"images": len(inventory), "readable_images": int(inventory["image_readable"].sum()),
                   "parsed_valid_rows": len(parsed), "target_rows": len(active),
                   "ignored_rows": int(parsed["is_ignored"].sum()), "qa_issues": len(issues)},
        "person_coverage": {**person_source, "combined_application_person": {
            "image_count": person["image_id"].nunique(), "bbox_count": len(person)}},
        "small_object_coverage": {
            "target_small_bbox_count": int((active["size_bin"] == "SMALL_LT_0.01").sum()),
            "target_small_percentage": float(100 * (active["size_bin"] == "SMALL_LT_0.01").mean()),
            "small_and_occluded_bbox_count": len(small_occ),
        },
        "duplicate_governance": {
            "exact_duplicate_image_count": int(inventory["sha256"].duplicated(keep=False).sum()),
            "exact_duplicate_groups": int(inventory.loc[inventory["sha256"].duplicated(keep=False), "sha256"].nunique()),
            "near_or_exact_multi_member_images": int((duplicates["group_size"] > 1).sum()),
            "multi_member_groups": int(duplicates.loc[duplicates["group_size"] > 1, "duplicate_group_id"].nunique()),
            "near_only_multi_member_images": int(((duplicates["group_size"] > 1) & (duplicates["exact_duplicate_group_size"] == 1)).sum()),
            "cross_official_partition_groups": int(
                inventory.loc[inventory["group_size"] > 1].groupby("duplicate_group_id")["official_partition"]
                .nunique().gt(1).sum()
            ),
            "phash": "dHash-64", "near_duplicate_hamming_threshold": config["qa"]["near_duplicate_hamming_threshold"],
        },
        "sequence_governance": {"group_count": len(sequences), "method": "INFERRED_NOT_OFFICIAL",
                                "official_sequence_metadata_available": False},
        "v1_stage18_locked_test_overlap": {**overlap, "stage18_usage": "OVERLAP_CHECK_ONLY"},
        "class_distribution": classes.to_dict("records"),
        "object_size_distribution": sizes.to_dict("records"),
        "occlusion_distribution_rows": len(occlusion),
        "acceptance": acceptance,
        "limitations": [
            "No explicit dataset license/usage grant was located; legal/owner review is required before training use.",
            "Filename-prefix grouping is a conservative project heuristic, not official sequence metadata.",
            "Coverage statistics do not prove that adding VisDrone will improve a future model.",
            "Stage 18 LOCKED_TEST was used only as a hash/perceptual-overlap reference, never for model selection.",
        ],
    }


def _write_previews(candidates: dict, destination: Path) -> None:
    colors = {"person": (0, 255, 255), "bicycle": (255, 255, 0),
              "car": (0, 255, 0), "motorcycle": (255, 0, 255),
              "bus": (255, 128, 0), "truck": (0, 128, 255)}
    for app_class, (path, annotations) in candidates.items():
        image = cv2.imread(str(path))
        for item in annotations[:200]:
            mapped, disposition = map_visdrone_class(item.source_class)
            if disposition != "MAPPED":
                continue
            color = colors[mapped]
            p1 = (round(item.left), round(item.top))
            p2 = (round(item.left + item.width), round(item.top + item.height))
            cv2.rectangle(image, p1, p2, color, 1)
            cv2.putText(image, f"{item.source_class}->{mapped} o{item.occlusion}", p1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(destination / f"{app_class}_preview.jpg"), image)


if __name__ == "__main__":
    raise SystemExit(main())
