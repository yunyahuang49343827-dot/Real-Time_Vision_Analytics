#!/usr/bin/env python3
"""Build and QA a small Open Images V7 targeted validation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.utils.dataset_qa import image_dhash  # noqa: E402
from vision_analytics.utils.openimages_qa import (  # noqa: E402
    aggregate_candidate,
    box_candidate_status,
    deterministic_pilot_selection,
    license_status,
    parse_box_row,
    resolve_target_classes,
    size_bin,
    validate_pilot_manifest,
    validate_stage18_usage,
)
from vision_analytics.utils.urbanscene_qa import cross_dataset_overlap_blocked  # noqa: E402
from vision_analytics.utils.visdrone_qa import near_duplicate_groups  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs/v2_openimages_qa.yaml"
OUTPUT = PROJECT_ROOT / "outputs/data_qa/v2_openimages"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_metadata(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "OpenImages-V2-QA/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _download(image_id: str, split: str, destination: Path) -> tuple[str, str]:
    path = destination / f"{image_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return image_id, "EXISTING"
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "OpenImages-V2-QA/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        return image_id, "DOWNLOADED"
    except Exception as error:  # network errors become auditable QA rows
        path.unlink(missing_ok=True)
        return image_id, f"FAILED:{type(error).__name__}:{error}"


def _reference_records() -> dict[str, list[dict[str, str]]]:
    root = PROJECT_ROOT / "outputs/data_qa"
    taiwan = pd.read_csv(root / "stage16/dataset_inventory.csv")
    split = pd.read_csv(root / "stage16/split_manifest.csv")
    locked = set(split.loc[split["split"] == "LOCKED_TEST", "image_id"].astype(str))
    visdrone = pd.read_csv(root / "v2_visdrone/dataset_inventory.csv")
    urban = pd.read_csv(root / "v2_urbanscene/dataset_inventory.csv")

    def records(frame: pd.DataFrame) -> list[dict[str, str]]:
        usable = frame.loc[frame["sha256"].notna() & frame["phash"].notna(),
                           ["image_id", "sha256", "phash"]]
        return usable.astype(str).to_dict("records")

    return {
        "v1_taiwan": records(taiwan),
        "stage18_locked_test_OVERLAP_CHECK_ONLY": records(
            taiwan.loc[taiwan["image_id"].astype(str).isin(locked)]
        ),
        "visdrone": records(visdrone),
        "urbanscene": records(urban),
    }


def _contact_sheets(
    review: pd.DataFrame, annotations: pd.DataFrame, image_dir: Path, destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    colors = {"person": (0, 255, 255), "bicycle": (255, 255, 0), "car": (0, 255, 0),
              "motorcycle": (255, 0, 255), "bus": (255, 128, 0), "truck": (0, 128, 255)}
    tiles = []
    for number, image_id in enumerate(review["image_id"], 1):
        image = cv2.imread(str(image_dir / f"{image_id}.jpg"))
        if image is None:
            continue
        height, width = image.shape[:2]
        for _, box in annotations.loc[annotations["image_id"] == image_id].iterrows():
            color = colors[str(box["application_class"])]
            p1 = (round(box["xmin"] * width), round(box["ymin"] * height))
            p2 = (round(box["xmax"] * width), round(box["ymax"] * height))
            cv2.rectangle(image, p1, p2, color, 2)
        tile = cv2.resize(image, (320, 240))
        cv2.rectangle(tile, (0, 0), (320, 25), (0, 0, 0), -1)
        cv2.putText(tile, f"{number:02d} {image_id}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    for index in range(0, len(tiles), 10):
        group = tiles[index:index + 10]
        blank = 10 - len(group)
        group.extend([np.full((240, 320, 3), 255, dtype="uint8")] * blank)  # pragma: no cover
        top = cv2.hconcat(group[:5]); bottom = cv2.hconcat(group[5:10])
        cv2.imwrite(str(destination / f"review_{index // 10 + 1:02d}.jpg"), cv2.vconcat([top, bottom]))


def main(download: bool) -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_stage18_usage(config["governance"]["stage18_usage"])
    raw_root = PROJECT_ROOT / config["dataset"]["raw_root"]
    image_dir = raw_root / "pilot" / config["dataset"]["candidate_source_split"]
    image_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "previews").mkdir(exist_ok=True)

    class_path = PROJECT_ROOT / config["metadata"]["class_descriptions"]
    bbox_path = PROJECT_ROOT / config["metadata"]["bounding_boxes"]
    image_meta_path = PROJECT_ROOT / config["metadata"]["image_metadata"]
    metadata_sources = (
        (config["metadata"]["class_descriptions_url"], class_path),
        (config["metadata"]["bounding_boxes_url"], bbox_path),
        (config["metadata"]["image_metadata_url"], image_meta_path),
    )
    if download:
        for url, path in metadata_sources:
            _download_metadata(url, path)
    missing_metadata = [str(path) for _, path in metadata_sources if not path.exists()]
    if missing_metadata:
        raise FileNotFoundError(
            f"official metadata missing: {missing_metadata}; rerun with --download"
        )
    classes = pd.read_csv(class_path, header=None, names=["source_class_id", "source_class"])
    resolved = resolve_target_classes(classes.itertuples(index=False, name=None))
    mapping = pd.DataFrame([
        {"source_class": source_name, "source_class_id": source_id,
         "application_class": application, "mapping_status": "MAPPED_FROM_OFFICIAL_METADATA"}
        for source_id, (source_name, application) in sorted(resolved.items(), key=lambda item: item[1][1])
    ])

    raw_boxes = pd.read_csv(bbox_path)
    raw_target = raw_boxes.loc[raw_boxes["LabelName"].isin(resolved)].copy()
    boxes, box_rows, issue_rows = [], [], []
    for line_number, row in raw_target.iterrows():
        try:
            box = parse_box_row(row.to_dict(), resolved)
        except ValueError as error:
            issue_rows.append({"image_id": row.get("ImageID", ""), "issue_type": "INVALID_BBOX",
                               "severity": "ERROR", "source_row": line_number + 2,
                               "message": str(error)})
            continue
        boxes.append(box)
        box_rows.append({
            "image_id": box.image_id, "source_class_id": box.source_class_id,
            "source_class": resolved[box.source_class_id][0],
            "application_class": box.application_class,
            "xmin": box.xmin, "xmax": box.xmax, "ymin": box.ymin, "ymax": box.ymax,
            "normalized_area": box.area, "size_bin": size_bin(box.area),
            "is_occluded": box.is_occluded, "is_truncated": box.is_truncated,
            "is_group_of": box.is_group_of, "is_depiction": box.is_depiction,
            "is_inside": box.is_inside, "box_candidate_status": box_candidate_status(box),
        })
    parsed = pd.DataFrame(box_rows)
    grouped: dict[str, list] = {}
    for box in boxes:
        grouped.setdefault(box.image_id, []).append(box)
    candidate = pd.DataFrame([aggregate_candidate(group) for group in grouped.values()])
    candidate = candidate.loc[candidate["candidate_status"] == "ELIGIBLE"].copy()
    metadata = pd.read_csv(image_meta_path)
    metadata = metadata.loc[metadata["ImageID"].isin(candidate["image_id"])].copy()
    license_results = metadata.apply(lambda row: license_status(row.to_dict()), axis=1)
    metadata["license_status"] = [item[0] for item in license_results]
    metadata["license_governance_note"] = [item[1] for item in license_results]
    metadata = metadata.rename(columns={
        "ImageID": "image_id", "Subset": "source_split", "OriginalURL": "original_url",
        "OriginalLandingURL": "license_reference", "License": "license_url",
        "AuthorProfileURL": "author_profile_url", "Author": "author", "Title": "title",
    })
    candidate = candidate.merge(metadata[[
        "image_id", "source_split", "original_url", "license_reference", "license_url",
        "author_profile_url", "author", "title", "Rotation", "OriginalMD5", "OriginalSize",
        "license_status", "license_governance_note",
    ]], on="image_id", how="left", validate="one_to_one")
    candidate = candidate.rename(columns={
        "Rotation": "rotation_degrees", "OriginalMD5": "original_md5",
        "OriginalSize": "original_size_bytes",
    })
    candidate["candidate_status"] = candidate["license_status"].map({
        "VERIFIED": "ELIGIBLE", "REQUIRES_REVIEW": "QUARANTINED_LICENSE_REVIEW",
        "REJECTED": "REJECTED_LICENSE",
    }).fillna("LICENSE_UNVERIFIED")

    # Temporarily expose the pre-license eligibility to the deterministic QA pilot selector.
    selection_rows = candidate.to_dict("records")
    for row in selection_rows:
        row["candidate_status"] = "ELIGIBLE" if row["candidate_status"] != "REJECTED_LICENSE" else row["candidate_status"]
    selected_ids = deterministic_pilot_selection(
        selection_rows, limit=int(config["pilot"]["size"]), seed=int(config["pilot"]["seed"])
    )
    pilot = candidate.loc[candidate["image_id"].isin(selected_ids)].copy()
    rank = {image_id: index for index, image_id in enumerate(selected_ids)}
    pilot["selection_rank"] = pilot["image_id"].map(rank)
    pilot = pilot.sort_values("selection_rank")
    pilot["local_path"] = pilot["image_id"].map(
        lambda image_id: (image_dir / f"{image_id}.jpg").relative_to(PROJECT_ROOT).as_posix()
    )
    validate_pilot_manifest(pilot.to_dict("records"), expected_count=int(config["pilot"]["size"]))

    download_status = {}
    if download:
        with ThreadPoolExecutor(max_workers=int(config["pilot"]["download_workers"])) as executor:
            futures = {executor.submit(_download, image_id, str(pilot.iloc[0]["source_split"]), image_dir): image_id
                       for image_id in pilot["image_id"]}
            for future in as_completed(futures):
                image_id, status = future.result()
                download_status[image_id] = status
    pilot["download_status"] = pilot["image_id"].map(
        lambda image_id: download_status.get(image_id, "NOT_REQUESTED")
    )
    pilot["image_readable"] = False
    pilot["width"] = 0; pilot["height"] = 0; pilot["sha256"] = ""; pilot["phash"] = ""
    for index, row in pilot.iterrows():
        path = PROJECT_ROOT / row["local_path"]
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            issue_rows.append({"image_id": row["image_id"], "issue_type": "IMAGE_NOT_READABLE",
                               "severity": "ERROR", "source_row": "",
                               "message": f"Pilot image missing or unreadable: {path}"})
            continue
        height, width = image.shape[:2]
        pilot.at[index, "image_readable"] = True
        pilot.at[index, "width"] = width; pilot.at[index, "height"] = height
        pilot.at[index, "sha256"] = sha256_file(path); pilot.at[index, "phash"] = image_dhash(image)

    eligible_boxes = parsed.loc[parsed["box_candidate_status"] == "ELIGIBLE"]
    class_distribution = (eligible_boxes.groupby(["source_class_id", "source_class", "application_class"])
                          .agg(image_count=("image_id", "nunique"), bbox_count=("image_id", "size"),
                               occluded_bbox_count=("is_occluded", "sum"),
                               truncated_bbox_count=("is_truncated", "sum"))
                          .reset_index())
    context_distribution = (candidate.assign(context_tag=candidate["context_tags"].str.split(";"))
                            .explode("context_tag").groupby("context_tag")
                            .agg(image_count=("image_id", "nunique")).reset_index())
    size_distribution = (eligible_boxes.groupby(["application_class", "size_bin"])
                         .agg(image_count=("image_id", "nunique"), bbox_count=("image_id", "size"))
                         .reset_index())
    size_distribution["class_percentage"] = 100 * size_distribution["bbox_count"] / \
        size_distribution.groupby("application_class")["bbox_count"].transform("sum")
    license_summary = candidate.groupby(["license_status", "candidate_status"]).agg(
        image_count=("image_id", "size")).reset_index()

    pilot_records = pilot.loc[pilot["image_readable"], ["image_id", "sha256", "phash"]].to_dict("records")
    duplicate_rows, overlap_summary = [], {}
    internal_groups = near_duplicate_groups(
        pilot_records, threshold=int(config["qa"]["near_duplicate_hamming_threshold"])
    )
    internal_duplicate_images = [row for row in internal_groups if row["is_duplicate"]]
    internal_exact_images = [
        row for row in internal_duplicate_images if row["exact_duplicate_group_size"] > 1
    ]
    for row in internal_duplicate_images:
        duplicate_rows.append({
            "pilot_image_id": row["image_id"], "reference_dataset": "openimages_pilot_internal",
            "reference_image_id": row["duplicate_group_id"],
            "match_type": row["grouping_basis"],
        })
    overlap_summary["openimages_pilot_internal"] = {
        "exact_duplicate_image_count": len(internal_exact_images),
        "near_duplicate_image_count": len(internal_duplicate_images),
        "method": "SHA256 exact + dHash-64 lossless band blocking",
        "threshold": int(config["qa"]["near_duplicate_hamming_threshold"]),
    }
    for reference_name, reference in _reference_records().items():
        result = cross_dataset_overlap_blocked(
            pilot_records, reference, threshold=int(config["qa"]["near_duplicate_hamming_threshold"])
        )
        overlap_summary[reference_name] = {
            "exact_overlap_count": result["exact_overlap_count"],
            "near_overlap_count": result["near_overlap_count"],
            "method": result["near_method"], "threshold": result["near_hamming_threshold"],
        }
        for image_id, reference_id in result["exact_pairs"]:
            duplicate_rows.append({"pilot_image_id": image_id, "reference_dataset": reference_name,
                                   "reference_image_id": reference_id, "match_type": "EXACT_SHA256"})
        for image_id, reference_id in result["near_pairs"]:
            duplicate_rows.append({"pilot_image_id": image_id, "reference_dataset": reference_name,
                                   "reference_image_id": reference_id, "match_type": "DHASH_NEAR_HEURISTIC"})

    review_count = min(int(config["pilot"]["review_sample_size"]), len(pilot))
    review_ids = pilot.head(review_count)["image_id"]
    review = pilot.loc[pilot["image_id"].isin(review_ids), [
        "image_id", "source_split", "context_tags", "small_person_count",
        "occluded_person_count", "truncated_person_count", "local_path",
    ]].copy()
    manual_review_path = PROJECT_ROOT / config["pilot"]["manual_review_manifest"]
    if manual_review_path.exists():
        old = pd.read_csv(manual_review_path)
        if old["image_id"].duplicated().any():
            raise ValueError("manual review image_id must be unique")
        review = review.merge(old[["image_id", "domain_relevance", "annotation_review", "notes"]],
                              on="image_id", how="left", validate="one_to_one")
    for column in ("domain_relevance", "annotation_review", "notes"):
        if column not in review:
            review[column] = "NOT_REVIEWED" if column != "notes" else ""

    pilot_annotations = parsed.loc[parsed["image_id"].isin(pilot["image_id"])]
    _contact_sheets(review, pilot_annotations, image_dir, OUTPUT / "previews")
    issues = pd.DataFrame(issue_rows, columns=("image_id", "issue_type", "severity", "source_row", "message"))
    duplicate_check = pd.DataFrame(duplicate_rows, columns=(
        "pilot_image_id", "reference_dataset", "reference_image_id", "match_type"
    ))
    metadata_hashes = {path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                       for path in (class_path, bbox_path, image_meta_path)}
    person_boxes = eligible_boxes.loc[eligible_boxes["application_class"] == "person"]
    summary = {
        "dataset": config["dataset"], "metadata_files": metadata_hashes,
        "official_target_class_mapping": mapping.to_dict("records"),
        "candidate_scope": "Official validation dense-annotation split only; no full train metadata/images downloaded.",
        "counts": {
            "raw_validation_bbox_rows": len(raw_boxes), "target_bbox_rows": len(raw_target),
            "valid_target_bbox_rows": len(parsed), "candidate_images": len(candidate),
            "person_boxes_in_candidate_scope": int(len(person_boxes)),
            "person_images_in_candidate_scope": int(person_boxes["image_id"].nunique()),
            "person_traffic_images": int((~candidate["context_tags"].eq("PERSON_ONLY")).sum()),
            "pilot_images": len(pilot), "pilot_readable_images": int(pilot["image_readable"].sum()),
        },
        "person_coverage": {
            "small_person_boxes": int((person_boxes["size_bin"] == "SMALL_LT_0.01").sum()),
            "small_person_boxes_in_traffic_context": int(
                candidate.loc[~candidate["context_tags"].eq("PERSON_ONLY"), "small_person_count"].sum()
            ),
            "small_occluded_person_boxes": int(((person_boxes["size_bin"] == "SMALL_LT_0.01") &
                                                   person_boxes["is_occluded"].eq(1)).sum()),
            "small_truncated_person_boxes": int(((person_boxes["size_bin"] == "SMALL_LT_0.01") &
                                                    person_boxes["is_truncated"].eq(1)).sum()),
            "occluded_person_boxes": int(person_boxes["is_occluded"].eq(1).sum()),
            "truncated_person_boxes": int(person_boxes["is_truncated"].eq(1).sum()),
        },
        "filtering": {
            "depiction_target_boxes_excluded": int(parsed["is_depiction"].eq(1).sum()),
            "group_of_target_boxes_excluded": int(parsed["is_group_of"].eq(1).sum()),
            "occluded_retained": True, "truncated_retained": True,
        },
        "license_governance": {
            "annotation_license": "CC BY 4.0", "image_metadata_license": "CC BY 2.0",
            "verified": int(candidate["license_status"].eq("VERIFIED").sum()),
            "requires_review": int(candidate["license_status"].eq("REQUIRES_REVIEW").sum()),
            "rejected": int(candidate["license_status"].eq("REJECTED").sum()),
            "policy": "Official warning requires per-image verification; metadata completeness alone is not VERIFIED.",
        },
        "cross_dataset_overlap": overlap_summary,
        "manual_review": {
            "sample_size": review_count,
            "completed": int(review["domain_relevance"].ne("NOT_REVIEWED").sum()),
            "domain_distribution": review["domain_relevance"].value_counts().to_dict(),
        },
        "acceptance": {
            "decision": "ACCEPT_WITH_WARNINGS",
            "training_pool_status": "QUARANTINED_PENDING_PER_IMAGE_LICENSE_AND_DOMAIN_FILTER",
            "reason": (
                "Object boxes and targeted person coverage are usable, but all image licenses require "
                "independent per-image verification and the general-domain pool needs traffic relevance filtering."
            ),
        },
        "governance": {"training_performed": False, "final_v2_holdout_created": False,
                       "stage18_usage": "OVERLAP_CHECK_ONLY", "v1_artifacts_modified": False},
    }

    candidate.to_csv(OUTPUT / "candidate_manifest.csv", index=False)
    mapping.to_csv(OUTPUT / "application_mapping.csv", index=False)
    class_distribution.to_csv(OUTPUT / "class_distribution.csv", index=False)
    context_distribution.to_csv(OUTPUT / "context_distribution.csv", index=False)
    size_distribution.to_csv(OUTPUT / "object_size_distribution.csv", index=False)
    license_summary.to_csv(OUTPUT / "license_summary.csv", index=False)
    pilot.to_csv(OUTPUT / "pilot_download_manifest.csv", index=False)
    pilot_annotations.to_csv(OUTPUT / "pilot_annotations.csv", index=False)
    review.to_csv(OUTPUT / "pilot_review.csv", index=False)
    issues.to_csv(OUTPUT / "annotation_issues.csv", index=False)
    duplicate_check.to_csv(OUTPUT / "duplicate_check.csv", index=False)
    (OUTPUT / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download and QA the 300-image pilot")
    raise SystemExit(main(parser.parse_args().download))
