#!/usr/bin/env python3
"""Prepare, review, and build the governed Taiwan + Open Images V2 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.utils.dataset_qa import (  # noqa: E402
    build_duplicate_groups,
    image_dhash,
    parse_yolo_annotation,
    sha256_file,
)
from vision_analytics.utils.openimages_qa import parse_box_row, resolve_target_classes, size_bin  # noqa: E402
from vision_analytics.utils.urbanscene_qa import cross_dataset_overlap_blocked  # noqa: E402
from vision_analytics.utils.v2_training_pool import (  # noqa: E402
    annotation_sha256,
    auditable_annotation_review,
    auditable_domain_review,
    deterministic_source_group_split,
    license_review_status,
    map_application_class,
    openimages_training_gate,
    parse_flickr_license_evidence,
    select_review_candidates,
    validate_training_manifest,
)

CONFIG_PATH = PROJECT_ROOT / "configs/v2_training_pool.yaml"
OUTPUT = PROJECT_ROOT / "outputs/data_qa/v2_training_pool"
PREVIEWS = OUTPUT / "review_previews"
TAIWAN_CLASSES = ["bicycle", "bus", "car", "human", "motorbike", "truck"]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _fetch_license(row: dict[str, object]) -> dict[str, object]:
    url = str(row["license_reference"])
    status = 0
    page = b""
    error = ""
    try:
        request = urllib.request.Request(url, headers={
            "User-Agent": "V2-Training-Pool-License-Review/1.0",
            "Accept-Encoding": "identity",
        })
        with urllib.request.urlopen(request, timeout=45) as response:
            status = int(response.status)
            page = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code); error = f"HTTPError:{exc}"
    except Exception as exc:  # auditable unresolved row
        error = f"{type(exc).__name__}:{exc}"
    evidence = parse_flickr_license_evidence(page.decode("utf-8", errors="replace"))
    review_status, note = license_review_status(row, evidence, http_status=status)
    return {
        "image_id": str(row["image_id"]), "license_review_status": review_status,
        "license_review_notes": note if not error else f"{note}; {error}",
        "license_reviewed_at": datetime.now(timezone.utc).isoformat(),
        "landing_http_status": status, "landing_page_sha256": hashlib.sha256(page).hexdigest() if page else "",
        **evidence,
    }


def _download_image(image_id: str, destination: Path, existing: Path | None) -> tuple[str, str, str]:
    if existing is not None and existing.exists():
        return image_id, existing.relative_to(PROJECT_ROOT).as_posix(), "EXISTING_V2_2C_PILOT"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{image_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return image_id, path.relative_to(PROJECT_ROOT).as_posix(), "EXISTING_REVIEW_DOWNLOAD"
    url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "V2-Training-Pool-QA/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        return image_id, path.relative_to(PROJECT_ROOT).as_posix(), "DOWNLOADED"
    except Exception as exc:
        path.unlink(missing_ok=True)
        return image_id, "", f"FAILED:{type(exc).__name__}:{exc}"


def _openimages_boxes(image_ids: set[str]) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]]]:
    class_path = PROJECT_ROOT / "data/raw/openimages_v7/metadata/oidv7-class-descriptions-boxable.csv"
    bbox_path = PROJECT_ROOT / "data/raw/openimages_v7/metadata/validation-annotations-bbox.csv"
    classes = pd.read_csv(class_path, header=None, names=["source_class_id", "source_class"])
    resolved = resolve_target_classes(classes.itertuples(index=False, name=None))
    raw = pd.read_csv(bbox_path)
    raw = raw.loc[raw["ImageID"].astype(str).isin(image_ids) & raw["LabelName"].isin(resolved)].copy()
    records: list[dict[str, object]] = []
    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source_row, row in raw.iterrows():
        box = parse_box_row(row.to_dict(), resolved)
        if box.is_depiction == 1 or box.is_group_of == 1:
            continue
        source_class = resolved[box.source_class_id][0]
        record = {
            "dataset_source": "openimages_v7", "source_image_id": box.image_id,
            "source_class": source_class, "application_class": box.application_class,
            "xmin": box.xmin, "ymin": box.ymin, "xmax": box.xmax, "ymax": box.ymax,
            "normalized_area": box.area, "size_bin": size_bin(box.area),
            "is_occluded": box.is_occluded, "is_truncated": box.is_truncated,
            "is_group_of": box.is_group_of, "is_depiction": box.is_depiction,
            "is_inside": box.is_inside, "source_row": int(source_row) + 2,
            "source_annotation": json.dumps(row.to_dict(), sort_keys=True, default=str),
            "derived_annotation": json.dumps({
                "application_class": box.application_class,
                "x_center": (box.xmin + box.xmax) / 2, "y_center": (box.ymin + box.ymax) / 2,
                "width": box.xmax - box.xmin, "height": box.ymax - box.ymin,
            }, sort_keys=True),
        }
        records.append(record); by_image[box.image_id].append(record)
    return pd.DataFrame(records), by_image


def _draw_contact_sheets(
    worklist: pd.DataFrame, annotations: pd.DataFrame, destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("review_*.jpg"):
        stale.unlink()
    colors = {"person": (0, 255, 255), "bicycle": (255, 255, 0), "car": (0, 255, 0),
              "motorcycle": (255, 0, 255), "bus": (255, 128, 0), "truck": (0, 128, 255)}
    tiles: list[np.ndarray] = []
    for number, row in enumerate(worklist.itertuples(index=False), 1):
        image = cv2.imread(str(PROJECT_ROOT / row.local_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for box in annotations.loc[annotations["source_image_id"].astype(str) == str(row.image_id)].itertuples():
            cv2.rectangle(image, (round(box.xmin * width), round(box.ymin * height)),
                          (round(box.xmax * width), round(box.ymax * height)),
                          colors[str(box.application_class)], 2)
        tile = cv2.resize(image, (320, 240))
        cv2.rectangle(tile, (0, 0), (320, 39), (0, 0, 0), -1)
        cv2.putText(tile, f"{number:03d} {row.image_id}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, str(row.context_tags)[:44], (4, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    .32, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    for index in range(0, len(tiles), 10):
        group = tiles[index:index + 10]
        group.extend([np.full((240, 320, 3), 255, dtype="uint8")] * (10 - len(group)))
        sheet = cv2.vconcat([cv2.hconcat(group[:5]), cv2.hconcat(group[5:])])
        cv2.imwrite(str(destination / f"review_{index // 10 + 1:03d}.jpg"), sheet)


def prepare(config: dict[str, object]) -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(PROJECT_ROOT / config["openimages"]["candidate_manifest"], low_memory=False)
    candidates = candidates.loc[candidates["context_tags"].ne("PERSON_ONLY")].copy()
    prior_review = pd.read_csv(PROJECT_ROOT / "data/manifests/openimages_pilot_review.csv")
    forced = prior_review.loc[prior_review["domain_relevance"].isin(
        ["TRAFFIC_RELEVANT", "PARTIALLY_RELEVANT"]), "image_id"].astype(str).tolist()
    selected = select_review_candidates(
        candidates.to_dict("records"), limit=int(config["openimages"]["review_candidate_count"]),
        seed=int(config["seed"]),
    )
    selected = forced + [image_id for image_id in selected if image_id not in set(forced)]
    selected = selected[:int(config["openimages"]["review_candidate_count"])]
    work = candidates.loc[candidates["image_id"].astype(str).isin(selected)].copy()
    order = {image_id: index for index, image_id in enumerate(selected)}
    work["review_rank"] = work["image_id"].astype(str).map(order); work = work.sort_values("review_rank")

    existing_license_path = OUTPUT / "openimages_license_evidence.csv"
    existing_license = pd.read_csv(existing_license_path) if existing_license_path.exists() else pd.DataFrame()
    stable_license = existing_license.loc[
        existing_license["license_review_status"].isin(["LICENSE_APPROVED", "REJECTED"])
    ] if not existing_license.empty else pd.DataFrame()
    existing_ids = set(stable_license["image_id"].astype(str)) if not stable_license.empty else set()
    license_rows = stable_license.loc[
        stable_license["image_id"].astype(str).isin(work["image_id"].astype(str))
    ].to_dict("records") if not stable_license.empty else []
    pending_license = [row for row in work.to_dict("records") if str(row["image_id"]) not in existing_ids]
    with ThreadPoolExecutor(max_workers=int(config["openimages"]["license_workers"])) as executor:
        futures = [executor.submit(_fetch_license, row) for row in pending_license]
        license_rows.extend(future.result() for future in as_completed(futures))
    license_frame = pd.DataFrame(license_rows)
    work = work.merge(license_frame, on="image_id", how="left", validate="one_to_one")

    review_dir = PROJECT_ROOT / config["openimages"]["review_image_dir"]
    pilot_dir = PROJECT_ROOT / "data/raw/openimages_v7/pilot/validation"
    with ThreadPoolExecutor(max_workers=int(config["openimages"]["download_workers"])) as executor:
        futures = [executor.submit(
            _download_image, str(image_id), review_dir,
            pilot_dir / f"{image_id}.jpg" if (pilot_dir / f"{image_id}.jpg").exists() else None,
        ) for image_id in work["image_id"]]
        download_rows = [future.result() for future in as_completed(futures)]
    downloads = pd.DataFrame(download_rows, columns=["image_id", "local_path", "download_status"])
    work = work.merge(downloads, on="image_id", how="left", validate="one_to_one")
    work["image_readable"] = False; work["width"] = 0; work["height"] = 0
    work["image_sha256"] = ""; work["phash"] = ""
    for index, row in work.iterrows():
        path = PROJECT_ROOT / str(row["local_path"]) if str(row["local_path"]) else Path("/missing")
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            continue
        work.at[index, "image_readable"] = True
        work.at[index, "height"], work.at[index, "width"] = image.shape[:2]
        work.at[index, "image_sha256"] = sha256_file(path)
        work.at[index, "phash"] = image_dhash(image)

    annotations, _ = _openimages_boxes(set(work["image_id"].astype(str)))
    counts = annotations.groupby(["source_image_id", "application_class"]).size().unstack(fill_value=0)
    for name in ("person", "motorcycle", "bicycle", "car", "bus", "truck"):
        work[f"{name}_boxes"] = work["image_id"].map(counts[name] if name in counts else {}).fillna(0).astype(int)
    work["small_person_boxes"] = work["image_id"].map(
        annotations.loc[(annotations["application_class"] == "person") &
                        (annotations["size_bin"] == "SMALL_LT_0.01")]
        .groupby("source_image_id").size()).fillna(0).astype(int)
    work["occluded_person_boxes"] = work["image_id"].map(
        annotations.loc[(annotations["application_class"] == "person") &
                        (annotations["is_occluded"] == 1)].groupby("source_image_id").size()
    ).fillna(0).astype(int)
    work["domain_relevance"] = "NOT_REVIEWED"
    work["domain_review_notes"] = ""
    work["annotation_review_status"] = "NOT_REVIEWED"
    work["annotation_review_notes"] = ""
    prior = prior_review.rename(columns={"annotation_review": "prior_annotation_review",
                                        "notes": "prior_review_notes"})
    work = work.merge(prior[["image_id", "domain_relevance", "prior_annotation_review",
                             "prior_review_notes"]], on="image_id", how="left", suffixes=("", "_prior"))
    mask = work["domain_relevance_prior"].notna()
    work.loc[mask, "domain_relevance"] = work.loc[mask, "domain_relevance_prior"]
    work.loc[mask, "domain_review_notes"] = work.loc[mask, "prior_review_notes"]
    work.loc[mask & work["prior_annotation_review"].eq("ACCEPTABLE"),
             "annotation_review_status"] = "ACCEPTABLE"
    work.loc[mask & work["prior_annotation_review"].eq("ISSUE"),
             "annotation_review_status"] = "REJECTED_INCOMPLETE"
    work.loc[mask & work["prior_annotation_review"].eq("AMBIGUOUS"),
             "annotation_review_status"] = "REJECTED_AMBIGUOUS"
    work = work.drop(columns=["domain_relevance_prior", "prior_annotation_review", "prior_review_notes"])
    for index, row in work.iterrows():
        if work.at[index, "domain_relevance"] == "NOT_REVIEWED":
            domain, note, method = auditable_domain_review(row.to_dict())
            work.at[index, "domain_relevance"] = domain
            work.at[index, "domain_review_notes"] = note
            work.at[index, "domain_review_method"] = method
        else:
            work.at[index, "domain_review_method"] = "MANUAL_V2_2C_OVERRIDE"
        if work.at[index, "annotation_review_status"] == "NOT_REVIEWED":
            status, note, method = auditable_annotation_review(row.to_dict())
            work.at[index, "annotation_review_status"] = status
            work.at[index, "annotation_review_notes"] = note
            work.at[index, "annotation_review_method"] = method
        else:
            work.at[index, "annotation_review_method"] = "MANUAL_V2_2C_OVERRIDE"
    work[["image_id", "domain_relevance", "domain_review_notes", "domain_review_method",
          "annotation_review_status", "annotation_review_notes", "annotation_review_method"]].to_csv(
        OUTPUT / "governed_review_decisions.csv", index=False
    )
    work.to_csv(OUTPUT / "openimages_review_worklist.csv", index=False)
    license_frame.to_csv(OUTPUT / "openimages_license_evidence.csv", index=False)
    annotations.to_csv(OUTPUT / "openimages_review_annotations.csv", index=False)
    _draw_contact_sheets(work, annotations, PREVIEWS)
    approved_visual = work.loc[work["license_review_status"].eq("LICENSE_APPROVED")].copy()
    approved_visual["visual_review_rank"] = range(1, len(approved_visual) + 1)
    approved_visual.to_csv(OUTPUT / "openimages_license_approved_review_worklist.csv", index=False)
    _draw_contact_sheets(approved_visual, annotations, OUTPUT / "license_approved_review_previews")
    print(json.dumps({
        "review_candidates": len(work), "downloaded_readable": int(work["image_readable"].sum()),
        "license_status": work["license_review_status"].value_counts().to_dict(),
        "prior_reviews_carried": int(mask.sum()), "preview_sheets": len(list(PREVIEWS.glob("*.jpg"))),
        "license_approved_preview_sheets": len(list(
            (OUTPUT / "license_approved_review_previews").glob("*.jpg")
        )),
    }, indent=2))
    return 0


def _taiwan_annotations(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]]]:
    records: list[dict[str, object]] = []
    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        text = (PROJECT_ROOT / row.label_path).read_text(encoding="utf-8")
        parsed, issues = parse_yolo_annotation(text, class_count=len(TAIWAN_CLASSES))
        if any(issue.severity == "ERROR" for issue in issues):
            raise ValueError(f"governed Taiwan label became invalid: {row.image_id}")
        for line_number, box in enumerate(parsed, 1):
            source_class = TAIWAN_CLASSES[box.class_id]
            app = map_application_class("taiwan_cctv_v3", source_class)
            area = box.width * box.height
            record = {
                "dataset_source": "taiwan_cctv_v3", "source_image_id": row.image_id,
                "source_class": source_class, "application_class": app,
                "xmin": box.x_center - box.width / 2, "ymin": box.y_center - box.height / 2,
                "xmax": box.x_center + box.width / 2, "ymax": box.y_center + box.height / 2,
                "normalized_area": area, "size_bin": size_bin(area),
                "is_occluded": "UNAVAILABLE", "is_truncated": "UNAVAILABLE",
                "is_group_of": "UNAVAILABLE", "is_depiction": "UNAVAILABLE",
                "is_inside": "UNAVAILABLE", "source_row": line_number,
                "source_annotation": text.splitlines()[line_number - 1],
                "derived_annotation": json.dumps({
                    "application_class": app, "x_center": box.x_center, "y_center": box.y_center,
                    "width": box.width, "height": box.height,
                }, sort_keys=True),
            }
            records.append(record); by_image[str(row.image_id)].append(record)
    return pd.DataFrame(records), by_image


def build(config: dict[str, object]) -> int:
    review_source = PROJECT_ROOT / config["openimages"]["review_manifest"]
    if not review_source.exists():
        raise FileNotFoundError(f"manual review manifest missing: {review_source}")
    work = pd.read_csv(OUTPUT / "openimages_review_worklist.csv", low_memory=False)
    review = pd.read_csv(review_source)
    if review["image_id"].duplicated().any() or set(review["image_id"].astype(str)) != set(work["image_id"].astype(str)):
        raise ValueError("manual review must contain every worklist image exactly once")
    work = work.drop(columns=["domain_relevance", "domain_review_notes",
                              "annotation_review_status", "annotation_review_notes"])
    work = work.merge(review, on="image_id", how="left", validate="one_to_one")
    decisions = work.apply(lambda row: openimages_training_gate(row.to_dict()), axis=1)
    work["approved_for_v2_training"] = [item[0] for item in decisions]
    work["governance_decision_reason"] = [item[1] for item in decisions]
    approved = work.loc[work["approved_for_v2_training"]].copy()
    rejected = work.loc[~work["approved_for_v2_training"]].copy()
    for frame in (approved, rejected):
        frame["source_url"] = frame["original_url"]
        frame["attribution"] = frame["author"]
    target_min = int(config["openimages"]["approved_target_min"])
    target_max = int(config["openimages"]["approved_target_max"])
    if len(approved) > target_max:
        approved = approved.sort_values("review_rank").head(target_max).copy()
        overflow = work.loc[work["approved_for_v2_training"] & ~work["image_id"].isin(approved["image_id"])].copy()
        overflow["approved_for_v2_training"] = False
        overflow["governance_decision_reason"] = "BALANCE_CAP_EXCEEDED"
        rejected = pd.concat([rejected, overflow], ignore_index=True)

    taiwan_split = pd.read_csv(PROJECT_ROOT / config["taiwan"]["split_manifest"])
    taiwan = taiwan_split.loc[taiwan_split["split"].isin(["TRAIN", "VAL"])].copy()
    taiwan_annotations, taiwan_by_image = _taiwan_annotations(taiwan)
    oi_annotations = pd.read_csv(OUTPUT / "openimages_review_annotations.csv")
    oi_annotations = oi_annotations.loc[oi_annotations["source_image_id"].astype(str).isin(
        approved["image_id"].astype(str))].copy()
    oi_by_image = {image_id: group.to_dict("records") for image_id, group in
                   oi_annotations.groupby("source_image_id")}

    manifest_rows: list[dict[str, object]] = []
    for row in taiwan.itertuples(index=False):
        records = taiwan_by_image[str(row.image_id)]
        manifest_rows.append({
            "image_id": f"taiwan_cctv_v3::{row.image_id}", "dataset_source": "taiwan_cctv_v3",
            "source_image_id": row.image_id, "image_path": row.image_path,
            "source_annotation_path": row.label_path, "source_group_id": f"TW::{row.group_id}",
            "image_sha256": row.sha256, "phash": row.phash,
            "annotation_sha256": annotation_sha256(records), "annotation_count": len(records),
        })
    for row in approved.itertuples(index=False):
        records = oi_by_image[str(row.image_id)]
        manifest_rows.append({
            "image_id": f"openimages_v7::{row.image_id}", "dataset_source": "openimages_v7",
            "source_image_id": row.image_id, "image_path": row.local_path,
            "source_annotation_path": "data/raw/openimages_v7/metadata/validation-annotations-bbox.csv",
            "source_group_id": f"OI::{row.image_id}", "image_sha256": row.image_sha256,
            "phash": row.phash, "annotation_sha256": annotation_sha256(records),
            "annotation_count": len(records),
        })
    dataset_manifest = pd.DataFrame(manifest_rows)
    group_input = dataset_manifest.rename(columns={"image_sha256": "sha256"}).to_dict("records")
    groups = pd.DataFrame(build_duplicate_groups(
        group_input, near_duplicate_threshold=int(config["split"]["near_duplicate_hamming_threshold"])
    ))
    groups = groups.rename(columns={"sha256": "image_sha256"})
    dataset_manifest = dataset_manifest.drop(columns=["source_group_id"]).merge(
        groups[["image_id", "group_id", "group_size", "exact_duplicate_count",
                "source_group_id", "grouping_basis"]], on="image_id", validate="one_to_one")
    assignments = deterministic_source_group_split(
        dataset_manifest.to_dict("records"), val_fraction=float(config["split"]["val_fraction"]),
        seed=int(config["seed"]),
    )
    split_manifest = dataset_manifest.copy()
    split_manifest["split"] = split_manifest["group_id"].map(assignments)
    validate_training_manifest(split_manifest.to_dict("records"))

    annotations = pd.concat([taiwan_annotations, oi_annotations], ignore_index=True)
    annotations["image_id"] = annotations.apply(
        lambda row: f"{row['dataset_source']}::{row['source_image_id']}", axis=1)
    annotations = annotations.merge(split_manifest[["image_id", "split"]], on="image_id", validate="many_to_one")
    class_distribution = (annotations.groupby(["split", "dataset_source", "application_class"])
                          .agg(image_count=("image_id", "nunique"), bbox_count=("image_id", "size"))
                          .reset_index())
    source_distribution = (annotations.groupby(["split", "dataset_source"])
                           .agg(image_count=("image_id", "nunique"), bbox_count=("image_id", "size"))
                           .reset_index())
    object_size = (annotations.groupby(["split", "dataset_source", "application_class", "size_bin"])
                   .agg(image_count=("image_id", "nunique"), bbox_count=("image_id", "size"))
                   .reset_index())

    locked_inventory = pd.read_csv(PROJECT_ROOT / config["taiwan"]["split_manifest"])
    locked = locked_inventory.loc[locked_inventory["split"].eq("LOCKED_TEST"),
                                  ["image_id", "sha256", "phash"]].astype(str).to_dict("records")
    overlap = cross_dataset_overlap_blocked(
        split_manifest[["image_id", "image_sha256", "phash"]]
        .rename(columns={"image_sha256": "sha256"}).to_dict("records"), locked,
        threshold=int(config["split"]["near_duplicate_hamming_threshold"]),
    )
    cross_split_groups = int((split_manifest.groupby("group_id")["split"].nunique() > 1).sum())
    taiwan_raw_hash = _tree_sha256(PROJECT_ROOT / config["taiwan"]["raw_root"])
    expected_raw_hash = str(config["taiwan"]["expected_raw_tree_sha256"])
    openimages_metadata_hashes = {
        path: sha256_file(PROJECT_ROOT / path)
        for path in config["openimages"]["immutable_metadata_sha256"]
    }
    openimages_metadata_immutable = all(
        openimages_metadata_hashes[path] == expected
        for path, expected in config["openimages"]["immutable_metadata_sha256"].items()
    )
    v1_train_person = int(pd.read_csv(PROJECT_ROOT / "outputs/data_qa/stage16/split_class_distribution.csv")
                          .query("split == 'TRAIN' and application_class == 'person'")["bbox_count"].sum())
    v2_train_person = int(annotations.loc[(annotations["split"] == "TRAIN") &
                                          (annotations["application_class"] == "person")].shape[0])
    pool_status = "APPROVED_FOR_V2_TRAINING" if (
        target_min <= len(approved) <= target_max
        and overlap["exact_overlap_count"] == overlap["near_overlap_count"] == 0
        and cross_split_groups == 0 and taiwan_raw_hash == expected_raw_hash
        and openimages_metadata_immutable
        and work.loc[work["approved_for_v2_training"], "license_review_status"].eq("LICENSE_APPROVED").all()
    ) else "NOT_APPROVED_FOR_V2_TRAINING"
    dataset_manifest["pool_status"] = (
        "FINAL_MANIFEST" if pool_status == "APPROVED_FOR_V2_TRAINING"
        else "PROVISIONAL_DO_NOT_TRAIN"
    )
    split_manifest["pool_status"] = dataset_manifest["pool_status"]
    approved_person = oi_annotations.loc[oi_annotations["application_class"].eq("person")]
    openimages_coverage = {
        name: {
            "image_count": int(oi_annotations.loc[oi_annotations["application_class"].eq(name),
                                                   "source_image_id"].nunique()),
            "bbox_count": int(oi_annotations["application_class"].eq(name).sum()),
        }
        for name in ("person", "bicycle", "car", "motorcycle", "bus", "truck")
    }
    openimages_coverage["person"].update({
        "small_bbox_count": int(approved_person["size_bin"].eq("SMALL_LT_0.01").sum()),
        "occluded_bbox_count": int(approved_person["is_occluded"].astype(str).eq("1").sum()),
    })
    summary = {
        "status": pool_status,
        "openimages": {
            "reviewed_images": len(work), "approved_images": len(approved),
            "rejected_images": len(rejected),
            "license_review": work["license_review_status"].value_counts().to_dict(),
            "domain_review": work["domain_relevance"].value_counts().to_dict(),
            "annotation_review": work["annotation_review_status"].value_counts().to_dict(),
            "governance_rejection_reasons": rejected["governance_decision_reason"].value_counts().to_dict(),
            "coverage": openimages_coverage,
            "target_minimum": target_min,
            "target_gap_images": max(0, target_min - len(approved)),
        },
        "combined": {
            "images": len(split_manifest),
            "train_images": int(split_manifest["split"].eq("TRAIN").sum()),
            "val_images": int(split_manifest["split"].eq("VAL").sum()),
            "v1_train_person_boxes": v1_train_person,
            "v2_train_person_boxes": v2_train_person,
            "v2_minus_v1_train_person_boxes": v2_train_person - v1_train_person,
        },
        "integrity": {
            "group_cross_split_count": cross_split_groups,
            "stage18_exact_overlap": overlap["exact_overlap_count"],
            "stage18_near_overlap": overlap["near_overlap_count"],
            "stage18_usage": "OVERLAP_CHECK_ONLY",
            "taiwan_raw_tree_sha256": taiwan_raw_hash,
            "taiwan_raw_immutable": taiwan_raw_hash == expected_raw_hash,
            "openimages_metadata_sha256": openimages_metadata_hashes,
            "openimages_metadata_immutable": openimages_metadata_immutable,
            "final_v2_holdout_created": False, "training_performed": False,
        },
        "blocking_reason": (
            "Open Images governed subset did not meet the required 400-image minimum; "
            "provisional manifests are DO_NOT_TRAIN."
            if len(approved) < target_min else ""
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    approved.to_csv(OUTPUT / "approved_openimages_manifest.csv", index=False)
    rejected.to_csv(OUTPUT / "rejected_openimages_manifest.csv", index=False)
    dataset_manifest.to_csv(OUTPUT / "v2_dataset_manifest.csv", index=False)
    split_manifest.to_csv(OUTPUT / "v2_split_manifest.csv", index=False)
    annotations.to_csv(OUTPUT / "v2_annotations.csv", index=False)
    class_distribution.to_csv(OUTPUT / "class_distribution.csv", index=False)
    source_distribution.to_csv(OUTPUT / "source_distribution.csv", index=False)
    object_size.to_csv(OUTPUT / "object_size_distribution.csv", index=False)
    groups.to_csv(OUTPUT / "duplicate_groups.csv", index=False)
    (OUTPUT / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if pool_status == "APPROVED_FOR_V2_TRAINING" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-review", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["governance"]["stage18_usage"] != "OVERLAP_CHECK_ONLY":
        raise ValueError("Stage 18 Locked Test usage must remain OVERLAP_CHECK_ONLY")
    if config["governance"]["final_v2_holdout_created"]:
        raise ValueError("V2-3 must not create a Final V2 Holdout")
    return prepare(config) if args.prepare_review else build(config)


if __name__ == "__main__":
    raise SystemExit(main())
