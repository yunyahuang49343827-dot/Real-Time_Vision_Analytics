#!/usr/bin/env python3
"""Prepare a governed, unannotated targeted frame candidate set for V2."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.utils.dataset_qa import image_dhash, sha256_file
from vision_analytics.video.targeted_sampling import (
    balanced_temporal_select,
    choose_duplicate_representatives,
    conservative_duplicate_groups,
    deterministic_candidate_id,
    frame_in_ranges,
    timestamp_seconds,
    validate_candidate_manifest,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "v2_targeted_sampling.yaml"
TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
MANIFEST_FIELDS = (
    "candidate_id", "video_id", "source_video", "frame_id", "timestamp_sec", "scene",
    "image_path", "image_sha256", "dhash", "duplicate_group_id", "duplicate_group_size",
    "sampling_method", "predicted_classes", "coverage_tags", "tag_source",
    "stage19_overlap", "selection_status", "selection_reason", "annotation_status",
    "priority_score", "blur_variance",
)


def _path(value: str) -> Path:
    return PROJECT_ROOT / value


def _load_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _stage19_ranges(path: Path) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["video_id"]].append((int(row["start_frame"]), int(row["end_frame"])))
    return dict(result)


def _bbox_overlap_ratio(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x1 = max(first[0], second[0]); y1 = max(first[1], second[1])
    x2 = min(first[2], second[2]); y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = min(area_a, area_b)
    return intersection / denominator if denominator else 0.0


def _frame_features(detections: pd.DataFrame, width: int, height: int) -> dict[int, dict[str, object]]:
    features: dict[int, dict[str, object]] = {}
    for frame_id, group in detections.groupby("frame_index", sort=False):
        group = group[group["class_name"].isin(TARGET_CLASSES)]
        classes = group["class_name"].astype(str).tolist()
        boxes = [tuple(values) for values in group[["x1", "y1", "x2", "y2"]].to_numpy()]
        person_areas = [
            max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (width * height)
            for box, class_name in zip(boxes, classes, strict=True) if class_name == "person"
        ]
        overlap = any(
            _bbox_overlap_ratio(boxes[a], boxes[b]) >= 0.20
            for a in range(len(boxes)) for b in range(a + 1, len(boxes))
        )
        features[int(frame_id)] = {
            "classes": classes,
            "person_count": classes.count("person"),
            "vehicle_count": sum(name in VEHICLE_CLASSES for name in classes),
            "small_person": any(area < 0.01 for area in person_areas),
            "overlap": overlap,
            "detection_count": len(classes),
        }
    return features


def _candidate_rows(
    *, scene: str, scene_config: dict[str, object], metadata: dict[str, str],
    detections: pd.DataFrame, excluded: list[tuple[int, int]], fractions: dict[str, float],
) -> list[dict[str, object]]:
    fps = float(metadata["fps"]); frame_count = int(metadata["frame_count"])
    features = _frame_features(detections, int(metadata["width"]), int(metadata["height"]))
    systematic_gap = max(1, round(fps * float(scene_config["systematic_interval_seconds"])))
    rows: list[dict[str, object]] = []
    static_tags = set(scene_config.get("static_tags", []))
    for frame_id in range(frame_count):
        if frame_in_ranges(frame_id, excluded):
            continue
        item = features.get(frame_id, {"classes": [], "person_count": 0, "vehicle_count": 0,
                                       "small_person": False, "overlap": False, "detection_count": 0})
        classes = list(item["classes"]); class_set = set(classes)
        tags = set(static_tags)
        sources = {"HEURISTIC"} if static_tags else set()
        if item["person_count"]:
            tags.add("PERSON_PRESENT"); sources.add("MODEL_ASSISTED")
        if item["person_count"] >= 2:
            tags.add("MULTIPLE_PERSON")
        if item["small_person"]:
            tags.add("SMALL_PERSON")
        for class_name, tag in (("motorcycle", "PERSON_MOTORCYCLE"),
                                ("bicycle", "PERSON_BICYCLE"), ("car", "PERSON_CAR")):
            if item["person_count"] and class_name in class_set:
                tags.add(tag)
        if item["detection_count"] >= 10:
            tags.add("DENSE_TRAFFIC"); sources.add("HEURISTIC")
        if item["overlap"]:
            tags.add("OCCLUSION"); sources.add("HEURISTIC")
        if not item["person_count"] and item["vehicle_count"] >= 3:
            tags.add("HARD_NEGATIVE"); sources.add("HEURISTIC")
        if frame_id % systematic_gap == 0:
            method = "SYSTEMATIC_TEMPORAL"; sources.add("SYSTEMATIC")
        elif "HARD_NEGATIVE" in tags:
            method = "HARD_NEGATIVE_CONTEXT"
        elif tags & {"SMALL_PERSON", "MULTIPLE_PERSON", "DENSE_TRAFFIC", "OCCLUSION", "AERIAL_SMALL_OBJECT"}:
            method = "DIFFICULT_HEURISTIC"
        else:
            method = "MODEL_ASSISTED_POSITIVE"; sources.add("MODEL_ASSISTED")
        score = (
            item["person_count"] * 3 + item["detection_count"] * 0.15
            + 5 * ("PERSON_MOTORCYCLE" in tags) + 5 * ("PERSON_BICYCLE" in tags)
            + 3 * ("SMALL_PERSON" in tags) + 2 * ("MULTIPLE_PERSON" in tags)
            + 2 * ("DENSE_TRAFFIC" in tags) + 1.5 * ("OCCLUSION" in tags)
            + 1 * (method == "SYSTEMATIC_TEMPORAL") + 0.5 * ("HARD_NEGATIVE" in tags)
        )
        rows.append({
            "candidate_id": deterministic_candidate_id(scene, frame_id), "video_id": metadata["video_id"],
            "frame_id": frame_id, "timestamp_sec": round(timestamp_seconds(frame_id, fps), 6),
            "scene": scene, "sampling_method": method,
            "predicted_classes": ";".join(sorted(class_set)), "coverage_tags": ";".join(sorted(tags)),
            "tag_source": ";".join(sorted(sources)) or "HEURISTIC", "priority_score": round(score, 3),
        })
    gap_frames = max(1, math.ceil(fps * float(scene_config["minimum_temporal_separation_seconds"])))
    return [dict(row) for row in balanced_temporal_select(
        rows, target=int(scene_config["preliminary_target"]), minimum_gap_frames=gap_frames,
        method_fractions=fractions,
    )]


def _extract_selected(video_path: Path, rows: list[dict[str, object]], output_dir: Path) -> None:
    requested = {int(row["frame_id"]): row for row in rows}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame_id = 0
        while requested:
            ok, frame = capture.read()
            if not ok:
                break
            row = requested.pop(frame_id, None)
            if row is not None:
                path = output_dir / f"{row['candidate_id']}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"failed to write {path}")
                row["image_path"] = path.relative_to(PROJECT_ROOT).as_posix()
                row["image_sha256"] = sha256_file(path)
                row["dhash"] = image_dhash(frame)
                row["blur_variance"] = round(float(cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 3)
            frame_id += 1
    finally:
        capture.release()
    if requested:
        raise RuntimeError(f"could not decode requested frames: {sorted(requested)[:5]}")


def _assign_duplicate_groups(
    rows: list[dict[str, object]], threshold: int, max_frame_gap_by_video: dict[str, int],
) -> None:
    groups = conservative_duplicate_groups(
        rows, threshold=threshold, max_frame_gap_by_video=max_frame_gap_by_video,
    )
    by_id = {str(row["candidate_id"]): row for row in groups}
    for row in rows:
        group = by_id[str(row["candidate_id"])]
        row["duplicate_group_id"] = str(group["duplicate_group_id"])
        row["duplicate_group_size"] = int(group["group_size"])
        row["duplicate_basis"] = group["grouping_basis"]
        row["exact_duplicate_group_size"] = int(group["exact_duplicate_group_size"])


def _final_selection(rows: list[dict[str, object]], config: dict[str, object]) -> set[str]:
    representatives, duplicate_reasons = choose_duplicate_representatives(rows)
    final_ids: set[str] = set()
    fractions = dict(config["sampling_method_fractions"])
    for scene, scene_config in config["scenes"].items():
        eligible = [row for row in rows if row["scene"] == scene and row["candidate_id"] in representatives
                    and float(row["blur_variance"]) >= 15.0]
        metadata_fps = float(scene_config["runtime_fps"])
        gap_frames = max(1, math.ceil(metadata_fps * float(scene_config["minimum_temporal_separation_seconds"])))
        chosen = balanced_temporal_select(
            eligible, target=int(scene_config["final_target"]), minimum_gap_frames=gap_frames,
            method_fractions=fractions,
        )
        final_ids.update(str(row["candidate_id"]) for row in chosen)
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in final_ids:
            row["selection_status"] = "SELECTED_FOR_ANNOTATION"
            row["selection_reason"] = "COVERAGE_BALANCED_DUPLICATE_REPRESENTATIVE"
        elif candidate_id not in representatives:
            row["selection_status"] = "REJECTED_NEAR_DUPLICATE"
            row["selection_reason"] = duplicate_reasons[candidate_id]
        elif float(row["blur_variance"]) < 15.0:
            row["selection_status"] = "REJECTED_LOW_SHARPNESS"
            row["selection_reason"] = "LAPLACIAN_VARIANCE_LT_15"
        else:
            row["selection_status"] = "NOT_SELECTED_BALANCE"
            row["selection_reason"] = "FINAL_SCENE_METHOD_BALANCE"
    return final_ids


def _copy_final(rows: list[dict[str, object]], mined_dir: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["selection_status"] != "SELECTED_FOR_ANNOTATION":
            continue
        source = mined_dir / f"{row['candidate_id']}.jpg"
        destination = images_dir / source.name
        shutil.copy2(source, destination)
        row["image_path"] = destination.relative_to(PROJECT_ROOT).as_posix()


def _contact_sheets(rows: list[dict[str, object]], output_dir: Path, page_size: int) -> None:
    selected = [row for row in rows if row["selection_status"] == "SELECTED_FOR_ANNOTATION"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene in sorted({str(row["scene"]) for row in selected}):
        scene_rows = [row for row in selected if row["scene"] == scene]
        for page_start in range(0, len(scene_rows), page_size):
            tiles = []
            for row in scene_rows[page_start:page_start + page_size]:
                image = cv2.imread(str(PROJECT_ROOT / str(row["image_path"])))
                thumb = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
                overlay = thumb.copy()
                cv2.rectangle(overlay, (0, 0), (320, 65), (0, 0, 0), -1)
                label = f"{row['candidate_id']}  t={float(row['timestamp_sec']):.2f}s"
                tags = str(row["coverage_tags"])[:54]
                cv2.putText(overlay, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                cv2.putText(overlay, str(row["video_id"])[:47], (5, 37),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.31, (180, 255, 180), 1)
                cv2.putText(overlay, tags, (5, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (80, 220, 255), 1)
                tiles.append(overlay)
            while len(tiles) < page_size:
                tiles.append(255 * __import__("numpy").ones((180, 320, 3), dtype="uint8"))
            sheet_rows = [cv2.hconcat(tiles[index:index + 5]) for index in range(0, page_size, 5)]
            sheet = cv2.vconcat(sheet_rows)
            page = page_start // page_size + 1
            cv2.imwrite(str(output_dir / f"{scene.lower()}_contact_sheet_{page:02d}.jpg"), sheet)


def _write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    config = _load_config()
    metadata_rows = pd.read_csv(_path(str(config["video_metadata"])), dtype={"video_id": str})
    metadata_by_video = {row["video_id"]: {key: str(value) for key, value in row.items()}
                         for row in metadata_rows.to_dict("records")}
    excluded = _stage19_ranges(_path(str(config["stage19_manifest"])))
    output_root = _path(str(config["output_root"])); mined_dir = output_root / "mined"
    images_dir = output_root / "images"; contact_dir = output_root / "contact_sheets"
    analytics_dir = PROJECT_ROOT / "outputs" / "data_qa" / "v2_targeted_sampling"
    for directory in (mined_dir, images_dir, contact_dir):
        if directory.exists():
            shutil.rmtree(directory)
    for directory in (mined_dir, images_dir, contact_dir, analytics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    raw_hashes_before: dict[str, str] = {}
    for scene, scene_config in config["scenes"].items():
        video_id = str(scene_config["video_id"]); metadata = metadata_by_video[video_id]
        scene_config["runtime_fps"] = float(metadata["fps"])
        video_path = PROJECT_ROOT / "data" / "raw" / "videos" / metadata["filename"]
        digest = sha256_file(video_path); raw_hashes_before[video_id] = digest
        expected = str(config["raw_video_sha256"][video_id])
        if digest != expected:
            raise RuntimeError(f"raw integrity mismatch before extraction: {video_id}")
        detection_path = _path(str(config["stage4_detection_dir"])) / f"{video_id}_detections.csv"
        detections = pd.read_csv(detection_path)
        scene_rows = _candidate_rows(
            scene=scene, scene_config=scene_config, metadata=metadata, detections=detections,
            excluded=excluded.get(video_id, []), fractions=dict(config["sampling_method_fractions"]),
        )
        _extract_selected(video_path, scene_rows, mined_dir)
        for row in scene_rows:
            row["source_video"] = video_path.relative_to(PROJECT_ROOT).as_posix()
            row["stage19_overlap"] = False
            row["annotation_status"] = "NOT_ANNOTATED"
        rows.extend(scene_rows)
        print(f"{scene}: preliminary={len(scene_rows)}")

    max_gaps = {
        video_id: max(1, round(float(metadata["fps"]) * float(config["near_duplicate_max_seconds"])))
        for video_id, metadata in metadata_by_video.items()
    }
    _assign_duplicate_groups(rows, int(config["duplicate_hamming_threshold"]), max_gaps)
    final_ids = _final_selection(rows, config)
    _copy_final(rows, mined_dir, images_dir)
    rows.sort(key=lambda row: (str(row["scene"]), int(row["frame_id"])))
    validate_candidate_manifest(rows, excluded_ranges=excluded)
    selected_by_scene = {
        scene: sum(row["candidate_id"] in final_ids and row["scene"] == scene for row in rows)
        for scene in config["scenes"]
    }
    print(f"selected_by_scene={selected_by_scene}")
    if not 600 <= len(final_ids) <= 800:
        raise RuntimeError(f"selected frame count outside governed range: {len(final_ids)}")
    selected_files = list(images_dir.glob("*.jpg"))
    if len(selected_files) != len(final_ids):
        raise RuntimeError("selected image file count does not match manifest")
    for video_id, before in raw_hashes_before.items():
        metadata = metadata_by_video[video_id]
        after = sha256_file(PROJECT_ROOT / "data" / "raw" / "videos" / metadata["filename"])
        if after != before:
            raise RuntimeError(f"raw video changed: {video_id}")

    _write_manifest(rows, _path(str(config["candidate_manifest"])))
    _contact_sheets(rows, contact_dir, int(config["contact_sheet_size"]))
    selected = [row for row in rows if row["selection_status"] == "SELECTED_FOR_ANNOTATION"]
    tag_counts: dict[str, int] = defaultdict(int)
    method_counts: dict[str, int] = defaultdict(int)
    scene_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        scene_counts[str(row["scene"])] += 1
        method_counts[str(row["sampling_method"])] += 1
        for tag in str(row["coverage_tags"]).split(";"):
            if tag: tag_counts[tag] += 1
    temporal_separation = {}
    for scene in sorted(scene_counts):
        scene_frames = sorted(int(row["frame_id"]) for row in selected if row["scene"] == scene)
        fps = float(config["scenes"][scene]["runtime_fps"])
        temporal_separation[scene] = {
            "configured_minimum_seconds": float(config["scenes"][scene]["minimum_temporal_separation_seconds"]),
            "observed_minimum_seconds": round(min(
                (right - left) / fps for left, right in zip(scene_frames, scene_frames[1:], strict=False)
            ), 6) if len(scene_frames) > 1 else None,
        }
    summary = {
        "dataset_status": "TARGETED_ANNOTATION_CANDIDATE_SET",
        "annotation_status": "NOT_ANNOTATED",
        "preliminary_candidate_count": len(rows), "selected_count": len(selected),
        "rejected_count": len(rows) - len(selected), "scene_counts": dict(sorted(scene_counts.items())),
        "sampling_method_counts": dict(sorted(method_counts.items())),
        "coverage_tag_counts": dict(sorted(tag_counts.items())),
        "duplicate_hamming_threshold": int(config["duplicate_hamming_threshold"]),
        "near_duplicate_max_seconds": float(config["near_duplicate_max_seconds"]),
        "duplicate_rejected_count": sum(row["selection_status"] == "REJECTED_NEAR_DUPLICATE" for row in rows),
        "exact_duplicate_rejected_count": sum(
            row["selection_status"] == "REJECTED_NEAR_DUPLICATE" and row["duplicate_basis"] == "EXACT"
            for row in rows
        ),
        "near_duplicate_rejected_count": sum(
            row["selection_status"] == "REJECTED_NEAR_DUPLICATE" and row["duplicate_basis"] != "EXACT"
            for row in rows
        ),
        "low_sharpness_rejected_count": sum(row["selection_status"] == "REJECTED_LOW_SHARPNESS" for row in rows),
        "temporal_separation": temporal_separation,
        "stage19_overlap_count": 0, "raw_hashes_before_and_after_match": True,
        "stage4_predictions_role": "SAMPLING_ASSISTANCE_ONLY_NOT_GROUND_TRUTH",
        "training_performed": False, "final_v2_holdout_created": False,
    }
    (analytics_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"video_id": video_id, "start_frame": start, "end_frame": end, "inclusive": True}
        for video_id, ranges in excluded.items() for start, end in ranges
    ]).to_csv(analytics_dir / "stage19_exclusions.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
