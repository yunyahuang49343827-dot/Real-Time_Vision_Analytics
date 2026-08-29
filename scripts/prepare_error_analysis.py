#!/usr/bin/env python3
"""Prepare Stage 5 confidence statistics and sampled visual-review artifacts."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.detection.error_analysis import (
    REVIEW_FIELDS,
    SAMPLE_FIELDS,
    SampleFrame,
    compute_confidence_statistics,
    select_review_samples,
)

DETECTION_DIR = PROJECT_ROOT / "outputs" / "detections" / "stage4"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
OVERLAY_VIDEO_DIR = PROJECT_ROOT / "outputs" / "videos" / "stage4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis" / "stage5"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"
BENCHMARK_PATH = ANALYTICS_DIR / "stage4_detection_benchmark.csv"
CONFIDENCE_PATH = ANALYTICS_DIR / "stage5_confidence_summary.csv"
SAMPLING_PATH = ANALYTICS_DIR / "stage5_sampling_summary.csv"
REVIEW_PATH = OUTPUT_DIR / "manual_review.csv"


def _extract_frame(capture: cv2.VideoCapture, frame_index: int) -> object:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    decoded, frame = capture.read()
    if not decoded or frame is None:
        raise RuntimeError(f"could not decode sampled frame {frame_index}")
    return frame


def _build_review_image(raw_frame: object, overlay_frame: object, sample: SampleFrame) -> object:
    target_width = 800
    target_height = round(raw_frame.shape[0] * target_width / raw_frame.shape[1])
    raw_small = cv2.resize(raw_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    overlay_small = cv2.resize(
        overlay_frame, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    header_height = 52
    canvas = np.zeros((target_height + header_height, target_width * 2, 3), dtype=np.uint8)
    canvas[header_height:, :target_width] = raw_small
    canvas[header_height:, target_width:] = overlay_small
    label = (
        f"{sample.source_id} frame={sample.frame_index} t={sample.timestamp_seconds:.3f}s "
        f"{sample.sample_type}/{sample.sample_reason} | RAW (left) / STAGE4 (right)"
    )
    cv2.putText(canvas, label, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _make_contact_sheets(image_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_index in range(0, len(image_paths), 6):
        page_paths = image_paths[page_index : page_index + 6]
        thumbnails = []
        for image_path in page_paths:
            image = cv2.imread(str(image_path))
            thumbnail = cv2.resize(image, (800, 476), interpolation=cv2.INTER_AREA)
            thumbnails.append(thumbnail)
        blank = np.zeros_like(thumbnails[0])
        while len(thumbnails) < 6:
            thumbnails.append(blank.copy())
        rows = [np.hstack(thumbnails[index : index + 2]) for index in range(0, 6, 2)]
        sheet = np.vstack(rows)
        cv2.imwrite(str(output_dir / f"contact_sheet_{page_index // 6 + 1:02d}.jpg"), sheet)


def main() -> int:
    benchmark = pd.read_csv(BENCHMARK_PATH)
    all_detections: list[pd.DataFrame] = []
    all_samples: list[SampleFrame] = []

    for video in benchmark.itertuples(index=False):
        detection_path = DETECTION_DIR / f"{video.video_id}_detections.csv"
        detections = pd.read_csv(detection_path)
        all_detections.append(detections)
        samples = select_review_samples(
            detections,
            video_id=video.video_id,
            source_id=video.source_id,
            frame_count=int(video.expected_frame_count),
            source_fps=float(video.source_fps),
        )

        raw_path = RAW_VIDEO_DIR / f"{video.video_id}.mp4"
        overlay_path = OVERLAY_VIDEO_DIR / f"{video.video_id}_stage4.mp4"
        raw_capture = cv2.VideoCapture(str(raw_path))
        overlay_capture = cv2.VideoCapture(str(overlay_path))
        if not raw_capture.isOpened() or not overlay_capture.isOpened():
            raise RuntimeError(f"could not open review video pair for {video.video_id}")

        image_dir = OUTPUT_DIR / "frames" / video.source_id
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths: list[Path] = []
        for sample in samples:
            raw_frame = _extract_frame(raw_capture, sample.frame_index)
            overlay_frame = _extract_frame(overlay_capture, sample.frame_index)
            review_image = _build_review_image(raw_frame, overlay_frame, sample)
            image_path = image_dir / f"frame_{sample.frame_index:06d}_{sample.sample_reason}.jpg"
            cv2.imwrite(str(image_path), review_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_paths.append(image_path)
            all_samples.append(
                replace(
                    sample,
                    image_path=image_path.relative_to(PROJECT_ROOT).as_posix(),
                )
            )
        raw_capture.release()
        overlay_capture.release()
        _make_contact_sheets(image_paths, OUTPUT_DIR / "contact_sheets" / video.source_id)
        print(f"{video.source_id}: prepared {len(samples)} unique review frames")

    combined = pd.concat(all_detections, ignore_index=True)
    confidence = compute_confidence_statistics(combined)
    confidence.insert(0, "scope", "all_stage4_videos")
    confidence.insert(1, "warning", "Confidence does not represent correctness")
    confidence.to_csv(CONFIDENCE_PATH, index=False, lineterminator="\n")

    with SAMPLING_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample.to_row() for sample in all_samples)

    if not REVIEW_PATH.exists():
        with REVIEW_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, lineterminator="\n")
            writer.writeheader()
            for sample in all_samples:
                writer.writerow(
                    {
                        "video_id": sample.video_id,
                        "frame_index": sample.frame_index,
                        "timestamp_seconds": round(sample.timestamp_seconds, 6),
                        "sample_type": sample.sample_type,
                        "predicted_class": sample.predicted_class,
                        "confidence": "" if sample.confidence is None else round(sample.confidence, 6),
                        "review_result": "",
                        "error_category": "",
                        "notes": "",
                    }
                )

    print(f"Wrote confidence statistics to {CONFIDENCE_PATH}")
    print(f"Wrote {len(all_samples)} samples to {SAMPLING_PATH}")
    print(f"Manual review template: {REVIEW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
