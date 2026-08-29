"""Sampling and descriptive statistics for qualitative detection review."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REVIEW_RESULTS = frozenset(
    {"", "CORRECT", "FALSE_POSITIVE", "FALSE_NEGATIVE", "CLASS_CONFUSION", "AMBIGUOUS"}
)

REVIEW_FIELDS = (
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "sample_type",
    "predicted_class",
    "confidence",
    "review_result",
    "error_category",
    "notes",
)

SAMPLE_FIELDS = (
    "video_id",
    "source_id",
    "frame_index",
    "timestamp_seconds",
    "sample_type",
    "sample_reason",
    "predicted_class",
    "confidence",
    "detection_count",
    "distinct_class_count",
    "image_path",
)


@dataclass(frozen=True, slots=True)
class SampleFrame:
    video_id: str
    source_id: str
    frame_index: int
    timestamp_seconds: float
    sample_type: str
    sample_reason: str
    predicted_class: str = ""
    confidence: float | None = None
    detection_count: int = 0
    distinct_class_count: int = 0
    image_path: str = ""

    def to_row(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "sample_type": self.sample_type,
            "sample_reason": self.sample_reason,
            "predicted_class": self.predicted_class,
            "confidence": "" if self.confidence is None else round(self.confidence, 6),
            "detection_count": self.detection_count,
            "distinct_class_count": self.distinct_class_count,
            "image_path": self.image_path,
        }


def compute_confidence_statistics(detections: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive confidence statistics per predicted class."""
    columns = [
        "class_name",
        "occurrence_count",
        "mean",
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
    ]
    if detections.empty:
        return pd.DataFrame(columns=columns)

    grouped = detections.groupby("class_name", sort=True)["confidence"]
    rows = []
    for class_name, confidences in grouped:
        values = confidences.to_numpy(dtype=float)
        rows.append(
            {
                "class_name": class_name,
                "occurrence_count": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p25": float(np.quantile(values, 0.25)),
                "p75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ranked_frames(
    detections: pd.DataFrame,
    *,
    reason: str,
    class_name: str | None = None,
    limit: int = 6,
    smallest_bbox: bool = False,
) -> list[tuple[int, str, str, float | None]]:
    subset = detections
    if class_name:
        subset = subset[subset["class_name"] == class_name]
    if subset.empty:
        return []

    if smallest_bbox:
        subset = subset.assign(
            bbox_area=(subset["x2"] - subset["x1"]) * (subset["y2"] - subset["y1"])
        ).sort_values(["bbox_area", "confidence", "frame_index"])
    elif reason == "low_confidence":
        subset = subset.sort_values(["confidence", "frame_index"])
    else:
        density = subset.groupby("frame_index").size().rename("density")
        subset = (
            subset.join(density, on="frame_index")
            .sort_values(["density", "confidence", "frame_index"], ascending=[False, True, True])
        )

    output: list[tuple[int, str, str, float | None]] = []
    for frame_index, group in subset.groupby("frame_index", sort=False):
        focus = group.iloc[0]
        output.append(
            (int(frame_index), reason, str(focus["class_name"]), float(focus["confidence"]))
        )
        if len(output) >= limit:
            break
    return output


def select_review_samples(
    detections: pd.DataFrame,
    *,
    video_id: str,
    source_id: str,
    frame_count: int,
    source_fps: float,
    uniform_count: int = 12,
    targeted_count: int = 12,
) -> list[SampleFrame]:
    """Select deterministic uniform and scene-aware targeted unique frames."""
    if frame_count <= 0 or source_fps <= 0:
        raise ValueError("frame_count and source_fps must be positive")
    if uniform_count + targeted_count > frame_count:
        raise ValueError("requested samples exceed available unique frames")

    frame_summary = detections.groupby("frame_index").agg(
        detection_count=("class_name", "size"),
        distinct_class_count=("class_name", "nunique"),
    )

    uniform_indices = np.linspace(0, frame_count - 1, uniform_count, dtype=int).tolist()
    selected: dict[int, SampleFrame] = {}
    for frame_index in uniform_indices:
        counts = frame_summary.loc[frame_index] if frame_index in frame_summary.index else None
        selected[frame_index] = SampleFrame(
            video_id=video_id,
            source_id=source_id,
            frame_index=frame_index,
            timestamp_seconds=frame_index / source_fps,
            sample_type="UNIFORM",
            sample_reason="uniform_temporal",
            detection_count=0 if counts is None else int(counts["detection_count"]),
            distinct_class_count=0 if counts is None else int(counts["distinct_class_count"]),
        )

    candidate_groups: list[list[tuple[int, str, str, float | None]]] = [
        _ranked_frames(detections, reason="low_confidence", limit=3),
        _ranked_frames(detections, reason="high_detection_density", limit=3),
    ]
    if source_id == "pexels_2103099":
        candidate_groups.extend(
            [
                _ranked_frames(detections, reason="highway_person", class_name="person", limit=3),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="truck", limit=2),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="bus", limit=1),
            ]
        )
    elif source_id == "pexels_13258685":
        candidate_groups.extend(
            [
                _ranked_frames(detections, reason="scooter_motorcycle", class_name="motorcycle", limit=4),
                _ranked_frames(detections, reason="bicycle", class_name="bicycle", limit=2),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="truck", limit=1),
            ]
        )
    elif source_id == "pexels_37258214":
        candidate_groups.extend(
            [
                _ranked_frames(detections, reason="scooter_motorcycle", class_name="motorcycle", limit=2),
                _ranked_frames(detections, reason="bicycle", class_name="bicycle", limit=2),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="bus", limit=2),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="truck", limit=1),
            ]
        )
    elif source_id == "pexels_9322363":
        candidate_groups.extend(
            [
                _ranked_frames(detections, reason="aerial_small_object", limit=4, smallest_bbox=True),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="truck", limit=2),
                _ranked_frames(detections, reason="commercial_vehicle", class_name="bus", limit=1),
            ]
        )

    candidates = [item for group in candidate_groups for item in group]
    for frame_index, reason, predicted_class, confidence in candidates:
        if len(selected) >= uniform_count + targeted_count:
            break
        if frame_index in selected or not 0 <= frame_index < frame_count:
            continue
        counts = frame_summary.loc[frame_index] if frame_index in frame_summary.index else None
        selected[frame_index] = SampleFrame(
            video_id=video_id,
            source_id=source_id,
            frame_index=frame_index,
            timestamp_seconds=frame_index / source_fps,
            sample_type="TARGETED",
            sample_reason=reason,
            predicted_class=predicted_class,
            confidence=confidence,
            detection_count=0 if counts is None else int(counts["detection_count"]),
            distinct_class_count=0 if counts is None else int(counts["distinct_class_count"]),
        )

    if len(selected) < uniform_count + targeted_count:
        density_order = frame_summary.sort_values(
            ["detection_count", "distinct_class_count"], ascending=False
        ).index.tolist()
        for frame_index in density_order + list(range(frame_count)):
            if len(selected) >= uniform_count + targeted_count:
                break
            frame_index = int(frame_index)
            if frame_index in selected:
                continue
            counts = frame_summary.loc[frame_index] if frame_index in frame_summary.index else None
            selected[frame_index] = SampleFrame(
                video_id=video_id,
                source_id=source_id,
                frame_index=frame_index,
                timestamp_seconds=frame_index / source_fps,
                sample_type="TARGETED",
                sample_reason="targeted_fill",
                detection_count=0 if counts is None else int(counts["detection_count"]),
                distinct_class_count=0 if counts is None else int(counts["distinct_class_count"]),
            )

    return sorted(selected.values(), key=lambda sample: sample.frame_index)


def validate_review_row(row: dict[str, str]) -> None:
    """Validate the controlled review schema, allowing blank pending reviews."""
    missing = set(REVIEW_FIELDS) - set(row)
    if missing:
        raise ValueError(f"missing review fields: {sorted(missing)}")
    if row["review_result"] not in REVIEW_RESULTS:
        raise ValueError(f"invalid review_result: {row['review_result']}")
    if row["review_result"] == "FALSE_NEGATIVE" and row["predicted_class"]:
        raise ValueError("FALSE_NEGATIVE rows must not claim a predicted class")
