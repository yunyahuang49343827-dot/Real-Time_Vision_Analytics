"""Stateful Ultralytics YOLO26n and ByteTrack integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from vision_analytics.detection.detector import CLASS_COLORS
from vision_analytics.detection.schema import BoundingBox

from .schema import ALLOWED_TRACK_CLASSES, TrackRecord


def build_track_records(
    boxes: Sequence[Sequence[float]],
    track_ids: Sequence[int],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    class_names: Mapping[int, str],
    *,
    video_id: str,
    source_id: str,
    frame_index: int,
    timestamp_seconds: float,
) -> list[TrackRecord]:
    """Convert tracked model boxes to validated application records."""
    records: list[TrackRecord] = []
    for coordinates, raw_track_id, raw_class_id, raw_confidence in zip(
        boxes, track_ids, class_ids, confidences, strict=True
    ):
        class_id = int(raw_class_id)
        class_name = class_names.get(class_id, "")
        if class_name not in ALLOWED_TRACK_CLASSES:
            continue
        records.append(
            TrackRecord(
                video_id=video_id,
                source_id=source_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                track_id=int(raw_track_id),
                class_id=class_id,
                class_name=class_name,
                confidence=float(raw_confidence),
                bbox=BoundingBox(*(float(value) for value in coordinates)),
            )
        )
    return records


class StatefulByteTracker:
    """One model and persistent ByteTrack state scoped to one video."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "mps",
        imgsz: int = 640,
        confidence_threshold: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        if device != "mps":
            raise ValueError("Stage 6 requires device='mps'")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not built and available; CPU fallback is disabled")
        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.model = YOLO(str(self.model_path))
        self.class_names = {
            int(class_id): name for class_id, name in self.model.names.items()
        }
        self.target_class_ids = sorted(
            class_id
            for class_id, name in self.class_names.items()
            if name in ALLOWED_TRACK_CLASSES
        )
        resolved_names = {self.class_names[class_id] for class_id in self.target_class_ids}
        if resolved_names != ALLOWED_TRACK_CLASSES:
            missing = sorted(ALLOWED_TRACK_CLASSES - resolved_names)
            raise RuntimeError(f"model taxonomy is missing target classes: {missing}")

    def track_frame(
        self,
        frame: object,
        *,
        video_id: str,
        source_id: str,
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[TrackRecord]:
        """Track one sequential frame while preserving tracker state."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            classes=self.target_class_ids,
            save=False,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []
        return build_track_records(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.id.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            self.class_names,
            video_id=video_id,
            source_id=source_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )


def draw_tracks(frame: object, records: Sequence[TrackRecord]) -> None:
    """Draw class, diagnostic track ID, and confidence without trajectories."""
    for record in records:
        color = CLASS_COLORS[record.class_name]
        x1, y1, x2, y2 = map(
            round, (record.bbox.x1, record.bbox.y1, record.bbox.x2, record.bbox.y2)
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{record.class_name} ID:{record.track_id} {record.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 6, label_top + text_height + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
