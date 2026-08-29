"""Ultralytics YOLO26 pretrained detector and target-class overlay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .schema import BoundingBox, DetectionRecord

TARGET_CLASS_NAMES = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
)

CLASS_COLORS = {
    "person": (255, 191, 0),
    "bicycle": (0, 215, 255),
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 255),
    "bus": (255, 128, 0),
    "truck": (0, 128, 255),
}


def filter_detection_candidates(
    boxes: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    class_names: Mapping[int, str],
    *,
    confidence_threshold: float,
    target_classes: frozenset[str] = TARGET_CLASS_NAMES,
) -> list[tuple[int, str, float, BoundingBox]]:
    """Filter raw model candidates and convert them to validated bbox values."""
    filtered: list[tuple[int, str, float, BoundingBox]] = []
    for coordinates, raw_class_id, raw_confidence in zip(
        boxes, class_ids, confidences, strict=True
    ):
        class_id = int(raw_class_id)
        confidence = float(raw_confidence)
        class_name = class_names.get(class_id, "")
        if class_name not in target_classes or confidence < confidence_threshold:
            continue
        filtered.append(
            (
                class_id,
                class_name,
                confidence,
                BoundingBox(*(float(value) for value in coordinates)),
            )
        )
    return filtered


class PretrainedDetector:
    """One loaded YOLO model configured for the Stage 4 MPS baseline."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "mps",
        imgsz: int = 640,
        confidence_threshold: float = 0.25,
    ) -> None:
        if device != "mps":
            raise ValueError("Stage 4 requires device='mps'")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not built and available; CPU fallback is disabled")
        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(str(self.model_path))
        self.class_names = {
            int(class_id): name for class_id, name in self.model.names.items()
        }
        self.target_class_ids = sorted(
            class_id
            for class_id, name in self.class_names.items()
            if name in TARGET_CLASS_NAMES
        )
        resolved_names = {self.class_names[class_id] for class_id in self.target_class_ids}
        if resolved_names != TARGET_CLASS_NAMES:
            missing = sorted(TARGET_CLASS_NAMES - resolved_names)
            raise RuntimeError(f"model taxonomy is missing target classes: {missing}")

    def detect(
        self,
        frame: object,
        *,
        video_id: str,
        source_id: str,
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[DetectionRecord]:
        results = self.model.predict(
            source=frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            classes=self.target_class_ids,
            save=False,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        candidates = filter_detection_candidates(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            self.class_names,
            confidence_threshold=self.confidence_threshold,
        )
        return [
            DetectionRecord(
                video_id=video_id,
                source_id=source_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
            )
            for class_id, class_name, confidence, bbox in candidates
        ]


def draw_detections(frame: object, detections: Sequence[DetectionRecord]) -> None:
    """Draw class name and confidence for each detection, without tracking data."""
    for detection in detections:
        bbox = detection.bbox
        color = CLASS_COLORS[detection.class_name]
        x1, y1, x2, y2 = map(round, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
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
