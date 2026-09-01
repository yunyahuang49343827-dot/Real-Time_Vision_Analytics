"""Supervision-only rendering adapters for existing tracking observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from vision_analytics.events.schema import EventRecord
from vision_analytics.spatial.line_crossing import CountingLine
from vision_analytics.spatial.zone import NormalizedZone
from vision_analytics.tracking.schema import TrackRecord


@dataclass(frozen=True, slots=True)
class VisualizationSettings:
    heatmap_classes: frozenset[str] = frozenset({"car", "motorcycle", "bus", "truck"})
    trace_length: int = 30
    heatmap_opacity: float = 0.28
    heatmap_radius: int = 36
    heatmap_kernel_size: int = 25

    def __post_init__(self) -> None:
        if self.trace_length <= 0 or self.heatmap_radius <= 0 or self.heatmap_kernel_size <= 0:
            raise ValueError("visualization dimensions must be positive")
        if not 0.0 < self.heatmap_opacity <= 1.0:
            raise ValueError("heatmap opacity must be within (0, 1]")


def tracks_to_supervision(records: Sequence[TrackRecord]) -> sv.Detections:
    """Adapt existing Track IDs to Supervision without tracking or ID assignment."""
    return sv.Detections(
        xyxy=np.asarray(
            [[item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2] for item in records],
            dtype=np.float32,
        ).reshape((-1, 4)),
        confidence=np.asarray([item.confidence for item in records], dtype=np.float32),
        class_id=np.asarray([item.class_id for item in records], dtype=int),
        tracker_id=np.asarray([item.track_id for item in records], dtype=int),
    )


def _draw_context(
    frame: np.ndarray,
    lines: Sequence[CountingLine],
    zones: Sequence[NormalizedZone],
    events: Sequence[EventRecord],
) -> None:
    height, width = frame.shape[:2]
    for line in lines:
        start, end = line.pixel_endpoints(width, height)
        start_i, end_i = tuple(map(round, start)), tuple(map(round, end))
        cv2.line(frame, start_i, end_i, (0, 210, 255), 2)
        cv2.putText(frame, line.line_id, start_i, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2)
    zone_layer = frame.copy()
    for zone in zones:
        polygon = zone.pixel_polygon(width, height).astype(np.int32)
        cv2.polylines(zone_layer, [polygon], True, (148, 163, 184), 1)
        anchor = tuple(polygon[0].tolist())
        cv2.putText(zone_layer, zone.zone_id, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
    if zones:
        # Preserve spatial context without allowing polygons to dominate the view.
        cv2.addWeighted(zone_layer, 0.32, frame, 0.68, 0, frame)
    for offset, event in enumerate(events[-3:]):
        text = f"{event.event_type} | {event.severity} | T{event.track_id if event.track_id is not None else '-'}"
        cv2.putText(frame, text, (12, 28 + offset * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 240), 2)


class SupervisionVisualizer:
    """Stateful annotators that consume, but never create, Track IDs."""

    def __init__(self, settings: VisualizationSettings) -> None:
        self.settings = settings
        self.box = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.trace = sv.TraceAnnotator(
            trace_length=settings.trace_length,
            position=sv.Position.CENTER,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.heatmap = sv.HeatMapAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            opacity=settings.heatmap_opacity,
            radius=settings.heatmap_radius,
            kernel_size=settings.heatmap_kernel_size,
        )

    def render_tracking(
        self,
        frame: np.ndarray,
        records: Sequence[TrackRecord],
        *,
        lines: Sequence[CountingLine] = (),
        zones: Sequence[NormalizedZone] = (),
        events: Sequence[EventRecord] = (),
    ) -> np.ndarray:
        detections = tracks_to_supervision(records)
        labels = [f"{item.class_name} {item.confidence:.2f} | Track ID {item.track_id}" for item in records]
        rendered = self.trace.annotate(frame.copy(), detections)
        rendered = self.box.annotate(rendered, detections)
        rendered = self.label.annotate(rendered, detections, labels=labels)
        _draw_context(rendered, lines, zones, events)
        return rendered

    def render_heatmap(
        self,
        frame: np.ndarray,
        records: Sequence[TrackRecord],
        *,
        lines: Sequence[CountingLine] = (),
        zones: Sequence[NormalizedZone] = (),
        events: Sequence[EventRecord] = (),
    ) -> np.ndarray:
        selected = [item for item in records if item.class_name in self.settings.heatmap_classes]
        rendered = self.heatmap.annotate(frame.copy(), tracks_to_supervision(selected))
        _draw_context(rendered, lines, zones, events)
        cv2.putText(
            rendered, "Image-space traffic activity", (12, rendered.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        return rendered
