from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.detection.schema import BoundingBox  # noqa: E402
from vision_analytics.services.visualization import (  # noqa: E402
    SupervisionVisualizer,
    VisualizationSettings,
    tracks_to_supervision,
)
from vision_analytics.tracking.schema import TrackRecord  # noqa: E402


def track(track_id: int, x: float, *, class_name: str = "car", class_id: int = 2) -> TrackRecord:
    return TrackRecord(
        video_id="video", source_id="scene", frame_index=1, timestamp_seconds=0.1,
        track_id=track_id, class_id=class_id, class_name=class_name, confidence=0.81,
        bbox=BoundingBox(x, 20, x + 20, 45),
    )


def test_supervision_import_and_existing_track_ids_are_preserved() -> None:
    assert sv.__version__ == "0.30.1"
    detections = tracks_to_supervision([track(41, 10), track(99, 50)])
    assert detections.tracker_id.tolist() == [41, 99]


def test_tracking_and_heatmap_render_from_same_observations() -> None:
    visualizer = SupervisionVisualizer(VisualizationSettings())
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    records = [track(7, 20)]
    tracking = visualizer.render_tracking(frame, records)
    heatmap = visualizer.render_heatmap(frame, records)
    assert tracking.shape == frame.shape and np.any(tracking != frame)
    assert heatmap.shape == frame.shape and np.any(heatmap != frame)


def test_heatmap_class_filter_and_empty_tracks_are_supported() -> None:
    visualizer = SupervisionVisualizer(VisualizationSettings(
        heatmap_classes=frozenset({"car"}),
    ))
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    assert tracks_to_supervision([]).tracker_id.size == 0
    rendered = visualizer.render_heatmap(
        frame, [track(2, 30, class_name="person", class_id=0)],
    )
    assert rendered.shape == frame.shape


def test_production_does_not_instantiate_supervision_bytetrack() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "src").rglob("*.py")
    )
    forbidden = "sv." + "ByteTrack("  # avoid creating the forbidden production token in this test
    assert forbidden not in production
    assert "models/finetuned/stage17/best.pt" not in (
        PROJECT_ROOT / "src/vision_analytics/services/pipeline.py"
    ).read_text(encoding="utf-8")
