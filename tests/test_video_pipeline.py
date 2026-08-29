from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_analytics.video.pipeline import process_video


def create_synthetic_video(path: Path, *, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create the synthetic MJPG test video")
    for index in range(frame_count):
        frame = np.full((64, 96, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_process_video_completes_and_preserves_frame_count(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.mp4"
    create_synthetic_video(input_path)

    result = process_video(
        input_path,
        output_path,
        video_id="synthetic",
        source_id="test_source",
    )

    assert result["status"] == "PASS"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert result["expected_frame_count"] == 8
    assert result["frames_processed"] == 8
    assert result["output_frame_count"] == 8
    assert result["processing_fps"] > 0
    assert (result["output_width"], result["output_height"]) == (96, 64)


def test_process_video_fails_for_invalid_input(tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"

    result = process_video(
        tmp_path / "missing.mp4",
        output_path,
        video_id="missing",
        source_id="test_source",
    )

    assert result["status"] == "FAIL"
    assert result["frames_processed"] == 0
    assert result["processing_fps"] == 0
    assert "input validation failed" in result["validation_message"]
    assert not output_path.exists()
