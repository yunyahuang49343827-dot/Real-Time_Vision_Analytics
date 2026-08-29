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

from vision_analytics.video.metadata import decode_fourcc, profile_video
from vision_analytics.video.validation import validate_video_metadata


def test_profile_video_with_synthetic_artifact(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create the synthetic MJPG test video")

    for value in range(5):
        frame = np.full((48, 64, 3), value * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    result = profile_video(video_path, video_id="synthetic", source_id="test_source")

    assert result["validation_status"] == "PASS"
    assert result["opencv_opened"] is True
    assert result["first_frame_decoded"] is True
    assert result["width"] == 64
    assert result["height"] == 48
    assert result["fps"] == pytest.approx(10.0)
    assert result["frame_count"] == 5
    assert result["duration_seconds"] == pytest.approx(0.5)


def test_missing_video_returns_fail_without_crashing(tmp_path: Path) -> None:
    result = profile_video(
        tmp_path / "missing.mp4", video_id="missing", source_id="test_source"
    )

    assert result["validation_status"] == "FAIL"
    assert "file does not exist" in result["validation_message"]


def test_missing_codec_is_a_warning() -> None:
    metadata = {
        "opencv_opened": True,
        "first_frame_decoded": True,
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "frame_count": 300,
        "duration_seconds": 10.0,
        "codec": "",
    }

    status, message = validate_video_metadata(metadata, file_exists=True)

    assert status == "WARNING"
    assert message == "codec is unavailable"


def test_decode_fourcc_handles_valid_and_unavailable_values() -> None:
    assert decode_fourcc(float(cv2.VideoWriter_fourcc(*"avc1"))) == "avc1"
    assert decode_fourcc(0.0) == ""
