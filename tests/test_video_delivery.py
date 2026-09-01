from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.services.video_delivery import transcode_browser_video  # noqa: E402


def _raw_file(job_directory: Path) -> Path:
    raw_path = job_directory / "processed_raw.mp4"
    raw_path.write_bytes(b"raw-video")
    return raw_path


def test_missing_ffmpeg_returns_structured_warning(tmp_path: Path) -> None:
    raw_path = _raw_file(tmp_path)
    browser_path = tmp_path / "processed_browser.mp4"
    result = transcode_browser_video(
        raw_path, browser_path, job_directory=tmp_path, which=lambda _: None,
    )
    assert result.browser_path is None
    assert result.warning_code == "VIDEO_TRANSCODE_UNAVAILABLE"
    assert not browser_path.exists()


def test_transcoding_success_generates_browser_artifact(tmp_path: Path) -> None:
    raw_path = _raw_file(tmp_path)
    browser_path = tmp_path / "processed_browser.mp4"
    observed_command: list[str] | None = None

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal observed_command
        observed_command = command
        browser_path.write_bytes(b"browser-video")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = transcode_browser_video(
        raw_path,
        browser_path,
        job_directory=tmp_path,
        which=lambda _: "/usr/local/bin/ffmpeg",
        run_command=fake_run,
    )
    assert result.browser_path == browser_path.resolve()
    assert result.warning_code is None
    assert observed_command is not None
    assert "libx264" in observed_command
    assert "yuv420p" in observed_command
    assert "+faststart" in observed_command
    assert "-an" in observed_command
    assert observed_command[-1] == str(browser_path.resolve())


def test_transcoding_failure_removes_partial_output(tmp_path: Path) -> None:
    raw_path = _raw_file(tmp_path)
    browser_path = tmp_path / "processed_browser.mp4"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        browser_path.write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1, "", "encoder failed")

    result = transcode_browser_video(
        raw_path,
        browser_path,
        job_directory=tmp_path,
        which=lambda _: "/usr/local/bin/ffmpeg",
        run_command=fake_run,
    )
    assert result.browser_path is None
    assert result.warning_code == "VIDEO_TRANSCODE_FAILED"
    assert not browser_path.exists()


def test_transcode_paths_must_be_contained_in_job_directory(tmp_path: Path) -> None:
    job_directory = tmp_path / "job"
    job_directory.mkdir()
    raw_path = _raw_file(job_directory)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="job directory"):
        transcode_browser_video(
            outside, job_directory / "processed_browser.mp4", job_directory=job_directory,
        )
    with pytest.raises(ValueError, match="job directory"):
        transcode_browser_video(
            raw_path, tmp_path / "browser-outside.mp4", job_directory=job_directory,
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the real codec smoke test",
)
def test_real_transcode_produces_h264_yuv420p(tmp_path: Path) -> None:
    raw_path = tmp_path / "processed_raw.mp4"
    browser_path = tmp_path / "processed_browser.mp4"
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48),
    )
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()

    result = transcode_browser_video(raw_path, browser_path, job_directory=tmp_path)
    assert result.available
    probe = subprocess.run(
        (
            shutil.which("ffprobe") or "ffprobe",
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt", "-of", "json",
            str(browser_path),
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"codec_name": "h264", "pix_fmt": "yuv420p"}
