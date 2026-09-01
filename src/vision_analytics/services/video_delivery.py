"""Browser-compatible delivery transcoding for completed processed videos."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class VideoDeliveryResult:
    browser_path: Path | None
    warning_code: str | None
    warning_message: str | None
    command: tuple[str, ...] | None

    @property
    def available(self) -> bool:
        return self.browser_path is not None


def _contained(path: Path, job_directory: Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(job_directory).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("video delivery path must be a file inside the job directory")
    return resolved


def transcode_browser_video(
    raw_path: Path,
    browser_path: Path,
    *,
    job_directory: Path,
    which: Callable[[str], str | None] = shutil.which,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VideoDeliveryResult:
    """Transcode raw OpenCV MP4 to H.264/yuv420p/faststart without fallback."""
    raw = _contained(raw_path, job_directory)
    browser = _contained(browser_path, job_directory)
    if raw == browser:
        raise ValueError("raw and browser video paths must be distinct")
    if not raw.is_file() or raw.stat().st_size <= 0:
        raise ValueError("raw processed video is missing or empty")
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return VideoDeliveryResult(
            browser_path=None,
            warning_code="VIDEO_TRANSCODE_UNAVAILABLE",
            warning_message="FFmpeg is unavailable; browser-compatible processed video was not generated.",
            command=None,
        )
    browser.parent.mkdir(parents=True, exist_ok=True)
    command = (
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(browser),
    )
    try:
        completed = run_command(
            list(command), capture_output=True, text=True, check=False, timeout=1800,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is None or completed.returncode != 0 or not browser.is_file() or browser.stat().st_size <= 0:
        if browser.exists():
            browser.unlink()
        return VideoDeliveryResult(
            browser_path=None,
            warning_code="VIDEO_TRANSCODE_FAILED",
            warning_message="FFmpeg could not generate a browser-compatible processed video.",
            command=command,
        )
    return VideoDeliveryResult(
        browser_path=browser,
        warning_code=None,
        warning_message=None,
        command=command,
    )
