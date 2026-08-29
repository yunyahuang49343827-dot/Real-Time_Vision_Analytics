"""Read basic video metadata and decode only the first frame."""

from __future__ import annotations

import math
from pathlib import Path

import cv2

from .validation import validate_video_metadata

VIDEO_METADATA_FIELDS = (
    "video_id",
    "filename",
    "source_id",
    "file_size_mb",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "codec",
    "opencv_opened",
    "first_frame_decoded",
    "validation_status",
    "validation_message",
)


def decode_fourcc(value: float) -> str:
    """Convert OpenCV's numeric FOURCC value into a printable codec string."""
    if not math.isfinite(value) or value <= 0:
        return ""
    integer = int(value)
    codec = "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))
    return codec.strip("\x00 ") if all(character.isprintable() for character in codec) else ""


def profile_video(path: Path, *, video_id: str, source_id: str) -> dict[str, object]:
    """Collect container metadata and attempt one first-frame decode."""
    path = Path(path)
    file_exists = path.is_file()
    metadata: dict[str, object] = {
        "video_id": video_id,
        "filename": path.name,
        "source_id": source_id,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3) if file_exists else 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "codec": "",
        "opencv_opened": False,
        "first_frame_decoded": False,
        "validation_status": "FAIL",
        "validation_message": "probe did not run",
    }

    capture = None
    capture_error = ""
    if file_exists:
        try:
            capture = cv2.VideoCapture(str(path))
            opened = capture.isOpened()
            metadata["opencv_opened"] = opened
            if opened:
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                fourcc = capture.get(cv2.CAP_PROP_FOURCC)

                metadata["width"] = int(round(width)) if math.isfinite(width) and width > 0 else 0
                metadata["height"] = int(round(height)) if math.isfinite(height) and height > 0 else 0
                metadata["fps"] = round(fps, 6) if math.isfinite(fps) and fps > 0 else 0.0
                metadata["frame_count"] = (
                    int(round(frame_count))
                    if math.isfinite(frame_count) and frame_count > 0
                    else 0
                )
                metadata["codec"] = decode_fourcc(fourcc)
                if metadata["fps"] and metadata["frame_count"]:
                    metadata["duration_seconds"] = round(
                        metadata["frame_count"] / metadata["fps"], 3
                    )

                decoded, frame = capture.read()
                metadata["first_frame_decoded"] = bool(decoded and frame is not None)
        except (OSError, cv2.error) as exc:
            capture_error = f"OpenCV probe error: {exc}"
        finally:
            if capture is not None:
                capture.release()

    status, message = validate_video_metadata(metadata, file_exists=file_exists)
    if capture_error:
        status = "FAIL"
        message = f"{capture_error}; {message}"
    metadata["validation_status"] = status
    metadata["validation_message"] = message
    return metadata
