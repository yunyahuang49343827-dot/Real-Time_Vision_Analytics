"""Validation rules for video metadata probes."""

from __future__ import annotations

from collections.abc import Mapping


def validate_video_metadata(
    metadata: Mapping[str, object], *, file_exists: bool
) -> tuple[str, str]:
    """Return a status and human-readable message for a metadata probe."""
    failures: list[str] = []
    warnings: list[str] = []

    if not file_exists:
        failures.append("file does not exist")
    if not metadata.get("opencv_opened", False):
        failures.append("OpenCV VideoCapture could not open the file")
    if not metadata.get("first_frame_decoded", False):
        failures.append("first frame could not be decoded")

    for field in ("width", "height", "fps", "frame_count"):
        value = metadata.get(field, 0)
        if not isinstance(value, (int, float)) or value <= 0:
            failures.append(f"{field} is unavailable or non-positive")

    duration = metadata.get("duration_seconds", 0)
    if not isinstance(duration, (int, float)) or duration <= 0:
        warnings.append("duration_seconds is unavailable or non-positive")
    if not metadata.get("codec"):
        warnings.append("codec is unavailable")

    if failures:
        message = "; ".join(failures + warnings)
        return "FAIL", message
    if warnings:
        return "WARNING", "; ".join(warnings)
    return "PASS", "All required video checks passed"
