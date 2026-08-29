"""OpenCV-only video processing pipeline for Stage 3."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .metadata import profile_video

BENCHMARK_FIELDS = (
    "video_id",
    "source_id",
    "input_width",
    "input_height",
    "source_fps",
    "expected_frame_count",
    "frames_processed",
    "elapsed_seconds",
    "processing_fps",
    "output_path",
    "output_width",
    "output_height",
    "output_frame_count",
    "status",
    "validation_message",
)


def add_overlay(
    frame: object, *, video_id: str, frame_index: int, source_fps: float
) -> None:
    """Draw Stage 3 identifiers in place on one OpenCV frame."""
    timestamp_seconds = frame_index / source_fps if source_fps > 0 else 0.0
    minutes, seconds = divmod(timestamp_seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    lines = (
        f"video_id: {video_id}",
        f"frame: {frame_index}",
        f"timestamp: {timestamp}",
    )

    frame_height, frame_width = frame.shape[:2]
    font_scale = max(0.5, min(frame_width, frame_height) / 1080.0)
    line_height = max(22, int(32 * font_scale))
    origin_x = max(12, int(20 * font_scale))
    origin_y = max(30, int(40 * font_scale))
    thickness = max(1, int(round(2 * font_scale)))

    for line_index, text in enumerate(lines):
        position = (origin_x, origin_y + line_index * line_height)
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _empty_result(
    *, video_id: str, source_id: str, output_path: Path
) -> dict[str, object]:
    return {
        "video_id": video_id,
        "source_id": source_id,
        "input_width": 0,
        "input_height": 0,
        "source_fps": 0.0,
        "expected_frame_count": 0,
        "frames_processed": 0,
        "elapsed_seconds": 0.0,
        "processing_fps": 0.0,
        "output_path": str(output_path),
        "output_width": 0,
        "output_height": 0,
        "output_frame_count": 0,
        "status": "FAIL",
        "validation_message": "pipeline did not run",
    }


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    video_id: str,
    source_id: str,
    output_codec: str = "mp4v",
) -> dict[str, object]:
    """Process every frame with an overlay and validate the generated MP4."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    result = _empty_result(
        video_id=video_id, source_id=source_id, output_path=output_path
    )

    input_metadata = profile_video(
        input_path, video_id=video_id, source_id=source_id
    )
    result.update(
        {
            "input_width": input_metadata["width"],
            "input_height": input_metadata["height"],
            "source_fps": input_metadata["fps"],
            "expected_frame_count": input_metadata["frame_count"],
        }
    )
    if input_metadata["validation_status"] == "FAIL":
        result["validation_message"] = (
            f"input validation failed: {input_metadata['validation_message']}"
        )
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        result["validation_message"] = "input could not be reopened for processing"
        return result

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*output_codec),
        float(input_metadata["fps"]),
        (int(input_metadata["width"]), int(input_metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        result["validation_message"] = (
            f"VideoWriter could not open with codec {output_codec}"
        )
        return result

    frames_processed = 0
    processing_error = ""
    started_at = time.perf_counter()
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            add_overlay(
                frame,
                video_id=video_id,
                frame_index=frames_processed,
                source_fps=float(input_metadata["fps"]),
            )
            writer.write(frame)
            frames_processed += 1
    except (OSError, cv2.error) as exc:
        processing_error = f"OpenCV processing error: {exc}"
    finally:
        capture.release()
        writer.release()
    elapsed_seconds = time.perf_counter() - started_at

    result["frames_processed"] = frames_processed
    result["elapsed_seconds"] = round(elapsed_seconds, 6)
    result["processing_fps"] = (
        round(frames_processed / elapsed_seconds, 3)
        if frames_processed > 0 and elapsed_seconds > 0
        else 0.0
    )

    failures: list[str] = []
    warnings: list[str] = []
    if processing_error:
        failures.append(processing_error)
    if frames_processed <= 0:
        failures.append("no frames were processed")
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        failures.append("output MP4 is missing or empty")

    output_capture = cv2.VideoCapture(str(output_path))
    if not output_capture.isOpened():
        failures.append("generated output could not be opened")
    else:
        output_width = int(round(output_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        output_height = int(round(output_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output_frame_count = int(
            round(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        result["output_width"] = output_width
        result["output_height"] = output_height
        result["output_frame_count"] = output_frame_count
        if (output_width, output_height) != (
            input_metadata["width"],
            input_metadata["height"],
        ):
            failures.append("output resolution does not match input resolution")
        if output_frame_count != frames_processed:
            warnings.append(
                "output frame count does not match processed frame count: "
                f"{output_frame_count} != {frames_processed}"
            )
    output_capture.release()

    expected_frame_count = int(input_metadata["frame_count"])
    if expected_frame_count != frames_processed:
        warnings.append(
            "expected frame count does not match processed frame count: "
            f"{expected_frame_count} != {frames_processed}"
        )

    if failures:
        result["status"] = "FAIL"
        result["validation_message"] = "; ".join(failures + warnings)
    elif warnings:
        result["status"] = "WARNING"
        result["validation_message"] = "; ".join(warnings)
    else:
        result["status"] = "PASS"
        result["validation_message"] = "All pipeline and output checks passed"
    return result
