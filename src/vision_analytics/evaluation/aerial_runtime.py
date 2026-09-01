"""Governed helpers for the Stage 21.1A aerial runtime diagnostic."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import yaml

from vision_analytics.training.governance import sha256_file

APPROVED_RUNTIME_MODEL = Path("models/pretrained/yolo26n.pt")
EXPECTED_EXPERIMENTS = (
    ("baseline_640_025", 640, 0.25),
    ("lowconf_640_015", 640, 0.15),
    ("highres_960_025", 960, 0.25),
    ("highres_lowconf_960_015", 960, 0.15),
)
SMALL_VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})
SAFE_EXPERIMENT_NAME = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    imgsz: int
    conf: float


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    project_root: Path
    runtime_model: Path
    expected_model_sha256: str
    rejected_model: Path
    input_video: Path
    expected_input_sha256: str
    video_id: str
    source_id: str
    output_directory: Path
    device: str
    tracker: str
    scene_config: Path
    start_frame: int
    end_frame: int
    comparison_relative_seconds: tuple[float, ...]
    small_bbox_area_threshold: float
    low_confidence_band: tuple[float, float]
    max_history_length: int
    minimum_displacement_pixels: float
    experiments: tuple[Experiment, ...]


def _project_path(project_root: Path, value: object) -> Path:
    root = project_root.resolve()
    path = (root / str(value)).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("diagnostic paths must be inside the project root")
    return path


def load_diagnostic_config(path: Path, project_root: Path) -> DiagnosticConfig:
    """Load and strictly validate the controlled four-run experiment matrix."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("diagnostic config must be a mapping")
    experiments = tuple(
        Experiment(str(row["name"]), int(row["imgsz"]), float(row["conf"]))
        for row in payload["experiments"]
    )
    observed_matrix = tuple((row.name, row.imgsz, row.conf) for row in experiments)
    if observed_matrix != EXPECTED_EXPERIMENTS:
        raise ValueError("Stage 21.1A experiment matrix does not match the governed matrix")
    if str(payload["runtime_model"]) != APPROVED_RUNTIME_MODEL.as_posix():
        raise ValueError("Stage 21.1A runtime model must remain pretrained yolo26n.pt")

    frame_range = payload["frame_range"]
    trajectory = payload["trajectory"]
    low_band = tuple(float(value) for value in payload["low_confidence_band"])
    if len(low_band) != 2 or not 0 <= low_band[0] < low_band[1] <= 1:
        raise ValueError("low confidence band must be an increasing pair within [0, 1]")
    config = DiagnosticConfig(
        project_root=Path(project_root).resolve(),
        runtime_model=_project_path(project_root, payload["runtime_model"]),
        expected_model_sha256=str(payload["runtime_model_sha256"]),
        rejected_model=_project_path(project_root, payload["rejected_model"]),
        input_video=_project_path(project_root, payload["input_video"]),
        expected_input_sha256=str(payload["input_video_sha256"]),
        video_id=str(payload["video_id"]),
        source_id=str(payload["source_id"]),
        output_directory=_project_path(project_root, payload["output_directory"]),
        device=str(payload["device"]),
        tracker=str(payload["tracker"]),
        scene_config=_project_path(project_root, payload["scene_config"]),
        start_frame=int(frame_range["start"]),
        end_frame=int(frame_range["end"]),
        comparison_relative_seconds=tuple(float(value) for value in payload["comparison_relative_seconds"]),
        small_bbox_area_threshold=float(payload["small_bbox_area_threshold"]),
        low_confidence_band=(low_band[0], low_band[1]),
        max_history_length=int(trajectory["max_history_length"]),
        minimum_displacement_pixels=float(trajectory["minimum_displacement_pixels"]),
        experiments=experiments,
    )
    if config.device != "mps" or config.tracker != "bytetrack.yaml":
        raise ValueError("device and ByteTrack configuration are fixed for Stage 21.1A")
    if config.start_frame < 0 or config.end_frame < config.start_frame:
        raise ValueError("frame range is invalid")
    if config.max_history_length != 30 or config.minimum_displacement_pixels != 5.0:
        raise ValueError("trajectory configuration must remain unchanged")
    if not 0 < config.small_bbox_area_threshold < 1:
        raise ValueError("small bbox area threshold must be within (0, 1)")
    if not config.video_id or not config.source_id:
        raise ValueError("video_id and source_id are required")
    return config


def validate_runtime_assets(config: DiagnosticConfig) -> dict[str, object]:
    """Prove the approved model and input exist without opening rejected weights."""
    if not config.runtime_model.is_file():
        raise FileNotFoundError(config.runtime_model)
    actual_hash = sha256_file(config.runtime_model)
    if actual_hash != config.expected_model_sha256:
        raise ValueError("runtime model SHA256 does not match the governed pretrained weight")
    if config.runtime_model == config.rejected_model:
        raise ValueError("rejected Stage 17 model cannot be used as runtime")
    if not config.input_video.is_file():
        raise FileNotFoundError(config.input_video)
    input_hash = sha256_file(config.input_video)
    if input_hash != config.expected_input_sha256:
        raise ValueError("input video SHA256 does not match the governed aerial clip")
    if not config.scene_config.is_file():
        raise FileNotFoundError(config.scene_config)
    return {
        "runtime_model": config.runtime_model.relative_to(config.project_root).as_posix(),
        "runtime_model_sha256": actual_hash,
        "input_video_sha256": input_hash,
        "rejected_model_used": 0,
        "training_runs": 0,
    }


def experiment_output_directory(output_root: Path, experiment_name: str) -> Path:
    if not SAFE_EXPERIMENT_NAME.fullmatch(experiment_name):
        raise ValueError("experiment name is unsafe")
    root = Path(output_root).resolve()
    destination = (root / experiment_name).resolve()
    if destination.parent != root:
        raise ValueError("experiment output must be isolated under the output root")
    return destination


def normalized_bbox_area(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    if frame_width <= 0 or frame_height <= 0 or x2 < x1 or y2 < y1:
        raise ValueError("bbox and frame dimensions must be valid")
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (frame_width * frame_height)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def confidence_distribution(values: Sequence[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "count": len(clean),
        "mean": round(mean(clean), 6),
        "median": round(median(clean), 6),
        "p10": round(_percentile(clean, 0.10), 6),
        "p25": round(_percentile(clean, 0.25), 6),
        "p75": round(_percentile(clean, 0.75), 6),
        "p90": round(_percentile(clean, 0.90), 6),
    }


def aggregate_runtime_metrics(
    detections: Sequence[Mapping[str, object]],
    tracks: Sequence[Mapping[str, object]],
    *,
    total_frames: int,
    processing_seconds: float,
    small_area_threshold: float,
    low_confidence_band: tuple[float, float],
) -> dict[str, object]:
    """Aggregate descriptive observations; none are formal accuracy metrics."""
    class_counts = Counter(str(row["class_name"]) for row in detections)
    small_class_counts = Counter(
        str(row["class_name"])
        for row in detections
        if str(row["class_name"]) in SMALL_VEHICLE_CLASSES
        and float(row["normalized_bbox_area"]) < small_area_threshold
    )
    confidence_by_class: dict[str, list[float]] = defaultdict(list)
    for row in detections:
        confidence_by_class[str(row["class_name"])].append(float(row["confidence"]))
    all_confidences = [float(row["confidence"]) for row in detections]
    small_confidences = [
        float(row["confidence"])
        for row in detections
        if str(row["class_name"]) in SMALL_VEHICLE_CLASSES
        and float(row["normalized_bbox_area"]) < small_area_threshold
    ]
    low, high = low_confidence_band
    track_frames: dict[int, set[int]] = defaultdict(set)
    for row in tracks:
        track_frames[int(row["track_id"])].add(int(row["frame_index"]))
    gap_candidates = sum(
        current - previous > 1
        for frames in track_frames.values()
        for previous, current in zip(sorted(frames), sorted(frames)[1:])
    )
    return {
        "total_frames": int(total_frames),
        "processing_seconds": round(float(processing_seconds), 6),
        "processing_fps": round(total_frames / processing_seconds, 3) if total_frames > 0 and processing_seconds > 0 else 0.0,
        "total_detection_observations": len(detections),
        "per_class_detection_observations": dict(sorted(class_counts.items())),
        "small_vehicle_detection_observations": sum(small_class_counts.values()),
        "small_vehicle_per_class_observations": dict(sorted(small_class_counts.items())),
        "low_confidence_015_025_observations": sum(low <= value < high for value in all_confidences),
        "confidence_distribution": confidence_distribution(all_confidences),
        "per_class_confidence_distribution": {
            name: confidence_distribution(values) for name, values in sorted(confidence_by_class.items())
        },
        "small_vehicle_confidence_distribution": confidence_distribution(small_confidences),
        "tracking_observations": len(tracks),
        "diagnostic_unique_track_ids": len(track_frames),
        "frame_gap_fragmentation_candidates": gap_candidates,
    }


def render_comparison_report(
    config: DiagnosticConfig,
    summaries: Sequence[Mapping[str, object]],
    model_sha256: str,
) -> str:
    rows = [
        "| Config | imgsz | conf | Detection obs | Small vehicle obs | Track IDs | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        rows.append(
            f"| {item['experiment']} | {item['imgsz']} | {float(item['conf']):.2f} | "
            f"{item['total_detection_observations']} | {item['small_vehicle_detection_observations']} | "
            f"{item['diagnostic_unique_track_ids']} | {float(item['processing_fps']):.3f} |"
        )
    return "\n".join((
        "# Stage 21.1A Aerial Small-Object Runtime Diagnostic",
        "",
        "## Governance",
        "",
        f"- Runtime model: `{APPROVED_RUNTIME_MODEL.as_posix()}`",
        f"- Runtime model SHA256: `{model_sha256}`",
        "- Stage 17 rejected model usage: `0`",
        "- Training runs: `0`",
        f"- Input: `{config.input_video.relative_to(config.project_root).as_posix()}`",
        f"- Input SHA256: `{config.expected_input_sha256}`",
        f"- Inclusive frame range: `{config.start_frame}–{config.end_frame}`",
        "- Fixed tracker: `bytetrack.yaml`",
        "- Fixed trajectory: history `30`, minimum displacement `5.0` image pixels",
        "",
        "## Automated comparison",
        "",
        *rows,
        "",
        "Detection observations are repeated per-frame occurrences. Track IDs are video-scoped diagnostics, not formal traffic counts.",
        "",
        "**No detection GT → no formal accuracy claim. Precision, Recall, and mAP are not computed.**",
        "",
        "Automated counts cannot determine visually obvious misses or false positives. Final runtime selection requires checking the synchronized comparison frames and videos.",
        "",
        "`MANUAL_VISUAL_REVIEW_REQUIRED`",
        "",
    ))
