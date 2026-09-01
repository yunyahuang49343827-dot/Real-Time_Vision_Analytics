"""Validated Stage 20 API configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

APPROVED_RUNTIME_MODEL = Path("models/pretrained/yolo26n.pt")
REJECTED_CANDIDATE = Path("models/finetuned/stage17/best.pt")


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    imgsz: int
    confidence_threshold: float

    def __post_init__(self) -> None:
        if self.imgsz <= 0 or not 0 < self.confidence_threshold <= 1:
            raise ValueError("invalid runtime profile")


@dataclass(frozen=True, slots=True)
class ApiConfig:
    project_root: Path
    upload_directory: Path
    job_output_directory: Path
    supported_extensions: frozenset[str]
    max_upload_size_bytes: int
    worker_threads: int
    runtime_model: Path
    scene_config: Path
    default_scene_source_id: str
    device: str
    imgsz: int
    confidence_threshold: float
    tracker: str
    runtime_profiles: Mapping[str, RuntimeProfile] = field(default_factory=dict)
    scene_runtime_profiles: Mapping[str, str] = field(default_factory=dict)
    heatmap_classes: frozenset[str] = frozenset({"car", "motorcycle", "bus", "truck"})

    def __post_init__(self) -> None:
        root = self.project_root.resolve()
        for path in (self.upload_directory, self.job_output_directory, self.runtime_model, self.scene_config):
            if not path.resolve().is_relative_to(root):
                raise ValueError("configured paths must remain inside project_root")
        if self.runtime_model.resolve() != (root / APPROVED_RUNTIME_MODEL).resolve():
            raise ValueError("Stage 20 runtime model must be models/pretrained/yolo26n.pt")
        if self.runtime_model.resolve() == (root / REJECTED_CANDIDATE).resolve():
            raise ValueError("rejected Stage 17 candidate is forbidden")
        if not self.supported_extensions or any(not item.startswith(".") for item in self.supported_extensions):
            raise ValueError("supported_extensions must contain dotted suffixes")
        if self.max_upload_size_bytes <= 0 or self.worker_threads <= 0:
            raise ValueError("upload limit and worker_threads must be positive")
        if self.device != "mps" or self.imgsz <= 0 or not 0 < self.confidence_threshold <= 1:
            raise ValueError("invalid runtime inference configuration")
        if any(name not in self.runtime_profiles for name in self.scene_runtime_profiles.values()):
            raise ValueError("scene references an unknown runtime profile")

    def runtime_profile_for(self, source_id: str) -> RuntimeProfile:
        name = self.scene_runtime_profiles.get(source_id, "standard")
        return self.runtime_profiles.get(name, RuntimeProfile(self.imgsz, self.confidence_threshold))


def _inside(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError("API paths must be project-relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("API path escapes project root")
    return resolved


def load_api_config(path: Path, *, project_root: Path | None = None) -> ApiConfig:
    config_path = Path(path).resolve()
    root = Path(project_root).resolve() if project_root else config_path.parents[1]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("API config must be a mapping")
    profile_payload = payload.get("runtime_profiles", {})
    runtime_profiles = {
        str(name): RuntimeProfile(
            imgsz=int(values["imgsz"]),
            confidence_threshold=float(values.get("confidence_threshold", values.get("conf"))),
        )
        for name, values in profile_payload.items()
    }
    heatmap = payload.get("visualization", {}).get("heatmap", {})
    return ApiConfig(
        project_root=root,
        upload_directory=_inside(root, payload["upload_directory"]),
        job_output_directory=_inside(root, payload["job_output_directory"]),
        supported_extensions=frozenset(str(value).lower() for value in payload["supported_extensions"]),
        max_upload_size_bytes=int(float(payload["max_upload_size_mb"]) * 1024 * 1024),
        worker_threads=int(payload.get("worker_threads", 1)),
        runtime_model=_inside(root, payload["runtime_model"]),
        scene_config=_inside(root, payload["scene_config"]),
        default_scene_source_id=str(payload["default_scene_source_id"]),
        device=str(payload.get("device", "mps")),
        imgsz=int(payload.get("imgsz", 640)),
        confidence_threshold=float(payload.get("confidence_threshold", 0.25)),
        tracker=str(payload.get("tracker", "bytetrack.yaml")),
        runtime_profiles=runtime_profiles,
        scene_runtime_profiles={
            str(source): str(profile)
            for source, profile in payload.get("scene_runtime_profiles", {}).items()
        },
        heatmap_classes=frozenset(str(value) for value in heatmap.get(
            "classes", ["car", "motorcycle", "bus", "truck"],
        )),
    )
