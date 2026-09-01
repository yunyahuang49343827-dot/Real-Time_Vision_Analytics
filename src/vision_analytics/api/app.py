"""Stage 20 FastAPI job-based video analytics service."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from vision_analytics.api.config import APPROVED_RUNTIME_MODEL, ApiConfig, load_api_config
from vision_analytics.api.schemas import (
    ErrorDetail,
    ErrorResponse,
    EventResponse,
    HealthResponse,
    JobCreateResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    AnalysisMode,
)
from vision_analytics.services.jobs import JobManager, PipelineRunner
from vision_analytics.services.pipeline import ExistingAnalyticsPipeline
from vision_analytics.video.metadata import profile_video

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "api.yaml"
SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code; self.code = code; self.message = message
        super().__init__(message)


def _status_response(record: object) -> JobStatusResponse:
    error = None
    if record.error_code:
        error = ErrorDetail(code=record.error_code, message=record.error_message or "Job failed")
    return JobStatusResponse(
        job_id=record.job_id, status=record.status, progress=record.progress,
        created_at=record.created_at, started_at=record.started_at,
        completed_at=record.completed_at, error=error,
        analysis_mode=record.analysis_mode,
        processed_frames=record.processed_frames,
        total_frames=record.total_frames,
    )


def _job_or_404(manager: JobManager, job_id: str):
    try:
        return manager.get(job_id)
    except KeyError as exc:
        raise ApiError(404, "JOB_NOT_FOUND", "Unknown job_id") from exc


def _safe_job_file(job_directory: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ApiError(404, "ARTIFACT_NOT_FOUND", "Artifact path is invalid")
    resolved = (job_directory / relative).resolve()
    if not resolved.is_relative_to(job_directory.resolve()):
        raise ApiError(404, "ARTIFACT_NOT_FOUND", "Artifact path escapes job directory")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=False)
    size = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ApiError(413, "UPLOAD_TOO_LARGE", "Uploaded video exceeds configured size limit")
            handle.write(chunk)
    return size


def create_app(*, config: ApiConfig | None = None, runner: PipelineRunner | None = None) -> FastAPI:
    active_config = config or load_api_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    active_runner = runner or ExistingAnalyticsPipeline(active_config)
    manager = JobManager(
        active_config.job_output_directory, active_runner,
        worker_threads=active_config.worker_threads,
    )
    runtime_model_sha256 = _sha256_file(active_config.runtime_model)
    health_profiles = {
        name: {
            "imgsz": profile.imgsz,
            "confidence_threshold": profile.confidence_threshold,
        }
        for name, profile in active_config.runtime_profiles.items()
    }
    health_profiles.setdefault("standard", {
        "imgsz": active_config.imgsz,
        "confidence_threshold": active_config.confidence_threshold,
    })

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown(wait=True)

    application = FastAPI(
        title="Real-Time Vision Analytics API", version="20.0.0", lifespan=lifespan,
        description="Local job-based orchestration over the existing vision analytics pipeline.",
    )
    application.state.job_manager = manager
    application.state.api_config = active_config
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_config.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        payload = ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message))
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(error=ErrorDetail(
            code="REQUEST_VALIDATION_ERROR", message="Request does not match the API contract",
        ))
        return JSONResponse(status_code=422, content=payload.model_dump())

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", service="vision-analytics",
            runtime_model=APPROVED_RUNTIME_MODEL.as_posix(),
            runtime_model_sha256=runtime_model_sha256,
            device=active_config.device,
            runtime_profiles=health_profiles,
        )

    @application.post(
        "/jobs", response_model=JobCreateResponse, status_code=202,
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    async def create_job(
        video: UploadFile = File(...),
        analysis_mode: AnalysisMode = Form(AnalysisMode.STANDARD),
    ) -> JobCreateResponse:
        original = Path(video.filename or "").name
        suffix = Path(original).suffix.lower()
        if suffix not in active_config.supported_extensions:
            raise ApiError(400, "UNSUPPORTED_EXTENSION", "Unsupported video extension")
        job_id = manager.new_job_id()
        staging = (active_config.upload_directory / job_id).resolve()
        if not staging.is_relative_to(active_config.upload_directory.resolve()):
            raise ApiError(400, "INVALID_UPLOAD_PATH", "Generated upload path is invalid")
        staged_file = staging / f"input{suffix}"
        try:
            size = await _save_upload(video, staged_file, active_config.max_upload_size_bytes)
            if size == 0:
                raise ApiError(400, "EMPTY_UPLOAD", "Uploaded video is empty")
            source_id = active_config.source_for_analysis_mode(analysis_mode.value)
            metadata = profile_video(staged_file, video_id=job_id, source_id=source_id)
            if metadata["validation_status"] == "FAIL":
                raise ApiError(400, "INVALID_VIDEO", "OpenCV could not validate the uploaded video")
            destination = active_config.job_output_directory / job_id / "input" / f"input{suffix}"
            record = manager.create(
                job_id, destination, analysis_mode=analysis_mode.value, source_id=source_id,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_file.replace(destination)
            shutil.rmtree(staging)
            manager.submit(job_id)
            return JobCreateResponse(job_id=record.job_id, status=JobStatus.CREATED)
        except ApiError:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        finally:
            await video.close()

    @application.get("/jobs/{job_id}", response_model=JobStatusResponse,
                     responses={404: {"model": ErrorResponse}})
    def job_status(job_id: str) -> JobStatusResponse:
        return _status_response(_job_or_404(manager, job_id))

    @application.get("/jobs/{job_id}/results", response_model=JobResultResponse,
                     responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
    def job_results(job_id: str) -> JobResultResponse:
        record = _job_or_404(manager, job_id)
        if record.status is not JobStatus.COMPLETED:
            raise ApiError(409, "JOB_NOT_COMPLETED", f"Job status is {record.status.value}")
        result_path = Path(record.output_directory) / "result.json"
        if not result_path.is_file():
            raise ApiError(404, "RESULT_ARTIFACT_MISSING", "Completed job result artifact is missing")
        try:
            return JobResultResponse.model_validate_json(result_path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ApiError(500, "RESULT_ARTIFACT_INVALID", "Result artifact does not match API schema") from exc

    @application.get("/jobs/{job_id}/events", response_model=list[EventResponse],
                     responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
    def job_events(job_id: str) -> list[EventResponse]:
        record = _job_or_404(manager, job_id)
        if record.status is not JobStatus.COMPLETED:
            raise ApiError(409, "JOB_NOT_COMPLETED", f"Job status is {record.status.value}")
        path = Path(record.output_directory) / "events.csv"
        if not path.is_file():
            raise ApiError(404, "EVENT_ARTIFACT_MISSING", "Event artifact is missing")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [EventResponse.model_validate({
            **row,
            "frame_index": int(row["frame_index"]),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "track_id": int(row["track_id"]) if row["track_id"] else None,
            "secondary_track_id": int(row["secondary_track_id"]) if row["secondary_track_id"] else None,
            "class_name": row["class_name"] or None,
            "secondary_class_name": row["secondary_class_name"] or None,
            "zone_id": row["zone_id"] or None, "line_id": row["line_id"] or None,
            "evidence_path": row.get("evidence_path") or None,
        }) for row in rows]

    @application.get(
        "/jobs/{job_id}/artifacts/{artifact_key}", response_class=FileResponse,
        responses={
            200: {"content": {"application/octet-stream": {}}},
            404: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
        },
    )
    def job_artifact(job_id: str, artifact_key: str):
        record = _job_or_404(manager, job_id)
        if record.status is not JobStatus.COMPLETED:
            raise ApiError(409, "JOB_NOT_COMPLETED", f"Job status is {record.status.value}")
        result_path = Path(record.output_directory) / "result.json"
        if not result_path.is_file():
            raise ApiError(404, "RESULT_ARTIFACT_MISSING", "Completed job result artifact is missing")
        try:
            result = JobResultResponse.model_validate_json(result_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ApiError(500, "RESULT_ARTIFACT_INVALID", "Result artifact does not match API schema") from exc
        if artifact_key not in type(result.artifacts).model_fields:
            raise ApiError(404, "ARTIFACT_KEY_NOT_FOUND", "Unknown artifact key")
        relative = getattr(result.artifacts, artifact_key)
        if not relative:
            raise ApiError(404, "ARTIFACT_NOT_AVAILABLE", "Artifact is not available for this job")
        path = _safe_job_file(Path(record.output_directory), relative)
        if not path.is_file():
            raise ApiError(404, "ARTIFACT_NOT_FOUND", "Artifact file is missing")
        video_keys = {
            "processed_video", "processed_raw_video", "processed_browser_video",
            "tracking_raw_video", "tracking_browser_video",
            "heatmap_raw_video", "heatmap_browser_video",
        }
        media_type = "video/mp4" if artifact_key in video_keys else "text/csv"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @application.get(
        "/jobs/{job_id}/evidence/{event_id}", response_class=FileResponse,
        responses={
            200: {"content": {"image/jpeg": {}}},
            404: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
        },
    )
    def job_evidence(job_id: str, event_id: str):
        record = _job_or_404(manager, job_id)
        if record.status is not JobStatus.COMPLETED:
            raise ApiError(409, "JOB_NOT_COMPLETED", f"Job status is {record.status.value}")
        if not SAFE_EVENT_ID.fullmatch(event_id):
            raise ApiError(404, "INVALID_EVENT_ID", "event_id contains unsafe characters")
        events = job_events(job_id)
        event = next((item for item in events if item.event_id == event_id), None)
        if event is None:
            raise ApiError(404, "EVENT_NOT_FOUND", "Unknown event_id for this job")
        if not event.evidence_path:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "Event has no evidence snapshot")
        path = _safe_job_file(Path(record.output_directory), event.evidence_path)
        if not path.is_file() or path.stem != event_id:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "Evidence snapshot is missing")
        return FileResponse(path, media_type="image/jpeg", filename=f"{event_id}.jpg")

    return application


app = create_app()
