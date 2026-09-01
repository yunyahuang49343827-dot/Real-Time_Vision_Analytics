"""Stable public Pydantic contracts for the Stage 20 API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    CREATED = "CREATED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    service: str
    runtime_model: str


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: ErrorDetail | None = None


class VideoMetadataResponse(BaseModel):
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    validation_status: str


class TrafficAnalyticsResponse(BaseModel):
    total_line_crossing_count: int = 0
    person_crossing_count: int = 0
    motorized_vehicle_crossing_count: int = 0
    bicycle_crossing_count: int = 0
    peak_interval_start_seconds: float | None = None
    peak_interval_end_seconds: float | None = None
    peak_interval_count: int = 0
    zone_peak_occupancy: int = 0
    density: str | None = None
    reconciliation_status: str = "PASS"


class EventSummaryResponse(BaseModel):
    event_type: str
    severity: str
    status: str
    count: int


class ArtifactReferences(BaseModel):
    processed_video: str | None = None
    processed_raw_video: str | None = None
    processed_browser_video: str | None = None
    tracking_raw_video: str | None = None
    tracking_browser_video: str | None = None
    heatmap_raw_video: str | None = None
    heatmap_browser_video: str | None = None
    events_csv: str
    evidence_manifest: str
    traffic_summary_csv: str
    class_distribution_csv: str | None = None
    direction_distribution_csv: str | None = None
    traffic_over_time_csv: str | None = None
    event_summary_csv: str | None = None


class ResultWarning(BaseModel):
    code: str
    message: str


class JobResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    video_metadata: VideoMetadataResponse
    traffic_analytics: TrafficAnalyticsResponse
    event_summary: list[EventSummaryResponse]
    artifacts: ArtifactReferences
    warnings: list[ResultWarning] = Field(default_factory=list)


class EventResponse(BaseModel):
    event_id: str
    video_id: str
    source_id: str
    event_type: str
    frame_index: int
    timestamp_seconds: float
    track_id: int | None = None
    secondary_track_id: int | None = None
    class_name: str | None = None
    secondary_class_name: str | None = None
    zone_id: str | None = None
    line_id: str | None = None
    severity: str
    status: str
    rule_source: str
    rule_value: str
    threshold: str
    evidence_path: str | None = None
