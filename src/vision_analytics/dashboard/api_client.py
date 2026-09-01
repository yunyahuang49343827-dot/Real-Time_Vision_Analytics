"""Single typed HTTP client for the Stage 20 FastAPI service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

import requests
import yaml
from pydantic import ValidationError

from vision_analytics.api.schemas import (
    EventResponse,
    HealthResponse,
    JobCreateResponse,
    JobResultResponse,
    JobStatusResponse,
)


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    base_url: str
    timeout_seconds: float
    poll_interval_seconds: float
    supported_extensions: tuple[str, ...]
    page_title: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) URL")
        if self.timeout_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("timeout and poll interval must be positive")
        if not self.supported_extensions or any(
            not item or item.startswith(".") for item in self.supported_extensions
        ):
            raise ValueError("supported_extensions must omit leading dots")
        if not self.page_title.strip():
            raise ValueError("page_title is required")


def load_dashboard_config(path: Path) -> DashboardConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("dashboard config must be a mapping")
    return DashboardConfig(
        base_url=str(payload["base_url"]).rstrip("/"),
        timeout_seconds=float(payload["timeout_seconds"]),
        poll_interval_seconds=float(payload["poll_interval_seconds"]),
        supported_extensions=tuple(str(value).lower() for value in payload["supported_extensions"]),
        page_title=str(payload["page_title"]),
    )


class DashboardApiError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        self.code = code; self.message = message; self.status_code = status_code
        super().__init__(message)


class VisionAnalyticsApiClient:
    def __init__(self, config: DashboardConfig, *, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def health(self) -> HealthResponse:
        return self._validate(HealthResponse, self._json("GET", "/health"))

    def create_job(self, *, filename: str, content: bytes, content_type: str) -> JobCreateResponse:
        return self._validate(JobCreateResponse, self._json(
            "POST", "/jobs", files={"video": (Path(filename).name, content, content_type)},
        ))

    def get_job(self, job_id: str) -> JobStatusResponse:
        return self._validate(JobStatusResponse, self._json("GET", f"/jobs/{job_id}"))

    def get_results(self, job_id: str) -> JobResultResponse:
        return self._validate(JobResultResponse, self._json("GET", f"/jobs/{job_id}/results"))

    def get_events(self, job_id: str) -> list[EventResponse]:
        payload = self._json("GET", f"/jobs/{job_id}/events")
        if not isinstance(payload, list):
            raise DashboardApiError("MALFORMED_RESPONSE", "Backend events response is malformed")
        try:
            return [EventResponse.model_validate(item) for item in payload]
        except ValidationError as exc:
            raise DashboardApiError("MALFORMED_RESPONSE", "Backend events response is malformed") from exc

    def get_evidence(self, job_id: str, event_id: str) -> bytes:
        return self._bytes("GET", f"/jobs/{job_id}/evidence/{event_id}")

    def get_artifact(self, job_id: str, artifact_key: str) -> bytes:
        return self._bytes("GET", f"/jobs/{job_id}/artifacts/{artifact_key}")

    def _request(self, method: str, path: str, **kwargs):
        try:
            return self.session.request(
                method, f"{self.config.base_url}{path}",
                timeout=self.config.timeout_seconds, **kwargs,
            )
        except requests.Timeout as exc:
            raise DashboardApiError("BACKEND_TIMEOUT", "Backend request timed out") from exc
        except requests.RequestException as exc:
            raise DashboardApiError("BACKEND_UNAVAILABLE", "Backend unavailable") from exc

    @staticmethod
    def _validate(model, payload):
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise DashboardApiError("MALFORMED_RESPONSE", "Backend response is malformed") from exc

    def _check(self, response):
        if response.status_code < 400:
            return response
        code, message = "BACKEND_ERROR", "Backend request failed"
        try:
            payload = response.json()
            detail = payload.get("error", {}) if isinstance(payload, Mapping) else {}
            code = str(detail.get("code") or code)
            message = str(detail.get("message") or message)
        except ValueError:
            pass
        raise DashboardApiError(code, message, status_code=response.status_code)

    def _json(self, method: str, path: str, **kwargs):
        response = self._check(self._request(method, path, **kwargs))
        try:
            return response.json()
        except ValueError as exc:
            raise DashboardApiError("MALFORMED_RESPONSE", "Backend returned invalid JSON") from exc

    def _bytes(self, method: str, path: str) -> bytes:
        response = self._check(self._request(method, path))
        if not response.content:
            raise DashboardApiError("EMPTY_ARTIFACT", "Backend returned an empty artifact")
        return bytes(response.content)
