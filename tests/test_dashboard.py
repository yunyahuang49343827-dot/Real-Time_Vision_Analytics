from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.dashboard.api_client import (  # noqa: E402
    DashboardApiError,
    VisionAnalyticsApiClient,
    load_dashboard_config,
)
from vision_analytics.dashboard.formatting import (  # noqa: E402
    PROXIMITY_WORDING,
    SUPPORTED_STATUSES,
    WRONG_WAY_REVIEW_WORDING,
    event_interpretation,
    event_table_rows,
    format_timestamp,
    normalize_progress,
    status_label,
)


class FakeResponse:
    def __init__(self, status_code: int = 200, *, payload=None, content: bytes = b"data",
                 json_error: bool = False) -> None:
        self.status_code = status_code; self.payload = payload
        self.content = content; self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError("invalid json")
        return self.payload


class FakeSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses); self.calls = []

    def request(self, method, url, timeout, **kwargs):
        self.calls.append((method, url, timeout, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def config():
    return load_dashboard_config(PROJECT_ROOT / "configs/dashboard.yaml")


def job_payload(status: str = "COMPLETED") -> dict[str, object]:
    return {
        "job_id": "job-1", "status": status, "progress": 1.0,
        "created_at": "2026-09-01T00:00:00Z", "started_at": "2026-09-01T00:00:01Z",
        "completed_at": "2026-09-01T00:00:02Z", "error": None,
    }


def result_payload() -> dict[str, object]:
    return {
        "job_id": "job-1",
        "video_metadata": {
            "filename": "input.mp4", "width": 1920, "height": 1080, "fps": 30,
            "frame_count": 300, "duration_seconds": 10, "codec": "h264",
            "validation_status": "PASS",
        },
        "traffic_analytics": {"total_line_crossing_count": 3, "density": "LOW"},
        "event_summary": [{
            "event_type": "WRONG_WAY", "severity": "CRITICAL",
            "status": "REVIEW_REQUIRED", "count": 1,
        }],
        "artifacts": {
            "processed_video": "processed_video.mp4", "events_csv": "events.csv",
            "evidence_manifest": "evidence_manifest.csv",
            "traffic_summary_csv": "traffic_summary.csv",
            "class_distribution_csv": "class_distribution.csv",
        },
    }


def event_payload() -> dict[str, object]:
    return {
        "event_id": "evt-1", "video_id": "job-1", "source_id": "scene",
        "event_type": "PROXIMITY_WARNING", "frame_index": 4, "timestamp_seconds": 0.4,
        "track_id": 1, "secondary_track_id": 2, "class_name": "person",
        "secondary_class_name": "motorcycle", "zone_id": "mixed", "line_id": None,
        "severity": "WARNING", "status": "REVIEW_REQUIRED",
        "rule_source": "spatial.proximity", "rule_value": "normalized_distance=0.01",
        "threshold": "trigger<=0.02", "evidence_path": "evidence/evt-1.jpg",
    }


def test_api_client_health_and_create_job() -> None:
    session = FakeSession(
        FakeResponse(payload={"status": "ok", "service": "vision-analytics",
                              "runtime_model": "models/pretrained/yolo26n.pt"}),
        FakeResponse(status_code=202, payload={"job_id": "job-1", "status": "CREATED"}),
    )
    client = VisionAnalyticsApiClient(config(), session=session)
    assert client.health().status == "ok"
    created = client.create_job(filename="../traffic.mp4", content=b"video", content_type="video/mp4")
    assert created.job_id == "job-1"
    files = session.calls[1][3]["files"]
    assert files["video"][0] == "traffic.mp4"


def test_api_client_job_results_events_evidence_and_artifact() -> None:
    session = FakeSession(
        FakeResponse(payload=job_payload()), FakeResponse(payload=result_payload()),
        FakeResponse(payload=[event_payload()]), FakeResponse(content=b"jpeg"),
        FakeResponse(content=b"mp4"),
    )
    client = VisionAnalyticsApiClient(config(), session=session)
    assert client.get_job("job-1").status.value == "COMPLETED"
    assert client.get_results("job-1").traffic_analytics.total_line_crossing_count == 3
    assert client.get_events("job-1")[0].event_id == "evt-1"
    assert client.get_evidence("job-1", "evt-1") == b"jpeg"
    assert client.get_artifact("job-1", "processed_video") == b"mp4"


def test_api_error_and_malformed_response_are_user_safe() -> None:
    error = FakeResponse(status_code=409, payload={
        "error": {"code": "JOB_NOT_COMPLETED", "message": "Job is processing"},
    })
    client = VisionAnalyticsApiClient(config(), session=FakeSession(error))
    with pytest.raises(DashboardApiError) as caught:
        client.get_results("job-1")
    assert caught.value.code == "JOB_NOT_COMPLETED"
    assert "Traceback" not in caught.value.message

    malformed = VisionAnalyticsApiClient(
        config(), session=FakeSession(FakeResponse(payload=None, json_error=True)),
    )
    with pytest.raises(DashboardApiError, match="invalid JSON"):
        malformed.health()

    wrong_schema = VisionAnalyticsApiClient(
        config(), session=FakeSession(FakeResponse(payload={"status": "ok"})),
    )
    with pytest.raises(DashboardApiError, match="response is malformed"):
        wrong_schema.health()


@pytest.mark.parametrize(
    ("exception", "code"),
    [(requests.Timeout("slow"), "BACKEND_TIMEOUT"),
     (requests.ConnectionError("offline"), "BACKEND_UNAVAILABLE")],
)
def test_timeout_and_offline_handling(exception: Exception, code: str) -> None:
    client = VisionAnalyticsApiClient(config(), session=FakeSession(exception))
    with pytest.raises(DashboardApiError) as caught:
        client.health()
    assert caught.value.code == code


def test_progress_timestamp_and_supported_status_formatting() -> None:
    assert normalize_progress(-1) == 0
    assert normalize_progress(0.456) == 46
    assert normalize_progress(2) == 100
    assert normalize_progress("bad") == 0
    assert format_timestamp(65.5) == "00:01:05.50"
    assert format_timestamp(-1) == "—"
    assert SUPPORTED_STATUSES == {"CREATED", "PROCESSING", "COMPLETED", "FAILED"}
    for status in SUPPORTED_STATUSES:
        assert status_label(status) != "UNKNOWN"
    assert status_label("OTHER") == "UNKNOWN"


def test_event_formatting_uses_governed_interpretation_wording() -> None:
    assert event_interpretation("PROXIMITY_WARNING", "REVIEW_REQUIRED") == PROXIMITY_WORDING
    assert event_interpretation("WRONG_WAY", "REVIEW_REQUIRED") == WRONG_WAY_REVIEW_WORDING
    combined = f"{PROXIMITY_WORDING} {WRONG_WAY_REVIEW_WORDING}".lower()
    for unsafe in ("collision risk", "near-miss probability", "physical distance",
                   "confirmed violation", "accident prediction"):
        assert unsafe not in combined
    row = event_table_rows([event_payload()])[0]
    assert row["frame_id"] == 4 and row["primary_track_id"] == 1
    assert row["timestamp"] == "00:00:00.40"


def test_dashboard_extensions_match_stage20_and_no_direct_ai_runtime() -> None:
    dashboard = config()
    api = yaml.safe_load((PROJECT_ROOT / "configs/api.yaml").read_text(encoding="utf-8"))
    assert set(dashboard.supported_extensions) == {
        str(value).lstrip(".") for value in api["supported_extensions"]
    }
    dashboard_root = PROJECT_ROOT / "src/vision_analytics/dashboard"
    source = "\n".join(path.read_text(encoding="utf-8") for path in dashboard_root.glob("*.py"))
    assert "ultralytics" not in source.lower()
    assert "StatefulByteTracker" not in source
    assert "models/finetuned/stage17/best.pt" not in source
    assert "training" not in source.lower()
