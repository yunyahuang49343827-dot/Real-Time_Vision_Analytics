from __future__ import annotations

import csv
import shutil
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_analytics.api.app import create_app  # noqa: E402
from vision_analytics.api.config import APPROVED_RUNTIME_MODEL, ApiConfig  # noqa: E402
from vision_analytics.api.schemas import JobStatus  # noqa: E402
from vision_analytics.events.schema import EVENT_FIELDS  # noqa: E402
from vision_analytics.services.jobs import JobManager  # noqa: E402
from vision_analytics.services.pipeline import OpenCVPipelineSmokeRunner  # noqa: E402


def api_config(tmp_path: Path) -> ApiConfig:
    model = tmp_path / APPROVED_RUNTIME_MODEL
    model.parent.mkdir(parents=True); model.write_bytes(b"approved-pretrained")
    scene = tmp_path / "configs/scenes.yaml"
    scene.parent.mkdir(parents=True); scene.write_text("scenes: {}\n", encoding="utf-8")
    return ApiConfig(
        project_root=tmp_path, upload_directory=tmp_path / "outputs/api/uploads",
        job_output_directory=tmp_path / "outputs/api/jobs",
        supported_extensions=frozenset({".mp4", ".avi"}), max_upload_size_bytes=5_000_000,
        worker_threads=1, runtime_model=model, scene_config=scene,
        default_scene_source_id="scene", device="mps", imgsz=640,
        confidence_threshold=0.25, tracker="bytetrack.yaml",
    )


def synthetic_video(path: Path, frames: int = 3) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 60, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


class FakeRunner:
    def __init__(self, *, gate: threading.Event | None = None, fail: bool = False) -> None:
        self.gate = gate; self.fail = fail; self.calls: list[tuple[Path, Path, str]] = []

    def run(self, input_path: Path, output_directory: Path, job_id: str, progress_callback):
        self.calls.append((input_path, output_directory, job_id))
        progress_callback(0.25)
        if self.gate is not None:
            assert self.gate.wait(timeout=5)
        if self.fail:
            raise RuntimeError("controlled pipeline failure")
        evidence = output_directory / "evidence"; evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "evt-1.jpg").write_bytes(b"jpeg-evidence")
        event_row = {
            "event_id": "evt-1", "video_id": job_id, "source_id": "scene",
            "event_type": "WRONG_WAY", "frame_index": 2, "timestamp_seconds": 0.2,
            "track_id": 7, "secondary_track_id": "", "class_name": "car",
            "secondary_class_name": "", "zone_id": "zone", "line_id": "",
            "severity": "CRITICAL", "status": "REVIEW_REQUIRED",
            "rule_source": "spatial.direction", "rule_value": "LEFT",
            "threshold": "allowed=RIGHT", "evidence_path": "evidence/evt-1.jpg",
        }
        with (output_directory / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS); writer.writeheader(); writer.writerow(event_row)
        (output_directory / "evidence_manifest.csv").write_text("event_id\nevt-1\n", encoding="utf-8")
        (output_directory / "traffic_summary.csv").write_text("total_line_crossing_count\n1\n", encoding="utf-8")
        (output_directory / "processed_raw.mp4").write_bytes(b"raw-processed")
        (output_directory / "processed_browser.mp4").write_bytes(b"browser-processed")
        progress_callback(0.9)
        return {
            "job_id": job_id,
            "video_metadata": {
                "filename": input_path.name, "width": 64, "height": 48, "fps": 10.0,
                "frame_count": 3, "duration_seconds": 0.3, "codec": "MJPG",
                "validation_status": "PASS",
            },
            "traffic_analytics": {"total_line_crossing_count": 1},
            "event_summary": [{
                "event_type": "WRONG_WAY", "severity": "CRITICAL",
                "status": "REVIEW_REQUIRED", "count": 1,
            }],
            "artifacts": {
                "processed_video": "processed_browser.mp4",
                "processed_raw_video": "processed_raw.mp4",
                "processed_browser_video": "processed_browser.mp4",
                "events_csv": "events.csv",
                "evidence_manifest": "evidence_manifest.csv",
                "traffic_summary_csv": "traffic_summary.csv",
            },
        }


def wait_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        payload = client.get(f"/jobs/{job_id}").json()
        if payload["status"] in {"COMPLETED", "FAILED"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not reach terminal status")


def post_video(client: TestClient, video_bytes: bytes, filename: str = "traffic.avi"):
    return client.post("/jobs", files={"video": (filename, video_bytes, "video/x-msvideo")})


def test_health_and_swagger(tmp_path: Path) -> None:
    with TestClient(create_app(config=api_config(tmp_path), runner=FakeRunner())) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["runtime_model"] == "models/pretrained/yolo26n.pt"
        assert client.get("/docs").status_code == 200
        assert "/jobs" in client.get("/openapi.json").json()["paths"]


def test_create_job_is_non_blocking_and_uses_uuid(tmp_path: Path) -> None:
    gate = threading.Event(); runner = FakeRunner(gate=gate)
    video = synthetic_video(tmp_path / "source.avi")
    with TestClient(create_app(config=api_config(tmp_path), runner=runner)) as client:
        response = post_video(client, video)
        assert response.status_code == 202 and response.json()["status"] == "CREATED"
        UUID(response.json()["job_id"])
        assert client.get(f"/jobs/{response.json()['job_id']}").json()["status"] in {"CREATED", "PROCESSING"}
        gate.set()
        assert wait_terminal(client, response.json()["job_id"])["status"] == "COMPLETED"


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [("bad.txt", b"video", "UNSUPPORTED_EXTENSION"),
     ("empty.avi", b"", "EMPTY_UPLOAD"),
     ("invalid.avi", b"not-a-video", "INVALID_VIDEO")],
)
def test_upload_validation(tmp_path: Path, filename: str, content: bytes, code: str) -> None:
    with TestClient(create_app(config=api_config(tmp_path), runner=FakeRunner())) as client:
        response = post_video(client, content, filename)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == code


def test_missing_upload_uses_typed_validation_error(tmp_path: Path) -> None:
    with TestClient(create_app(config=api_config(tmp_path), runner=FakeRunner())) as client:
        response = client.post("/jobs")
        assert response.status_code == 422
        assert response.json() == {"error": {
            "code": "REQUEST_VALIDATION_ERROR",
            "message": "Request does not match the API contract",
        }}


def test_lifecycle_and_illegal_transition(tmp_path: Path) -> None:
    config = api_config(tmp_path); runner = FakeRunner()
    input_path = tmp_path / "input.avi"; input_path.write_bytes(b"x")
    manager = JobManager(config.job_output_directory, runner)
    record = manager.create(manager.new_job_id(), input_path)
    assert record.status is JobStatus.CREATED
    manager.transition(record.job_id, JobStatus.PROCESSING)
    with pytest.raises(ValueError, match="illegal job transition"):
        manager.transition(record.job_id, JobStatus.CREATED)
    manager.transition(record.job_id, JobStatus.COMPLETED)
    with pytest.raises(ValueError, match="illegal job transition"):
        manager.transition(record.job_id, JobStatus.FAILED)
    manager.shutdown()


def test_unknown_job_and_result_before_completion(tmp_path: Path) -> None:
    gate = threading.Event(); video = synthetic_video(tmp_path / "source.avi")
    with TestClient(create_app(config=api_config(tmp_path), runner=FakeRunner(gate=gate))) as client:
        assert client.get("/jobs/unknown").status_code == 404
        created = post_video(client, video).json()
        response = client.get(f"/jobs/{created['job_id']}/results")
        assert response.status_code == 409
        gate.set(); wait_terminal(client, created["job_id"])


def test_completed_results_events_and_evidence(tmp_path: Path) -> None:
    video = synthetic_video(tmp_path / "source.avi")
    with TestClient(create_app(config=api_config(tmp_path), runner=FakeRunner())) as client:
        job_id = post_video(client, video).json()["job_id"]
        status = wait_terminal(client, job_id)
        assert status["progress"] == 1.0 and status["error"] is None
        result = client.get(f"/jobs/{job_id}/results")
        assert result.status_code == 200
        assert result.json()["traffic_analytics"]["total_line_crossing_count"] == 1
        events = client.get(f"/jobs/{job_id}/events")
        assert events.status_code == 200 and events.json()[0]["event_id"] == "evt-1"
        evidence = client.get(f"/jobs/{job_id}/evidence/evt-1")
        assert evidence.status_code == 200 and evidence.content == b"jpeg-evidence"
        browser = client.get(f"/jobs/{job_id}/artifacts/processed_browser_video")
        assert browser.status_code == 200 and browser.content == b"browser-processed"
        assert browser.headers["content-type"] == "video/mp4"
        raw = client.get(f"/jobs/{job_id}/artifacts/processed_raw_video")
        assert raw.status_code == 200 and raw.content == b"raw-processed"
        assert client.get(f"/jobs/{job_id}/artifacts/../../job.json").status_code == 404
        assert client.get(f"/jobs/{job_id}/artifacts/not_a_key").status_code == 404
        assert client.get(f"/jobs/{job_id}/evidence/missing").status_code == 404


def test_failed_job_is_contained_without_traceback(tmp_path: Path) -> None:
    video = synthetic_video(tmp_path / "source.avi")
    with TestClient(create_app(config=api_config(tmp_path), runner=FakeRunner(fail=True))) as client:
        job_id = post_video(client, video).json()["job_id"]
        status = wait_terminal(client, job_id)
        assert status["status"] == "FAILED"
        assert status["error"]["code"] == "PIPELINE_FAILED"
        assert "Traceback" not in status["error"]["message"]
        assert client.get(f"/jobs/{job_id}/results").status_code == 409


def test_evidence_path_traversal_and_cross_job_isolation(tmp_path: Path) -> None:
    video = synthetic_video(tmp_path / "source.avi")
    config = api_config(tmp_path)
    with TestClient(create_app(config=config, runner=FakeRunner())) as client:
        first = post_video(client, video).json()["job_id"]
        second = post_video(client, video).json()["job_id"]
        wait_terminal(client, first); wait_terminal(client, second)
        assert first != second
        assert (config.job_output_directory / first).is_dir()
        assert (config.job_output_directory / second).is_dir()
        assert client.get(f"/jobs/{first}/evidence/%2e%2e%2fevt-1").status_code == 404
        assert client.get(f"/jobs/{first}/evidence/{second}").status_code == 404


def test_rejected_runtime_model_is_forbidden(tmp_path: Path) -> None:
    config = api_config(tmp_path)
    assert config.runtime_model.relative_to(config.project_root) == APPROVED_RUNTIME_MODEL
    with pytest.raises(ValueError, match="runtime model"):
        ApiConfig(
            project_root=tmp_path, upload_directory=config.upload_directory,
            job_output_directory=config.job_output_directory,
            supported_extensions=config.supported_extensions,
            max_upload_size_bytes=config.max_upload_size_bytes, worker_threads=1,
            runtime_model=tmp_path / "models/finetuned/stage17/best.pt",
            scene_config=config.scene_config, default_scene_source_id="scene", device="mps",
            imgsz=640, confidence_threshold=0.25, tracker="bytetrack.yaml",
        )


def test_small_service_integration_calls_existing_opencv_pipeline(tmp_path: Path) -> None:
    video = synthetic_video(tmp_path / "source.avi", frames=4)
    config = api_config(tmp_path)
    with TestClient(create_app(config=config, runner=OpenCVPipelineSmokeRunner())) as client:
        job_id = post_video(client, video).json()["job_id"]
        assert wait_terminal(client, job_id)["status"] == "COMPLETED"
        result = client.get(f"/jobs/{job_id}/results").json()
        output = config.job_output_directory / job_id / result["artifacts"]["processed_raw_video"]
        assert output.is_file() and output.stat().st_size > 0
        assert result["video_metadata"]["frame_count"] == 4
        if shutil.which("ffmpeg") is None:
            assert result["artifacts"]["processed_video"] is None
            assert result["artifacts"]["processed_browser_video"] is None
            assert result["warnings"][0]["code"] == "VIDEO_TRANSCODE_UNAVAILABLE"
