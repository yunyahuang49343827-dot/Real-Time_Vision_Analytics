"""Thread-safe local job lifecycle and persistence."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from vision_analytics.api.schemas import JobStatus

ProgressCallback = Callable[[float], None]


class PipelineRunner(Protocol):
    def run(self, input_path: Path, output_directory: Path, job_id: str,
            progress_callback: ProgressCallback) -> dict[str, object]: ...


@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None
    completed_at: str | None
    input_video_path: str
    output_directory: str
    progress: float
    error_code: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class JobManager:
    _TRANSITIONS = {
        JobStatus.CREATED: frozenset({JobStatus.PROCESSING}),
        JobStatus.PROCESSING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED}),
        JobStatus.COMPLETED: frozenset(),
        JobStatus.FAILED: frozenset(),
    }

    def __init__(self, job_root: Path, runner: PipelineRunner, *, worker_threads: int = 1) -> None:
        self.job_root = Path(job_root).resolve()
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self._executor = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="vision-job")
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}

    @staticmethod
    def new_job_id() -> str:
        return str(uuid4())

    def create(self, job_id: str, input_path: Path) -> JobRecord:
        output = (self.job_root / job_id).resolve()
        if not output.is_relative_to(self.job_root) or output == self.job_root:
            raise ValueError("invalid job_id path")
        with self._lock:
            if job_id in self._jobs:
                raise ValueError("duplicate job_id")
            output.mkdir(parents=True, exist_ok=False)
            record = JobRecord(
                job_id=job_id, status=JobStatus.CREATED, created_at=_now(), started_at=None,
                completed_at=None, input_video_path=str(Path(input_path).resolve()),
                output_directory=str(output), progress=0.0, error_code=None, error_message=None,
            )
            self._jobs[job_id] = record
            self._persist(record)
            return record

    def submit(self, job_id: str) -> None:
        with self._lock:
            record = self.get(job_id)
            if record.status is not JobStatus.CREATED:
                raise ValueError("only CREATED jobs can be submitted")
            self._futures[job_id] = self._executor.submit(self._run, job_id)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(job_id) from exc

    def transition(self, job_id: str, target: JobStatus) -> JobRecord:
        with self._lock:
            record = self.get(job_id)
            if target not in self._TRANSITIONS[record.status]:
                raise ValueError(f"illegal job transition: {record.status.value} -> {target.value}")
            record.status = target
            if target is JobStatus.PROCESSING:
                record.started_at = _now()
            elif target in {JobStatus.COMPLETED, JobStatus.FAILED}:
                record.completed_at = _now()
                if target is JobStatus.COMPLETED:
                    record.progress = 1.0
            self._persist(record)
            return record

    def _progress(self, job_id: str, value: float) -> None:
        value = min(0.999, max(0.0, float(value)))
        with self._lock:
            record = self.get(job_id)
            if record.status is not JobStatus.PROCESSING or value < record.progress:
                return
            if value - record.progress < 0.01 and value < 0.99:
                return
            record.progress = value
            self._persist(record)

    def _run(self, job_id: str) -> None:
        try:
            record = self.transition(job_id, JobStatus.PROCESSING)
            result = self.runner.run(
                Path(record.input_video_path), Path(record.output_directory), job_id,
                lambda value: self._progress(job_id, value),
            )
            result_path = Path(record.output_directory) / "result.json"
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.transition(job_id, JobStatus.COMPLETED)
        except Exception as exc:  # job boundary: contain all pipeline failures
            with self._lock:
                record = self.get(job_id)
                record.error_code = "PIPELINE_FAILED"
                message = str(exc).replace("\n", " ").strip()
                record.error_message = (message[:500] or "Video analytics pipeline failed")
                if record.status is JobStatus.CREATED:
                    self.transition(job_id, JobStatus.PROCESSING)
                if record.status is JobStatus.PROCESSING:
                    self.transition(job_id, JobStatus.FAILED)

    def _persist(self, record: JobRecord) -> None:
        directory = Path(record.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(directory / "job.json")

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)


def _now() -> str:
    return datetime.now(UTC).isoformat()
