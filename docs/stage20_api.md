# Stage 20 FastAPI Job Service

Stage 20 exposes the existing vision analytics engines through a local,
non-blocking job API. It does not define new detection, tracking, event, or
traffic-analytics semantics.

## Run locally

```bash
.venv/bin/uvicorn vision_analytics.api.app:app --app-dir src --reload
```

Then open:

- Health: <http://127.0.0.1:8000/health>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Governed artifact download: `GET /jobs/{job_id}/artifacts/{artifact_key}`

Create a job:

```bash
curl -F "video=@/absolute/path/to/video.mp4" http://127.0.0.1:8000/jobs
```

Poll `GET /jobs/{job_id}` and request `/results`, `/events`, or an event
evidence JPG only after status becomes `COMPLETED`.

## Architecture and governance

The API streams an upload into a server-generated staging directory, validates
its suffix, size, OpenCV metadata, and first-frame decode, then creates an
isolated `outputs/api/jobs/{uuid}/` directory. A one-worker
`ThreadPoolExecutor` changes the job from `CREATED` to `PROCESSING` and invokes
the callable analytics service. Terminal states are `COMPLETED` or `FAILED`;
all other transitions are rejected.

The production runner composes the existing `StatefulByteTracker`,
`TrajectoryEngine`, line, zone, direction, dwell, proximity, `EventEngine`,
`EvidenceCapture`, and pandas traffic-analytics functions. The only shared
pipeline refactor is an optional frame progress callback in the existing
OpenCV `process_video` loop.

The runtime model is hard-gated to `models/pretrained/yolo26n.pt` with MPS,
`imgsz=640`, and confidence `0.25`. The rejected Stage 17 candidate and all V2
artifacts are forbidden by configuration validation.

Uploaded videos use the configured `default_scene_source_id` because spatial
rules require a known calibrated scene. Stage 20 defaults to the Taipei scene
configuration (`pexels_13258685`); changing it is an operator configuration
decision, never an arbitrary client filesystem parameter.

## Job artifacts

```text
outputs/api/jobs/{job_id}/
  job.json
  result.json
  input/input.<extension>
  processed_raw.mp4
  processed_browser.mp4     # only when FFmpeg delivery transcode succeeds
  video_metadata.json
  events.csv
  crossings.csv
  traffic_summary.csv
  class_distribution.csv
  direction_distribution.csv
  traffic_over_time.csv
  event_summary.csv
  evidence_manifest.csv
  evidence/{event_id}.jpg
```

The OpenCV artifact remains `processed_raw.mp4`. After the core pipeline
completes, the delivery layer looks up FFmpeg with `shutil.which("ffmpeg")` and
generates `processed_browser.mp4` using H.264 (`libx264`), `yuv420p`, no audio,
and MP4 `+faststart`. Both paths are server-generated and must resolve inside
the job directory. The `processed_video` compatibility reference aliases the
browser artifact only; it never silently falls back to the raw OpenCV file.

If FFmpeg is unavailable or conversion fails, the job can remain `COMPLETED`
and retain analytics, events, evidence, and the raw video. `result.json`
contains `VIDEO_TRANSCODE_UNAVAILABLE` or `VIDEO_TRANSCODE_FAILED`, while the
browser artifact references remain empty.

`outputs/api/` is already covered by the repository-wide `outputs/**` Git
ignore rule. Evidence lookup first resolves `event_id` from that job's own
unified events CSV, then verifies the stored relative path remains inside the
job directory. Python tracebacks are never returned through the API.
