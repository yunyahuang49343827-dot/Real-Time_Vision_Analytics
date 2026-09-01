# Stage 21 Streamlit Dashboard

The Stage 21 dashboard is a presentation layer over the Stage 20 FastAPI job
service. It never imports or invokes YOLO, ByteTrack, spatial rules, the event
engine, or traffic aggregation code.

## Launch

Terminal 1:

```bash
.venv/bin/uvicorn vision_analytics.api.app:app --app-dir src --reload
```

Terminal 2:

```bash
.venv/bin/streamlit run src/vision_analytics/dashboard/app.py
```

The dashboard URL defaults to <http://127.0.0.1:8501>. Backend URL, HTTP
timeout, polling interval, and upload extensions are centralized in
`configs/dashboard.yaml`.

## User flow

1. Dashboard checks `GET /health`. Submission remains disabled and the UI shows
   `Backend unavailable` when FastAPI cannot be reached.
2. The uploader displays filename and byte size. `Analyze Video` posts the file
   once to `POST /jobs` and stores the returned job ID in Streamlit session state.
3. While status is `CREATED` or `PROCESSING`, the dashboard polls at the
   configured interval and displays backend progress as 0–100%.
4. `COMPLETED` jobs load typed results and unified events. A `FAILED` job shows
   only the backend error code/message, never a traceback.
5. Processed MP4 and structured CSVs are downloaded through the governed
   `/jobs/{job_id}/artifacts/{artifact_key}` endpoint. The dashboard never reads
   a backend artifact path directly.
6. Evidence JPGs are requested only through the existing event-scoped evidence
   endpoint. Missing snapshots become warnings rather than application errors.
7. `New Analysis` clears frontend session state and the upload widget. It does
   not delete backend job artifacts.

## Dashboard sections

- Header and backend connection state
- Traffic video upload
- Job lifecycle and progress
- Overview KPIs from backend values only
- Video metadata and processed video
- Class, direction, and interval analytics from backend CSV artifacts
- Unified Event Schema review table
- Selected evidence preview

Interpretation copy deliberately describes proximity as an **image-space
proximity warning / review candidate** and wrong-way output as a **rule candidate
requiring human review**. Line crossing is identified as a track-based crossing
count, not a complete traffic census. No collision, physical-distance, accident,
or unique-vehicle claims are generated.

## Manual final integration

A short generated smoke clip is prepared at:

```text
outputs/api/smoke/stage21_short_traffic.mp4
```

It contains the first 90 frames (3 seconds) of the governed Highway runtime
video and is ignored by Git. Use it to manually exercise upload, lifecycle,
processed video, analytics, events, and evidence in the browser.

Implementation and automated startup checks do not substitute for visual browser
confirmation. Final state remains:

```text
MANUAL_INTEGRATION_TEST_REQUIRED
```
