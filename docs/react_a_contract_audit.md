# React A Contract Audit

## Scope

React A uses the existing job API as the only analytics boundary. The browser
does not load a model, assign Track IDs, derive events, aggregate analytics, or
generate visualizations.

## Existing contracts retained

- `GET /health`
- `POST /jobs` multipart upload
- `GET /jobs/{job_id}` lifecycle polling
- `GET /jobs/{job_id}/results`
- `GET /jobs/{job_id}/events`
- governed artifact and evidence endpoints

## Minimal gaps addressed

Three orchestration-level gaps prevented the requested upload experience:

1. Job status exposed only normalized progress. It now also exposes
   `processed_frames` and `total_frames`.
2. Job creation could not select the governed standard or aerial runtime
   profile. Multipart `analysis_mode` accepts only `standard` or `aerial`, and
   the server resolves each value to a configured scene/source ID.
3. Browser requests from the local Vite development origin required a narrow
   CORS allowlist.

No detection, ByteTrack, trajectory, spatial rule, event, evidence, or traffic
analytics implementation was changed. Runtime model governance still rejects
the Stage 17 model and keeps `models/pretrained/yolo26n.pt`.

## Frontend boundary

TanStack Query owns health checks and status polling. The upload mutation sends
only the selected video and analysis mode. Polling stops at `COMPLETED` or
`FAILED`. Other sidebar areas remain unavailable until completion; their data
views are intentionally outside React A.
