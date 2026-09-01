# React B Dashboard

## API contract audit

The existing FastAPI contract is sufficient for React B. The frontend reads only:

- `GET /jobs/{job_id}/results` for governed KPI summaries and artifact references;
- `GET /jobs/{job_id}/events` for the event timeline and review workspace;
- `GET /jobs/{job_id}/artifacts/{artifact_key}` for existing tracking, heatmap,
  and analytics artifacts; and
- `GET /jobs/{job_id}/evidence/{event_id}` for an existing Evidence Snapshot.

React B does not add or modify a backend endpoint or schema. Navigation and video
mode changes are frontend state changes. They never submit `POST /jobs`, rerun
inference, or regenerate a Heatmap.

## Overview semantics

- **通過計數線**: `traffic_analytics.total_line_crossing_count`.
- **需關注事件**: event-summary rows whose severity is `WARNING` or
  `CRITICAL`, or whose status is `REVIEW_REQUIRED`. Ordinary `INFO` events are not
  included unless their status explicitly requires review.
- **主要車種**: the non-person class with the largest track-based line-crossing
  count in `class_distribution.csv`.
- **車流高峰區間**: the backend-provided peak interval start and end.

The analysis video remains the main visual surface. The selector switches between
`tracking_browser_video` and `heatmap_browser_video`. It never creates a new Job.

## Analytics and event review

Traffic charts map the existing class, direction, and 10-second traffic CSV
artifacts. Peak Zone occupancy uses the reliable summary exposed by the backend;
React does not rerun the CV pipeline to invent an occupancy time series. Raw rows
remain available only in collapsed details.

The event workspace filters by severity/status, then loads the selected event's
Evidence Snapshot. A missing or unreadable image produces a Traditional Chinese
empty-state warning rather than an application error.

## Interpretation boundaries

- Proximity is image-space proximity, not physical distance or collision risk.
- Wrong-way is a rule candidate requiring manual confirmation.
- Line crossing is a virtual-line count, not a complete traffic census.
- Heatmap is image-space traffic activity, not road density or incident risk.
