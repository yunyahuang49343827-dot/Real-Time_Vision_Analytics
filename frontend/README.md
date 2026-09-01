# React Vision Analytics Dashboard

This React + TypeScript dashboard connects to the local FastAPI analytics service.
It includes the Traditional Chinese upload and job-lifecycle flow, completed-job
overview, tracking/heatmap artifact switching, traffic charts, event filtering,
and Evidence Snapshot review.

## Local development

Start FastAPI from the repository root:

```bash
.venv/bin/uvicorn vision_analytics.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Start Vite in another terminal:

```bash
cd frontend
npm install
npm run dev
```

The default API origin is `http://127.0.0.1:8000`. Copy `.env.example` to
`.env.local` only when an override is needed.

## Verification

```bash
npm test
npm run build
```
