# React Vision Analytics Dashboard

This is the React A frontend foundation for the local FastAPI analytics service.
It contains the Traditional Chinese upload and job-lifecycle flow only; charts,
event review, evidence, and processed-video views remain future frontend slices.

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
