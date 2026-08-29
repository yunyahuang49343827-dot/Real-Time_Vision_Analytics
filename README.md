# Real-Time Vision Analytics & Event Detection System

This repository currently contains project setup, source governance, and
**Stage 2: Video Profiling & Validation**. Detection, tracking, spatial analysis,
event detection, APIs, and dashboards are deliberately not implemented yet.

## Requirements

- Apple Silicon Mac
- Python 3.11
- Apple Metal Performance Shaders (MPS)
- Git

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Verify Stage 0

The environment check downloads `yolo26n.pt` through Ultralytics on its first run,
stores it under `models/pretrained/`, and performs one inference on an in-memory
640 x 640 synthetic image using MPS. Model weights are ignored by Git.

```bash
python --version
python scripts/check_environment.py
pytest
git status
```

The check must report `PASS`. If MPS is not built or available, it exits with a
failure instead of silently falling back to CPU.

## Profile Stage 1 runtime videos

The profiler reads container metadata and decodes only the first frame of each
runtime video listed in `data/manifests/sources.csv`. It does not run detection
or a full-frame benchmark.

```bash
python scripts/profile_videos.py
```

Results are written to `data/interim/video_metadata.csv`. Source and license
governance details are documented in `docs/data_sources.md`.

## Project layout

```text
configs/
data/{raw,interim,processed,manifests}/
models/{pretrained,finetuned}/
outputs/{videos,detections,tracks,events,analytics,evidence}/
src/vision_analytics/{video,detection,tracking,spatial,events,analytics,services,utils}/
scripts/
tests/
```
