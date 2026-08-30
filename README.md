# Real-Time Vision Analytics & Event Detection System

This repository currently contains project setup, source governance, video
profiling, an OpenCV pipeline, the Stage 4 YOLO26n pretrained detection baseline,
and **Stage 5 structured qualitative error analysis**. Stage 6 adds video-scoped
ByteTrack IDs as tracking diagnostics. Spatial analysis, event detection, APIs,
and dashboards are deliberately not implemented yet.

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

## Run the Stage 3 OpenCV pipeline

The pipeline decodes each complete runtime video, adds a simple identifier/frame
index/timestamp overlay, and writes an MP4 at the original resolution. Its
reported processing FPS is an end-to-end decode + overlay + write baseline, not
a pure decode or model-inference benchmark.

```bash
python scripts/process_videos.py
```

Processed videos are written under `outputs/videos/stage3/`; benchmark results
are written to `outputs/analytics/stage3_video_benchmark.csv`. Both are generated
artifacts and remain Git-ignored.

## Run the Stage 4 pretrained detection baseline

The detector runs `yolo26n.pt` on Apple MPS at image size 640 and confidence
threshold 0.25. It retains person, bicycle, car, motorcycle, bus, and truck
detection occurrences. It does not perform tracking or object counting.

```bash
python scripts/run_detection.py
```

Generated overlay MP4s, per-video detection CSV/summary JSON files, and the
end-to-end Stage 4 benchmark remain under `outputs/` and are Git-ignored.

## Prepare the Stage 5 qualitative review

The preparation script computes per-class confidence distributions and selects
96 unique frames using uniform temporal and scene-aware targeted sampling. It
extracts raw/overlay comparison images without rerunning YOLO. Visual findings
and limitations are documented in `docs/stage5_error_analysis.md`.

```bash
python scripts/prepare_error_analysis.py
```

Generated review frames, contact sheets, sampling summaries, and the completed
manual review CSV remain under `outputs/` and are Git-ignored. Confidence is a
model score, not correctness; Stage 5 does not calculate Precision, Recall, or
mAP.

## Run Stage 6 multi-object tracking

The tracking script keeps one Ultralytics ByteTrack state alive across all
sequential frames of each video. It uses the unchanged Stage 4 YOLO26n/MPS
settings and the stock `bytetrack.yaml` configuration.

```bash
python scripts/run_tracking.py
```

Stage 6 overlay MP4s, track-observation CSV files, summary JSON files, and the
end-to-end benchmark remain under `outputs/` and are Git-ignored. Track IDs are
scoped to one video and are diagnostic identifiers, not traffic or business
counts. Qualitative observations are recorded in
`docs/stage6_tracking_review.md`.

## Run the Stage 7 trajectory engine

The trajectory runner adds a 30-observation bounded recent trail to each
video-scoped Track ID and derives image-space delta, displacement, frame-gap,
and recent-window direction features. A 5-pixel net-displacement threshold maps
small movement to `STATIONARY`; this is a movement label, not an event.

```bash
python scripts/run_trajectory.py
```

Generated trajectory MP4s, CSV files, summary JSON files, and the end-to-end
benchmark remain under `outputs/` and are Git-ignored. All displacement values
are pixels, not physical distance or speed. Qualitative observations are in
`docs/stage7_trajectory_review.md`.

## Run Stage 8 line-crossing counts

Stage 8 converts normalized scene lines from `configs/scenes.yaml` to finite
pixel segments and counts a Track ID at most once per video and line. A crossing
requires opposite line sides, finite-segment intersection, a frame gap of at
most five, and at least three pixels of observed movement.

```bash
python scripts/run_line_crossing.py
```

Generated crossing MP4s, CSV files, summary JSON files, and the Stage 8
benchmark remain under `outputs/` and are Git-ignored. Counts are Track-ID-based
line crossings, not perfect Ground Truth traffic counts. Configuration and
qualitative review are documented in `docs/stage8_line_crossing_review.md`.

## Run Stage 9 polygon-zone analysis

Stage 9 converts normalized polygons from `configs/scenes.yaml` to pixels and
maintains OUTSIDE/ENTER/INSIDE/EXIT membership keyed by video, zone, and Track
ID. Boundary points are inside; missing observations never synthesize EXIT.

```bash
python scripts/run_zone_analysis.py
```

Generated Zone overlay MP4s, ENTER/EXIT CSV files, summary JSON files, and the
benchmark remain under `outputs/` and are Git-ignored. Zone diagnostics are not
Ground Truth unique visitors or formal traffic analytics. Review details are in
`docs/stage9_zone_review.md`.

## Run Stage 10 wrong-way monitoring

Stage 10 combines recent-window trajectory direction with polygon-zone context
and config-defined allowed directions. A disallowed, non-stationary movement
must meet the pixel-displacement threshold for consecutive observations before
one video/zone/Track diagnostic is confirmed.

```bash
python scripts/run_wrong_way.py
```

Generated overlays, per-video detection CSV/summary JSON files, and the Stage 10
benchmark remain under `outputs/` and are Git-ignored. Natural videos may
legitimately produce zero confirmed wrong-way rows; rules are not reversed to
manufacture detections. Review details are in
`docs/stage10_wrong_way_review.md`.

## Run Stage 11 temporal rules

Stage 11 maintains separate observed episodes for `LONG_DWELL` and
`STATIONARY_VEHICLE`. Dwell uses source timestamps and Zone state; stationary
monitoring additionally requires an explicitly configured zone, vehicle class,
duration, and low frame-diagonal-normalized image displacement.

```bash
python scripts/run_temporal_rules.py
```

Temporary missing observations do not synthesize EXIT, while gaps beyond the
configured limit restart observable continuity. Generated overlays, CSV/summary
files, and the benchmark remain Git-ignored under `outputs/`. These diagnostics
are not physical speed measurements or formal accuracy results. Review details
are in `docs/stage11_temporal_rules_review.md`.

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
