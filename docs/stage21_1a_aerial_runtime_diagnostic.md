# Stage 21.1A Aerial Small-Object Runtime Diagnostic

## Governance

- Runtime model: `models/pretrained/yolo26n.pt`
- Runtime model SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- Stage 17 rejected model usage: `0`
- Training runs: `0`
- Input: `data/raw/videos/pexels_9322363_aerial_busy_intersection.mp4`
- Input SHA256: `7506c1997481f4df107e4c4f17b8bb5eccbd706f077f54980fa2ac8a73a57c07`
- Inclusive frame range: `280–459`
- Fixed tracker: `bytetrack.yaml`
- Fixed trajectory: history `30`, minimum displacement `5.0` image pixels

## Automated comparison

| Config | imgsz | conf | Detection obs | Small vehicle obs | Track IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline_640_025 | 640 | 0.25 | 2778 | 2762 | 58 | 41.301 |
| lowconf_640_015 | 640 | 0.15 | 3277 | 3255 | 55 | 43.031 |
| highres_960_025 | 960 | 0.25 | 3878 | 3859 | 45 | 40.550 |
| highres_lowconf_960_015 | 960 | 0.15 | 4204 | 4176 | 47 | 39.922 |

Detection observations are repeated per-frame occurrences. Track IDs are video-scoped diagnostics, not formal traffic counts.

**No detection GT → no formal accuracy claim. Precision, Recall, and mAP are not computed.**

Automated counts cannot determine visually obvious misses or false positives. Final runtime selection requires checking the synchronized comparison frames and videos.

`MANUAL_VISUAL_REVIEW_REQUIRED`

## Per-class detection observations

| Config | person | bicycle | car | motorcycle | bus | truck |
|---|---:|---:|---:|---:|---:|---:|
| baseline_640_025 | 0 | 0 | 2730 | 0 | 0 | 48 |
| lowconf_640_015 | 0 | 0 | 3213 | 0 | 0 | 64 |
| highres_960_025 | 0 | 0 | 3698 | 0 | 8 | 172 |
| highres_lowconf_960_015 | 0 | 0 | 4014 | 0 | 8 | 182 |

## Small-vehicle observations (`normalized bbox area < 0.01`)

| Config | car | motorcycle | bus | truck | Total |
|---|---:|---:|---:|---:|---:|
| baseline_640_025 | 2726 | 0 | 0 | 36 | 2762 |
| lowconf_640_015 | 3208 | 0 | 0 | 47 | 3255 |
| highres_960_025 | 3687 | 0 | 0 | 172 | 3859 |
| highres_lowconf_960_015 | 3994 | 0 | 0 | 182 | 4176 |

## Confidence diagnostics

- `baseline_640_025`: overall mean/median 0.4644/0.4610; small-vehicle mean/median 0.4648/0.4617; observations in [0.15, 0.25): 0.
- `lowconf_640_015`: overall mean/median 0.4260/0.4228; small-vehicle mean/median 0.4265/0.4241; observations in [0.15, 0.25): 459.
- `highres_960_025`: overall mean/median 0.5349/0.5554; small-vehicle mean/median 0.5355/0.5559; observations in [0.15, 0.25): 0.
- `highres_lowconf_960_015`: overall mean/median 0.5094/0.5374; small-vehicle mean/median 0.5105/0.5383; observations in [0.15, 0.25): 324.

## Automated findings

- `lowconf_640_015` vs baseline: detection observations +499, small-vehicle observations +493, diagnostic Track IDs -3, FPS +1.730.
- `highres_960_025` vs baseline: detection observations +1100, small-vehicle observations +1097, diagnostic Track IDs -13, FPS -0.751.
- `highres_lowconf_960_015` vs baseline: detection observations +1426, small-vehicle observations +1414, diagnostic Track IDs -11, FPS -1.379.

These deltas measure emitted observations, not correctness. Added low-confidence boxes may be true small vehicles or false positives; visual review is required.

## Tracking and trajectory diagnostics

| Config | Tracking obs | Diagnostic IDs | Frame-gap candidates | Trajectory obs |
|---|---:|---:|---:|---:|
| baseline_640_025 | 2778 | 58 | 205 | 2778 |
| lowconf_640_015 | 3277 | 55 | 84 | 3277 |
| highres_960_025 | 3878 | 45 | 154 | 3878 |
| highres_lowconf_960_015 | 4204 | 47 | 54 | 4204 |

A frame-gap candidate is one gap greater than one frame within an observed Track ID. It can indicate intermittent detection but is not verified physical-object fragmentation or an ID switch.

## Sampled visual observations (three synchronized stills)

- At source frame 310, both 960 configurations recover additional edge/bottom and construction-side vehicles that are visibly present but unboxed at 640. Lowering confidence at 640 changes less in this still.
- At source frame 370, low-confidence 640 adds at least one visible blue vehicle missed by the 640 baseline; 960 configurations cover more of the central/right traffic. The yellow commercial van also illustrates car/truck taxonomy ambiguity across configurations.
- At source frame 430, 960 configurations again cover additional central/edge vehicles. Several visually obvious scooters/motorcycles in the upper intersection remain unboxed across all four configurations, consistent with zero motorcycle observations in the aggregate output.
- The three stills do not show an unequivocal false positive. Low-confidence partial/edge detections and parked/construction-side vehicles still require full-video review before any runtime choice.

This is a limited sampled-frame review, not Ground Truth annotation or a complete visual audit of all 180 frames.

## Synchronized comparison frames

- Relative t=1.0s / source frame 310: `outputs/evaluation/stage21_1a/comparison_relative_001.0s_frame_000310.jpg`
- Relative t=3.0s / source frame 370: `outputs/evaluation/stage21_1a/comparison_relative_003.0s_frame_000370.jpg`
- Relative t=5.0s / source frame 430: `outputs/evaluation/stage21_1a/comparison_relative_005.0s_frame_000430.jpg`

## Limitations

- No detection or MOT Ground Truth was created; visually obvious misses, false positives, fragmentation, and trajectory usefulness require manual review.
- FPS is an end-to-end local runtime diagnostic for decode + YOLO/ByteTrack + trajectory/overlay + MP4 writing. Model-load time is recorded separately and excluded, matching earlier runtime baselines.
- `small bbox area < 0.01` is a descriptive image-space heuristic, not a physical-size category or accuracy metric.
- Detection observations are Ultralytics tracking-output box occurrences. In these runs every returned box had a Track ID, so detection and tracking observation totals are equal; this does not make Track IDs formal object counts.
- No configuration is automatically selected or promoted.

`MANUAL_VISUAL_REVIEW_REQUIRED`
