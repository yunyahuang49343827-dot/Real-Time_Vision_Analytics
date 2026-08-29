# Stage 7 Trajectory Engine Review

## Scope and definitions

Stage 7 extends each video-scoped ByteTrack identity with a bounded recent
trajectory. Histories are keyed by `(video_id, track_id)` and use
`deque(maxlen=30)`. They never retain a full-video trail.

All centers, deltas, and displacements are measured in image-space pixels.
They are not meters, speed, or real-world distance. Image `x` increases to the
right and image `y` increases downward. Direction is a recent-window movement
classification, not an event or an allowed-direction decision.

Direction uses the vector from the oldest point in the current 30-observation
window to the newest point. A net displacement below 5 pixels is classified as
`STATIONARY`. Here, `STATIONARY` only means small image-space movement in the
recent window; it is not a stationary-vehicle event.

## Qualitative inspection

Selected Stage 7 overlay sequences were inspected from all four videos. Review
contact sheets are generated under `outputs/trajectory_review/stage7/` and are
Git-ignored.

### Highway

Nearby car and commercial-vehicle trails generally remain attached to the bbox
center and progress continuously as vehicles approach the camera. The visible
trails are limited to recent points. Small or distant vehicles still inherit
Stage 6 detection and tracking gaps, so some trails are shorter or interrupted.

### Taipei

Pedestrians and scooters show plausible image-space trails while continuously
tracked. Dense crossings, the foreground pole, mutual occlusion, and short-lived
Track IDs interrupt scooter trails or cause a new trail to begin after tracking
fragmentation. This is expected behavior for histories keyed by Track ID; the
trajectory engine does not attempt identity repair.

### Urban multi-class

The reviewed large truck sequence moves diagonally toward the lower-right image
region, and its recent-window direction is `DOWN_RIGHT`. Predominantly horizontal
road traffic is reflected by the high `RIGHT` and `LEFT` diagnostic counts.
Occasional detection gaps produce `frame_gap > 1` without mixing histories from
different Track IDs.

### Aerial

Cars that remain detected produce clear, compact trails across the intersection.
Tiny scooters, motorcycles, and riders frequently lack upstream detections, so
they have no trajectory or only short fragments. The trajectory engine cannot
recover movement for an object that YOLO/ByteTrack did not observe.

## Direction diagnostics

Direction counts summarize trajectory observations, not unique objects and not
accuracy. The largest categories were:

- Highway: `DOWN` 7,671, `DOWN_LEFT` 4,857, `DOWN_RIGHT` 4,468, `RIGHT` 4,462.
- Taipei: `STATIONARY` 18,200, `RIGHT` 8,215, `LEFT` 7,547.
- Urban: `RIGHT` 4,799, `LEFT` 2,860, `STATIONARY` 966.
- Aerial: `LEFT` 3,554, `RIGHT` 1,606, `STATIONARY` 954.

Camera viewpoint, camera motion, perspective, detector jitter, and tracking
fragmentation all affect these image-space classifications. No formal
trajectory accuracy is claimed.

## Limitations

- No trajectory Ground Truth or formal accuracy metric was created.
- `frame_gap` records missing observations, but no interpolation is performed.
- A Track-ID change deliberately starts a separate history.
- Camera motion can produce image-space movement for physically stationary
  objects.
- No line crossing, counting, zones, allowed direction, wrong-way logic, dwell,
  stationary-vehicle event, proximity, or event engine was implemented.
