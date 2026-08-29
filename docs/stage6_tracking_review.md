# Stage 6 ByteTrack Qualitative Review

## Scope

Stage 6 uses the unchanged Stage 4 detector configuration (`yolo26n.pt`, Apple
MPS, image size 640, confidence 0.25, and the six target classes) with the stock
Ultralytics `bytetrack.yaml`. One tracker instance is created per video and is
kept alive across its sequential frames with `persist=True`.

The generated Track IDs are scoped to one video. `unique_track_ids_observed` and
`class_track_ids_observed` are tracking diagnostics only; they are not total
traffic, unique-vehicle, or business counts.

## Inspection method

Stage 6 overlay sequences were inspected at multiple consecutive frames for:

- highway cars under smooth motion;
- Taipei pedestrians and dense scooter crossings;
- urban scooters near partial occlusion;
- aerial cars and small road users.

The inspected contact sheets are generated under
`outputs/tracking_review/stage6/` and remain Git-ignored. This review is
qualitative. No MOTA, HOTA, IDF1, or ground-truth identity annotations were
created.

## Observations

### Highway cars

Ordinary nearby cars generally retained their IDs as they moved toward the
camera. For example, car Track ID 296 remains present from frame 400 through
frame 500 while its center moves consistently down and right in the image. The
inspection also showed stable commercial-vehicle IDs over tens of frames.
Fragmentation candidates remain more common for distant, small, or intermittently
detected vehicles.

### Taipei dense traffic

Foreground pedestrians commonly retained IDs during continuous visibility, and
several crossing scooters retained IDs for their short visible passage. Dense
overlap, the foreground pole, pedestrian occlusion, and scooters entering or
leaving behind other road users produced short tracks and plausible
fragmentation candidates. The reviewed sequence does not provide ground truth
for declaring a definite ID switch.

### Urban occlusion

Motorcycle Track ID 1025 was observed across frames 1330–1358 despite several
missing observations, showing that ByteTrack can reconnect a short interruption.
The same numeric ID briefly appeared with a `person` class observation at frame
1332, which is a class-instability candidate in the detector/tracker output.
Other heavily overlapped scooters produced short-lived IDs, consistent with
detection dropouts and fragmentation risk.

### Aerial small objects

Larger cars showed useful continuity across the aerial sequence; multiple car
IDs remained active for dozens of frames. Tiny scooters, motorcycles, and riders
often had no detector boxes, so ByteTrack could not create or preserve their
identities. This is primarily an upstream small-object detection limitation,
not evidence that tracking alone can recover them.

## Limitations

- The inspection covers selected sequences from four videos, not exhaustive
  identity ground truth.
- Apparent fragmentation or ID-switch cases are candidates unless verified with
  manually annotated identities.
- Track IDs can be absent whenever the detector does not return an associated
  box.
- Class-specific Track-ID diagnostics can overlap because a track's predicted
  class may change between observations.
- No trajectories, movement vectors, direction, line crossing, zones, dwell
  logic, events, or traffic counting were implemented.
