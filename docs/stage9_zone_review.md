# Stage 9 Zone / ROI Review

## Configuration

Each scene has one normalized polygon in `configs/scenes.yaml`:

| Source | Zone | Type | Points |
| --- | --- | --- | --- |
| Highway | highway_approach | vehicle_monitoring | (0.18,0.54), (0.82,0.54), (0.96,0.98), (0.04,0.98) |
| Taipei | taipei_crossing | mixed_road_users | (0.18,0.35), (0.82,0.35), (0.88,0.88), (0.12,0.88) |
| Urban | urban_intersection | mixed_traffic | (0.18,0.36), (0.82,0.36), (0.88,0.90), (0.12,0.90) |
| Aerial | aerial_intersection | vehicle_monitoring | (0.20,0.22), (0.80,0.22), (0.82,0.84), (0.18,0.84) |

Runtime converts points with `(width - 1, height - 1)`. Track bbox center is
tested with OpenCV `pointPolygonTest`; polygon boundary is inside.

## State semantics

State identity is `(video_id, zone_id, track_id)`; predicted class is not part
of identity. Initial observations produce `INSIDE` or `OUTSIDE`, never `ENTER`.
Subsequent transitions follow OUTSIDE→INSIDE=`ENTER`, INSIDE→INSIDE=`INSIDE`,
INSIDE→OUTSIDE=`EXIT`, and OUTSIDE→OUTSIDE=`OUTSIDE`.

Missing Tracks do not update stored membership and never synthesize EXIT. The
overlay's current occupancy is narrower: Tracks observed inside on the current
frame. This prevents a missing Track from staying indefinitely in the displayed
occupancy while preserving its membership for a later valid observation.

## ENTER / EXIT diagnostics

| Scene | ENTER | EXIT | Tracks observed inside | Peak observed occupancy |
| --- | ---: | ---: | ---: | ---: |
| Highway | 48 | 16 | 114 | 10 |
| Taipei | 106 | 109 | 530 | 12 |
| Urban | 50 | 59 | 236 | 7 |
| Aerial | 24 | 16 | 81 | 13 |

Class breakdown is stored in each Stage 9 summary JSON. These are Track-state
diagnostics, not Ground Truth unique visitors or formal traffic analytics.

## Qualitative review

- Highway cars visibly transition into the approach polygon and remain inside
  while their bbox centers stay within it.
- Taipei scooters and pedestrians are observed inside the mixed-use polygon.
  Foreground occlusion interrupts observations but does not itself emit EXIT.
- Urban cars, motorcycles, people, buses, and trucks are represented inside the
  intersection polygon when their centers are visible there.
- Aerial zone observations are dominated by cars. Tiny scooters and riders
  missed upstream have no Zone observation or transition.

Review contact sheets are under `outputs/zone_review/stage9/` and are Git-ignored.

## Limitations

- Detection misses omit Zone observations and can hide valid ENTER/EXIT pairs.
- Tracking fragmentation can create separate state identities for one physical
  object; an ID switch can therefore produce additional transitions.
- A reappearing persistent ID retains its previous membership until a valid new
  observation; there is no timeout-generated EXIT.
- Camera motion, perspective, and bbox-center jitter affect polygon membership.
- No Ground Truth Zone labels exist, so formal Zone accuracy is not claimed.
- No dwell, allowed direction, wrong way, stationary-vehicle event, pedestrian
  intrusion event, proximity, severity, evidence capture, or event engine was
  implemented.
