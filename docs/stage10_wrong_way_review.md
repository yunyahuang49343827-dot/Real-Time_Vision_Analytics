# Stage 10 — Direction & Wrong-Way Review

## Scope and rule design

Stage 10 combines the existing bounded trajectory direction, current polygon-zone
membership, and allowed directions loaded from `configs/scenes.yaml`. It does not
derive direction from a single frame and does not add dwell, severity, evidence,
or a general event engine.

Only `pexels_2103099` has a defensible one-direction monitoring region. The
`highway_inbound_direction` polygon covers the approaching carriageway and allows
`DOWN_LEFT`, `DOWN`, and `DOWN_RIGHT` for car, motorcycle, bus, and truck. Taipei,
Urban, and Aerial scenes are intersections with legitimate multi-direction
traffic, so Stage 10 deliberately leaves them without a wrong-way rule instead of
inventing a one-way restriction.

A rule violation requires all of the following:

- the Track center is currently inside the configured zone;
- the class is applicable;
- trajectory `frame_gap` is exactly one;
- direction is not `STATIONARY`;
- recent-window net displacement is at least 20 image-space pixels;
- direction is outside the configured allowed set for 8 consecutive observations.

The confirmation key is `(video_id, zone_id, track_id)`, so one key can be emitted
at most once. An allowed/outside/stationary/insufficient/gapped observation resets
the streak; a missing observation cannot add to it.

## Results and visual review

All four videos completed their full frame loops. The Highway rule emitted three
rule-confirmed candidates; the other videos emitted zero because no directional
rule is configured for their multi-direction intersection zones.

| Scene | Frames | Rule-confirmed rows | Manual review |
|---|---:|---:|---|
| Highway | 1,800 | 3 | All three appear to be false alarms, not visible wrong-way travel |
| Taipei intersection | 5,958 | 0 | No rule configured; normal multi-direction traffic was not evaluated as wrong-way |
| Urban multi-class | 2,091 | 0 | No rule configured |
| Aerial intersection | 473 | 0 | No rule configured |

The Highway candidates were reviewed at and around frames 167, 348, and 1501:

- Track 70, frame 167, `RIGHT`: the car is visibly travelling normally toward the
  camera. Its recent bbox-center path temporarily became more horizontal, then
  returned to `DOWN_RIGHT`.
- Track 302, frame 348, `RIGHT`: this new Track begins with an approximately
  28-pixel horizontal center jump. The bounded net vector remains `RIGHT` long
  enough to meet the consecutive rule even though the vehicle travels normally.
- Track 1660, frame 1501, `UP`: a newly initialized Track has two large upward
  bbox-center jumps (about 44 and 15 pixels). Subsequent centers move downward,
  but the early jumps keep the recent net vector `UP`/`UP_RIGHT` long enough to
  confirm. The visible vehicle is not driving against traffic.

Normal Highway vehicles with `DOWN`, `DOWN_LEFT`, or `DOWN_RIGHT` movement did not
trigger the rule. `STATIONARY` observations are excluded before streak updates.
There is no Ground Truth, so this inspection is qualitative and is not a formal
wrong-way accuracy measurement.

## Limitations

The rule consumes image-space centers and inherits detection bbox jitter,
occlusion, Track initialization, ID switches, fragmentation, and trajectory-window
effects. A transient early center jump can dominate the net direction for several
observations. Fragmentation can create a fresh Track with an unstable initial
history, while a missed detection creates a frame gap that correctly prevents
continued accumulation but may also miss a real violation. Perspective means
image directions are scene-specific and cannot be interpreted as map headings or
physical velocity.

The three Highway outputs should therefore be read as rule-confirmed diagnostic
candidates with human-reviewed false alarms, not proven traffic offences. Stage 10
does not tune thresholds against this small natural-video sample and does not
claim that the absence of rows in the intersection scenes proves absence of
wrong-way behaviour.
