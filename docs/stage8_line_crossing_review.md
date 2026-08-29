# Stage 8 Line Crossing and Unique Counting Review

## Scope

Stage 8 adds one normalized virtual counting line per runtime scene. A crossing
is evaluated from the previous observed Track center to the current center. The
movement segment must intersect the configured **finite** line segment and the
centers must be strictly on opposite sides.

The oriented config segment `start → end` defines A-side as its positive
cross-product side and B-side as its negative side. Positive-to-negative
crossing is `A_TO_B`; the reverse is `B_TO_A`. These names report crossing
direction only and do not encode allowed direction or wrong-way semantics.

## Scene configuration

All coordinates in `configs/scenes.yaml` are normalized to 0.0–1.0 and converted
to pixels at runtime using `(width - 1, height - 1)`.

| Source | Line | Normalized start | Normalized end | Geometry |
| --- | --- | --- | --- | --- |
| pexels_2103099 | highway_main | (0.08, 0.62) | (0.92, 0.62) | horizontal |
| pexels_13258685 | taipei_center | (0.50, 0.15) | (0.50, 0.92) | vertical |
| pexels_37258214 | urban_center | (0.50, 0.30) | (0.50, 0.90) | vertical |
| pexels_9322363 | aerial_center | (0.50, 0.08) | (0.50, 0.92) | vertical |

The geometry implementation also supports diagonal finite segments. That case
is covered by tests rather than adding a second demo line.

The shared safeguards are `maximum_frame_gap: 5` and
`minimum_movement_pixels: 3.0`. No per-scene tolerance tuning was performed.

## Deduplication

After a successful crossing, `(video_id, line_id, track_id)` is stored in the
engine and can never increment that line again. All four generated crossing CSV
files contain zero duplicate keys under this definition. Different Track IDs
can count independently, and one Track ID can count once on each distinct line.

This removes repeat counts caused by the same Track jittering around a line. It
cannot merge two Track IDs that actually belong to one physical object; tracking
fragmentation can therefore still over-count.

## Observed counts

### Highway — 45

| Class | A_TO_B | B_TO_A |
| --- | ---: | ---: |
| car | 0 | 41 |
| person | 0 | 1 |
| truck | 0 | 3 |

The dominant direction is consistent with vehicles approaching through the
horizontal line. The `person` crossing inherits the Stage 5 taxonomy limitation:
highway `person` can be a visible vehicle occupant and should not automatically
be interpreted as a pedestrian traffic count.

### Taipei — 88

| Class | A_TO_B | B_TO_A |
| --- | ---: | ---: |
| bus | 4 | 3 |
| car | 11 | 5 |
| motorcycle | 16 | 8 |
| person | 28 | 13 |

### Urban — 50

| Class | A_TO_B | B_TO_A |
| --- | ---: | ---: |
| bicycle | 0 | 1 |
| bus | 2 | 1 |
| car | 20 | 4 |
| motorcycle | 4 | 1 |
| person | 9 | 1 |
| truck | 7 | 0 |

### Aerial — 7

| Class | A_TO_B | B_TO_A |
| --- | ---: | ---: |
| car | 3 | 4 |

These are real Track-ID-based line-crossing counts under the configured logic,
not perfect Ground Truth traffic counts.

## Qualitative review

Stage 8 overlay sequences were inspected around reported crossing frames:

- Highway Track ID 9 (`truck`) crosses `highway_main` at frame 400, and the live
  count increments once.
- Taipei Track ID 298 (`motorcycle`) crosses `taipei_center` at frame 617. Several
  pedestrian crossings are also visible, although foreground occlusion makes
  some individual reviews less certain.
- Urban Track ID 3 (`motorcycle`) crosses `urban_center` at frame 46 while its
  trajectory passes through the finite yellow segment.
- Aerial Track ID 327 (`car`) crosses `aerial_center` at frame 327.

Review contact sheets are generated under
`outputs/line_crossing_review/stage8/` and remain Git-ignored.

## Limitations

- Upstream detection misses cause missed crossings; this is especially visible
  for small aerial scooters and riders.
- Tracking fragmentation can assign a new ID after occlusion and over-count one
  physical object. Conversely, a lost track may never supply the observation
  pair needed to count.
- Camera motion and perspective affect where image-space centers cross a line.
- The configured lines are reasonable demo lines, not calibrated traffic-survey
  infrastructure.
- No complete Ground Truth exists, so formal counting accuracy is not claimed.
- No zones, zone entry/exit, allowed direction, wrong way, dwell, stationary
  vehicle event, proximity, severity, evidence capture, or event engine was
  implemented.
