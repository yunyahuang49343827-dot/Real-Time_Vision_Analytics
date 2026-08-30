# Stage 13 — Event Engine & Severity

## Scope

Stage 13 runs the existing line crossing, zone, wrong-way, dwell, stationary, and
proximity engines without changing their semantics. Only newly emitted upstream
records are normalized into one Event schema. No snapshot, clip, evidence path,
new CV model, or traffic analytics layer is added.

The schema preserves video, frame, timestamp, primary/secondary Track IDs,
primary/secondary classes, zone or line, rule source, observed rule value, and
threshold. Event IDs use a deterministic readable sequence scoped by the video
name, for example `pexels_2103099_traffic_flow_highway-EVT-000001`. The video name
makes sequences globally unique across the four runs.

## Severity and status policy

Severity and status are loaded from `configs/scenes.yaml`, not assigned inside
the upstream engines and not derived from YOLO confidence.

| Event type | Severity | Status |
|---|---|---|
| LINE_CROSSING | INFO | DETECTED |
| ZONE_ENTRY | INFO | DETECTED |
| ZONE_EXIT | INFO | DETECTED |
| WRONG_WAY | CRITICAL | REVIEW_REQUIRED |
| LONG_DWELL | WARNING | DETECTED |
| STATIONARY_VEHICLE | WARNING | REVIEW_REQUIRED |
| PEDESTRIAN_INTRUSION | zone override, otherwise WARNING | REVIEW_REQUIRED |
| PROXIMITY_WARNING | WARNING | REVIEW_REQUIRED |

`CRITICAL` describes operational review priority, not certainty. `DETECTED` means
the configured rule emitted a record. `REVIEW_REQUIRED` identifies heuristic or
candidate outputs. `CONFIRMED` is supported by the schema but Stage 13 produces no
confirmed events because it does not perform manual verification.

## Pedestrian intrusion

Only `highway_shoulder_stationary` is marked `restricted_for_person`, with
`CRITICAL` operational severity. A `person` must produce a real Zone `ENTER` in
that polygon. Initial `INSIDE`, other classes, and all unrestricted-zone entries
remain ordinary Zone events and do not create intrusion.

No pedestrian intrusion was emitted in the four videos because no person Track
produced `ENTER` for the restricted shoulder zone. The zero result is retained;
the zone and rule were not changed to manufacture a candidate.

## Results

| Event type | Count | Severity | Status |
|---|---:|---|---|
| LINE_CROSSING | 190 | INFO | DETECTED |
| ZONE_ENTRY | 241 | INFO | DETECTED |
| ZONE_EXIT | 218 | INFO | DETECTED |
| WRONG_WAY | 3 | CRITICAL | REVIEW_REQUIRED |
| LONG_DWELL | 16 | WARNING | DETECTED |
| STATIONARY_VEHICLE | 0 | WARNING | REVIEW_REQUIRED |
| PEDESTRIAN_INTRUSION | 0 | CRITICAL for configured zone | REVIEW_REQUIRED |
| PROXIMITY_WARNING | 180 | WARNING | REVIEW_REQUIRED |

There are 848 total events with 848 unique IDs: 649 `INFO`, 196 `WARNING`, and 3
`CRITICAL`; 665 have `DETECTED`, 183 have `REVIEW_REQUIRED`, and none have
`CONFIRMED`.

Per-video totals are 152 Highway, 469 Taipei, 173 Urban, and 54 Aerial. Counts are
rule-generated system events and are not Ground Truth incidents, traffic totals,
or verified offences.

## Candidate boundary and limitations

The three Highway wrong-way records remain the same rule candidates visually
reviewed in Stage 10 as false alarms, despite their `CRITICAL` priority. The 180
proximity records remain Stage 12 normalized image-space warnings, not collision
risk. Long dwell remains observed Zone time and does not imply stationary or
abnormal behaviour.

Every Event inherits upstream detector misses and class errors, bbox jitter,
ByteTrack fragmentation and ID switches, trajectory-window effects, zone and line
placement, and heuristic thresholds. Normalization improves schema consistency
and traceability; it does not improve upstream accuracy or convert candidates into
verified incidents.
