# Stage 11 — Dwell Time & Stationary Detection Review

## Scope and configuration

Stage 11 keeps `LONG_DWELL` and `STATIONARY_VEHICLE` as separate rules. Both use
source `timestamp_seconds`; no duration is inferred from a fixed FPS. State keys
are `(video_id, zone_id, track_id)` and class is a rule scope, not identity.

Configured dwell zones and observed thresholds are:

| Scene | Zone | Threshold | Classes |
|---|---|---:|---|
| Highway | `highway_approach` | 8 s | car, motorcycle, bus, truck |
| Taipei | `taipei_crossing` | 12 s | all six target classes |
| Urban | `urban_intersection` | 8 s | all six target classes |
| Aerial | `aerial_intersection` | 5 s | car, motorcycle, bus, truck |

Stationary monitoring is enabled only for `highway_shoulder_stationary`, a
dedicated shoulder-side polygon. It applies to car, motorcycle, bus, and truck,
requires five observed seconds, and defines low movement as recent trajectory net
displacement no greater than `0.003` of the 3840×2160 frame diagonal. This is a
normalized image-space measure, not speed or physical distance. Taipei, Urban,
and Aerial have no stationary rule, so waiting at signals and ordinary slow
intersection movement cannot emit `STATIONARY_VEHICLE`.

## Episode and missing-observation semantics

An `ENTER` starts a dwell episode. An initial `INSIDE` observation also starts an
observed episode at that timestamp without pretending it is the true entry time.
`INSIDE` continues it, `EXIT` ends it, and a later `ENTER` starts a new episode.
Each episode can trigger once.

A missing Track does not synthesize `EXIT`. Missing time up to one second can
preserve observable continuity when the same Track returns. If the gap between
observations exceeds one second, both dwell and stationary continuity restart at
the new observation; unseen time is not accumulated indefinitely. A new Track ID
after fragmentation necessarily starts separate state.

Stationary episodes additionally reset on movement above the normalized threshold,
outside/exit observations, inapplicable classes, missing trajectory data, or a
frame gap. A later low-movement episode may trigger independently.

## Results

| Scene | Frames | LONG_DWELL | STATIONARY_VEHICLE |
|---|---:|---:|---:|
| Highway | 1,800 | 9 cars | 0 |
| Taipei | 5,958 | 0 | 0 — rule disabled |
| Urban | 2,091 | 0 | 0 — rule disabled |
| Aerial | 473 | 7 cars | 0 — rule disabled |

All four full frame loops passed output validation. The 16 dwell records are
observed zone episodes; they are not stationary detections and are not unique
visitor or business counts.

## Qualitative review

Highway start/trigger pairs show cars progressing through congested traffic while
remaining in the large approach polygon for eight seconds. They are visibly
moving, and no vehicle in the dedicated shoulder polygon met the five-second
low-movement rule. Short bbox-center jitter therefore did not create a stationary
alert.

Taipei produced no long dwell and cannot produce stationary alerts because the
rule is deliberately disabled; signal waiting is not treated as abnormal. Urban
slow-moving traffic likewise produced neither rule, and slow motion alone would
not qualify without an explicitly configured stationary zone.

Aerial start/trigger pairs show cars traversing or queuing within the large ROI
for at least five observed seconds. Small-object misses can pause observations;
same-ID gaps within one second preserve continuity, gaps beyond one second reset
it, and tracking fragmentation creates a new episode. Consequently, misses can
both suppress a dwell trigger and split one physical object's observed episode.

## Limitations

Both rules inherit detector misses, bbox jitter, occlusion, Track ID switches and
fragmentation, zone placement, and the bounded trajectory window. Dwell begins at
the first observed inside timestamp, not necessarily physical entry. Normalized
displacement is resolution-relative image motion and is affected by perspective;
it is not real-world velocity. There is no temporal Ground Truth, so Stage 11 does
not report formal Dwell or Stationary accuracy.
