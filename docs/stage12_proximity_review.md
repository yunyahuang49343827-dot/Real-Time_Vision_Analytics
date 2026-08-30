# Stage 12 — Person–Vehicle Proximity Review

## Scope and configuration

Stage 12 produces normalized image-space proximity warnings only. It does not
estimate physical distance, collision probability, near-miss risk, or time to
collision. Pair state is keyed by
`(video_id, zone_id, person_track_id, vehicle_track_id)`.

Proximity is enabled only in the mixed-traffic zones `taipei_crossing` and
`urban_intersection`. `highway_approach` and `aerial_intersection` explicitly set
`enabled: false`; they still run through the full video pipeline and generate
schema-valid empty outputs.

Both enabled rules use:

- vehicle classes: bicycle, car, motorcycle, bus, and truck;
- normalized trigger distance: `0.012`;
- normalized release distance: `0.020`;
- minimum consecutive observations: 8;
- rider overlap exclusion ratio: `0.25`.

The trigger is intentionally below the release threshold. A pair triggers only
after eight consecutive close observations, remains active while distance stays
within the release threshold, and emits at most once per active episode. Once the
distance exceeds the release threshold, a future close approach can create a new
episode. A missing person or vehicle resets both streak and active state, so
unobserved frames never accumulate a false streak.

## Distance and pair filtering

The engine first keeps only Tracks whose centers are inside an enabled Zone, then
separates `person` Tracks from configured vehicle classes. Only those filtered
person–vehicle combinations are compared; person–person and vehicle–vehicle pairs
are never evaluated.

For two axis-aligned bboxes, horizontal and vertical gaps are clamped to zero and
combined with Euclidean distance. Overlap or boundary contact therefore has zero
pixel distance. The value is divided by the frame diagonal. It is resolution-
relative image geometry, not a safety distance.

## Rider/self-vehicle exclusion

For bicycle and motorcycle pairs only, a pair is excluded when bbox intersection
covers at least 25% of the smaller box or when the person's bottom-center lies
inside the two-wheeler bbox. This reduced repeated rider/self-bike warnings but is
not a rider classifier. It can miss riders whose boxes do not overlap enough and
can exclude an adjacent pedestrian under strong occlusion.

## Results

| Scene | Frames | Enabled | Warnings | Distinct rider-pair exclusions | Filtered comparisons |
|---|---:|---:|---:|---:|---:|
| Highway | 1,800 | No | 0 | 0 | 0 |
| Taipei | 5,958 | Yes | 166 | 494 | 29,994 |
| Urban | 2,091 | Yes | 14 | 18 | 1,256 |
| Aerial | 473 | No | 0 | 0 | 0 |

Taipei warnings consist of 105 motorcycle, 55 car, and 6 bus pairs. Urban has 11
car, 2 truck, and 1 bus pair. Of the 166 Taipei episodes, 124 trigger with bbox
overlap (`normalized_distance = 0`); 12 of 14 Urban episodes also trigger at zero.
The results therefore strongly reflect occlusion, perspective, and dense-scene
bbox overlap rather than a calibrated real-world risk threshold.

## Qualitative review

Taipei samples show pedestrians, scooter riders, adjacent scooters, cars, and a
bus sharing the image-space ROI. The exclusion removes many visibly overlapping
rider/two-wheeler pairs, but remaining warnings include neighboring road users and
some plausible rider leakage where detector boxes do not meet the exclusion
geometry. Dense traffic can generate several simultaneous pair episodes around
one large vehicle; those are distinct Track pairs, not distinct safety incidents.

Urban samples include people or riders near cars as well as perspective overlap
with trucks and buses. A large foreground vehicle bbox can overlap a distant or
partially occluded person in the image while their physical separation is
unknown. These are correctly represented only as image-space warnings.

## Limitations

The heuristic inherits detector class errors, the Stage 5 person/pedestrian
taxonomy mismatch, bbox size and jitter, perspective, occlusion, Track ID switches,
fragmentation, and zone placement. Missing observations end an episode and may
later permit a duplicate warning after reacquisition. Bbox overlap cannot identify
depth ordering or a rider's ownership relationship. No Ground Truth proximity
labels exist, so Stage 12 does not report formal accuracy or collision risk.
