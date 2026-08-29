# Stage 5 Multi-Scene Detection Error Analysis

## Scope and method

This is a structured qualitative review of the Stage 4 pretrained YOLO26n
outputs. It is not an accuracy evaluation. No ground-truth dataset was created,
and no Precision, Recall, or mAP was calculated.

The review set contains 96 unique frames: 24 from each of the four runtime
videos. Each video contributes 12 uniformly spaced temporal samples and 12
targeted samples. Targeted selection prioritizes low-confidence detections,
high detection density, and scene-specific cases such as highway `person`,
scooter/`motorcycle`, `bicycle`, commercial vehicles, and aerial small objects.
The targeted sample is deliberately diagnostic and is not representative of the
natural frequency of errors.

Each sample was inspected as a side-by-side raw frame and Stage 4 overlay.
Six-frame contact sheets were reviewed for all 96 samples, followed by
full-resolution inspection of representative cases. The completed qualitative
labels are in `outputs/error_analysis/stage5/manual_review.csv`; sampled-frame
metadata is in `outputs/analytics/stage5_sampling_summary.csv`.

The frame-level review outcomes are 55 `CORRECT`, 29 `FALSE_NEGATIVE`, 11
`AMBIGUOUS`, and one `FALSE_POSITIVE`. These counts describe this deliberately
selected review set only. They must not be interpreted as rates or model
performance metrics, and a `CORRECT` row means that the sampled prediction was
plausible rather than that every object in the frame was exhaustively annotated.

## Confidence summary

The confidence summary covers all 93,998 Stage 4 detection occurrences:

| Class | Occurrences | Mean | Median | p10 | p25 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 147 | 0.483 | 0.469 | 0.289 | 0.351 | 0.628 | 0.681 |
| bus | 1,955 | 0.626 | 0.639 | 0.294 | 0.399 | 0.859 | 0.926 |
| car | 42,268 | 0.595 | 0.591 | 0.297 | 0.389 | 0.813 | 0.882 |
| motorcycle | 15,398 | 0.653 | 0.682 | 0.393 | 0.553 | 0.778 | 0.854 |
| person | 29,354 | 0.671 | 0.777 | 0.318 | 0.466 | 0.852 | 0.879 |
| truck | 4,876 | 0.413 | 0.382 | 0.273 | 0.313 | 0.470 | 0.589 |

**Confidence does not represent correctness.** For example, several detections
near 0.25 were visually plausible, while the questionable Taipei bicycle box
also had confidence above the configured threshold. The lower truck and bicycle
distributions identify useful review strata, not error rates.

## Observed failure modes

### Highway

Nearby cars were generally detected plausibly. In dense or distant traffic,
small and partially occluded vehicles were visibly missed. Commercial vans,
minibuses, and utility vehicles moved between `car`, `truck`, and `bus`; these
cases were recorded as taxonomy ambiguity instead of automatically being called
model errors.

The targeted highway `person` detections at frames 959, 982, and 997 correspond
to visible drivers or occupants through windshields. They are semantically valid
COCO `person` detections, but do not mean pedestrian activity on the highway.
This is an application taxonomy mismatch between `person` and the desired
roadway-pedestrian concept.

Representative example:
[`frame 982 — highway person`](../outputs/error_analysis/stage5/frames/pexels_2103099/frame_000982_highway_person.jpg)

### Taipei intersection and scooters

Visible scooters were usually represented by the pretrained COCO `motorcycle`
class, while riders generated separate `person` occurrences. This mapping is
workable for a baseline, but dense queues and mutual occlusion produced missed
scooters and riders. Detection occurrences must therefore not be interpreted as
unique vehicles or riders.

At frame 5576, a low-confidence `bicycle` box does not align with a clearly
visible bicycle and is a likely false positive or bicycle/motorcycle confusion
candidate. The inspected sample does not provide enough evidence to claim a
systematic confusion rate. At frame 5905, a blurred commercial van labeled
`truck` is taxonomically ambiguous rather than a definitive class error.

Representative examples:

- [`frame 5576 — bicycle candidate`](../outputs/error_analysis/stage5/frames/pexels_13258685/frame_005576_bicycle.jpg)
- [`frame 5905 — commercial vehicle`](../outputs/error_analysis/stage5/frames/pexels_13258685/frame_005905_commercial_vehicle.jpg)

### Urban multi-class scene

The detector handled many visible cars, buses, scooters, riders, and the sampled
cyclist. The cyclist at frames 1907 and 1910 was detected as `bicycle` at low
confidence, showing that low confidence alone is not an error. In the densest
groups, overlapping riders and scooters were visibly missed. Large cropped
commercial vehicles also exposed the same bus/truck/van boundary seen in the
other scenes.

Representative examples:

- [`frame 1351 — dense traffic`](../outputs/error_analysis/stage5/frames/pexels_37258214/frame_001351_high_detection_density.jpg)
- [`frame 1907 — visible bicycle`](../outputs/error_analysis/stage5/frames/pexels_37258214/frame_001907_bicycle.jpg)

### Aerial intersection

This scene showed the clearest scale-related gap. Cars were detected much more
consistently than tiny scooters, motorcycles, and their riders. Across uniform
and targeted samples, multiple visible small road users had no boxes even when
nearby cars were detected. The oblique aerial viewpoint, small pixel footprint,
and overlap at the intersection jointly contribute to these false negatives.
A small yellow commercial minibus/van labeled `truck` remains a taxonomy issue.

Representative examples:

- [`frame 314 — dense aerial traffic`](../outputs/error_analysis/stage5/frames/pexels_9322363/frame_000314_high_detection_density.jpg)
- [`frame 366 — aerial small objects`](../outputs/error_analysis/stage5/frames/pexels_9322363/frame_000366_aerial_small_object.jpg)

## Taxonomy findings

- `motorcycle` is the available pretrained label for scooters; the application
  should document that mapping explicitly rather than silently renaming raw
  model output.
- `bicycle` and `motorcycle` deserve focused review in small or occluded cases,
  but this sample supports only a confusion candidate, not a measured trend.
- Passenger van, commercial van, minibus, truck, and bus boundaries are visually
  and semantically ambiguous. An application annotation policy is required
  before these can be judged consistently.
- COCO `person` includes drivers, passengers, riders, and pedestrians. A traffic
  application that needs pedestrians must add contextual rules or a more precise
  downstream taxonomy; `person` occurrences are not pedestrian counts.

## Taiwan domain gap and next-step rationale

The scooter-heavy Taipei and urban scenes, the local commercial-vehicle mix,
and the aerial scene's missed small scooters/riders provide qualitative evidence
of a domain gap worth investigating. The evidence supports evaluating a
Taiwan-specific annotated dataset and a clearly defined local taxonomy in a
later stage. It does **not** establish that fine-tuning will necessarily improve
performance; that claim would require controlled ground truth, held-out
evaluation, and comparison against this pretrained baseline.

## Limitations

- The 96 frames come from only four videos and include temporally nearby frames.
- Targeted selection intentionally overrepresents suspected difficult cases.
- Review was qualitative and frame-level, not exhaustive object annotation.
- There is no ground truth, so Precision, Recall, mAP, or error rates cannot be
  computed or inferred from the review labels.
- Stage 4's confidence threshold remained fixed at 0.25; no threshold tuning was
  performed.
- No tracking, identity reasoning, unique counting, trajectory, zone, or event
  analysis was performed.
