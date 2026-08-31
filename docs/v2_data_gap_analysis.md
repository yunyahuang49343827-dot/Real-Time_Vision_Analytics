# V2-1 Data Gap Analysis & Improvement Plan

## Scope and guardrails

This is a planning-only analysis of existing Stage 16–19 artifacts. It did not download data, change labels, run inference, train a model, or modify V1 artifacts. The Stage 18 `LOCKED_TEST` is sealed: its frozen reports are cited as historical evidence, but its images, predictions, and outcomes may not be used to choose V2 thresholds, training parameters, augmentations, architectures, or checkpoints.

Evidence statements below are explicitly separated from hypotheses and planned experiments. A hypothesis is not a proven root cause.

## V1 evidence summary

### Observed Evidence — dataset shape and class balance

Stage 16 retained 1,808 images and 48,380 valid boxes. The source-to-application mapping remained `human → person` and `motorbike → motorcycle`; raw labels were not rewritten.

| Application class | Images | Boxes | Share of boxes |
|---|---:|---:|---:|
| bicycle | 333 | 461 | 0.95% |
| bus | 884 | 1,217 | 2.52% |
| car | 1,802 | 21,136 | 43.69% |
| person | 190 | 772 | 1.60% |
| motorcycle | 1,792 | 23,464 | 48.50% |
| truck | 888 | 1,330 | 2.75% |

Person occurs in only 190 images (10.5% of included images), and bicycle in 333 (18.4%). Motorcycle has about 30.4 times as many boxes as person and 50.9 times as many as bicycle. The training split contains 570 person boxes and 333 bicycle boxes, compared with 16,133 motorcycle and 14,056 car boxes.

This confirms class imbalance. It does not by itself prove that imbalance caused the person failure: diversity, annotation completeness, object size, role semantics, and optimization behaviour can matter more than box count alone.

### Observed Evidence — small-object distribution

Stage 16 reports 40,850/48,380 boxes (84.44%) with normalized area below 0.01. Median normalized area is 0.00214. Stage 18 frozen diagnostics for the fine-tuned model show that small boxes account for:

- 34/35 person miss-or-localization candidates (97.1%).
- 317/317 motorcycle miss-or-localization candidates (100%).
- 144/146 car miss-or-localization candidates (98.6%).

These diagnostics describe where failures occurred; they do not establish that input resolution alone caused the failures. Occlusion, label policy, limited pixels, crowding, and localization tolerance are confounded.

### Observed Evidence — person performance and operating point

Stage 17 VAL person metrics were precision 1.0000, recall 0.0000, mAP50 0.1041, and mAP50-95 0.0358. On the sealed Stage 18 test, fine-tuned person recall remained 0.0000 at the common Ultralytics maximum-mean-F1 reporting point, versus pretrained recall 0.2957; this -0.2957 regression triggered the predeclared rejection gate.

The sealed low-confidence diagnostic nevertheless found same-class IoU≥0.5 candidates for 79/115 fine-tuned person labels, versus 83/115 for pretrained. Fine-tuned person predictions had maximum confidence 0.2495 and none reached 0.25, while person AP remained non-zero. This is evidence that score distribution/operating-point behaviour is part of the observed failure. It is not permission to tune a threshold on the sealed test.

Stage 18 person miss/localization candidates were dominated by small objects, and one person confusion candidate mapped to bicycle. Stage 16 manual review also found sparse `human` labels and cases where riders were represented only inside `motorbike` boxes.

### Observed Evidence — downstream system effects

Stage 19 used the pretrained runtime model, not the rejected candidate. In sampled review:

- 23 physical objects produced two fragmentation candidates and no verified ID switch.
- Three natural wrong-way candidates were all false events driven by perspective-sensitive direction/jitter.
- Nine proximity warnings were reviewed: four useful review candidates, three false events, and two ambiguous.
- Rider/self-motorcycle association, person coverage, occlusion, and image-space geometry limited proximity interpretation.
- Natural pedestrian-intrusion positives were absent; only a clearly labeled controlled synthetic positive path was validated.

These system observations show how person misses, rider semantics, detection gaps, and bbox jitter can propagate. They are manually sampled and are not full-dataset accuracy estimates.

## Confirmed observations versus hypotheses

| Topic | Observed Evidence | Hypothesis | Planned Experiment |
|---|---|---|---|
| Person scarcity | 772 person boxes; 570 in TRAIN; person is 1.60% of boxes. | Insufficient diversity/coverage may weaken representation and confidence calibration. | Add governed person-focused data, then retrain the same YOLO26n/640 baseline before changing architecture. |
| Small persons | 34/35 fine-tuned person miss/localization candidates are small. | More retained pixels may improve feature quality for some small people. | Compare YOLO26n at 640 and 960 on the same frozen V2 Train/Val; do not assume 960 wins. |
| Operating point | Fine-tuned person max score was 0.2495; reported recall was 0 despite low-floor matches. | Class imbalance, calibration, semantics, or training dynamics may suppress person scores. | Measure per-class PR/score distributions on V2 VAL only under identical protocols; thresholds remain an explicitly governed later decision. |
| Rider semantics | Stage 16 found riders sometimes only inside `motorbike` boxes; Stage 19 found rider/self-vehicle proximity errors. | Inconsistent person+rider completeness may teach conflicting person/motorcycle boundaries. | Freeze a rider/person/vehicle annotation policy and audit annotation completeness before training. |
| Occlusion/density | Qualitative reviews show dense scooters, partial occlusion, and temporary misses. | More occlusion and crowd coverage may reduce missing tracks and proximity ambiguity. | Add coverage-tagged scenes and report metrics segmented by occlusion/density on V2 VAL. |
| Commercial vehicles | car/bus/truck confusion and class instability appear in Stages 16, 18, and 19. | Ambiguous van/minibus/pickup rules create inconsistent supervision and analytics. | Apply a written vehicle policy, log ambiguous cases, and inspect per-class confusion on V2 VAL. |

## Person gap analysis

The evidence supports a multi-factor risk, not a single proven cause:

1. **Training coverage risk:** 570 TRAIN person boxes are small relative to car and motorcycle, and image-level occurrence is sparse.
2. **Small-object dominance:** nearly all fine-tuned person miss/localization candidates are below 0.01 normalized area.
3. **Taxonomy translation:** `human → person` is an application mapping, not evidence that the source's `human` inclusion policy matches COCO or the V2 operational definition.
4. **Rider semantics:** a visible rider may be separately labeled, omitted, or subsumed in a motorbike box. This affects both person learning and downstream rider/self-vehicle exclusion.
5. **Occlusion and density:** dense scooter/pedestrian scenes reduce visible extent and can cause detection/track gaps.
6. **Operating-point behaviour:** low-floor localization candidates coexist with zero reported recall. Score calibration/training dynamics are plausible contributors, but the sealed test cannot be used to select a threshold.

## V2 annotation policy proposal

Source taxonomy and application taxonomy remain separate. Every source label must be retained unchanged with a versioned mapping and policy metadata.

- **person:** detection class for a separately localizable visible human extent. Record `occluded`, `truncated`, `small`, and role attributes when the format supports them.
- **pedestrian:** role attribute under `person` for walking, standing, or crossing; not a competing detection class.
- **rider:** role attribute under `person`. When sufficiently visible, annotate the person and the ridden vehicle separately.
- **motorcycle:** scooter/motorcycle vehicle extent. Exclude the rider from the vehicle extent where consistently separable; document the chosen bounding convention.
- **bicycle:** bicycle vehicle extent; separately annotate a visible rider as person.
- **car:** passenger car/SUV/taxi and passenger van unless a source-specific documented policy says otherwise.
- **bus:** passenger-carrying bus/minibus under a consistent size/use rule.
- **truck:** freight truck/pickup/goods van under a consistent use/body rule.
- **commercial van ambiguity:** retain the source label, flag the sample for policy review, and do not silently force uncertain cases across car/bus/truck.

V1 raw annotations remain immutable. Any V2 correction or harmonization must be a derived, versioned annotation layer with provenance back to the raw label.

## V2 data acquisition specification

No data is acquired in V2-1. A future acquisition should prioritize coverage rather than headline image count:

- Taiwan or closely comparable urban traffic context.
- Crosswalks and pedestrian-heavy intersections.
- Pedestrian–scooter and pedestrian–car interaction.
- Distant/small people, motorcycles, and cars.
- Partial occlusion, dense traffic, and multiple viewing angles.
- Clearly separated rider/person and motorcycle/bicycle examples.
- Commercial vans, minibuses, pickups, buses, and trucks near taxonomy boundaries.
- Multiple independent camera/sequence groups and varied lighting.

Required annotation format is object-detection bounding boxes. Each source needs an explicit license, intended-use compatibility decision, source URL/owner/version/access date, immutable archive hash, and documented taxonomy. Person annotation completeness and rider/vehicle pairing must be audited; a large dataset with missing people fails acceptance even if its image count is high.

## Data acceptance criteria

A candidate source is accepted only when:

1. License and provenance are explicit and compatible with intended use; conflicts are rejected or quarantined.
2. Images decode, annotations pass class/coordinate checks, and exclusions retain reasons.
3. Raw labels remain immutable and source-to-application mapping is versioned.
4. A stratified manual audit checks person completeness, rider/vehicle pairing, and commercial-vehicle ambiguity.
5. Coverage matrix includes every required scene/object tag and reports person size/occlusion distributions.
6. SHA256 exact duplicates, explainable near-duplicate grouping, and camera/sequence grouping prevent leakage.
7. No V2 asset/group overlaps V1 Stage 18 Locked Test.

Hard numeric completeness targets should be fixed before QA is run on candidate data, not invented after seeing results. The machine-readable config defines stable binary gates now and leaves distribution targets to a pre-registered acquisition protocol.

## V2 experiment priority

Experiments will run in fixed order, with V2 Train/Val as the only model-selection data:

1. Expand governed data and freeze/refine the annotation policy.
2. Retrain YOLO26n at `imgsz=640` as the controlled V2 baseline.
3. Compare YOLO26n at `imgsz=960` using the same split and otherwise controlled protocol. Higher resolution is a hypothesis, not a promised improvement; MPS memory, throughput, localization, and per-class metrics must all be reported.
4. Compare YOLO26s only if data improvements and both YOLO26n experiments remain insufficient.
5. After model selection is frozen, run one final evaluation on a new sealed V2 Holdout and then conduct downstream Stage 19-style system regression review.

Per-class gates must include person recall/PR behaviour, bicycle, motorcycle, and car rather than relying on aggregate mAP. Runtime cost and system propagation are secondary promotion gates; no candidate replaces the V1 runtime solely because aggregate detection metrics improve.

## V2 evaluation governance

The old Stage 18 set remains `SEALED_DIAGNOSTIC_REFERENCE_ONLY`. It is excluded from V2 Train, Val, threshold selection, checkpoint selection, experiment selection, and architecture choice.

The new V2 Holdout will be created only after acquisition, duplicate grouping, annotation-policy freeze, and V2 Train/Val definition. It must:

- Be an independent group-aware partition with no camera/sequence or duplicate-group overlap.
- Have no image/group overlap with V1 Stage 18 Locked Test.
- Store image IDs, source/group IDs, image and annotation hashes, manifest hash, and dataset-tree hash.
- Remain inaccessible for training and model selection.
- Be used once for final evaluation and promotion decision.
- Permit post-decision diagnostic explanation only; any resulting model change requires a future holdout.

## Limitations and open questions

- Stage 18 diagnostics are frozen post-hoc explanations, not a tuning source.
- Occlusion and rider roles were not encoded as structured attributes in V1, so current counts cannot isolate their causal effects.
- Stage 19 review is sampled and cannot estimate population tracking/event accuracy.
- Acquisition quotas and completeness thresholds must be preregistered after candidate-source discovery but before inspecting candidate QA outcomes.
- A 960-pixel input may improve, degrade, or leave small-object performance unchanged and will reduce throughput; only a controlled V2 VAL experiment can resolve this.
