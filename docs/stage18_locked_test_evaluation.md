# Stage 18 — Pretrained vs Fine-tuned Locked Test

## Evaluation integrity

Stage 18 is the first and only use of the 271-image `LOCKED_TEST` established
in Stage 16. Materialization copied exactly those 271 image/label pairs and no
`TRAIN`, `VAL`, or `EXCLUDED` samples. Raw label bytes were unchanged. The
sealed evaluation manifest contains every Locked Test image ID, both model
hashes, the Stage 16 manifest hash, the common protocol, and the predeclared
promotion policy.

COCO pretrained class IDs differ from the Taiwan dataset's raw IDs. Therefore,
both models' prediction IDs were mapped by class name to the shared application
taxonomy before evaluation. Ground-truth labels remained unchanged. This avoids
silently comparing, for example, COCO `person=0` against Taiwan `bicycle=0`.

Both models used MPS, image size 640, batch 8, inference confidence floor
0.001, NMS IoU 0.7, maximum 300 detections, and AP IoU thresholds 0.50–0.95.
Precision and recall follow the Ultralytics maximum-mean-F1 reporting point;
mAP integrates the confidence-ranked precision/recall curve above the common
inference floor.

## Locked Test metrics

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Pretrained | 0.3731 | 0.2683 | 0.2253 | 0.0897 |
| Fine-tuned | 0.8250 | 0.5592 | 0.6462 | 0.3712 |
| Delta | +0.4519 | +0.2910 | +0.4209 | +0.2815 |

| Class | Pretrained P/R/mAP50/mAP50-95 | Fine-tuned P/R/mAP50/mAP50-95 | Delta P/R/mAP50/mAP50-95 |
|---|---|---|---|
| bicycle | .2356 / .0172 / .0233 / .0042 | .7808 / .6207 / .6882 / .3577 | +.5453 / +.6034 / +.6649 / +.3534 |
| bus | .6551 / .5508 / .5618 / .2643 | .8814 / .7754 / .8628 / .5921 | +.2263 / +.2246 / +.3010 / +.3278 |
| car | .6668 / .3733 / .4163 / .1223 | .8197 / .7461 / .8358 / .4699 | +.1530 / +.3727 / +.4195 / +.3476 |
| person | .0375 / .2957 / .0213 / .0056 | 1.0000 / 0.0000 / .1150 / .0307 | +.9625 / **-.2957** / +.0937 / +.0251 |
| motorcycle | .3215 / .0671 / .0608 / .0122 | .7921 / .6438 / .7436 / .3598 | +.4706 / +.5767 / +.6829 / +.3475 |
| truck | .3221 / .3056 / .2683 / .1294 | .6759 / .5694 / .6317 / .4170 | +.3538 / +.2639 / +.3634 / +.2875 |

## Person regression

Fine-tuning raises person AP but reports zero person recall at the model-wide
maximum-mean-F1 operating point, versus 0.2957 for pretrained. The low
confidence-floor diagnostic still finds IoU≥0.5 same-class candidates for 79
of 115 person labels, explaining why person AP is non-zero and improves. This
does not negate the operating-point regression: an application-critical class
would be absent at the reported common operating policy. No threshold was
changed or explored after observing this result.

## Diagnostic error review

Diagnostic analysis reused the already sealed prediction CSVs and did not run
inference again. It is heuristic explanatory analysis, not another metric or a
tuning source.

- Fine-tuned same-class IoU≥0.5 candidates increase sharply for bicycle, bus,
  car, motorcycle, and truck.
- Fine-tuned miss/localization candidates are dominated by small objects:
  34/35 person, 317/317 motorcycle, and 144/146 car candidates have normalized
  bbox area below 0.01.
- Fine-tuned confusion candidates include bicycle→motorcycle (3), bus→car or
  truck (3 each), motorcycle→car (12), and truck→car (8). These align with
  scooter/bicycle and commercial-vehicle taxonomy ambiguity.
- Visual samples show distant, very small targets, dense scooter groups,
  partial occlusion, and low-detail boxes. Occlusion is a qualitative
  observation because the source annotations do not encode an occlusion flag.

## Promotion decision

Decision: **REJECT**.

The candidate delivers large aggregate and five-class improvements, including
the Taiwan-critical motorcycle class. However, the promotion policy was fixed
before evaluation and rejects a person recall regression larger than 0.10. The
observed delta is -0.2957, so aggregate mAP cannot override that gate. Stage 18
evaluation integrity is PASS even though model promotion is REJECT.

The Stage 17 model remains a candidate, is not installed as the runtime model,
and must not be modified and reevaluated on this Locked Test. Future model work
requires a new development/validation cycle and a separately governed future
test set; this sealed set cannot be used for tuning.
