# V2 Targeted Annotation Candidate Set

Dataset status: `TARGETED_ANNOTATION_CANDIDATE_SET`  
Annotation status: `NOT_ANNOTATED`

This directory contains deterministic JPG frames selected from the four governed
Pexels runtime videos. The images are prepared for manual review and later import
into an annotation tool. They do not contain labels or Ground Truth annotations.

## Frozen V2 annotation classes

Annotators must use only these application classes:

1. `person`
2. `bicycle`
3. `car`
4. `motorcycle`
5. `bus`
6. `truck`

`pedestrian` is an application role, not a separate detection class. A visible
rider must be annotated with separate boxes: one `person` box and one
`motorcycle` or `bicycle` box. Do not infer or annotate an invisible rider.

Commercial vans remain a taxonomy-review case; do not silently force an ambiguous
vehicle into `car`, `bus`, or `truck`. Record the ambiguity for policy review.

## Governance and import procedure

- `images/` contains only rows whose manifest status is
  `SELECTED_FOR_ANNOTATION`.
- Stage 19 evaluation frame ranges are hard-excluded and have zero overlap.
- Stage 4 pretrained predictions and coverage tags are sampling assistance only:
  `model_prediction != ground_truth`.
- Every row remains `annotation_status=NOT_ANNOTATED` until a human annotation
  workflow is completed and separately governed.
- Raw MP4 files are immutable and were SHA256-verified before and after export.
- Import only `images/` into the annotation tool; never import `mined/` or contact
  sheets as annotation images.
- No fake labels, pseudo-labels, V2 Holdout, or training artifacts are present.

The auditable candidate manifest is
`data/manifests/v2_targeted_annotation_candidates.csv`. Generated JPGs and contact
sheets are intentionally excluded from Git.
