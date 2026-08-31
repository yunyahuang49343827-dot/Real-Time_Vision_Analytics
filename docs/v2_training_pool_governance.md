# V2-3 governed training pool construction

Date: 2026-08-31

Stage result: **PARTIAL**

Training pool status: **NOT_APPROVED_FOR_V2_TRAINING**

## Executive decision

The governance pipeline, per-image license checks, traffic-domain filters, unified taxonomy, duplicate-aware split and leakage validations were completed. However, only **191 Open Images images** passed every hard gate, versus the required **400–500**. The generated Taiwan + Open Images manifests are therefore marked `PROVISIONAL_DO_NOT_TRAIN`; they are integrity-test artifacts, not authorization to start V2 training.

No gate was relaxed to reach the requested count. No training or Final V2 Holdout was performed.

## Inputs and intended grain

The Open Images review grain is one source image. A source image becomes individually eligible only when all of the following pass:

1. current per-image license evidence;
2. traffic-domain relevance;
3. annotation review status;
4. source/application taxonomy mapping;
5. image decode and bbox geometry;
6. duplicate and leakage governance.

The combined dataset manifest grain is one image. The unified annotation table grain is one source bbox. Raw source taxonomy remains in `source_class`; mapped V2 taxonomy is stored separately in `application_class` and `derived_annotation`.

## Open Images review population

The V2-2C candidate pool had 5,190 validation images. V2-3 constructed a deterministic, coverage-balanced worklist of 650 images, prioritizing Person with motorcycle, bicycle, bus/truck, small Person, occlusion and multi-traffic context. All 650 review images decoded successfully.

The initial 50-image manual contact-sheet review from V2-2C was retained as an explicit override. An additional 150-image contact-sheet calibration confirmed that generic Person+Car metadata frequently represented static cars, exhibitions, motorsport, stunts or other non-operational contexts. Remaining statuses were assigned by conservative, inspectable rules recorded in code and in `governed_review_decisions.csv`:

- explicit street/traffic/road/city/transit terms: `TRAFFIC_RELEVANT`;
- motorcycle/bicycle/bus/truck/multi-traffic without explicit negative context: `PARTIALLY_RELEVANT` with a rare-coverage reason;
- race, rally, show, museum, stunt, velodrome, exhibition and similar terms: `NON_TRAFFIC`;
- generic Person+Car without auditable road semantics: `AMBIGUOUS`.

`PARTIALLY_RELEVANT` passes only for rare coverage with a non-empty justification. `NON_TRAFFIC` and `AMBIGUOUS` never pass.

## Per-image license governance

Open Images metadata was not treated as approval. Each selected Flickr landing page was fetched independently. `LICENSE_APPROVED` required:

- HTTP 200 from the current individual photo page;
- complete image ID, original URL, landing page, author and metadata license URL;
- current Flickr JSON-LD `ImageObject.license` equal to CC BY 2.0;
- `acquireLicensePage` equal to the reviewed landing page;
- Flickr licensed `contentUrl` containing the same photo ID.

The evidence table retains review time, HTTP status, landing-page SHA256, observed license, acquire-license page, content URL and creator. Unresolved pages were retried once; only three additional approvals resulted.

| License status | Images |
|---|---:|
| `LICENSE_APPROVED` | 509 |
| `REQUIRES_REVIEW` | 123 |
| `REJECTED` | 18 |

The 141 unresolved/rejected images cannot enter training.

## Domain and annotation review

| Domain status | Images |
|---|---:|
| `TRAFFIC_RELEVANT` | 28 |
| `PARTIALLY_RELEVANT` | 170 |
| `NON_TRAFFIC` | 23 |
| `AMBIGUOUS` | 429 |

| Annotation status | Images |
|---|---:|
| `ACCEPTABLE` | 46 |
| `ACCEPTABLE_WITH_NOTE` | 462 |
| `REJECTED_INCOMPLETE` | 2 |
| `REJECTED_AMBIGUOUS` | 140 |

`ACCEPTABLE_WITH_NOTE` means normalized geometry, official MID mapping, depiction/group-of filtering and image decode passed, but the rule-assisted/contact-sheet review is not proof of exhaustive annotation completeness. Very dense scenes beyond contact-sheet resolution were conservatively rejected as ambiguous. Two prior manual-review images were rejected for apparent missing individual targets.

## Final Open Images hard-gate result

Only 191 images passed license, domain, annotation and source-candidate gates. Rejection causes use the first failed hard gate:

| Rejection cause | Images |
|---|---:|
| Domain ambiguous | 296 |
| License requires review | 123 |
| Domain non-traffic | 19 |
| License rejected | 18 |
| Annotation incomplete | 2 |
| Annotation ambiguous | 1 |

The approved count is **209 images below** the required minimum. These 191 images are individually governed candidates, but they do not constitute an approved V2 Open Images pool at the requested scale.

### Governed Open Images coverage

| Application class | Images | Boxes |
|---|---:|---:|
| person | 191 | 414 |
| bicycle | 71 | 130 |
| car | 49 | 110 |
| motorcycle | 43 | 63 |
| bus | 24 | 31 |
| truck | 44 | 52 |

Person coverage includes 170 small boxes and 256 occluded boxes. This is useful incremental coverage, but no claim is made that it will improve a model.

## Taiwan CCTV integration and taxonomy

Only the Stage 16 `TRAIN` and `VAL` records were eligible for the provisional V2 pool: 1,537 Taiwan images. Stage 16 `LOCKED_TEST` and `EXCLUDED` were not included.

The six application classes remain:

```text
person, bicycle, car, motorcycle, bus, truck
```

Taiwan `human → person` and `motorbike → motorcycle` are derived mappings; raw labels are unchanged. Open Images official classes/MIDs are also preserved in source annotations. Pedestrian remains an application role, not a competing class. Rider semantics were not rewritten or relabeled.

The unified `v2_annotations.csv` retains:

- `dataset_source` and `source_image_id`;
- `source_class` and `application_class`;
- original annotation representation and derived normalized representation;
- source row, normalized geometry, size bin and available Open Images attributes.

## Provisional TRAIN / VAL integrity result

The 191 governed Open Images candidates were combined with 1,537 non-test Taiwan images solely to exercise the split and integrity pipeline.

| Split | Taiwan images | Open Images images | Total |
|---|---:|---:|---:|
| TRAIN | 1,233 | 153 | 1,386 |
| VAL | 304 | 38 | 342 |
| Total | 1,537 | 191 | 1,728 |

Splitting is deterministic with seed 2301, source-aware, source-group-aware and dHash/SHA256 duplicate-aware. Both sources appear in TRAIN and VAL. Every row records `split`, `dataset_source`, `group_id`, `image_sha256` and `annotation_sha256`.

All split manifests are explicitly marked `PROVISIONAL_DO_NOT_TRAIN` because the Open Images target-size gate failed.

## Class distribution

### TRAIN bbox counts

| Class | Taiwan | Open Images | Combined |
|---|---:|---:|---:|
| bicycle | 331 | 108 | 439 |
| bus | 815 | 23 | 838 |
| car | 13,979 | 90 | 14,069 |
| motorcycle | 15,719 | 52 | 15,771 |
| person | 579 | 343 | 922 |
| truck | 871 | 44 | 915 |

### VAL bbox counts

| Class | Taiwan | Open Images | Combined |
|---|---:|---:|---:|
| bicycle | 72 | 22 | 94 |
| bus | 215 | 8 | 223 |
| car | 3,727 | 20 | 3,747 |
| motorcycle | 4,064 | 11 | 4,075 |
| person | 78 | 71 | 149 |
| truck | 243 | 8 | 251 |

Stage 16 V1 TRAIN contained 570 Person boxes. The provisional V2 TRAIN contains 922, a descriptive increase of 352. This is not evidence of improved model performance, and the provisional pool must not be trained.

## Duplicate, leakage and immutability checks

- Cross-split governance groups: 0.
- Stage 18 exact overlap: 0.
- Stage 18 dHash-near overlap at threshold 6: 0.
- Stage 18 usage: `OVERLAP_CHECK_ONLY`.
- Taiwan raw tree SHA256 remained `8440bba4d4f374c0532970a8e4dbbb0928db9fe20992646804517216f9acfa24`.
- All three V2-2C Open Images metadata SHA256 values matched their recorded immutable values.
- Additional review images were written under `data/interim/`, not into Taiwan raw data or the existing Open Images raw metadata/pilot artifacts.
- Final V2 Holdout created: false.
- Training performed: false.

dHash grouping is a documented near-duplicate heuristic, not ground-truth scene identity.

## Required remediation

The smallest safe next step is not training. To make a future pool approvable, obtain at least 209 additional images that independently pass all current gates. Given the high rate of generic/non-traffic Open Images candidates, this likely requires a new targeted source or a new acquisition specification rather than loosening domain or license policy. Any continuation must preserve the current Stage 18 seal and rerun combined duplicate/leakage checks.

## Artifacts

- Config: `configs/v2_training_pool.yaml`
- Builder: `scripts/build_v2_training_pool.py`
- Governance helpers: `src/vision_analytics/utils/v2_training_pool.py`
- Tests: `tests/test_v2_training_pool.py`
- Generated outputs: `outputs/data_qa/v2_training_pool/`
- Summary: `outputs/data_qa/v2_training_pool/coverage_summary.json`

Generated images, contact sheets and data-QA outputs remain Git-ignored. Only code, configuration, tests and this governance decision document are committed.
