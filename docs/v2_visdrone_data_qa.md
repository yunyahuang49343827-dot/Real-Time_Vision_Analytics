# V2-2 VisDrone2019-DET Data Acquisition and QA

## Decision

**Acceptance decision: `ACCEPT_WITH_WARNINGS`.** VisDrone2019-DET materially expands person, small-object, dense aerial traffic, occlusion, bicycle, motorcycle, and vehicle coverage. It is technically suitable as a candidate V2 source, but it remains `QUARANTINED_PENDING_LICENSE_REVIEW` and is **not yet approved for training** because the owner does not publish an explicit dataset license/usage grant in the official repository. This unresolved gate makes V2-2 `PARTIAL`, not `PASS`.

This stage downloaded and inspected raw data only. It did not train a model, create a final V2 Holdout, modify V1 raw data, or modify/use Stage 18 Locked Test for model selection.

## Source, provenance, and usage terms

- Dataset: VisDrone2019-DET, detection in static images, 2019 release.
- Owner: AISKYEYE team, Lab of Machine Learning and Data Mining, Tianjin University, China.
- Official source: [VisDrone dataset repository](https://github.com/VisDrone/VisDrone-Dataset).
- Official source coverage: train, val, and annotated test-dev. Test-challenge was not downloaded because its annotations are unavailable.
- Access date: 2026-08-31.
- License: `NOT_EXPLICITLY_STATED`.
- License status: `REQUIRES_REVIEW`.

The official repository publishes the dataset, owner, archive links, and citation but no dataset `LICENSE` or explicit usage grant. The official DET toolkit's “research purpose only” statement describes the toolkit/code library and is not treated as a dataset license. Commercial, redistribution, derivative-annotation, and production-training permissions must not be inferred. See [source research](v2_visdrone_source_research.md) for primary-source evidence.

## Raw acquisition and integrity

Three owner-published Google Drive archives were saved under Git-ignored `data/raw/visdrone2019_det/archives/`. All passed ZIP CRC testing.

| Archive | Bytes | SHA256 |
|---|---:|---|
| VisDrone2019-DET-train.zip | 1,549,875,511 | `86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84` |
| VisDrone2019-DET-val.zip | 81,638,851 | `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9` |
| VisDrone2019-DET-test-dev.zip | 311,045,829 | `78b0c5078a14ee43d0b803a354e76016d7260d1704cfd1c2dc821858d839e261` |

- Archive total: 1,942,560,191 bytes (1.81 GiB).
- Extracted tree: 1,950,992,687 bytes (1.82 GiB).
- Extracted raw-tree SHA256: `32fca3168463205b8c4e8afc7f229797938b369d0ba49439c8027428b48523fe`.
- Raw archives, images, and annotations remain byte-preserved and Git-ignored.

The official source does not publish archive hashes, so these local hashes identify the acquired artifacts but cannot independently prove equivalence to an upstream checksum.

## Annotation format and parsed representation

VisDrone DET is not YOLO format. Each raw row is an eight-column pixel-space CSV record:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,class_id,truncation,occlusion
```

The official specification defines `score=0` rows/ignored regions as excluded from evaluation; truncation is 0/1 and occlusion is 0/1/2 (none/partial/heavy). Raw `.txt` files were not rewritten. A separate normalized representation was generated at `outputs/data_qa/v2_visdrone/parsed_annotations.csv`, retaining pixel boxes, source class ID/name, score, truncation, occlusion, normalized center/size/area, size bin, application mapping, and disposition.

## Image and annotation QA

- Images/annotation pairs: 8,629 / 8,629.
- OpenCV-readable images: 8,629 (100%).
- Valid parsed rows: 471,260.
- V2 target rows after mapping/ignore disposition: 413,597.
- Ignored rows: 14,197, including 12,369 `ignored_regions` and 1,828 `others`/score-zero rows.
- Annotation issues: 6.
  - Three exact duplicate annotation rows; omitted only from parsed representation, raw retained.
  - Three zero-size bbox rows; excluded from parsed representation with explicit issue records, raw retained.
- Missing images/annotations: 0.
- Invalid class IDs, malformed rows, out-of-bounds boxes, unreadable images: 0.

Six generated annotation previews were visually inspected. They show dense aerial scenes, numerous very small people/vehicles, separate `pedestrian`/`people` source labels mapping to person, and retained occlusion metadata. Preview density makes text overlap, but bbox placement and small-object character are evident; previews are QA artifacts, not Ground Truth corrections.

## Source taxonomy and application mapping

Raw IDs and names remain unchanged.

| ID | Source class | V2 application class | Disposition |
|---:|---|---|---|
| 0 | ignored_regions | — | IGNORED_OR_OTHER |
| 1 | pedestrian | person | MAPPED |
| 2 | people | person | MAPPED |
| 3 | bicycle | bicycle | MAPPED |
| 4 | car | car | MAPPED |
| 5 | van | — | EXCLUDED_FROM_V2_TARGET |
| 6 | truck | truck | MAPPED |
| 7 | tricycle | — | EXCLUDED_FROM_V2_TARGET |
| 8 | awning-tricycle | — | EXCLUDED_FROM_V2_TARGET |
| 9 | bus | bus | MAPPED |
| 10 | motor | motorcycle | MAPPED |
| 11 | others | — | IGNORED_OR_OTHER |

`van`, `tricycle`, and `awning-tricycle` were not forced into incompatible target classes. Any later policy change requires a new derived mapping version; raw IDs remain immutable.

## Source class distribution

Counts below exclude ignored score-zero rows while preserving them separately in parsed data.

| Source class | Images | Boxes |
|---|---:|---:|
| pedestrian | 7,083 | 109,186 |
| people | 5,226 | 38,560 |
| bicycle | 3,496 | 13,069 |
| car | 8,178 | 187,003 |
| van | 6,537 | 32,702 |
| truck | 4,567 | 16,284 |
| tricycle | 2,270 | 6,387 |
| awning-tricycle | 1,604 | 4,377 |
| bus | 2,992 | 9,117 |
| motor | 5,516 | 40,378 |

## Person coverage

- `pedestrian`: 7,083 images / 109,186 boxes.
- `people`: 5,226 images / 38,560 boxes.
- Combined application `person`: 7,482 distinct images / 147,746 boxes.

This is substantially broader person coverage than V1 Taiwan's 190 images / 772 person boxes, particularly for distant aerial people and dense groups. It does not prove transfer benefit to Taiwan CCTV: geography, camera altitude, source label semantics, and domain appearance remain different.

## Small-object coverage

Size bins match V1 Taiwan QA: small `<0.01`, medium `0.01–<0.09`, large `≥0.09` normalized bbox area.

- All mapped target classes: 404,026 small boxes, 97.69% of 413,597 target boxes.
- Person: 147,617/147,746 small (99.91%).
- Motorcycle: 40,348/40,378 small (99.93%).
- Bicycle: 13,047/13,069 small (99.83%).
- Car: 180,481/187,003 small (96.51%).

Compared with V1 Taiwan's 84.44% small-box share, VisDrone clearly supplies additional small-object coverage. This is a coverage finding, not a claim that a future model will improve.

## Occlusion and truncation

| Application class | None | Partial | Heavy | Small + occluded |
|---|---:|---:|---:|---:|
| person | 79,574 | 58,269 | 9,903 | 68,128 |
| motorcycle | 12,960 | 20,990 | 6,428 | 27,400 |
| bicycle | 3,755 | 7,539 | 1,775 | 9,302 |
| car | 96,739 | 70,976 | 19,288 | 88,242 |

Truncation counts are also retained: person 2,478 partial, motorcycle 812, bicycle 316, and car 10,174. `occlusion_distribution.csv` preserves the full application class × size × occlusion × truncation breakdown for later V2 diagnostics.

## Duplicate and grouping governance

- Exact duplicates: 6 images in 3 SHA256 groups.
- Exact/near multi-member components: 135 images in 35 groups.
- Near-only members: 129 images in 32 groups.
- Near-duplicate heuristic: dHash-64, Hamming distance ≤ 6.
- Four near/exact components cross official source partitions and must remain grouped in any future V2 split.

Near-duplicate components are conservative leakage controls, not proof of identical scene identity. Some low-detail aerial images can collide under perceptual hashing and require manual review before final split governance.

The official DET documentation describes static images and does not publish a filename sequence grammar. A conservative prefix-based grouping produced 345 groups (train 208, val 76, test-dev 61), but every such group is explicitly labeled `INFERRED_NOT_OFFICIAL`. A future V2 splitter must keep both duplicate components and accepted conservative capture groups intact; this stage does not create a Holdout.

## V1 Stage 18 leakage protection

VisDrone inventory hashes/dHashes were compared with the 271-image V1 Stage 18 Locked Test inventory only for overlap protection:

- Exact SHA256 overlap: 0.
- Near dHash overlap at Hamming ≤ 6: 0.
- Stage 18 use: `OVERLAP_CHECK_ONLY`.

The old Locked Test remains sealed and was not used for taxonomy, model selection, threshold selection, training, or experiment choice.

## Acceptance rationale and next gate

VisDrone is an `ACCEPT_WITH_WARNINGS` **coverage candidate** because provenance, archive integrity, annotation readability, target coverage, metadata, duplicate controls, and V1 overlap checks are adequate. Warnings are material:

1. Dataset license/usage rights remain `REQUIRES_REVIEW`; training pool status is quarantined.
2. Three invalid zero-size boxes and three duplicate rows must remain excluded in any derived labels.
3. Unsupported classes remain excluded until a versioned taxonomy policy exists.
4. Filename grouping is inferred, not official.
5. Domain transfer to Taiwan traffic is unproven.

Before any V2 training, an authorized reviewer must resolve usage terms and record the decision. A later data-materialization stage must also combine VisDrone duplicate/capture groups with all other V2 sources and create only Train/Val; final V2 Holdout governance remains a separate future step.
