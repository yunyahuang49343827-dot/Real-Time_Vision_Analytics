# V2-2B UrbanScene Data Acquisition and QA

Review date: 2026-08-31 (Asia/Taipei)

## Decision

**Dataset acceptance: `REJECT`.**

- Suitability: `NOT_SUITABLE_FOR_YOLO_OBJECT_DETECTION`
- Training-pool decision: `REJECT_FOR_SUPERVISED_YOLO_TRAINING`
- V2-2B execution status: the acquisition and QA protocol completed successfully; rejecting the dataset for the requested supervised use is the gate's intended outcome.

The official Version 1 download contains exactly 16,426 readable JPG images arranged in image-category folders. It contains **zero** TXT, XML, JSON, CSV, or other paired object-annotation files and documents no bbox schema. Folder membership must not be converted into fabricated bounding boxes. UrbanScene may be useful as an image-level scene reference, but it cannot directly provide YOLO object-detection supervision unless a future, separately governed annotation effort is authorized.

## Provenance and raw integrity

- Dataset: *UrbanScene: An Extensive Multi-Object Dataset for Pedestrian, Traffic, and Motorbike Detection*
- DOI/version: `10.17632/5gt4fg4rvp.1`, Version 1
- Owner/contributors: Kailas PATIL; prawit chumchu; Siddharth Pashankar; Darshana Gatagat; Omkar Rumane
- Repository/institution: Mendeley Data; Kasetsart University Sri Racha Campus is listed on the record
- Dataset license: **CC BY 4.0, verified on the official Mendeley record**
- Access date: 2026-08-31
- Raw location: `data/raw/urbanscene/` (Git ignored)

The accompanying *Data in Brief* paper has its own CC BY-NC 4.0 article license. That article license was not substituted for the Mendeley dataset record's explicit CC BY 4.0 license. See [the official dataset record](https://data.mendeley.com/datasets/5gt4fg4rvp/1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the separate [primary-source research note](v2_urbanscene_source_research.md).

Three official per-file downloads were retained without modification. Each passed byte-count, repository-published SHA256, ZIP CRC, and unsafe-member-path checks:

| Archive | Bytes | SHA256 |
|---|---:|---|
| `Motorbikes_&_Cyclist.zip` | 1,168,975,616 | `42fd6393116e9817d456a03fce5b85afb81b309b0588dcae29d0e64b0c07c109` |
| `Pedestrians.zip` | 1,133,335,639 | `c6ebd3a86441d6109226dd3679c9e192d4f9ec73292365bc18b0a10949fcd33e` |
| `Traffic.zip` | 1,371,652,798 | `88732a107bc2ec1c2aea39503db895bb9c1e069f7e3b87621bd5528eacb7ff8c` |
| **Total** | **3,673,964,053** | — |

The deterministic extracted raw-tree SHA256 is `865ae4e0e66e2dfe0de7306c851d7ee2d2b0a69161cda0025c2198c90c6e2a5b`.

## Actual inventory and annotation gate

| Check | Result |
|---|---:|
| Files/images | 16,426 / 16,426 |
| Readable images | 16,426 |
| Image format | JPG only |
| Candidate annotation files | 0 |
| Paired annotation files | 0 |
| Object bboxes | 0 |
| Dominant resolution | 768 × 1024 (16,424 images) |
| Resolution exceptions | 638 × 1024 (1); 654 × 1024 (1) |

Actual folder tree:

```text
Motorbikes_and_Cyclist/
  Cyclist data {Morning, Evening, Night}/
  Motorbike data {Morning, Evening, Night}/
Pedestrians/output/
  {mrngresized, everesized, ngtresized}/
Traffic/Traffic/
  {Morning, Evening, Night}/
```

The publication title and prose use “detection,” but the delivered raw archive and author paper describe category-organized images and an image-classification experiment. No object-annotation parser is run because no published annotation schema or annotation artifacts exist. The QA layer deliberately returns `IMAGE_LEVEL_CATEGORY_ONLY`; possible future annotation-looking artifacts would be quarantined as `UNVERIFIED_ANNOTATION_CANDIDATES`, not guessed.

`annotation_issues.csv` records the dataset-level blocker and the two non-dominant-resolution warnings. There were no unreadable images. Empty-annotation, malformed-row, invalid-class, invalid-bbox, bbox-outside-image, duplicate-annotation, and pairing checks are not applicable because there are no annotation files; they are not reported as successful bbox QA.

## Source taxonomy and application semantics

These are source **folder categories**, not object class IDs:

| Source category | Images | Application semantic | Disposition |
|---|---:|---|---|
| Pedestrians | 4,106 | person | image-level semantic only; not bbox supervision |
| Motorbikes | 3,876 | motorcycle | image-level semantic only; rider and vehicle not separated |
| Cyclists | 1,217 | bicycle | ambiguous image-level rider/cycle scene; not bbox supervision |
| Traffic | 7,227 | unmapped | broad traffic scene; cannot split into car/bus/truck |

Source names remain unchanged in inventory and manifests. The semantic mapping is descriptive only and cannot be materialized as detection labels. In particular, `Traffic` is not decomposed, and a cyclist image does not establish separate `person` plus `bicycle` object boxes.

## Coverage findings

### Person and small objects

The `Pedestrians` folder contains 4,106 images, but **person image count under a verified detection annotation policy, person bbox count, and bbox completeness are unavailable**. It therefore cannot be compared numerically with V1 Taiwan's 772 person boxes as detection supervision.

Normalized bbox-area distributions for person, motorcycle, bicycle, and car are also `NOT_AVAILABLE_WITHOUT_BBOX_ANNOTATIONS`. No small/medium/large counts, occlusion statistics, ignored regions, or completeness claims were fabricated.

### Lighting and scene semantics

Folder-derived, metadata-confirmed time grouping is:

| Time | Images |
|---|---:|
| Morning | 6,453 |
| Evening | 3,915 |
| Night | 6,058 |

Manual inspection of pedestrian, motorbike, cyclist, and traffic samples across these folders confirmed visibly different lighting and urban road contexts. Examples included pedestrians near roads/crosswalks, a rider on a scooter, a cyclist with bicycle, broad mixed traffic, and night roadway scenes. This supports image-level scene diversity only. It does not establish per-object coverage, annotation completeness, crosswalk metadata, or detection utility.

## Duplicate and grouping governance

- Exact duplicate: 3 images in 1 SHA256 group (`night (1365).jpg`, `night (1366).jpg`, `night (1367).jpg`).
- dHash heuristic: 64-bit dHash, Hamming distance ≤ 6, with exact/near transitive components.
- Multi-member exact/near components: 2,125 groups containing 11,603 images.

The large near-duplicate result is consistent with many adjacent, visually similar captures; it is a heuristic grouping signal, not Ground Truth sequence identity. No official per-image sequence, capture, device, or location ID is present. Only category/time folders are retained as context and explicitly marked `FOLDER_GROUP_ONLY_NOT_SEQUENCE_ID`. If these images are ever re-annotated and split, exact/near components must remain intact; no final V2 Holdout was created here.

## Cross-dataset leakage checks

| Reference | Exact SHA256 overlap | dHash candidates (≤ 6) | Governance |
|---|---:|---:|---|
| V1 Taiwan, all images | 0 | 5 | heuristic review candidates only |
| Stage 18 old LOCKED_TEST | 0 | 0 | `OVERLAP_CHECK_ONLY` |
| VisDrone2019-DET | 0 | 253 | heuristic review candidates only |

There is no byte-identical overlap with any reference. A manually viewed V1 dHash candidate paired a dark Indian road image with an unrelated Taiwan CCTV intersection, demonstrating that low-texture/perceptual-hash collisions can occur; the candidate count must not be represented as confirmed shared source imagery. Stage 18 remains sealed and was read only for SHA256/dHash overlap comparison—never for model selection or tuning.

## Outputs and reproducibility

Generated, Git-ignored artifacts are under `outputs/data_qa/v2_urbanscene/`:

- `dataset_inventory.csv`
- `class_distribution.csv`
- `application_mapping.csv`
- `annotation_issues.csv`
- `duplicate_groups.csv`
- `coverage_summary.json`
- twelve image-level structure/category previews (explicitly marked “NO BBOX ANNOTATION AVAILABLE”)

`object_size_distribution.csv` and bbox annotation previews were intentionally not created because object boxes do not exist.

The governing implementation is `scripts/run_urbanscene_qa.py` with reusable helpers in `src/vision_analytics/utils/urbanscene_qa.py` and configuration in `configs/v2_urbanscene_qa.yaml`.

## Limitations and allowed future use

CC BY 4.0 provenance, raw integrity, image readability, category/time coverage, and duplicate governance are confirmed. Object-detection taxonomy, bbox coverage, annotation completeness, small-object coverage, and per-object occlusion are not available. The current data must not enter supervised YOLO training. A future independent manual bbox annotation project could reconsider it only with a new annotation policy, QA, provenance, leakage grouping, and fresh Train/Val/V2 Holdout governance.

No training, fine-tuning, final V2 Holdout construction, V1 artifact mutation, or Stage 18 tuning occurred in V2-2B.
