# V2-2C Open Images V7 targeted subset data QA

Date: 2026-08-31

Decision: **ACCEPT_WITH_WARNINGS**

Training-pool status: **QUARANTINED_PENDING_PER_IMAGE_LICENSE_AND_DOMAIN_FILTER**

## Scope and governance

This stage used only the official Open Images V7 class descriptions, validation bounding-box annotations, validation image metadata and the documented Open Images S3 image mechanism. It did not download the complete Open Images corpus. Candidate design was intentionally limited to the official, more densely annotated validation split; it produced a 5,190-image metadata pool and a 300-image pilot.

No training, fine-tuning, model selection or final V2 Holdout construction occurred. The sealed Stage 18 Locked Test was read only for SHA256/dHash overlap checking and was not used to select samples or models.

The official source and schema research is recorded in [v2_openimages_source_research.md](v2_openimages_source_research.md). Primary sources are the [official V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html), [official V7 description and license notice](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), and [official subset downloader](https://github.com/openimages/dataset/blob/main/downloader.py).

## Source and license governance

- Dataset: Open Images V7, released October 2022.
- Candidate source split: official `validation` dense annotations only.
- Annotation license: CC BY 4.0.
- Underlying images: official metadata lists CC BY 2.0, but Open Images explicitly disclaims a warranty of individual image license status and requires users to verify each image independently.
- Access date: 2026-08-31.
- Pilot download: 300 official S3 mirror JPGs, 103,085,911 bytes; full local metadata plus pilot tree is approximately 139 MB.

For that reason, complete attribution fields and a CC BY 2.0 URL produce `REQUIRES_REVIEW`, not `VERIFIED`. All 5,190 candidates remain quarantined. The annotation license is never propagated to the underlying pixels.

| Official metadata file | Bytes | Local SHA256 |
|---|---:|---|
| `oidv7-class-descriptions-boxable.csv` | 12,064 | `1839e0e7e84130ae281f7f67413768601b031581c0c42e7fc17527b8e2a99aa9` |
| `validation-annotations-bbox.csv` | 25,105,048 | `d8bbd59410af14835d7733165a7bb8a3f0213981b22dd5077b0b9f7878991ff2` |
| `validation-images-with-rotation.csv` | 15,245,485 | `ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17` |

## Official class resolution

The mappings were resolved from the official V7 boxable class-description CSV, not guessed from strings. Raw MIDs and annotations remain unchanged.

| Source class | Official MID | Application class |
|---|---|---|
| Person | `/m/01g317` | `person` |
| Bicycle | `/m/0199g` | `bicycle` |
| Car | `/m/0k4j` | `car` |
| Motorcycle | `/m/04_sv` | `motorcycle` |
| Bus | `/m/01bjv` | `bus` |
| Truck | `/m/07r04` | `truck` |

## Candidate pool and coverage

The validation annotation file contained 303,980 rows. All 27,784 target-class rows parsed successfully. After excluding depiction and group-of boxes at object level, 5,190 images retained at least one eligible Person box.

| Class | Images | Eligible boxes | Small boxes |
|---|---:|---:|---:|
| person | 5,190 | 12,792 | 5,389 (42.13%) |
| bicycle | 239 | 393 | 46 (11.70%) |
| car | 4,658 | 9,142 | 1,865 (20.40%) |
| motorcycle | 144 | 222 | 10 (4.50%) |
| bus | 80 | 101 | 3 (2.97%) |
| truck | 295 | 343 | 10 (2.92%) |

The descriptive size bins are normalized area `<0.01` small, `0.01–<0.09` medium and `>=0.09` large. They are not physical-size measurements.

Person difficulty coverage was substantial: 6,963 occluded Person boxes, 3,129 truncated Person boxes, 3,596 small-and-occluded Person boxes, and 549 small-and-truncated Person boxes. Of the 5,389 small Person boxes, 1,441 were in candidate images with at least one eligible traffic class.

### Person plus traffic context

Context tags overlap; one image may contribute to several rows.

| Context tag | Images |
|---|---:|
| `PERSON_BICYCLE` | 106 |
| `PERSON_MOTORCYCLE` | 68 |
| `PERSON_CAR` | 562 |
| `PERSON_BUS` | 27 |
| `PERSON_TRUCK` | 59 |
| `PERSON_MULTI_TRAFFIC` | 56 |
| `PERSON_ONLY` | 4,424 |

There were 766 candidates with some Person-plus-traffic context. This is metadata co-occurrence, not proof of road-scene relevance.

## Bounding-box and attribute QA

The parser preserved `IsOccluded`, `IsTruncated`, `IsGroupOf`, `IsDepiction` and `IsInside`, including the documented `-1` unknown value. It required finite normalized coordinates, positive width/height, coordinates within `[0,1]`, a mapped official MID and valid attribute values.

- Invalid target bbox rows: 0.
- Depiction target boxes excluded: 1,970.
- Group-of target boxes excluded: 2,952.
- Occluded and truncated boxes: retained by design.
- Pilot images readable: 300/300.
- Pilot eligible target annotations: 1,722, including 1,072 Person boxes.
- Automated pilot annotation issues: 0.

The zero automated-issue count means the metadata and geometry passed deterministic checks; it does not establish exhaustive annotation completeness.

## Pilot design and manual review

The deterministic 300-image pilot used seed 2203 and prioritized Person with motorcycle, bicycle, car, bus or truck plus small/occluded/truncated Person cases. It contained no `PERSON_ONLY` selection. All pilot images and generated contact sheets are Git-ignored.

Fifty pilot images were visually reviewed across the required contexts. Results:

| Domain relevance | Count | Share |
|---|---:|---:|
| `TRAFFIC_RELEVANT` | 12 | 24% |
| `PARTIALLY_RELEVANT` | 15 | 30% |
| `NON_TRAFFIC` | 23 | 46% |
| `AMBIGUOUS` | 0 | 0% |

Annotation review classified 46 as acceptable, 2 as ambiguous and 2 with obvious completeness concerns. The problematic examples were dense/parade or market scenes where visible people appeared incompletely represented by individual eligible boxes. General-domain contamination included bicycle and motorcycle racing, stunts, exhibitions, static vehicle portraits, amusement vehicles, recreational cycling and non-operational vehicle scenes.

The review demonstrates that target-class co-occurrence alone is insufficient. A formal V2 pool needs an explicit traffic-domain review/filter and should retain `PARTIALLY_RELEVANT` only when it covers a documented failure mode.

## Duplicate and leakage checks

SHA256 exact matching and dHash-64 near-duplicate grouping with Hamming threshold 6 found:

- Open Images pilot internal: 0 exact and 0 dHash-near duplicate images.
- V1 Taiwan CCTV: 0 exact and 0 dHash-near overlaps.
- Stage 18 Locked Test (`OVERLAP_CHECK_ONLY`): 0 exact and 0 dHash-near overlaps.
- VisDrone: 0 exact and 0 dHash-near overlaps.
- UrbanScene: 0 exact and 0 dHash-near overlaps.

dHash is a documented heuristic, not proof that two scenes are unrelated. Any later materialization must repeat overlap checks on the final accepted sample list.

## Acceptance decision

**ACCEPT_WITH_WARNINGS** means the source is technically suitable for a governed V2 candidate pool, not that the current pilot may enter training.

Positive evidence:

- Official, traceable object-level bbox metadata converts cleanly to detection labels.
- Person and small/occluded Person coverage materially exceeds the V1 gap in raw coverage terms.
- 766 metadata candidates combine Person with at least one target traffic class.
- Pilot images decode successfully and mapped geometry is valid.
- Leakage checks are implementable and currently clear.

Blocking warnings:

- `VERIFIED=0`, `REQUIRES_REVIEW=5,190`, `REJECTED=0`; every image needs independent license/source verification before training eligibility.
- Nearly half of the 50-image review was non-traffic, so domain relevance must be manually or conservatively filtered.
- Dense scenes showed some apparent annotation-completeness limitations.

If those gates are later cleared, the next acquisition should cap the governed pool at **400–500 total images including this pilot**, drawn first from the 766 Person-plus-traffic candidates and balanced toward motorcycle, bicycle, small Person and occlusion. This is a review budget, not permission to bulk-download or train. No additional image should become training-eligible merely because it appears in the candidate manifest.

## Reproducibility and artifacts

- Configuration: `configs/v2_openimages_qa.yaml`
- Governed manual review source: `data/manifests/openimages_pilot_review.csv`
- Candidate and QA outputs: `outputs/data_qa/v2_openimages/`
- Main command: `.venv/bin/python scripts/run_openimages_qa.py --download`
- Coverage summary: `outputs/data_qa/v2_openimages/coverage_summary.json`

Raw metadata, images, QA CSV/JSON outputs and previews are generated artifacts and remain Git-ignored. The tracked configuration, code, tests, provenance rows and documentation are the reproducible governance layer.
