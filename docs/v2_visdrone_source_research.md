# VisDrone2019-DET source research

Research date: 2026-08-31 (Asia/Taipei)

Scope: official/primary sources only. This note verifies provenance, official download locations, annotation semantics, usage-term evidence, and sequence/grouping evidence for V2-2. No dataset archive was downloaded for this research task.

## Governance conclusion

**License status: `REQUIRES_REVIEW`.** The official `VisDrone/VisDrone-Dataset` repository exposes the dataset and citation but, as of the research date, its repository root lists only `README.md` and no `LICENSE`/terms file. The official DET toolkit says that the **code library** is “for research purpose only”; that wording is not an explicit license grant for the dataset images or annotations. Consequently, it would be unsafe to infer commercial permission, redistribution rights, derivative-data rights, or a standard SPDX license from the toolkit notice. Obtain written clarification from the VisDrone/AISKYEYE owners before treating VisDrone2019-DET as approved training data for a production or commercial system. [Official dataset repository](https://github.com/VisDrone/VisDrone-Dataset#readme), [official DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit#introduction)

This status is a legal/provenance gate, not a statement that download or academic analysis is technically impossible.

## Dataset identity and owner

- Dataset/release: **VisDrone2019**, detection-in-images subset **VisDrone2019-DET** (also called “VisDrone-DET” in the current official repository).
- Owner/organization stated by the publisher: the **AISKYEYE team at the Lab of Machine Learning and Data Mining, Tianjin University, China**. The current AISKYEYE datasets page links back to the `VisDrone` GitHub organization as the VisDrone dataset source. [Official dataset README](https://github.com/VisDrone/VisDrone-Dataset#visdrone-dataset), [AISKYEYE datasets page](https://aiskyeye.com/datasets/)
- Coverage described by the official benchmark: data were captured in 14 Chinese cities with multiple drone platforms, environments, weather, and lighting conditions. The complete benchmark has 10,209 static images plus separate video clips; the image-detection task uses those static images. [Official benchmark paper](https://arxiv.org/abs/1804.07437)
- The official DET repository currently distributes four source partitions: train, val, test-dev, and test-challenge. Test-dev ground truth is available; test-challenge annotations are unavailable. [Official download section](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images)

The official historical benchmark paper reports 6,471 training images, 548 validation images, and 3,190 testing images. The current download page divides the testing material into test-dev and test-challenge; its published archive sizes are 1.44 GB, 0.07 GB, 0.28 GB, and 0.28 GB respectively. [Official benchmark paper](https://arxiv.org/abs/1804.07437), [official download section](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images)

## Official download URLs

The following URLs are the links presently published by the dataset owner. They are preferable to transformed third-party mirrors because the project requires raw source-taxonomy preservation.

| Partition | Published size | Official Google Drive | Official BaiduYun |
|---|---:|---|---|
| train | 1.44 GB | [file `1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn`](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing) | [owner-published link](https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A) |
| val | 0.07 GB | [file `1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59`](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing) | [owner-published link](https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g) |
| test-dev (GT available) | 0.28 GB | [file `1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V`](https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V) | [owner-published link](https://pan.baidu.com/s/1RdRfSWV-1IFK7aWljLU_LQ) |
| test-challenge (annotations unavailable) | 0.28 GB | [file `1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT`](https://drive.google.com/file/d/1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT/view?usp=sharing) | [owner-published link](https://pan.baidu.com/s/1lvEkCgy1WWK4B7TLki4yBQ) |

Source for every link and published size: [official VisDrone dataset README](https://raw.githubusercontent.com/VisDrone/VisDrone-Dataset/master/README.md).

The official page does not publish archive SHA256 values. V2-2 should therefore compute SHA256 immediately after acquisition and record the exact source URL, access date, archive filename/size, ZIP integrity result, and extracted raw-tree manifest/hash. A locally computed digest establishes the acquired artifact’s identity; it does not independently prove equivalence to an upstream checksum because none is published in the cited official material.

## Raw annotation format

VisDrone DET ground truth is not native YOLO. It is one comma-separated text file per image, with one object instance per row:

```text
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

The four bbox values are pixel coordinates/dimensions, using top-left `x`, top-left `y`, width, and height. In ground truth, `score=1` means the box participates in evaluation and `score=0` means it is ignored. The source category IDs are: [official results-format specification](https://aiskyeye.com/evaluate/results-format/), [official DET toolkit README](https://github.com/VisDrone/VisDrone2018-DET-toolkit#det-submission-format)

| Source ID | Official source category | V2 application disposition |
|---:|---|---|
| 0 | ignored regions | preserve metadata; exclude from target training labels |
| 1 | pedestrian | map to `person` |
| 2 | people | map to `person` |
| 3 | bicycle | map to `bicycle` |
| 4 | car | map to `car` |
| 5 | van | `EXCLUDED_FROM_V2_TARGET` |
| 6 | truck | map to `truck` |
| 7 | tricycle | `EXCLUDED_FROM_V2_TARGET` |
| 8 | awning-tricycle | `EXCLUDED_FROM_V2_TARGET` |
| 9 | bus | map to `bus` |
| 10 | motor | map to `motorcycle` |
| 11 | others | preserve metadata; ambiguous/unsupported and exclude |

The paper’s taxonomy prose uses **person** for a human not standing/walking, while the official file-format table names source ID 2 **people**. V2 should preserve the released numeric ID and the file-format label (`people`) in raw/parsed source fields, then apply the requested application mapping separately. It must not rewrite raw IDs. [Official benchmark paper, Task 1 taxonomy](https://arxiv.org/abs/1804.07437), [official results-format specification](https://aiskyeye.com/evaluate/results-format/)

### Difficulty metadata

- `truncation=0`: no truncation; `truncation=1`: partial truncation, documented as 1–50%. The benchmark paper additionally says targets with truncation over 50% are skipped in evaluation.
- `occlusion=0`: none; `occlusion=1`: partial (1–50%); `occlusion=2`: heavy (50–100% in the web/toolkit specification; the paper describes heavy as over 50%).
- Ignored regions (category 0), rows with ground-truth score 0, and `others` require explicit preservation and exclusion logic rather than being silently converted to target classes.

Sources: [official results-format specification](https://aiskyeye.com/evaluate/results-format/), [official benchmark paper](https://arxiv.org/abs/1804.07437), [official DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit#det-submission-format).

## Partition and sequence/grouping evidence

### Confirmed official evidence

- DET is the **object detection in images** task. The official paper describes its 10,209 items as **static images** and explicitly says they do not overlap the benchmark’s video clips. VisDrone-VID is a separate task/dataset. [Official benchmark paper](https://arxiv.org/abs/1804.07437)
- The official paper states that the Task 1 training, validation, and testing subsets were captured at different locations. The historical DET toolkit also states that its train, validation, and test-challenge sets do not overlap. [Official benchmark paper](https://arxiv.org/abs/1804.07437), [official DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit#dataset)
- The DET annotation row has no sequence ID, frame ID, camera ID, location ID, or capture-session field. It contains only bbox, score/evaluation flag, category, truncation, and occlusion. [Official results-format specification](https://aiskyeye.com/evaluate/results-format/)

### What cannot be claimed from official documentation

No official source located in this review documents a VisDrone2019-DET filename grammar or guarantees that a substring of a filename is a true sequence, camera, flight, location, or capture-session identifier. Therefore, a group ID **cannot be authoritatively derived from the published annotation schema alone**. DET should not be described as continuous video frames merely because some filenames share prefixes.

### Conservative V2-2 treatment

After authorized acquisition, retain at least these separate fields:

1. `official_partition` from the archive (`train`, `val`, `test-dev`, or `test-challenge`);
2. `source_stem`, the full immutable filename stem;
3. `sequence_group_id`, initially an explicit conservative/derived value rather than an asserted official sequence ID;
4. `group_method` and `group_confidence`, recording whether grouping came from exact hashes, perceptual near-duplicate analysis, or a filename heuristic.

Use exact and near-duplicate evidence to form leakage-safe connected components. If a filename-prefix heuristic is added, document and test it as a **project policy**, not an official VisDrone fact. Do not split any resulting duplicate/near-duplicate component across future V2 Train/Val/Holdout. The official source partitions should also remain recorded even if a later V2 governance decision uses only a subset of them.

## Acquisition recommendations for V2-2

1. Download only from the owner-published Google Drive or BaiduYun links above; retain raw archives read-only.
2. Compute archive SHA256 and byte size before extraction; run ZIP CRC/integrity verification.
3. Produce a sorted extracted-tree manifest containing relative path, byte size, and file SHA256, then hash the manifest itself.
4. Preserve raw `.txt` annotations byte-for-byte. Put normalized parsed rows in a separate output tree/table.
5. Validate all eight fields and retain ignored/unsupported rows with an explicit disposition.
6. Keep the license gate at `REQUIRES_REVIEW` until an authorized reviewer records owner terms or written permission that covers the intended use.

## Primary sources consulted

- [VisDrone official dataset repository and downloads](https://github.com/VisDrone/VisDrone-Dataset)
- [AISKYEYE official datasets index](https://aiskyeye.com/datasets/)
- [AISKYEYE official results/ground-truth format](https://aiskyeye.com/evaluate/results-format/)
- [Official VisDrone2019 DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)
- [Original benchmark paper, *Vision Meets Drones: A Challenge*](https://arxiv.org/abs/1804.07437)

