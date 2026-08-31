# Stage 16 Dataset QA and Split Governance

This stage performs descriptive QA and leakage-aware split governance only. It
does not train a model, tune thresholds, or modify source images/labels.

## Provenance and integrity

- Source: Hsiang / `taiwan CCTV`, Roboflow Universe version 3
- License: CC BY 4.0
- Export: official YOLO26 ZIP, acquired 2026-08-31
- Raw assets: 1,824 images and 1,824 label files
- Raw location: `data/raw/taiwan_cctv_v3` (Git-ignored)
- ZIP SHA-256: `040672837f3345d6a3d6ffeb999a4e466209db69a4ce5791fa802bb308d7a918`
- Extracted raw-tree SHA-256: `8440bba4d4f374c0532970a8e4dbbb0928db9fe20992646804517216f9acfa24`

Raw class IDs remain `bicycle`, `bus`, `car`, `human`, `motorbike`, `truck`.
Application names are a manifest-only mapping: `human -> person` and
`motorbike -> motorcycle`; raw label rows are not rewritten.

## QA and exclusions

OpenCV decoded all 1,824 images. Every image had a matching label and every
label had a matching image. Sixteen samples contain annotation errors and are
recorded as `EXCLUDED`, without deleting or editing them: 14 contain a zero-sized
bbox and 2 contain a bbox whose normalized extent crosses the image boundary.
The governed 1,808-image population contains 48,380 valid boxes.

The dominant resolution is 1940 x 1454 (1,678 images). Of the valid normalized
bbox areas, 40,850 (84.4%) are below 0.01, 6,795 are from 0.01 to below 0.09,
and 735 are at least 0.09. The median area is 0.00214, so small and occluded
objects are a material characteristic of this dataset.

## Duplicate grouping and splits

Exact duplicates use SHA-256. Near duplicates use 64-bit difference hash
(`dHash-64`) with Hamming distance <= 6. This heuristic is transparent but is
not Ground Truth scene identity. No exact duplicate group was found; 146
multi-image governance groups were formed by the near-duplicate rule.

The export does not provide reliable camera/sequence metadata beyond generic
`traffic3` and its original train/valid/test filename tokens. Treating all
`traffic3` frames as one group would make three governed splits impossible, so
the new deterministic split (seed 1601) groups exact/near duplicates and does
not trust the source split tokens as scene identity. Group integrity takes
priority over exact ratios:

- TRAIN: 1,266 images
- VAL: 271 images
- LOCKED_TEST: 271 images
- EXCLUDED: 16 images

No image, exact-duplicate group, near-duplicate group, or governance group
crosses splits. LOCKED_TEST must not be used for Stage 17 training, threshold
tuning, or model selection.

## Manual preview review

Eighteen generated previews (three per raw class) were inspected. Boxes generally
align with visible objects, including dense scooter traffic and small distant
objects. The review also confirms limitations that should not be silently
relabeled in Stage 16:

- `human` annotations are sparse and sometimes occur beside riders, while many
  riders are represented only within `motorbike` boxes.
- `motorbike` boxes commonly contain rider plus scooter; the application rename
  to `motorcycle` does not create a separate rider label.
- Bicycle examples are uncommon and can be visually close to rider/motorbike
  cases, especially at low resolution or under occlusion.
- Vans/minibuses/pickups expose `car`/`bus`/`truck` taxonomy ambiguity; examples
  include passenger vans labeled `car`, minibuses labeled `bus`, and pickups
  labeled `truck`.
- Dense scenes make overlay labels overlap, and small/occluded objects dominate
  the bbox-size distribution. These are review limitations, not automatic
  relabel decisions.

Machine-readable results are generated under `outputs/data_qa/stage16/` and are
Git-ignored. The deterministic script and governance configuration are committed;
raw data and generated QA artifacts are not.
