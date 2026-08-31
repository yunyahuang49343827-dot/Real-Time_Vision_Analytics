# Data Sources and Governance

Access date for all web sources: **2026-08-29**.

This document records provenance and intended use only. Stage 1 does not profile
videos, inspect frames, run detection, perform dataset QA, create splits, or train
models. Raw media and datasets must remain under `data/raw/` and must not be
committed to Git.

## Data A | Runtime Videos

Role: runtime/demo input for later stages. Source: Pexels. The four source pages
were reachable on the access date and each exposed a **Free download** control and
marked the video **Free to use**:

| Source ID | Pexels video | Intended local path | Acquisition status |
|---|---|---|---|
| `pexels_2103099` | [Traffic Flow In The Highway](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/) | `data/raw/videos/pexels_2103099_traffic_flow_highway.mp4` | Downloaded from the page's official Free download link |
| `pexels_13258685` | [Traffic in City](https://www.pexels.com/video/traffic-in-city-13258685/) | `data/raw/videos/pexels_13258685_traffic_in_city.mp4` | Downloaded from the page's official Free download link |
| `pexels_37258214` | [Urban Traffic Scene with Cars and Motorcycles](https://www.pexels.com/video/urban-traffic-scene-with-cars-and-motorcycles-37258214/) | `data/raw/videos/pexels_37258214_urban_traffic_cars_motorcycles.mp4` | Downloaded from the page's official Free download link |
| `pexels_9322363` | [An Aerial View of a Busy Intersection with Cars and Buses](https://www.pexels.com/video/an-aerial-view-of-a-busy-intersection-with-cars-and-buses-9322363/) | `data/raw/videos/pexels_9322363_aerial_busy_intersection.mp4` | Downloaded from the page's official Free download link |

License: **Pexels License**. Pexels' official license page states that its photos
and videos are free to download and use, attribution is not required, and
modification is allowed. It also prohibits selling unaltered copies,
redistribution on stock-photo/wallpaper platforms, implied endorsement, and
trademark use; the full Terms additionally require consideration of third-party
rights in recognizable people, logos, brands, buildings, and similar content.
See the [Pexels License](https://www.pexels.com/license/) and
[Pexels Terms of Service](https://www.pexels.com/terms-of-service/).

Governance:

- Intended use is project runtime/demo video only; do not redistribute the raw
  MP4 files.
- Acquisition used only the `Free download` URL presented on each asset page;
  no access control or site restriction was bypassed. SHA-256 checksums are
  retained in `data/manifests/sources.csv`.
- Keep the Pexels asset ID, source page, contributor attribution metadata, access
  date, and license snapshot in the manifest even though attribution is not
  mandatory.
- Re-check the asset page and current terms before any public or commercial
  release, and review visible people, brands, and license plates for third-party
  rights.

## Data B | Primary Taiwan Fine-tuning Candidate

Primary source: [Hsiang / taiwan CCTV on Roboflow Universe](https://universe.roboflow.com/hsiang-idaw8/taiwan-cctv).

The owner page identifies Hsiang, Object Detection, **CC BY 4.0**, six classes,
and 1,824 images. The selected immutable candidate is
[`taiwan-cctv/3`](https://universe.roboflow.com/hsiang-idaw8/taiwan-cctv/dataset/3),
generated 2024-07-12. Its version page records **1,824 total images**, no
preprocessing, and no augmentations. It offers downloads from the version page in
YOLO26/YOLOv11/YOLOv8/COCO JSON/Pascal VOC and other formats; using the download
control may require a Roboflow sign-in/API key.

The Stage 16 official YOLO26 export's `data.yaml` establishes this raw class-ID
order (the earlier Stage 1 web page did not expose class IDs):

`bicycle`, `bus`, `car`, `human`, `motorbike`, `truck`

The official v3 archive was acquired on 2026-08-31 through Roboflow's authenticated
download flow and extracted to `data/raw/taiwan_cctv_v3`. The ZIP SHA-256 is
`040672837f3345d6a3d6ffeb999a4e466209db69a4ce5791fa802bb308d7a918`; raw files
remain Git-ignored and unchanged.

Application taxonomy is a manifest-layer mapping only:

| Raw source class | Application class | Mapping |
|---|---|---|
| `human` | `person` | rename |
| `motorbike` | `motorcycle` | rename |
| `car` | `car` | identity |
| `truck` | `truck` | identity |
| `bus` | `bus` | identity |
| `bicycle` | `bicycle` | identity |

Do not edit raw annotations or raw class names.

### Version guardrail

Roboflow Universe also exposes a different endpoint,
[`mcutplate/taiwan-cctv-fqzwr/1`](https://universe.roboflow.com/mcutplate/taiwan-cctv-fqzwr/model/1),
whose model page reports **4,990 training images**. It is a distinct project and
is not the provenance target above. Do not treat it, or any approximately
5,000-image fork/augmented export, as Hsiang's 1,824-image source dataset. Only
the owner/project slug plus version `hsiang-idaw8/taiwan-cctv/3` is approved as
the Stage 1 primary candidate.

License: **CC BY 4.0**, as displayed on the Hsiang project and version pages.
Attribution and license-link requirements must be preserved in any later use;
see the [Creative Commons legal code](https://creativecommons.org/licenses/by/4.0/legalcode).

## Data C | External Taiwan Evaluation Candidate

Candidate family: **FishEye8K** (2023/2024 archive) and the hosting platform's
newer **FE-DETRAC** record. These are related records, but their dataset versions
and sizes must not be conflated.

The [authors' FishEye8K repository](https://github.com/MoyoG/FishEye8K) records
8,000 annotated images, approximately 157,000 bounding boxes, and five classes:

`Bus`, `Bike`, `Car`, `Pedestrian`, `Truck`

The repository directs users to the official NCHC dataset platform and specifies
the current complete archive
`Fisheye8K_all_including_train&test_update_2024Jan.zip`. Users select the
resource on the dataset page, then choose **Explore** and **Go to resource**. It
also warns that an older pre-2024-01-28 archive requires corrected COCO
`train.json` and `test.json` labels. For future acquisition, use the current
archive and record its resource metadata/checksum; do not silently mix label
revisions.

Official/data-platform records:

- [NCHC FE-DETRAC dataset page](https://scidm.nchc.org.tw/en/dataset/fe-detrac)
- [NCHC catalog result filtered as CC-BY-NC-4.0](https://scidm.nchc.org.tw/en/dataset?_license_id_limit=0&_organization_limit=0&_res_format_limit=0&_tags_limit=0&license_id=CC-BY-NC-4.0&organization=hsinchu_police&tags=Fish-Eye+Camera)
- [FishEye8K 2024 archive resource](https://scidm.nchc.org.tw/dataset/fe-detrac/resource/f6e7500d-1d6d-48ea-9d38-c4001a17170e)
- DOI: [`10.30193/scidm-ds-571593m`](https://doi.org/10.30193/scidm-ds-571593m)
- Dataset repository: [MoyoG/FishEye8K](https://github.com/MoyoG/FishEye8K)

The current NCHC record labels itself **FE-DETRAC version 2026.B1** and reports
20,000 continuous frames, more than 470,000 bounding boxes, five classes
(`Pedestrian`, `Bike`, `Car`, `Bus`, `Truck`), 22 fisheye IoT cameras at Hsinchu
intersections, and VOC/COCO/MOT/YOLO formats. The same record still exposes the
legacy-named FishEye8K 2024 archive. Accordingly, the Stage 1 candidate must
record the exact selected resource/version: FishEye8K's 8,000-image archive and
FE-DETRAC 2026.B1's 20,000-frame description are not interchangeable counts.

### License reconciliation hold

The NCHC catalog metadata/filter identifies the dataset under
**CC BY-NC 4.0**, while the current FE-DETRAC/FishEye8K dataset page text states
**CC BY-NC-SA 4.0** and adds ShareAlike and citation obligations. These are
materially different terms. No inference is made that the less restrictive
record controls.

```text
license = CC BY-NC 4.0 | CC BY-NC-SA 4.0 (conflicting source records)
license_status = REQUIRES_RECONCILIATION
intended_use = research / portfolio external evaluation only
```

Until the dataset owner/platform resolves the discrepancy in writing:

- Do not use this candidate for commercial purposes.
- Do not use it for fine-tuning or mix it into the primary training dataset.
- Do not redistribute raw data or derivatives.
- Restrict any later acquisition to research/portfolio external evaluation and
  preserve the dataset/paper citation and both observed license records.

## Provenance controls

- `data/manifests/sources.csv` is the source-of-truth inventory for source URL,
  license status, purpose, access date, selected version, and local path.
- `data/manifests/class_mapping.csv` translates source taxonomy to application
  taxonomy without mutating source labels.
- Raw assets remain immutable and Git-ignored. Record transformations later in
  separate interim/processed manifests rather than editing raw content.
- A URL being reachable is not proof that an automated downloader is permitted.
  Use only the publisher's documented download flow; if blocked, record manual
  acquisition rather than bypassing controls.
- Dataset QA, frame inspection, deduplication, splitting, model inference, and
  training are explicitly outside Stage 1.
