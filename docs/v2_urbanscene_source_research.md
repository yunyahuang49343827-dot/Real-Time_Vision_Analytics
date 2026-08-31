# UrbanScene source research

Research date: 2026-08-31 (Asia/Taipei)

Scope: primary sources only. This note verifies the provenance, licence, version, published files, described data structure, and official access endpoints for DOI `10.17632/5gt4fg4rvp.1`. No dataset file was downloaded or inspected during this research task.

## Governance conclusion

The Mendeley Data record explicitly licenses **dataset version 1 under CC BY 4.0**. This is distinct from the accompanying *Data in Brief* article, which is published under **CC BY-NC 4.0**; the article's licence must not be substituted for the dataset licence. The Mendeley licence notice permits sharing and modification with attribution, a licence link, and change indication, while warning that separately identified third-party content may require further permission. [Official Mendeley Data record](https://data.mendeley.com/datasets/5gt4fg4rvp/1), [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/), [author article and article-licence notice](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/)

**Technical suitability warning:** the authoritative sources describe and distribute the resource as category-organized JPG images, but they do **not** document object-detection bounding-box annotations, annotation files, bbox formats, per-object classes, occlusion/truncation metadata, or official train/validation/test partitions. The authors' reported experiment is a four-class image-classification problem. Therefore, UrbanScene must not be represented as a verified bbox-annotated object-detection dataset based on the published record alone. Before V2 acquisition, this is a hard schema/fitness gate: either obtain owner confirmation of annotation contents or treat the source as image-level/category data only. [Author article, Data Description and folder structure](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#sec3), [author article, four-class experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#tbl3)

## Dataset identity, owner, and contributors

- Dataset: **UrbanScene: An Extensive Multi-Object Dataset for Pedestrian, Traffic, and Motorbike Detection**.
- DOI/version: **`10.17632/5gt4fg4rvp.1`, Version 1**.
- Published: **2024-05-28**; repository metadata records the last modification as **2024-05-26**.
- Repository owner: **Kailas PATIL**. The official Mendeley page identifies **Kasetsart University Sri Racha Campus** as the associated institution.
- Contributors, in repository order: **Kailas PATIL; prawit chumchu; Siddharth Pashankar; Darshana Gatagat; Omkar Rumane**.
- The accompanying author article gives affiliations of Vishwakarma University, Pune, India (Patil, Gatagat, Rumane, Pashankar) and Kasetsart University, Sriracha, Thailand (Patil and Chumchu). It identifies Kailas Patil and Prawit Chumchu as corresponding authors. [Official Mendeley Data record](https://data.mendeley.com/datasets/5gt4fg4rvp/1), [author article and affiliations](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/)

The Mendeley record describes **16,426 urban photographs** of vehicles, cyclists, motorbikes, and pedestrians across morning, evening, and night conditions. The author article says the images were collected in five locations in Pune, Maharashtra, India, principally using an iPhone 13, then saved as JPG and resized to `768 × 1024` pixels. [Official Mendeley Data record](https://data.mendeley.com/datasets/5gt4fg4rvp/1), [author article, specifications table](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#sec1)

## Official file inventory

The anonymous Mendeley public file API returned exactly three completed root-level ZIP files on the research date. The API publishes a SHA256 digest for each archive, so acquisition should verify bytes against these upstream hashes before extraction.

| Published file | File ID | Bytes | GiB | Upstream SHA256 |
|---|---|---:|---:|---|
| `Motorbikes_&_Cyclist.zip` | `f1424a0c-9242-4964-b092-0779137a8dda` | 1,168,975,616 | 1.089 | `42fd6393116e9817d456a03fce5b85afb81b309b0588dcae29d0e64b0c07c109` |
| `Pedestrians.zip` | `77e68631-0483-42c0-909d-121772eada73` | 1,133,335,639 | 1.056 | `c6ebd3a86441d6109226dd3679c9e192d4f9ec73292365bc18b0a10949fcd33e` |
| `Traffic.zip` | `e65d5757-7b05-4881-92ec-d66e29bbe424` | 1,371,652,798 | 1.277 | `88732a107bc2ec1c2aea39503db895bb9c1e069f7e3b87621bd5528eacb7ff8c` |
| **Total** |  | **3,673,964,053** | **3.422** |  |

Primary metadata endpoint: [Mendeley public file API for version 1](https://data.mendeley.com/public-api/datasets/5gt4fg4rvp/files?folder_id=root&version=1).

The published API inventory is archive-level only; it does not expose the ZIP members. Because this research task did not download archives, exact internal filenames/counts and extracted-tree hashes remain unverified.

## Described image and category structure

The authors report the following image-level distribution:

| Published category | Images | Description in the author article |
|---|---:|---|
| Traffic | 7,227 | cars, buses, and trucks in chaotic, light, and free-flowing traffic |
| Pedestrians | 4,106 | standing or walking people in sidewalks, crosswalks, and open spaces |
| Motorbikes & Cyclists | 5,093 | motorcycles/motorbikes and bicycles in roads, bike lanes, and varying traffic |
| **Total** | **16,426** |  |

The article says categories are stored in distinct folders and environments are organized as **Morning, Evening, and Night**. It also discusses four image-level subjects/classes—Cyclist, Motorbike, Pedestrian, and Traffic—and reports a four-class classification experiment, even though the repository exposes the first two in a combined archive. This is a documented structural ambiguity to resolve after any authorized acquisition; it is not evidence of object-level labels. [Author article, Data Description and Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#sec3), [author article, folder structure](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#fig1), [author article, classification experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#tbl3)

### Annotation status

**Confirmed:** JPG images, `768 × 1024`, grouped by subject/category and time-of-day environment.

**Not confirmed by any primary source reviewed:** bbox annotation files; annotation syntax; normalized or pixel bbox coordinates; per-instance person/vehicle labels; ignored regions; occlusion/truncation/difficulty metadata; source class IDs; image-to-annotation pairing; or official split/sequence identifiers.

Although the title, keywords, and prose use “object detection,” the actual published data specification says `Data format: Raw`, `Type of data: Image`, and the article's measured models/confusion matrices are for image classification. Do not infer missing bbox annotations from the phrase “object detection.” [Author article, specifications table](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#sec1), [author article, model evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#tbl3)

## Official access endpoints

Dataset landing page and DOI:

- [Mendeley Data version 1](https://data.mendeley.com/datasets/5gt4fg4rvp/1)
- [DOI resolver](https://doi.org/10.17632/5gt4fg4rvp.1)

Anonymous metadata endpoints used for this review:

- File list: `https://data.mendeley.com/public-api/datasets/5gt4fg4rvp/files?folder_id=root&version=1`
- Folder list: `https://data.mendeley.com/public-api/datasets/5gt4fg4rvp/folders/1`

Official per-file download URLs published by the Mendeley API:

- `Motorbikes_&_Cyclist.zip`: `https://data.mendeley.com/public-files/datasets/5gt4fg4rvp/files/f1424a0c-9242-4964-b092-0779137a8dda/file_downloaded`
- `Pedestrians.zip`: `https://data.mendeley.com/public-files/datasets/5gt4fg4rvp/files/77e68631-0483-42c0-909d-121772eada73/file_downloaded`
- `Traffic.zip`: `https://data.mendeley.com/public-files/datasets/5gt4fg4rvp/files/e65d5757-7b05-4881-92ec-d66e29bbe424/file_downloaded`
- Whole-dataset ZIP route exposed by the official page: `https://data.mendeley.com/public-api/zip/5gt4fg4rvp/download/1`

Mendeley's official API documentation says a specific published version is selected with the `version` query parameter, and that published versions are permanent public records. The anonymous `data.mendeley.com/public-api` endpoints above are the routes used by the official dataset page; the separate developer API at `api.mendeley.com/datasets/{id}` requires OAuth. [Mendeley Datasets API guide](https://dev.mendeley.com/code/datasets_quick_start_guides.html), [Mendeley API reference](https://dev.mendeley.com/methods/)

## Acquisition and acceptance implications

1. If acquisition is authorized, download the three immutable version-1 files from the official per-file URLs and verify byte size plus the upstream SHA256 values above before extraction.
2. Preserve the three ZIP files and extracted raw tree read-only; produce an independent sorted extracted-tree manifest and manifest SHA256.
3. Verify whether any annotation files actually exist. If the archives contain only category folders/JPGs, record `ANNOTATION_TYPE=IMAGE_LEVEL_CATEGORY` and **reject UrbanScene as a direct object-detection-bbox training source** for V2. Do not manufacture bounding boxes or reinterpret folder names as object annotations.
4. If annotations unexpectedly exist, quarantine them until their schema, provenance, completeness, class semantics, and licence coverage are documented and tested; repository metadata alone does not validate them.
5. Keep source categories and application taxonomy separate. The published traffic grouping merges cars/buses/trucks at image-description level, and the combined motorbike/cyclist archive may contain separate subfolders but does not establish per-object identities.
6. Record privacy/ethics for review. The paper says the dataset contains people, asserts that privacy was maintained, and says dataset authors appear in images; it does not describe consent, face blurring, or a person-level release procedure. [Author article, Ethics Statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/#sec8)

## Primary sources consulted

- [Official Mendeley Data dataset record, version 1](https://data.mendeley.com/datasets/5gt4fg4rvp/1)
- [Official Mendeley anonymous file metadata API](https://data.mendeley.com/public-api/datasets/5gt4fg4rvp/files?folder_id=root&version=1)
- [Owner-authored *Data in Brief* article](https://doi.org/10.1016/j.dib.2024.110887)
- [Full author article in PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405908/)
- [Mendeley official Datasets API guide](https://dev.mendeley.com/code/datasets_quick_start_guides.html)
