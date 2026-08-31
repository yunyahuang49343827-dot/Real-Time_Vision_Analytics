# Open Images V7 official-source research

Research date: 2026-08-31 (Asia/Taipei)

Scope: official/primary sources only. This note verifies the Open Images V7 identity, target-class MIDs, bounding-box and image-metadata schemas, image acquisition mechanism, and the distinction between annotation and underlying-image licenses for V2-2C. No image corpus or training data was downloaded for this research task.

## Governance conclusion

Open Images V7 is technically suitable for building a targeted object-detection candidate pool: the official release provides normalized object-level bounding boxes, source split identifiers, box attributes, class-description metadata, image attribution/license metadata, and an official subset downloader. The V7 description reports 16 million boxes over 600 boxable classes on 1.9 million images. [Official V7 description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), [official V7 downloads and formats](https://storage.googleapis.com/openimages/web/download_v7.html)

**Image licensing remains a per-image hard gate.** Google licenses the annotations under CC BY 4.0, whereas the V7 page says the images are *listed* as CC BY 2.0 and explicitly makes no representation or warranty about each image's license status. It instructs users to verify every image themselves. Therefore, the annotation license must not be propagated to image pixels, and neither an Open Images membership nor a valid bbox row is enough to mark an image `VERIFIED`. Candidate governance must retain and validate each image's `License`, original source/landing URL, author, and attribution fields. [Official V7 license statement](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses)

This research supports proceeding with metadata filtering and a small governed pilot. It does **not** pre-decide the final V2-2C `ACCEPT` / `ACCEPT_WITH_WARNINGS` / `REJECT` outcome, which also depends on actual per-image license checks, pilot annotation quality, traffic relevance, and leakage results.

## Dataset identity and release

- Dataset: **Open Images Dataset V7**.
- Version/release date: **V7, released October 2022** according to the official download page. [Official V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html)
- Publisher/annotation owner: Google LLC; the official license section identifies Google LLC as licensor of the annotations. [Official V7 license statement](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses)
- Dense-annotation scope: a subset of approximately 1.9M images carries bounding boxes and other dense annotations; the remaining images have only image-level labels. The reported bbox counts are 14,610,229 train, 303,980 validation, and 937,327 test boxes over 600 classes. [Official V7 description, bounding boxes](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#bounding-boxes)

The official V7 page deliberately links some annotation artifacts under older storage prefixes because those files are reused in V7. In particular, its current links resolve to the V6 train bbox CSV and V5 validation/test bbox CSVs. Acquisition code should record both the logical release (`Open Images V7`) and the exact upstream artifact URL/filename rather than renaming the source version silently. [Official V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html)

## Official target-class resolution

Open Images classes are identified by MIDs; the official documentation instructs users to resolve MIDs through the class-description CSV rather than infer IDs from names. The six requested mappings below were resolved from the official V7 **boxable** class-description artifact. [Official class-name format documentation](https://storage.googleapis.com/openimages/web/download_v7.html#class-names), [official V7 boxable class descriptions CSV](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)

| Official source class | Official source MID | Application class | Mapping status |
|---|---|---|---|
| Person | `/m/01g317` | `person` | `MAPPED_EXACT` |
| Bicycle | `/m/0199g` | `bicycle` | `MAPPED_EXACT` |
| Car | `/m/0k4j` | `car` | `MAPPED_EXACT` |
| Motorcycle | `/m/04_sv` | `motorcycle` | `MAPPED_EXACT` |
| Bus | `/m/01bjv` | `bus` | `MAPPED_EXACT` |
| Truck | `/m/07r04` | `truck` | `MAPPED_EXACT` |

These mappings apply to the six explicit source classes only. They do not imply that subclasses, related hierarchy nodes, or visually similar classes should be merged automatically. Raw `LabelName` MIDs must remain unchanged; the application mapping belongs in a separate governed table.

## Official metadata artifacts

The V7 download page publishes the following primary artifacts for the object-detection workflow. Downloaded metadata should be stored byte-for-byte, with URL, access date, byte size, and a locally computed SHA256. The official pages do not publish SHA256 values for these CSVs, so a local digest proves which artifact was used but is not an upstream authenticity checksum.

| Purpose | Split | Exact URL published through the V7 page |
|---|---|---|
| Boxable class names | all | [`oidv7-class-descriptions-boxable.csv`](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv) |
| Bounding boxes | train | [`oidv6-train-annotations-bbox.csv`](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) |
| Bounding boxes | validation | [`validation-annotations-bbox.csv`](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) |
| Bounding boxes | test | [`test-annotations-bbox.csv`](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv) |
| Image information / IDs | train | [`train-images-boxable-with-rotation.csv`](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv) |
| Image information / IDs | validation | [`validation-images-with-rotation.csv`](https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv) |
| Image information / IDs | test | [`test-images-with-rotation.csv`](https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv) |

Source index for all links: [official Open Images V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html).

For a training-oriented candidate pool, V2-2C should state explicitly which official source splits it filters. Merely calling all metadata “V7” without retaining the per-row `source_split` and exact artifact provenance would lose essential traceability.

## Bounding-box CSV schema and semantics

The official bbox schema is one object-level box per CSV row:

```text
ImageID,Source,LabelName,Confidence,
XMin,XMax,YMin,YMax,
IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,
XClick1X,XClick2X,XClick3X,XClick4X,
XClick1Y,XClick2Y,XClick3Y,XClick4Y
```

Official semantics relevant to V2-2C:

- `ImageID` identifies the image; `LabelName` is the source-class MID.
- `Source` describes how the box was produced. `xclick` boxes were manually drawn from four extreme clicks; `activemil` boxes were generated semi-automatically and human verified to IoU greater than 0.7.
- `Confidence` is a dummy value and is always 1 for these bbox rows; it is not model confidence.
- `XMin`, `XMax`, `YMin`, `YMax` are normalized image coordinates. X runs left-to-right from 0 to 1 and Y top-to-bottom from 0 to 1.
- `IsOccluded=1` means another object occludes the instance.
- `IsTruncated=1` means the object extends beyond the image boundary.
- `IsGroupOf=1` denotes one box around a touching/heavily occluded group of more than five instances.
- `IsDepiction=1` means a representation such as a cartoon or drawing rather than a real physical instance.
- `IsInside=1` means the photograph was taken from inside the object, such as a car interior.
- For all five attributes, 1 means present, 0 absent, and -1 unknown.
- Extreme-click coordinates are normalized; `activemil` rows use -1 dummy click values.

Source: [official V7 bbox format and attribute definitions](https://storage.googleapis.com/openimages/web/download_v7.html#bounding-boxes).

The requested project formula

```text
normalized_area = (XMax - XMin) * (YMax - YMin)
```

is compatible with the official normalized coordinate convention. Parser QA should still reject or quarantine rows with non-finite values, coordinates outside `[0, 1]`, `XMax <= XMin`, or `YMax <= YMin`; that is a project validation policy derived from the documented geometry, not a claim that the official source contains such errors.

### Filtering implications

- Preserve occluded and truncated objects: those attributes directly cover V2 failure modes.
- Exclude `IsDepiction=1` from the default real-world candidate pool.
- Treat `IsGroupOf=1` separately rather than converting a group box into one individual-object label.
- Retain raw values including `-1` unknown; never coerce unknown to false.
- `IsInside=1` is valuable metadata for traffic relevance review because a car-interior image may contain target labels yet not represent the desired roadside detection domain.

The official description also warns that train boxes were created for available positive, human-verified labels and emphasizes the most specific labels. Validation/test are described as more exhaustively annotated. Therefore, absence of a requested bbox class from a train image is not universal proof that the object is visually absent; pilot “annotation completeness” review should account for Open Images' class-specific annotation policy. [Official V7 description, bounding boxes](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#bounding-boxes)

## Image information, attribution, and license fields

The official image-information schema is:

```text
ImageID,Subset,OriginalURL,OriginalLandingURL,License,
AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,
Thumbnail300KURL,Rotation
```

The official documentation says this table contains image URLs, Open Images IDs, rotation, titles, authors, and license information. `OriginalMD5` is a base64-encoded binary MD5, `OriginalSize` is the original download size, and `Thumbnail300KURL` is an optional convenience thumbnail URL. If the thumbnail is missing, the original URL must be used. It also warns that generated thumbnails can change in contents or resolution over time. [Official V7 image-information format](https://storage.googleapis.com/openimages/web/download_v7.html#image-information)

Minimum governed fields to retain for every pilot image:

- `ImageID` and `Subset`;
- `OriginalURL` and `OriginalLandingURL`;
- `License` exactly as published;
- `AuthorProfileURL`, `Author`, and `Title` for attribution;
- `OriginalSize`, `OriginalMD5`, and a newly computed local SHA256 of the downloaded pilot file;
- `Thumbnail300KURL` and `Rotation`;
- exact metadata artifact URL, access date, and source-row provenance.

Suggested license-status policy:

- `VERIFIED`: the per-image metadata row exists; its license URL is an allowed license under project policy; required attribution/source fields are present; and the original landing/license reference can be checked at review time.
- `REQUIRES_REVIEW`: metadata is present but one or more attribution fields, URLs, or current-source checks are incomplete or inconsistent.
- `REJECTED`: image metadata is missing, its license is outside the approved policy, or the source/license evidence is unusable.

This is intentionally stricter than accepting the global dataset label. The official no-warranty warning means even `VERIFIED` should record the evidence and review date rather than imply Google guarantees the upstream license.

## Annotation license versus image license

The two rights layers are separate:

| Layer | Official statement | V2-2C treatment |
|---|---|---|
| Google-created annotations/metadata | Google LLC licenses annotations under **CC BY 4.0** | Record CC BY 4.0 and required attribution for annotation reuse. |
| Underlying image pixels | Images are listed as **CC BY 2.0**, but Google provides no representation or warranty and tells users to verify each image | Resolve and retain each image's own `License` and attribution metadata; independently gate every downloaded image. |

Source: [official V7 license section](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses).

Consequently, `annotation_license = CC BY 4.0` must never automatically set `image_license_status = VERIFIED`. Any candidate manifest should keep these as independent fields.

## Official targeted image-download mechanism

The Open Images V7 page provides an official Python subset downloader specifically to avoid downloading the full 1.9M dense-annotation images. Its input is one line per image in the form:

```text
train/f9e0434389a1d4dd
validation/<IMAGE_ID>
test/<IMAGE_ID>
```

The official procedure is:

```text
python downloader.py IMAGE_LIST_FILE --download_folder=DOWNLOAD_FOLDER --num_processes=5
```

Source: [official V7 manual subset-download instructions](https://storage.googleapis.com/openimages/web/download_v7.html#download-manually), [official `openimages/dataset` downloader source](https://github.com/openimages/dataset/blob/main/downloader.py).

The official script uses unsigned access to the CVDF AWS S3 bucket `open-images-dataset` and fetches key `{split}/{image_id}.jpg`. The CVDF mirror documentation states that bbox-annotated images are resized so their longest side is at most 1024 pixels while preserving aspect ratio. [Official downloader source](https://github.com/openimages/dataset/blob/main/downloader.py), [official CVDF mirror instructions](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations)

For V2-2C, the safest reproducible workflow is:

1. Filter official bbox metadata and join exact `ImageID`/split rows to official image-information metadata.
2. Apply bbox-attribute, context, and per-image license gates before downloading pixels.
3. Create the official downloader input as `split/ImageID`, with only the selected 200–500 pilot IDs.
4. Use the official downloader (or an implementation that fetches the same documented unsigned S3 keys) and record tool revision/URL.
5. Compute local SHA256 immediately after download, while retaining `OriginalMD5` as source metadata.
6. Validate decoded dimensions and account for `Rotation` when converting normalized boxes to displayed pixel coordinates.

Do not substitute the metadata `OriginalURL` blindly for the governed mirror download: upstream links can disappear, and the official image-information page says thumbnails can change. If the original landing page is consulted for license verification, record that as a separate review action from downloading the CVDF-resized image.

## Implementation constraints derived from official evidence

- Resolve target IDs from `oidv7-class-descriptions-boxable.csv`; reject name-only guessed mappings.
- Preserve raw MIDs and all five bbox attributes, including `-1` unknown values.
- Keep `source_split` and exact metadata-artifact version/URL.
- Join bbox rows to image-information rows by `ImageID` before pilot selection.
- Keep annotation license and image license in separate fields.
- Apply `LICENSE_UNVERIFIED`/`REQUIRES_REVIEW` before download when required metadata is absent.
- Favor traffic co-occurrence candidates only after exact MID matching; Person + Car alone remains a metadata context signal, not proof of road-scene relevance.
- Use manual pilot review for domain relevance and apparent annotation completeness; the official train annotation policy does not justify assuming every visually present class is boxed.
- No evidence in the official sources supports using Stage 18 Locked Test for selection; project governance must restrict that data to overlap checking only.

## Primary sources consulted

- [Open Images V7 official site](https://storage.googleapis.com/openimages/web/index.html)
- [Open Images V7 official description and licenses](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [Open Images V7 official downloads and data formats](https://storage.googleapis.com/openimages/web/download_v7.html)
- [Official V7 boxable class descriptions](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
- [Official Open Images subset downloader](https://github.com/openimages/dataset/blob/main/downloader.py)
- [Official CVDF Open Images mirror/download instructions](https://github.com/cvdfoundation/open-images-dataset)
