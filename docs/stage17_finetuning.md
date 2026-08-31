# Stage 17 — YOLO26n Taiwan Fine-tuning

## Scope and governance

Stage 17 materializes only the governed `TRAIN` (1,266 images) and `VAL`
(271 images) rows from the Stage 16 split manifest. The 271 `LOCKED_TEST`
images and 16 `EXCLUDED` images are not copied into the processed dataset and
are not exposed by the training data YAML. Raw images and YOLO label bytes are
left unchanged.

The Stage 16 split-manifest SHA256 is
`3a6c91aabf50c6500f1755e38143c60d3c41a9c00b8448a6bcde678778688b97`.
The base-model SHA256 is
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

## Controlled training run

- Base model: `models/pretrained/yolo26n.pt`
- Device: Apple MPS (no CPU fallback)
- Image size: 640
- Epochs: 50 completed / 50 requested
- Patience: 10
- Batch: 8
- Seed: 1601
- Augmentation: Ultralytics standard defaults; no custom augmentation or
  hyperparameter search
- Model status: `CANDIDATE`

The host interrupted the original process after epoch 28. Training resumed
from the same run's `last.pt`, including optimizer state and original training
arguments, and completed epochs 29–50. This was checkpoint continuation, not a
second experiment. PyTorch emitted a warning that
`index_put_with_accumulate_mps` has no deterministic MPS implementation even
with deterministic algorithms enabled in warn-only mode; therefore the fixed
seed improves repeatability but does not guarantee bit-for-bit reproduction on
this MPS stack.

## Validation-only results

Best epoch: 50. Overall VAL metrics: precision 0.8287, recall 0.5526, mAP50
0.6509, and mAP50-95 0.3683.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.7216 | 0.5571 | 0.6819 | 0.3330 |
| bus | 0.8930 | 0.7791 | 0.8545 | 0.5549 |
| car | 0.8312 | 0.7211 | 0.8304 | 0.4642 |
| person | 1.0000 | 0.0000 | 0.1041 | 0.0358 |
| motorcycle | 0.8260 | 0.6041 | 0.7370 | 0.3598 |
| truck | 0.7006 | 0.6540 | 0.6974 | 0.4625 |

The person class has only 87 VAL boxes across 25 images and zero recall at the
selected operating point; its apparent precision of 1.0 must not be interpreted
as strong performance. Bicycle also has only 70 VAL boxes. Bus and car are the
strongest classes by mAP, while truck remains subject to the source taxonomy's
commercial-vehicle ambiguity. Motorcycle has abundant Taiwan-scene examples
and moderate recall, but no final comparison against the pretrained model is
made here.

## Candidate artifacts

`best.pt` and `last.pt` are generated, Git-ignored artifacts under
`models/finetuned/stage17/`. The small model manifest is tracked there and
records hashes, configuration, split isolation, metrics, and
`model_status = CANDIDATE`. The candidate is not promoted to the runtime model.
Locked Test evaluation and pretrained-versus-fine-tuned comparison remain
reserved for Stage 18.
