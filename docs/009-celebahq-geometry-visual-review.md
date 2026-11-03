# 009 - CelebA-HQ-style geometry: private visual review before ablation

## Why review preprocessing first

The [source diagnostic](008-source-validation-spoof-diagnostic.md) measured the
implemented raw-residual recipe, not every possible diffusion PAD approach.
The legacy pipeline expands a rotated face rectangle by 1.30 and resizes it
to 256 x 256. A nonsquare rectangle can therefore be stretched anisotropically.
Synthetic execution confirmed a 158 x 236 context becoming 256 x 256.
The extent and effect on actual PAD performance have not yet been measured.

The user requested CelebA-compatible cropping and a private HTML contact sheet
to approve visually **before** model scoring or an ablation run.

## Reference facts and limits

- The [original DDPM loader](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/tpu_utils/datasets.py)
  uses `celeb_a_hq/256:2.0.0`, not unprocessed full-camera frames.
- The [TFDS 2.1 loader](https://github.com/tensorflow/datasets/blob/v2.1.0/tensorflow_datasets/image/celebahq.py)
  reads prebuilt images and points to the original CelebA-HQ construction.
- The [CelebA-HQ construction reference](https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py)
  uses eye/mouth landmarks for an oriented square, with context around the head.
- The pinned [Hugging Face model card](https://huggingface.co/google/ddpm-celebahq-256/blob/cd5c944777ea2668051904ead6cc120739b86c4d/README.md)
  does not provide a complete weights-conversion/training preprocessing manifest.
  We do not claim bit-exact equivalence for this checkpoint.

Matching geometry is distinct from reproducing the dataset's enhancement,
delta corrections, historic Pillow behavior and full multiresolution pipeline.
Do not silently substitute FFHQ preprocessing or a tight recognition crop.

## Versioned geometry

[Geometry implementation](../src/diffusionfas/celebahq_geometry.py) defines
`celebahq_geometry_v1` independently of the frozen legacy FAS preprocessing:

- The horizontal direction combines the eye baseline and eye-to-mouth axis;
  the vertical direction is perpendicular.
- Square half-side is the maximum of twice eye distance and 1.8 times
  eye-midpoint-to-mouth-midpoint distance.
- Center is eye midpoint plus 0.1 times the eye-to-mouth vector.
- The transform is isotropic rotation/scale/translation, not rectangular stretch.
- Render at 1024 with bilinear interpolation and reflected boundaries, then
  area-downsample to RGB256. Padding fraction is measured from a validity mask.
- No learned super-resolution, face restoration, beautification or global blur.
  Border reflection/interpolation are approximations, not bit-exact HQ processing.

This changes geometry, context and resampling together. A future A/B result
would test the combined preprocessing intervention, not isolate a single cause.
Additional background may itself contain attack/domain cues.

## Approved preview scope and stop

The user approved one CPU-only serial attempt:

| Limit | Approved |
|---|---:|
| Shared worker wall time, including HTML rendering | 1,200 seconds |
| Worker-tree RSS | 8 GiB |
| New private output | 512 MiB |
| DDPM/GPU scoring or training | None |
| Calibration/target pixel access | None |

Select **12 unique source-validation originals**, two per domain/binary class,
using seed `20261104`. The eligible pool is the 259 previously prepared
originals from the full 283-original diagnostic; the other 24 failures are not
silently rehabilitated. Selection never uses residuals, AUROC, visual quality
or new crop outcomes.

For each video, choose slots **0, 7, 15** of its existing lower-center 16-frame
middle window on the 8-FPS grid. Reuse the SHA-bound smoothed boxes/landmarks,
not new detector outputs. Recover pixels from the original video: resizing the
old cache cannot restore missing context.

Archive bytes, native order/count and RGB hashes remain bound to accepted
receipts; MSU uses strict FFmpeg without fallback. Every decoded native frame
is checked, including nonselected frames. Regenerated legacy crops must be
pixel-identical to their saved cache entries before they enter the preview.
One original is staged at a time and raw staging is removed afterward.

Failed examples remain failures, with no replacement, automatic retry or
automatic visual approval. Unexpected failures stop the worker explicitly.

## Review deliverable

[Preview runner](../scripts/crop_preview.py) has `plan` and `run`:

```bash
conda run --no-capture-output -n fas python scripts/crop_preview.py plan \
  --diagnostic-root /path/to/private/completed-diagnostic \
  --private-root /path/to/private/new-preview

conda run --no-capture-output -n fas python scripts/crop_preview.py run \
  --plan /path/to/private/new-preview/plan.json
```

The private `review.html` embeds all PNGs as data URLs: it opens offline without
a server, external requests or companion image paths. A restrictive CSP disables
scripts/network dependencies. HTML, individual metadata and image files are
prohibited inside any Git worktree.

Each selected frame shows:

1. Original frame thumbnail, with amber legacy and cyan new-square outlines.
2. Exact legacy RGB256 crop.
3. New RGB256 CelebA-HQ-style crop, without annotation on model pixels.

The original preview thumbnail is at most 640 pixels on its long edge. Both
model crops are lossless 256 x 256 PNGs. The page labels timestamp, padding
fraction, source square side and output eye distance. Check head/chin coverage,
face scale, aspect distortion, alignment, reflected boundaries and within-clip
stability; the page makes no claim that human approval has occurred.

Both completion receipt and UI remain **human review pending**. There is no
button or code path that launches an ablation. A future run needs explicit
visual approval and a separately agreed finite budget.

## Validation

[Validation v13](../results/validation-v13.json): **97 tests passed**, including
11 new preview tests. They cover isotropic/rotation-equivariant geometry,
degenerate landmarks, reflected color preservation, seeded split/class scope,
integer window slots, native metadata corruption, closed scoring authority,
actual synthetic video hashes, nonselected-frame hash mismatch, complete
12-video worker-to-HTML execution, visible failures and the no-retry/review stop.
Existing actual CUDA-resume and strict native FFmpeg tests remain passing.

The full synthetic worker exercises three panels on 36 frames and decodes all
108 embedded PNGs. This is code validation, not visual approval of PAD crops or
evidence that new preprocessing improves spoof separation.

## Executed preview: awaiting human review

[Aggregate-only receipt](../results/crop-preview-v1.json) records the real
CPU-only attempt. Plan SHA-256:
`68456b4c673e94b401ea3864ffd42e00334f2d205f82f64887ddd44cb76b290e`.

- Completed all **12/12 selected videos, 36/36 frames**, with no replacements.
- Wall time **41.896 seconds**, sampled peak worker-tree RSS about **262 MiB**,
  sampled peak output about **77.2 MiB**; final output about **41.5 MiB**.
- Self-contained HTML about **24.7 MB**, containing **108 PNG images**.
- Postflight decoded all embedded PNGs, verified asset/HTML/receipt hashes,
  checked all 36 legacy crops against cache pixels, native-frame commitments
  and isotropic transforms. No raw staging remains.
- Padding fractions range from **0% to 49.96%**. Large reflected areas are a
  material review issue, not evidence that the crop policy is acceptable.
- Model scoring/training/calibration/target pixel access remain **zero**.

The integrated browser refused the private file because it is outside the
trusted workspace. We did not move private images into Git, change workspace
trust or bypass that restriction. The user chose to download/open the
self-contained artifact in an external browser.

**Human review is still pending.** Browser visual approval has not been
performed by the agent, and model compatibility/performance has not been
certified. No ablation or further data run is authorized by this receipt.
