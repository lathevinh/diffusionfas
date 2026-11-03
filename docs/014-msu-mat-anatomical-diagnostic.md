# Fixed MSU MAT anatomical inpainting development diagnostic

## Predeclared protocol

After accepting the native-512 crop and four anatomical masks in
[report013](013-mat-native512-inpainting-preview.md), the user explicitly
approved the following bounded diagnostic. This protocol was written before
new MAT PAD scoring. Results are not an independent test.

- Exact same 22 geometry-qualified MSU validation originals as the
  [frozen DDPM crop diagnostic](012-msu-frozen-crop-ablation.md):
  six bona fide, sixteen attacks. No new dataset, window or subject selection.
- Existing lower-middle T16 window: sixteen frames, four independent masks
  per frame, one reconstruction per mask: **1,408 reconstructions**.
- Unchanged native RGB to real-pixel oriented square, render1024 to512.
  No padding, shifting, shrinking, stretching or upscaled cache256.
- Frozen, strictly loaded safetensors model from the previously approved
  third-party mirror. Equivalence to author weights remains **unverified**;
  research-only use, no new download and no training.
- Same addressed latent/internal-dropout seeds as the accepted preview:
  seed20261103 plus original key, clip slot, mask name and `MAT-preview`.
  Seeds and inference do not use class labels.
- RGB normalization `/127.5 - 1`; mask1 known,0 hole; missing normalized
  image content explicitly zeroed. Exact known-pixel preservation, deterministic
  replay and changed-hidden-ground-truth invariance remain mandatory.
- **Primary score:** RGB MSE within each hole, equal arithmetic mean across
  the four regions, then equal mean across sixteen frames. Not weighted by
  hole size, not maximum error, not best reconstruction. Higher means attack.
- Four per-region video means are predeclared **secondary descriptive**
  results. No choosing a winning region, sign reversal, threshold fitting,
  hyperparameter tuning or extra draws after seeing AUROC.
- Paired comparison to already measured real-pixel DDPM scores on exactly
  these originals. AUROC and paired subject-group bootstrap,2,000 repetitions,
  using the same fixed bootstrap seed as the DDPM crop report.
  Only three declared groups: uncertainty estimates are fragile.

The approved worker cap is30 minutes,8GiB process-tree RSS,8GiB **total-device**
GPU memory including other users, and1GiB new output. The watchdog also counts
the reused official-source directory conservatively. Any failed attempt is
preserved; no automatic retry. No fitting, calibration or target pixels,
CASIA/SiW access, threshold fitting or model training.

## Implementation and evidence

[Input policy](../src/diffusionfas/mat_input.py) and
[accepted model inference](../src/diffusionfas/mat_preview.py) are reused
without modification, preserving historical plan hashes.
[Diagnostic implementation](../src/diffusionfas/mat_diagnostic.py) checks
the accepted preview, explicit approval, completed baseline, exact population,
window, pinned weights/source and current implementation in parent and worker.
[Bounded CLI](../scripts/mat_diagnostic.py) requires conda environment `fas`.

Private evidence retains all352 original512 crops and **unquantized float32
hole predictions**, rather than recomputing losses from clipped preview PNGs.
This permits independent exact-mask/hole-only MSE and score recomputation.
Private identities, media, individual residuals and model weights stay outside
Git. Only aggregate verified results will be published.

Pre-run plan SHA256:
`38a5621ccd6b80347f55dccd1d8b340a15703de5be95f8c183aed73b2f7910f8`.
The fixed bootstrap seed is `149462085860204614`.
All **41 focused tests** passed (MAT input/inference/diagnostic and existing
crop preview/ablation). Python imports and the actual metadata-only plan CLI
succeeded in `fas`. Pylance could not initialize its MCP transport, so no
editor type-check success is claimed. No environment changes were needed.

```bash
conda run --no-capture-output -n fas python scripts/mat_diagnostic.py plan \
  --preview-root /private/mat-preview-v2 \
  --baseline-root /private/msu-crop-ablation-v2 \
  --private-root /private/mat-diagnostic-v1

conda run --no-capture-output -n fas python scripts/mat_diagnostic.py run \
  --plan /private/mat-diagnostic-v1/plan.json
```

The first command requires an immutable private visual/diagnostic approval
receipt with this exact protocol and budget. Neither command grants authority
for another run, another cohort or an automatic retry.

## Results

Pending the approved run and independent postflight.
