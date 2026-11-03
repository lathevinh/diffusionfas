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

**Completed and independently verified:**22/22 originals,352/352 frames,
1,408/1,408 reconstructions; **zero technical failures**. No retry.
Protocol and implementation were committed as `6bff041` before this run.
See [aggregate-only evidence](../results/mat-diagnostic-v1.json).

| Fixed video-level score | AUROC | Subject-bootstrap95% CI |
|---|---:|---|
| DDPM real-pixel baseline |0.6979166667|[0.625,0.750]|
| **MAT primary: equal four-region MSE** |**0.7500000000**|[0.625,0.8333333333]|

Primary paired difference is **+0.0520833333 AUROC**;95% paired bootstrap CI
**[-0.0833333333,+0.1041666667]**, including zero. All2,000 replicates were
usable, with only **three declared subject groups**. This does **not** establish
an improvement over DDPM, generalization, calibrated accuracy or liveness.
AUROC0.75 means observed pairwise ranking, **not75% classification accuracy**.

Predeclared secondary, descriptive only:

| Region | AUROC, fixed higher=attack |
|---|---:|
| Left eye |0.7812500000|
| Right eye |0.7916666667|
| Nose |0.2604166667|
| Mouth |0.6875000000|

The nose behaves in the opposite direction on this subset. It remains in the
primary score with the originally fixed positive sign and equal weight.
Neither it nor a "best eye" is selected, discarded or sign-flipped after
inspection. Different regions can reflect different face structure, lighting
or reconstruction difficulty rather than spoof evidence; these results alone
do not establish the cause. The highest secondary value is not the headline.

### Independent postflight

- Verified all22 private evidence archives and their receipt hashes.
- Recomputed all1,408 binary masks, addressed seeds, hole areas and losses
  from **unquantized float32 hole predictions** and original uint8 crops.
  Maximum MSE difference from independent float64 summation:
  **1.1102230246251565e-16**, attributable to reduction order.
- Verified352 crop hashes and native frame addresses; independently
  recomputed square geometry, landmark transforms and source margins.
  Minimum native margin **9.8715095592 pixels**; padding fraction0.
- All18 overlapping reviewed frames and72 reviewed reconstructions reproduced
  their PNG pixels **exactly**, with exact matching masked MSE.
- Independently verified22 strict replay receipts against **6,028 full native
  RGB frames**, original frame counts and decoded-sequence hashes:
  no substitutions, no decode fallback.
- sklearn independently reproduced all six AUROCs, including the DDPM baseline.
  A separate weighted pairwise implementation reproduced both subject-bootstrap
  intervals and the paired-difference interval.
- Actual GPU synthetic replay error and hidden-ground-truth-change error:
  **0**. Frozen model state digest unchanged:
  `c2fc813240578fbcf72180c128ea443c567462a43976d62edec0866e24a997f4`.
- No CUDA-extension setup/fallback in the successful worker log.
  No staged raw videos remain. Private crops/hole predictions remain as
  authorized experiment evidence.

The CPU postflight initially used a source-manifest `video_id` field for a
prepared-video row, whose actual schema uses `key`. Corrected only the private
verifier; the completed model run was **not rerun or changed**.

### Resources and model provenance

Worker wall time **235.637624574 seconds**. Watchdog0.2-second samples recorded
peak process-tree RSS1,910,349,824 bytes, total-device GPU5,282,725,888 bytes and
owned output293,002,503 bytes (including the reused source directory).
Actual PyTorch model peak allocated/reserved memory:
1,348,788,224 /1,656,750,080 bytes. These stayed within the approved caps;
sampled telemetry is not a continuous instantaneous-peak measurement.

Reused the approved safetensors mirror `not-lain/MAT-CelebA-HQ_512`, revision
`b9dd007aea6d7c9e53818f2d11b071b1be5bf48c`, SHA256
`b47eb8ff60bc25131b531eab96cf8f8d4e85a86da688c7595ea50003df93d8bb`,
and pinned official source
`d273d891ecdad2e1df106516423a75bc45b2d800`. No new acquisition, installation,
training, threshold fitting, fitting/calibration/target pixel access.
**Official-weight equivalence and exact training-crop equivalence remain
unverified.** Research-only provenance/licensing caveats from report013 still
apply.

## Interpretation and next gate

This fixed pretrained inpainting score has a **promising observed signal** on
the small, previously inspected development subset, but its apparent
advantage over DDPM is uncertain. The region-direction inconsistency is
important counterevidence to a universal "spoof always reconstructs worse"
assumption. Both scores process frames independently; averaging T16 is not
learning temporal liveness.

Do not train longer, select eye-only scoring or deploy a threshold on the
strength of this result. A next experiment should freeze the present primary
recipe and evaluate additional native-crop-qualified subjects/domains with
explicit population, failure denominators and a new access/resource approval.
Any region/method changes become a separate development recipe needing a
later independent evaluation. No extra experiment was authorized or run.
