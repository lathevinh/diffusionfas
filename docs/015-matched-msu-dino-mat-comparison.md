# Matched-video frozen DINO versus MAT development diagnostic

## Predeclared scope and approval

The user explicitly approved comparison on the same22 MSU validation originals
and lower-middle T16 windows as [MAT](014-msu-mat-anatomical-diagnostic.md):
six bona fide, sixteen attacks, three declared subject groups.
No new dataset/subject selection, fitting, threshold selection or fusion tuning.

Use the **actual saved paper324 MSU-excluded heads and monotone calibrators**
for both DINO-Reg and plain DINO, for **all three seeds**:
20260917,20260923,20261001. The entire MSU dataset is excluded from all
fitting/validation/calibration roles for these heads. Check roles against
the original sealed partitions, selected-frame hashes and source manifest;
bind the saved model files, original successful completion/reconstruction,
public evidence, official source and already acquired weight hashes.
These DINO heads used supervised PAD source labels; MAT has not been PAD
fine-tuned. This is a systems comparison, not equal training supervision.

Both encoders run frozen in FP32, batch1, no AMP or TF32,518 resolution.
Pool normalized CLS plus mean normalized patch tokens, excluding registers,
using the original helper. Reuse head coefficients and positive-slope
calibrators without optimization. Average **frame probabilities**, not
logits, across16 frames. Fixed two-DINO is the original50/50 probability
average, not a new learned fusion or an ensemble across the three seeds.
Report every seed and descriptive mean AUROC, never a best seed.

### Crop limitation explicitly accepted

This comparison matches **originals and native frame addresses**, not crops.
DINO preserves the historical head-training preprocessing: five-landmark
rotation about the eyes,1.3x face-box context, original bicubic resize of
the rectangle to518x518. Aspect ratio distortion is possible; this is
explicitly disclosed and accepted, not silently replaced by MAT geometry.
Render from native RGB, not enlarged cache256.

Require the inverse-mapped **4x4 bicubic source footprint** of the whole DINO
crop to be in real native pixels. Reject any unsupported crop; no zero or
reflected context, crop relocation or corrective resize. Fail on the first
technical failure, preserve the attempt, do not automatically retry.
MAT keeps its already accepted real-pixel oriented square512 input and
previously measured scores. Neither model is rerun with a modified crop
to improve measured results.

### Endpoints fixed before scoring

- Video-level AUROC and paired subject-bootstrap95% intervals,2,000
  repetitions, same fixed group/bootstrap seed as the MAT diagnostic.
- Per seed, compare MAT, DINO-Reg, plain DINO and fixed two-DINO on the
  same complete ordered originals. Retain the existing DDPM result as context.
- Complementarity: all16x6=96 attack/bona-fide pair orderings. Publish the
  complete3x3 matrix of wrong/tie/correct orderings for two-DINO versus MAT.
  Ties stay explicit; report both DINO-wrong/MAT-correct and the reverse.
  These pairs share videos and are **not96 independent trials**.
- MAT has no calibrated threshold. These counts are **not spoof rescues,
  false-accept reductions, accuracy, operating-point utility or fusion gain**.
  No MAT threshold, percentile cutoff, fitted weight or selected eye-only
  variant will be introduced after viewing these results.

New run cap:30minutes,8GiB process-tree RSS,8GiB **device-wide** GPU usage,
1GiB new output.704 image/encoder forwards and four synthetic repeat forwards.
No download, installation, new fitting, calibration/target pixels, other
datasets, threshold changes or retry. Existing source-trained parameters
and metadata are read-only; private media/results stay outside Git.
Bytecode writes into the source checkout are disabled.

## Implementation and reproduction

[Comparison](../src/diffusionfas/dino_comparison.py) reuses the original
read-only FAS normalization, pooling, crop, head-logit and calibration helpers.
[CLI](../scripts/dino_comparison.py) performs parent/worker commitment checks
and device-wide supervision. Private evidence retains352 cropped518 RGB
images,44 feature arrays, all frame probabilities and native replay receipts.

```bash
conda run --no-capture-output -n fas python scripts/dino_comparison.py plan \
  --mat-root /private/mat-diagnostic-v1 \
  --private-root /private/dino-comparison-v1 \
  --approval /private/dino-comparison-approval-v1.json

conda run --no-capture-output -n fas python scripts/dino_comparison.py run \
  --plan /private/dino-comparison-v1/plan.json
```

The explicit approval must bind this exact policy, all three seeds and caps.
The completed MAT execution is replayed only as saved evidence; its historical
implementation hashes remain checked without treating newly added diagnostic
modules as modifications to the old run.

Pre-run plan SHA256:
`93319fead64a6a3a0c0f9e76df3452e16605d84b81d836689860347e29b831a3`.
All **48 focused DINO/MAT/crop tests** passed in `fas`. Metadata-only audit
checked all352 source bicubic footprints using synthetic shape placeholders:
no real PAD pixels read, all footprints inside native bounds. No package
installation or upgrade.

The initial metadata isolation preflight found a legitimate failed source
original with null selected-frame evidence. The audit now checks dataset
exclusion for every original and exact-frame exclusion for successful inputs;
it explicitly rejects missing evidence on a successful input. A regression
test covers both cases. This occurred before plan creation, PAD decoding or
encoder inference; it is not an experiment retry.

## Results

Pending the approved run and independent postflight.
