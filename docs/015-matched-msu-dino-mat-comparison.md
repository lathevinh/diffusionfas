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

**Completed and independently verified, with no technical failures or retry.**
The protocol was committed as `4a46292` before inference. All22 originals,
352 native518 crops,704 image/encoder forwards and four synthetic repeat
forwards completed. Existing MAT/DDPM scores were reused, not rerun or tuned.
See [aggregate-only evidence](../results/dino-comparison-v1.json).

### Same originals and frame windows

| Existing head seed | DINO-Reg AUROC | Plain DINO AUROC | Fixed two-DINO AUROC |
|---|---:|---:|---:|
|20260917|0.9791666667|0.9687500000|0.9791666667|
|20260923|0.9791666667|0.9895833333|0.9895833333|
|20261001|0.9791666667|0.9895833333|0.9791666667|
|**Descriptive equal-seed mean**|**0.9791666667**|**0.9826388889**|**0.9826388889**|

On these same22 originals, MAT remains **0.7500000000** and the earlier
real-pixel DDPM score **0.6979166667**. The crops are deliberately model-specific,
not geometrically matched: do not attribute the entire difference to backbone
architecture or learning objective. DINO has supervised PAD source heads;
MAT has no PAD fine-tuning.

Each seed's two-DINO subject-bootstrap95% CI is
**[0.9761904762,1.0]**. Paired two-DINO-minus-MAT differences are respectively
**+0.2291666667,+0.2395833333,+0.2291666667**, each with conditional interval
**[+0.1666666667,+0.375]**. All2,000 replicates per seed are valid.
All30 per-system/paired intervals are published in the aggregate.
Only **three subject groups** underlie these discrete, conditional intervals:
there is no pooled mean CI or deployment/generalization guarantee.

Plain DINO and two-DINO have the same descriptive mean here; combining two
encoders is not proven better on this subset. Do not select a single best seed
or promote a new single-model policy based on these22 clips alone.

**This is not an improvement from paper324's0.927 to0.983 on the same test.**
Paper324 uses the official MSU target-test cohort and a representative image.
This diagnostic uses22 selected source-validation originals and a T16
probability average. The population, aggregation and coverage constraints
differ, despite reusing the actual same saved models.

### Does MAT add a different ranking signal?

These are **96 dependent attack/bona-fide pair orderings per seed**, not counts
of correctly classified videos, spoof rescues or newly safe authentication
decisions. Both model score directions were fixed before this comparison.

| Seed | Both order correctly | DINO wrong / MAT correct | DINO correct / MAT wrong | Both wrong | Pairs with ties |
|---|---:|---:|---:|---:|---:|
|20260917|70|2|24|0|0|
|20260923|71|1|24|0|0|
|20261001|70|2|24|0|0|

The complete3x3 wrong/tie/correct matrices are preserved for every seed in the
aggregate. MAT orders the1--2 DINO-inverted pairs correctly, so it is not
literally identical information. Conversely, DINO orders all24 MAT-inverted
pairs correctly. This small, asymmetric complementarity is **not evidence of
fusion utility**: no inference-time gate can choose the right model using the
evaluation labels. No MAT threshold, fixed new fusion, oracle fused score,
learned gate or threshold-dependent false-accept claim was computed.

### Independent verification and resources

- Verified all committed input/evidence hashes after the run, including all
  six supervised heads/calibrators, both backbone weights and source helpers.
  Complete MSU exclusion applies to all source fitting roles for all seeds.
- Verified352 crop hashes and inverse-mapped cubic footprints: **zero native
  padding**. Historical pre-resize width/height ratios range
  **0.7625570776--0.8664495114**; distortion disclosure is substantive, not
  a claim of equivalent square crops.
- Verified704 finite FP32 feature vectors, exact16x1536 per original/encoder.
  Independently reconstructed all3x2x352 frame probabilities from saved
  coefficients and features using scalar compensated summation and a stable
  sigmoid, not the original inference helper.
  Maximum probability difference **1.3877787808e-15**.
- sklearn independently reproduced every AUROC. Independent pair tables
  reproduced every count, including no ties. A weighted pairwise bootstrap
  reproduced all **30 intervals**, with original group draws and denominators.
- Verified all22 strict native-replay receipts against **6,028 full RGB
  frames**, counts and decoded-sequence hashes, without new native decoding.
  No substitutions. No staged raw videos remain.
- Actual GPU synthetic repeated extraction error is **0 for both encoders**.
  Each backbone state digest matched before/after inference.
- Worker wall time **60.99555798 seconds**. Watchdog0.2-second samples recorded
  peak tree RSS1,854,021,632 bytes, **device-wide** GPU4,062,183,424 bytes and
  owned output258,571,522 bytes. These are sampled, not instantaneous-peak
  guarantees; all approved guards passed.
- **48 focused tests passed**. No packages installed/upgraded, fitting,
  model download, new calibration/target pixels or fusion/threshold selection.
  The FAS source checkout stays read-only.

## Decision and next gate

On this small matched-video diagnostic, the existing DINO systems clearly
outrank raw MAT/DDPM reconstruction errors. Prioritize DINO as the strong
baseline; this result does **not** justify longer MAT training or deploying
an inpainting detector.

There is a small distinct MAT ranking signal, but no demonstrated operating
benefit from adding it. A fusion experiment would require an explicitly
declared source-only fitting/selection protocol and later independent data,
not weights or gates chosen on these22 clips.

The useful next question is low-false-accept threshold transfer and stability
on additional subjects/cameras, with no threshold chosen from the held-out
evaluation labels. The present near-perfect ranking does not resolve
paper324's operating-policy failures or establish security. No extra
experiment, threshold fitting or access beyond the approved22 was performed.
