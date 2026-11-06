# Spatial DINO: low-FA diagnosis and aspect-preserving crop preparation

The [completed spatial study](025-spatial-dino-completed-results.md) was
reconstructed from its hash-bound saved probabilities without fitting,
calibration or PAD pixel decode. The [receipt](../results/spatial-dino-lowfa-crop-preparation-v1.json)
also records a separately owner-selected CPU crop preview. **No new DINO
training or train-two/calibrate-third experiment has started.**

## 1. Separation versus threshold transfer

For each of five arms and four targets, at alpha1% and5%, the diagnostic
independently sorts source calibration attacks, recounts the saved policy,
and enumerates the grouped target score curve. Tied scores enter together;
threshold acceptance is strict. Original class denominators retain shared
technical failures as nonacceptances. All40 grouped-curve oracle counts
match the previously verified order-statistic oracle and were additionally
checked by brute-force enumeration of sample thresholds.

The oracle uses target labels and is **diagnostic only, not deployable**.
It shows the maximum attainable bona-fide acceptance on this fixed sample
at the specified attack acceptance budget; it does not prove a source-only
policy can attain that performance on future targets.

| Held-out target | Reference oracle BFA at FA <=1% | Reference oracle BFA at FA <=5% |
|---|---:|---:|
| OULU-NPU |52.222%|76.111%|
| CASIA-FASD |93.333%|98.889%|
| MSU-MFSD |42.500%|75.000%|
| SiW-Mv2 |26.437%|45.211%|

| Target | Reference oracle BFA at5% | Content oracle BFA at5% | Anatomical oracle BFA at5% |
|---|---:|---:|---:|
| OULU-NPU |76.111%|84.722%|80.556%|
| CASIA-FASD |98.889%|98.889%|98.889%|
| MSU-MFSD |75.000%|82.500%|70.000%|
| SiW-Mv2 |45.211%|47.893%|46.743%|

At5%, source-policy FA for reference is9.306%,0%,13.333%,51.934% respectively.
OULU and MSU have substantial empirical threshold-transfer headroom at the
internal BFA50% goal. At1%, MSU reference/content/anatomical all fall below
BFA50% even with the target-label oracle. SiW falls below50% at5% in all
three displayed arms: source-domain holdout calibration alone cannot fix
that observed separation deficit.

Across all40 cases,7 meet the current source-policy FA/BFA50% goal,17 have
oracle BFA50% headroom but fail that source-policy goal, and16 have
insufficient oracle separation at that goal. This classification depends
on the utility goal; the implementation also reports BFA85% diagnostically,
explicitly not an ISO/iBeta criterion.

**Decision:** do not combine a spatial architecture change with a new
calibration policy now. A train-two/calibrate-third study remains a possible
matched follow-up for threshold transfer, not a solution to the SiW
feature-separation problem and not an already validated contribution.

## 2. What changes when removing stretch

[Metadata implementation](../src/diffusionfas/spatial_diagnosis.py) examined
all7,505 originals, with7,472 usable historical landmark/crop transactions.
It audited two square candidates without reading images:

1. **Expanded legacy square:** keep the original crop center and rotation,
   take the longer crop dimension as square side, and expand the shorter
   dimension. Do not shift, shrink, pad or choose a different frame.
2. **CelebA-HQ-style square:** reuse the previous eye/mouth geometry as an
   alternative. It is not a requirement of DINO's pretrained checkpoint.

Both check source corner margins and transferred RGB224 landmark support.
A conservative2-source-pixel margin is required for candidate eligibility;
the historical crop must also have real-pixel support for a strict pair.
Records distinguish new-crop eligibility, strict-pair eligibility, original
failures and geometry reasons. Nothing is silently replaced or relabeled.

The historical rectangle's median anisotropy factors, by dataset/class,
range about1.285 to1.359, with95th percentiles about1.426 to1.543. Thus
distortion is not limited to a single isolated example.

| Held-out target | Original test population | Expanded-square strict pairs | CelebA-HQ strict pairs |
|---|---:|---:|---:|
| OULU-NPU |1800|1566|1190|
| CASIA-FASD |360|220|211|
| MSU-MFSD |160|147|127|
| SiW-Mv2 |623|501|446|

These are **metadata eligibility counts**, not a new test result. Original
and usable populations differ;623 is the SiW original target count, not
its618 historical usable target count.

The owner selected expanded-square **preview only**, keeping the change
closer to the old DINO crop than the DDPM-style alternative. Square
expansion adds context and changes sampling geometry as well as aspect
ratio, so any future gain cannot be attributed solely to "removing stretch."

## 3. Private sixteen-frame preview

[Preview implementation](../src/diffusionfas/spatial_crop_preview.py) selects
two originals per class per held-out target, seed20261106, by a stable
identity hash without scores. It reads the exact original saved native
PNG middle frames; no native video, new detector or GPU inference is used.
The selection includes16 unique frames, not alternative windows or
replacement subjects.

All16 legacy crops were regenerated and matched their pinned RGB224 pixels
exactly. Each new square is rendered at native crop scale with bicubic
interpolation, then bicubic-resized to224. The final landmark transform
includes the resize pixel-center convention. Real source footprint checks
cover affine extrema plus a two-pixel cubic radius and1/32-pixel OpenCV
quantization margin. No reflected or zero-filled source context is sampled.
Both old and square footprints passed for all16 selected frames.

The self-contained private HTML contains48 PNGs: native frame/square outline,
historical RGB224, and square RGB224 with landmark dots. **The orange dots
are geometry landmarks, not learned attention maps.** Raw images, identities,
HTML and individual scores remain outside Git. The paired preview was also
visually inspected for one sample from each of the four datasets.

Preview ran24.036s under a20-minute/2GiB/256MiB supervised cap, with peak
tree RSS1,526,673,408 bytes and output8,146,522 bytes. The48 embedded PNGs,
all native/crop/transaction bindings and the HTML hash were independently
checked. This validates only the selected frames, not the full cohort.

## 4. Before any new training

The proposed matched factorial is four targets x three arms
(`reference`, `content`, `anatomical`) x two crops = **24 new backbone fits**
and24 new source calibrators, subject to a separate exact plan. A
common-cohort source restriction changes the fitting distribution: the
old full-source12 corresponding checkpoints cannot substitute for
matched training on the restricted cohort. Six or12 "new fits only" would
confound preprocessing with training population.

Metadata predicts248,400 maximum optimizer steps at batch32 and200 epochs
for the24 proposed fits. This is an inventory estimate, not a measured
wall-time/GPU budget or an executable sealed training plan.

Required preparation after visual acceptance:

- Verify real saved PNG bytes/shape and actual interpolation support for
  **all** candidate role images, not just the16 preview frames.
- Freeze a common eligibility mask and retain original failure/exclusion
  accounting in both arms. Do not silently compare the restricted cohort
  to the old full-population table.
- Keep the original source role assignments; audit subject/group isolation
  and both classes in each source stratum after applying the common mask.
- Report both conditional common-cohort metrics and original-denominator
  system metrics. For example, treating crop-ineligible real targets as
  nonacceptances gives a BFA ceiling60/90=66.667% on CASIA and
 214/261=81.992% on SiW before classification errors. These ceilings must
  not be hidden by excluding difficult originals from denominators.
- Rebuild aligned priors from the new pixel-center transform; preserve
  augmentation/PDA donor transport and verify exact legacy crop parity.
- Measure a synthetic resource probe, seal input/code/cohort/budget
  commitments and obtain exact execution approval before training.

Learned patch-weight maps were **not** computed in this CPU stage: they
require token inference on bound PAD images. A separate small diagnostic
scope should specify fixed score-independent images, model checkpoints,
maps and resource caps; no anatomy-causality claim follows from landmark
overlays. Neither visual acceptance nor metadata eligibility alone
authorizes this inference or new training.

## 5. Validation and preserved preparation failures

48 focused tests passed in2.222s, including oracle ties/failures,
independent random brute-force curves, exact operating counts, crop
geometry, preview selection and native render parity, plus prior crop/MAT
input regressions. Six Python files parsed. No dependency was installed.

The first CPU diagnosis stopped because AUROC comparison demanded exact
floating-point equality; the observed differences were1e-16 to3e-16.
All operating counts matched. The fix reuses the existing1e-10 verifier
tolerance for AUROC while preserving exact count/FA/BFA checks, with a
regression that rejects actual changed decisions.

The first preview stopped because a floating bicubic support mask was
compared exactly with1. Fully supported interior pixels differed by about
1e-7. The fix verifies interpolation footprint geometrically rather than
relaxing border eligibility or accepting an approximate mask sum. Both
failed private roots/logs remain, and the same16 selected frames were used
for the corrected preview. No scientific model was retrained or replaced.

The diagnosis rerun took43.057s under30-minute/2GiB/64MiB CPU supervision,
peak RSS1,558,679,552 bytes, output26,404,587 bytes. Independent checks
recomputed40 oracle curves, source margins and square-matrix singular
values for7,472 usable transactions. Development-exposed targets, one
training seed and unknown SiW participant dependence remain limitations.
