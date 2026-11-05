# Fine-tuned DINO with soft anatomical patch weights

Date: 2026-11-05.
Status: implementation and synthetic/metadata preparation completed;
**scientific execution blocked, not a PAD result or execution approval**.

## 1. Hypothesis and five arms

Does a soft anatomical prior help a learned patch selector transfer spoof
evidence better than content-only selection, ordinary patch averaging and
an equally distributed but spatially permuted prior?

| Arm | Global prediction |
|---|---|
| reference | Original paper329 CLS classifier |
| uniform | Original CLS classifier plus linear mean-patch classifier |
| content | CLS plus learned weighted-patch classifier, no prior |
| anatomical: primary | Same classifier/gate with soft anatomical prior |
| permuted | Same classifier/gate with the anatomical prior's256 values permuted |

Primary attribution contrasts: anatomical minus content and anatomical minus
permuted. Anatomical minus reference/uniform is necessary context, not enough
by itself to attribute a gain to anatomy. Content versus uniform tests learned
selection. Reference is freshly retrained, not a historical score table.

This is not the previous frozen518 regional study. Every arm fine-tunes the
same generic pretrained DINOv2-Reg B/14 backbone, using the existing single
middle-frame RGB224 crop, original source fitting/validation/calibration
partitions and four held-out targets. No PAD-fine-tuned checkpoint is used as
initialization. One initial seed20260917; five arms x four targets =20 fits
and20 source calibrators proposed. SiW-Mv2 still replaces Replay-Attack.
All targets are already development-exposed, not an untouched confirmation.

## 2. Exactly what changes

Keep the original global head, patch head, FAS-Aug/PDA and detached final
CLS-attention APL. Add no new loss or patch annotation. Existing APL retains
its original inherited/synthetic patch-label assumptions; the new weighted
branch receives image-level supervision through the global classification
loss. Patch tokens are contextual transformer features, not independent
local image measurements.

For patch token `h_i`, the content selector is
`u_i = 2*tanh(Linear32_to1(tanh(Linear768_to32(h_i))))`.
The anatomical arm adds `log(p_i)`; content adds nothing.
Weights are `w_i = 0.25/256 + 0.75*softmax(u)_i`.
The floor preserves global support; it does not prevent every possible
concentration pattern or prove shortcut avoidance.

The local score is `Linear768_to2(sum_i w_i*h_i)`, added to the unchanged CLS
logits. This is additive linear fusion, not a new deep classifier.
The local head and final gate layer are zero-initialized. All five arms
therefore start with exactly the original reference logits; the four local
arms have identical initial tensor states. Uniform's unused gate is frozen.
After the local head moves, gradients can train the content gate.

### Soft geometry without another detector or model

Transfer the saved five landmarks through the existing rotation/crop/resize
using pixel-center conventions. At224 pixels, create:

`p(x,y) = 1 + mean_r exp(-0.5*((dx_r/rx_r)^2 + (dy_r/ry_r)^2))`.

The four anchors are left eye, right eye, nose and mouth center. Fixed
eye-distance radii are reused from the existing anatomical policy:
eyes(.25,.18), nose(.23,.30), mouth(.42,.23).
These are researcher-fixed hypotheses, not established spoof-localization
truth. Cheeks/forehead are not separately estimated from five landmarks.

The dense field has positive support everywhere; Gaussians may be truncated
at crop edges. There is no full-ellipse requirement, no new crop and no
silent uniform fallback. Invalid/degenerate geometry is a shared failure for
all five arms, including reference; corrupt metadata is a hard stop.
Original31 failures stay nonacceptances. Original denominators remain in
FA/BFA; usable-only AUROC is labeled accordingly.

Flip and rotation use exactly the same sampled transform as RGB, with
bilinear field interpolation and neutral field fill1. Brightness and
nongeometric FAS-Aug do not move the prior. PDA replaces donor prior cells
alongside donor RGB cells, using the actual augmented donor slot rather than
only the donor video ID. Average each14x14 prior block, then normalize by
its per-image mean.

Only after augmentation/PDA does the control permute these256 values.
One fixed seeded nonidentity permutation is shared across datasets, roles,
images and seeds in this initial study. It preserves the exact prior-value
histogram, but not anatomical locations or spatial smoothness. Thus this
control alone cannot distinguish anatomy from every possible structured
spatial bias. No "anatomy proven" conclusion is permitted from it alone.

## 3. Everything else stays fixed

- Original global CE + detached-attention APL, no hardest-attack term.
- FP32, no AMP/TF32; AdamW backbone LR5e-6, head/gate LR5e-5,
  matrix decay.05, bias decay0, batch32.
- Maximum200 epochs;20-epoch source-validation patience. Earliest strictly
  best equal-source/equal-class validation BCE, no post-selection refit.
- Identical addressed RGB/FAS-Aug/PDA batches at common epoch/step addresses.
  Different arm early-stopping lengths are not silently equalized.
- Original positive-affine source calibrator and minimum-source attack
  quantile policy; strict acceptance below threshold, ties rejected;
 5% primary FA budget and1%/10% sensitivities.
- No held-out-source calibration experiment simultaneously; all three
  source datasets remain training domains, with their disjoint role splits.
- No target-based threshold, prior radius, gate constant or checkpoint
  selection. Target-label low-FA oracle remains diagnostic only.

The sampler has to mirror the pinned source implementation because that
read-only implementation exposes no geometry callback. Tests enforce
byte-for-byte RGB, labels, donor/mask and original-trace parity, including
forced PDA and FAS-Aug. This limited duplication is deliberate; no source
repository changes are made.

## 4. Preparation boundary and commands

Implementation:
[model, sampler and fit/predict helpers](../src/diffusionfas/spatial_dino.py),
[metadata/budget/synthetic probe](../src/diffusionfas/spatial_preparation.py),
[CLI](../scripts/spatial_dino.py),
[synthetic tests](../tests/test_spatial_dino.py).

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src XFORMERS_DISABLED=1 \
  conda run --no-capture-output -n fas python -m unittest discover \
  -s tests -p test_spatial_dino.py -v

conda run --no-capture-output -n fas python scripts/spatial_dino.py smoke \
  --fas-root /mnt/e/FAS --private-root /path/to/new-private-synthetic-probe

conda run --no-capture-output -n fas python scripts/spatial_dino.py prepare \
  --fas-root /mnt/e/FAS --private-root /path/to/new-private-preparation \
  --probe /path/to/new-private-synthetic-probe/probe.json
```

The synthetic probe uses six generated images, three steps/arm
(one warmup plus two timed), total15 synthetic optimizer steps. Every arm
checks initialization parity, finite full-backbone gradients, same-process
and separate-process checkpoint replay. Temporary synthetic checkpoints
are removed. No PAD accuracy, calibration or target evaluation is performed.

Hard probe caps:600s,8GiB whole-controller-tree RSS,12GiB conservative
total-device GPU memory,2GiB owned outputs. It uses existing weights/assets
only, no download/install/environment changes or automatic retry.

Metadata preparation verifies existing record/crop hashes and landmark
support, but does not decode PAD crops. It binds code, runtime, generic
pretrained weights, source assets, partitions and synthetic evidence.
Individual records/paths/landmarks remain private; the public receipt is
aggregate only.

**There is intentionally no `run` or `approve` command in this preparation.**
Generic fit/predict helpers are implemented and tested, but a full scientific
execution controller, exact owner approval, nonresettable clocks,
independent scientific reconstruction and paired statistical analysis must
be wired and sealed before starting the20 real fits. This is not yet a
ready-to-approve scientific execution plan.

## 5. Budget and go/stop

Proposed future ceiling, not authorized:72h shared wall,8GiB process-tree
RSS,12GiB conservative total-device GPU use,16GiB owned outputs; no retry or
seed expansion. Use maximum200-epoch inventories, not historical selected
epochs, for cost estimation.

The preparation estimate multiplies each arm's slowest measured synthetic
step by its full optimizer-step inventory, includes full source validation
each epoch, then applies3x and adds6h for other IO/calibration/verification/
analysis/closure. This is a planning estimate, not a proven upper bound.
Missing full-population measurements and contention remain explicit.
If it exceeds the proposed cap, do not silently shorten the recipe.

Report every target and arm. Consider expansion only if the anatomical
candidate improves over content and permuted controls, improves low-FA
utility rather than merely AUROC/heatmaps, and does not create serious
cross-target regressions. Use the proposed operational gates in the
[audit](021-benchmark-novelty-and-next-decision.md), paired component-aware
uncertainty and then repeat seeds; a one-seed pilot does not establish
robustness or paper novelty. A visual attention map is not causal evidence.

## 6. Completed engineering evidence

The [aggregate receipt](../results/spatial-dino-preparation-v1.json) records:

- 47 DINO tests passed, including16 new spatial tests; four Python files pass
  AST parsing. Pylance transport initialization was unavailable; editor/type
  diagnostics are not claimed.
- All five full-backbone synthetic arms start with exactly the original
  reference logits. Same-process and independent-process replay are exact
  for all five.15 full-backbone synthetic optimizer steps,606 synthetic image
  forwards, no PAD fitting/calibration/evaluation or retained checkpoints.
- Synthetic supervised wall39.763s; sampled whole-controller-tree peak
  RSS3.984GiB, conservative total-device GPU9.632GiB and owned disk0.323GiB.
  No guard failure, download or scientific start.
- Metadata/hashing215.854s,14,975 file bindings.7,505 originals,7,472 usable:
 31 historical failures plus2 additional SiW invalid-landmark shared failures.
  No PAD image decode;7,474 existing crop files were read only for hashing.

| Target | Source fit | Source validation | Source calibration | Target usable | Max steps/arm |
|---|---:|---:|---:|---:|---:|
| OULU-NPU |836|277|282|1800|5400|
| CASIA-FASD |2491|949|864|354|15600|
| MSU-MFSD |2566|974|888|156|16200|
| SiW-Mv2 |2012|791|702|618|12600|

Maximum total inventory:249,000 optimizer steps. Synthetic extrapolation
gives21.834h training plus2.046h validation before the safety factor.
The fixed3x factor plus6h reserve gives **77.639h**, which **exceeds72h**.
The estimate includes about6.454GiB of checkpoint payload before traces and
other outputs. Neither historical early stopping nor the short raw benchmark
is used to pretend that the conservative planning gate passed.

Private preparation SHA256:
`61c8c126c15b3a2130318141b0c86a1ee68eaa118997ff6ce044bc97ec100ae0`.

**Stop here:** do not launch20 real fits, silently shorten200 epochs, raise
the ceiling or create owner approval. Before a scientific run, explicitly
resolve the budget and finish/seal its approval-gated execution,
reconstruction and analysis controller. Model/sampler/training preparation
is implemented; scientific execution readiness is not claimed.
