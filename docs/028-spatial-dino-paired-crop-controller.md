# Paired crop controller: sealed and verified, no PAD training started

The owner continued implementation and explicitly selected
`approve_synthetic_probe_and_metadata_seal_only`. This authorized a
10-minute/8GiB RSS/14GiB total-device GPU/2GiB output engineering probe,
then a30-minute metadata seal/check, **not scientific fitting or calibration**.

The controller,86 focused tests, actual generic-DINO synthetic GPU probe,
full input verification and execution seal completed. The sealed plan SHA is
`2df70a86325cf2f06baf20709773d42177d39708721981d4c218d0e08ce2fe86`.
No approval, attempt, worker-start or training log exists for this plan.
The source FAS repository is unchanged; private crop/subject records remain
outside Git. [The aggregate receipt](../results/spatial-crop-controller-preparation-v1.json)
contains sanitized hashes, scope and measurements.

## 1. Fixed24-fit experiment, not reuse of12 old checkpoints

[The crop contract](../src/diffusionfas/spatial_crop_study.py) extends the
existing [shared controller](../src/diffusionfas/spatial_study.py), fine-tuning
engine and [checkpoint/statistical verifier](../src/diffusionfas/spatial_verify.py).
It does not introduce a second independently implemented training recipe.

Four held-out targets x two crops x three arms = **24 new full-DINO fits**,
24 source calibrators, one unchanged seed20260917:

- `legacy_reference`, `legacy_content`, `legacy_anatomical`;
- `square_reference`, `square_content`, `square_anatomical`.

All use the already verified6,231/7,505 common-original mask and unchanged
source roles. All initialize from the generic DINO checkpoint. Historical
full-source adapted checkpoints cannot replace any legacy fit. The24-job
inventory has248,400 maximum optimizer steps at batch32/max200 epochs.
Source equal-domain/equal-class validation, earliest strict best, patience20,
restored best checkpoint, FP32 and original optimization/losses remain unchanged.

The crop loader requires an explicit system. Square RGB and priors cannot
fall back to legacy paths or coordinates. It verifies payload bytes before
decode and square decoded RGB hashes; retained failures reject pixel/prior
requests. Each checkpoint commits the actual model arm, crop, target,
preparation SHA, common-mask SHA and role-key SHA. Replay rejects a
wrong-crop checkpoint even if its tensor shapes and enclosing file hashes
are otherwise valid. The engine recipe hash is shared; the explicit
input commitment distinguishes preprocessing and cohort.

## 2. Matched augmentation and PDA

Within each crop, all three arms must have exactly equal addressed
batch/geometry traces on their common optimizer-step prefixes. Across crops,
the controller compares selected originals, RNG-derived flips/angles/
brightness, augmentation families, redraws, labels/domain/patch masks,
PDA donor identities and donor slots. Only RGB batch and spatial-prior
hashes are allowed to differ.

The original RGB/label sampler and losses are reused. Crop-specific dense
priors follow the same flip/rotation and actual PDA donor-pixel transport.
Early stopping may give different trace lengths, so pairing uses common
prefixes while verifying each full optimizer-step count. No source split,
calibration strategy, temporal architecture or threshold policy was changed.

## 3. Two denominator views, one set of scores/calibrators

Each model is calibrated once using its source calibration pool. The same
source attack-quantile thresholds and target predictions are used for:

1. `original`: all original target records, including shared nonacceptances;
2. `common`: only common usable target records.

There is no extra calibration or fit for the second view. At1%,5%,10%,
the inventory is4 targets x6 systems x3 alpha x2 views = **144 unique cases**.
Source thresholds use strict score<threshold. Target-label oracle values
remain diagnostic and cannot select checkpoints or deployed thresholds.

Primary5% bootstrap reporting includes6 system points and9 paired contrasts
per target/view: three within each crop (anatomical-content,
anatomical-reference, content-reference), plus square-legacy for each arm.
Three metrics, four targets and two views give **360 interval records**.
Original subject/group/content components are retained; conditional views
restrict their membership to usable originals. Each view resamples its
declared components. Thus even where usable AUROC points match across
views, bootstrap intervals need not be identical.

The separate production replay process reconstructs all24 checkpoints,
selected source-validation losses, affine calibrated probabilities,
strict source order-statistic thresholds, policy/oracle counts and bootstrap
records. It verifies144 case keys,360 interval keys,24 training starts,
24 calibrator starts and zero reused-fit receipts. Calibration coefficients
are used to independently reconstruct probabilities; verification does
not claim independent re-optimization of the calibrator solver.

## 4. Executed validation, no PAD-result claim

86 focused tests passed in39.241s, covering the new contract/probe gates and
all directly related DINO/controller/recovery/crop/MAT regressions. The new
CPU integration test actually trains24 tiny synthetic backbones, restores
them into fresh models and reconstructs144 cases/360 intervals using the
independent statistical helpers. It checks36 macro records, both original
and conditional denominators, no reused fits and crop-binding tamper rejection.
The synthetic test intercepts subprocess launching; it is not24 real-DINO
or PAD fits. Existing20-fit and6-inherited/14-new recovery tests still pass.

Six changed Python files were AST-parsed. Pylance cannot locate the external
destination repository files, so no editor type-check success is claimed.
No package was installed or environment changed.

The GPU resource probe uses the existing fixed five-arm engineering engine,
a superset of reference/content/anatomical. It runs15 optimizer steps on
synthetic RGB/prior arrays, with exact same-process and actual separate-process
checkpoint replay for all five variants. This measures shared tensor shapes
and backbone/head costs, **not paired-crop PAD inference, real PNG IO,
full epochs or scientific accuracy**. Temporary synthetic checkpoints were
removed by the existing probe.

The probe completed in41.322s, total-device GPU peak9,453,961,216 bytes,
whole-tree RSS peak4,282,355,712 bytes and peak transient output346,516,572
bytes; no guard failed. The supervisor's device cap is14GiB; the shared
synthetic worker/replay retain their12GiB per-process CUDA allocator limit.
Measured reference/content/anatomical steps were approximately0.307-0.333s.

## 5. Budget and exact approval gate

The sealed proposed execution caps are:

| Resource | Cap |
|---|---:|
| Shared wall clock, approval through verification/closure |120 hours|
| Whole process-tree RSS |8GiB|
| Conservative total-device GPU memory |14GiB|
| New execution output |16GiB|

Maximum-epoch synthetic training extrapolation is81,717.583s; validation
extrapolation is7,325.769s. The existing3x factor plus6h reserve yields
288,730.057s, approximately**80.203 hours**, below the120h cap.
Estimated24-checkpoint bytes are8,315,497,352; a2GiB trace/analysis reserve
also fits the16GiB output cap.

This is a planning estimate, **not a guaranteed upper bound or measured
24-model training duration**. Real crop IO, full validation batch efficiency,
thermal/device contention and full statistical closure are not measured
by the short synthetic probe. Supervisory guards retain failed receipts;
there is no retry, resume, clock reset, automatic expansion or stopping
another user's GPU process.

Sealing and a separate `check` verified the fixed private preparation/
summary/verification receipts, all27,488 file commitments, current code/
runtime, source roles, class strata, real-pixel footprints, transformed
landmarks,24 jobs, successful bounded probe and budget estimate.
The private check receipt records zero PAD decodes, fitting/calibrator starts
and approval/attempt creation.

The CLI is [the shared spatial command](../scripts/spatial_dino.py):

```bash
conda run --no-capture-output -n fas python scripts/spatial_dino.py crop-smoke \
  --fas-root /path/to/read-only/fas --private-root /path/to/new/private/probe

conda run --no-capture-output -n fas python scripts/spatial_dino.py crop-seal \
  --preparation /path/to/private/fullcrop/data/preparation.json \
  --verification /path/to/private/fullcrop/verification.json \
  --probe /path/to/private/probe/probe.json --private-root /path/to/new/private/run
```

The existing `check`, `approve`, `run`, worker and verifier routes recognize
this sealed crop study. Approval requires the exact plan SHA and
`approve_exact_spatial_crop_24_fits_120h_no_retry`; the old20-fit/recovery
answer or general "continue" instruction cannot approve it. The original
nonresettable boot-bound approval clock covers input checks through closure.

**Scientific training remains unapproved and unattempted.** The earlier
negative anatomical result, weak SiW low-FA separation, development-exposed
target data, one seed, nonstandard SiW-Mv2 protocol and unresolved SiW
participant dependence remain. No accuracy improvement, novelty, SOTA,
ISO/iBeta pass, learned attention-map interpretation or publication readiness
is established by these engineering tests.
