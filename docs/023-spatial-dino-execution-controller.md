# Spatial DINO: bounded scientific execution and separate reconstruction

Date: 2026-11-05.
Status: **the approved attempt stopped at its total-device GPU guard; scientific
execution did not complete.** See [failure and recovery preparation](024-spatial-dino-recovery.md)
and the historical [start receipt](../results/spatial-dino-start-v1.json).
Preparation and no-start statements below preserve their pre-approval history.

This wires the [five-arm specification](022-spatial-dino-preparation.md).
The120h budget increase is not treated as an exact scientific execution
approval. No new method/accuracy claim is made by this implementation.

## 1. Fixed execution contract

- Four held-out targets, seed20260917, five arms: reference, uniform,
  content, anatomical, permuted.
-20 full-backbone fine-tunes and20 source calibrators; no extra seed,
  hyperparameter search, post-selection refit, retry or resume.
- Generic pretrained DINO initialization, original single-frame RGB224
  crops, source-only fitting/validation/calibration, unchanged augmentation,
  APL and threshold rules.
-200 epochs maximum, original20-epoch patience; every checkpoint history
  must prove complete patience or the full200 epochs. Partial runs cannot
  be relabeled successful.
-7,505 originals,7,472 usable,31 historical plus2 shared geometry failures.
  Failures remain nonacceptances for all arms, with original denominators.
-60 operating cases (four targets x five arms x three alpha values) and
 132 conditional intervals (five points plus six contrasts, three metrics,
  four targets);2,000 component bootstrap replicates each.
- Primary alpha5%, sensitivities1% and10%. Strict `score < threshold`,
  source attack-quantile thresholds and target-label oracle diagnostics.
- Source train/validation/calibration and target identities are checked
  before PAD crop access. Known subjects, groups and exact frame content
  determine bootstrap components; unknown SiW participant dependence and
  development-exposed targets remain disclosed.

The six contrasts are anatomy minus content/permuted/uniform/reference,
content minus uniform, and uniform minus reference. No best-target arm
selection or unnoticed change of threshold information budget is allowed.

## 2. Authority, resources and persistence

[Controller](../src/diffusionfas/spatial_study.py) /
[CLI](../scripts/spatial_dino.py):

1. `prepare` hashes/binds original inputs, metadata and synthetic resources.
2. `seal` checks the current code/runtime, inventory, crop/landmark joins and
   component map, then writes a new immutable private execution plan.
3. `check` revalidates without PAD decode.
4. `approve` requires the exact plan SHA and exact owner answer, and refuses
   an already approved/attempted root.
5. `run` requires that approval and creates one immutable attempt marker.
6. Only the recorded supervisor may start the worker; only that worker may
   start its verifier. Both stop if their recorded parent disappears.

The approval records the boot ID and monotonic clock at approval-command
entry. No reboot, second approval, retry, resume or phase change resets the
shared120h. Input checking, fitting, validation, calibration, statistical
analysis, checkpoint reconstruction and closure share it. Computation leaves
a60s closure reserve; supervision leaves30s for final receipts.

Whole-controller-tree RSS8GiB, conservative **total-device** GPU12GiB
(including other users), owned outputs16GiB. The existing supervisor samples
every0.2s, kills only its owned process group on failure and retains the log.
No other user's process is stopped. No downloads, installs, new detector,
native video replay or scientific pixel cache is introduced.

Crop bytes are verified before every decode; priors use a bounded256-entry
in-memory cache. All crops are existing single-middle-frame inputs.
No raw PAD arrays or per-video records are published in Git.

Each fit records its initial state fingerprint, addressed batches,
source-selected checkpoint, raw role logits, calibrator and policies.
Common batch prefixes must match byte-for-byte across all five arms, even
when their legitimate early-stopping lengths differ.

## 3. Verification is another process, not another fit

[Verifier](../src/diffusionfas/spatial_verify.py) reopens all20 checkpoints:

- Rebuilds the paired generic initialization and loads each saved state.
- Validates the full history, first minimum, patience, optimizer steps and
  training/calibrator start inventory.
- Re-decodes bound crops and recomputes validation, calibration and target
  logits. Checks the selected checkpoint's source-validation loss.
- Reconstructs positive-affine calibration independently with SciPy expit;
  absolute tolerance1e-10. Also requires identical strict operating
  decisions on replay, not merely close probabilities.
- Sorts calibration attack scores independently for thresholds; counts
  original-denominator FA/BFA and uses sklearn AUROC.
- Checks the diagnostic oracle against grouped empirical curves and
  reconstructs the bootstrap by the existing independent vectorized code.
- Requires all60 unique cases,132 unique intervals and20 unique checkpoint
  replays, with zero new fitting.

The neural replay uses the same implementation in a separate process;
it is not an independently coded DINO implementation or a reproduction of
optimization. The statistics use separate reconstruction logic. Verification
does require PAD crop inference; it is not misrepresented as a feature-only
or pixel-free scientific verifier.

Successful supervision alone is insufficient. The controller checks the
bound verifier result and writes private `aggregate.json`, `completion.json`
and `terminal.json`. Failed work retains evidence and cannot produce a
success-shaped fallback. Human-readable publication of the aggregate can
follow completion without exposing per-video scores.

## 4. Synthetic validation

55 targeted/regression tests passed in30.264s. Coverage includes:

- Complete synthetic four-target/five-arm CPU execution with20 tiny models,
  real fitting/selection/calibration/artifact code,60 operating cases and
 132 reconstructed intervals. The subprocess call is intercepted for this
  tiny-model test; it is not claimed to be a full separate-process PAD run.
- Corrupted accepted-attack counts are rejected by reconstruction.
- Mismatched plan/code/runtime/scope, missing exact approval, expired/rebooted
  clocks, partial histories, parent mismatches and retries fail closed.
- The existing crop/augmentation/gradient/serialization and DINO/clip
  regressions remain passing.

Separately, the real DINO synthetic GPU probe validates all five full
backbones and five actual independent-process checkpoint replays, using
generated images only. It has15 synthetic optimizer steps and606 synthetic
image forwards; it does not measure PAD accuracy.

Pylance transport remains unavailable; no editor/type diagnostics are claimed.
The seven changed/new Python files are additionally parsed with Python AST.

## 5. Execution commands after explicit owner approval

```bash
conda run --no-capture-output -n fas python scripts/spatial_dino.py seal \
  --fas-root /mnt/e/FAS --preparation /private/preparation/plan.json \
  --preparation-sha256 EXACT_PREPARATION_SHA --private-root /private/new-run

conda run --no-capture-output -n fas python scripts/spatial_dino.py check \
  --plan /private/new-run/plan.json --plan-sha256 EXACT_EXECUTION_SHA

conda run --no-capture-output -n fas python scripts/spatial_dino.py approve \
  --plan /private/new-run/plan.json --plan-sha256 EXACT_EXECUTION_SHA \
  --answer approve_exact_spatial_dino_20_fits_120h_no_retry

conda run --no-capture-output -n fas python scripts/spatial_dino.py run \
  --plan /private/new-run/plan.json --plan-sha256 EXACT_EXECUTION_SHA \
  --approval /private/new-run/approval.json
```

Seal/check are metadata preparation. The approve/run lines are documentation,
not authority. Show the real sealed SHA and exact scope to the owner before
executing them. On startup, verify both supervisor/worker PIDs and real log
progress; do not claim a running experiment from an attempted launch alone.

Resource measurements, current plan hash and start status are recorded in
the aggregate preparation/start receipts, not guessed from this template.

## 6. Sealed execution preparation

The [aggregate execution-preparation receipt](../results/spatial-dino-execution-preparation-v1.json)
binds14,976 files and the complete20-fit inventory. Current execution-plan
SHA256:

`866233a31917bae7d73e943a533a72916edb15383ad13bd1af675259fc2c896f`.

The final code-bound GPU probe took39.127s under supervision, with sampled
process-tree peak3.980GiB and conservative device peak9.632GiB. Its fresh
conservative maximum-epoch planning estimate is **81.517h**, below the
authorized120h ceiling. The difference from earlier76.6/77.6h estimates is
synthetic measurement/contention variation, not a changed scientific recipe.
There are no PAD crop decodes, fitted PAD models, owner-approval record or
attempt marker at sealing.

Both approval and actual launch remain separate explicit steps. The
five-arm/20-fit scope above, including the verifier's PAD crop replay, must
be shown with this exact SHA before starting.

## 7. Exact owner approval and verified start

The owner selected `approve_exact_plan` for execution SHA
`866233a31917bae7d73e943a533a72916edb15383ad13bd1af675259fc2c896f`.
Approval receipt SHA:
`9dc6c802efcb98fb9313fe4fd8c70d0978d53a08d7ed495e282a4ff58eea201b`.

The persistent supervised run was launched and independently checked:
supervisor441567 and worker441575 were alive, and the OULU-NPU reference fit
had written its first real completed optimizer step (epoch1, zero-based
step0). This is training progress, not merely an attempted process launch.

At launch the shared120h clock became active, with no reset between phases. No successful
completion, target result, novel-method benefit or seed expansion is claimed.
The private run root retains `batch.log` and per-fit `batches.jsonl`;
on completion it must contain the separate verifier and aggregate/terminal
receipts before any success report.

## 8. GPU-guard termination

The attempt terminated after6217.954s with
`conservative_device_gpu_memory_cap`: sampled total-device use was
13,139,705,856 bytes against12GiB. This is a supervisory termination,
not a recorded CUDA out-of-memory exception. The original terminal remains
`completed: false`; no independent final reconstruction or successful
scientific aggregate exists.

All five OULU-NPU fits and CASIA-FASD/reference completed training.
CASIA-FASD/uniform was interrupted after its epoch28 log entry and has no
saved completed checkpoint. The remaining targets were not started.
The [explicit recovery preparation](024-spatial-dino-recovery.md) retains
this failed evidence and requires a separate exact-plan approval. It does
not reset, resume or relabel the failed attempt.
