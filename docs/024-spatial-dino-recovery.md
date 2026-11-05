# Spatial DINO: GPU-guard failure and explicit recovery preparation

Date: 2026-11-05.
Status: **following the failed first recovery, the owner explicitly approved
a fresh recovery attempt after GPU headroom returned; v2 is running and
passed real inherited-checkpoint replay/calibration readiness.** See the
[v2 start receipt](../results/spatial-dino-recovery-start-v2.json).
Preparation and failed-v1 statements below preserve their historical context.

## 1. What failed, and what is not established

The [original approved attempt](023-spatial-dino-execution-controller.md)
ended after6217.9535s, with return code-15 and guard
`conservative_device_gpu_memory_cap`. Sampled total-device GPU use reached
13,139,705,856 bytes (12.237GiB), exceeding12GiB. Process-tree peak RSS
was3,791,966,208 bytes and retained output peak2,134,397,727 bytes.

This was a deliberate supervisor termination, not a recorded PyTorch CUDA
out-of-memory error. Batch size was32. Six fits completed with that same
batch size, and the worker already deletes each model, collects garbage
and empties the CUDA cache between fits. These facts do not prove that no
memory-growth or peak-allocation issue exists.

After termination, a device snapshot showed3,087MiB in use while the WSL
compute-process query returned no attributable entries. This is not a
measurement of the background load at the failure instant. The receipts
retain only sampled aggregate peaks, not a per-process time series;
external contention, display/system allocations and the worker's own peak
cannot be separated retrospectively. Do not claim a proven leak or blame
another user's process.

## 2. Fixed recovery scope

The owner selected **preparation** with unchanged batch32 and a14GiB
total-device ceiling, preferring reuse only when completed checkpoints
pass verification. This decision does not authorize scientific execution.

The candidate inherited inventory is fixed by completion, not target scores:

| Target | Completed fits eligible for reuse |
|---|---|
| OULU-NPU | reference, uniform, content, anatomical, permuted |
| CASIA-FASD | reference |

CASIA-FASD/uniform is not eligible. Its interrupted training cannot be
exactly resumed: the run has no saved optimizer/RNG recovery state. It and
the other13 unfinished fits must start from the original generic pretrained
DINO initialization. No best-checkpoint picking across multiple attempts,
extra seeds, changed crop, changed batch, shortened patience, mixed precision
or target-driven tuning is introduced.

The proposed final study contains:

-6 inherited completed backbone fits plus14 new backbone fits;
-20 newly fitted source calibrators, including for inherited models;
-20 separate-process final checkpoint replays, with no fitting in that verifier;
- the original60 operating cases and132 conditional intervals;
- the same four contexts, source-only splits, geometry failures, input bytes,
  soft priors, augmentation, optimizer, seed, thresholds and bootstrap.

Recalibration and evaluation of inherited checkpoints happen only after a
new exact execution approval. No target scores are used to choose this
inventory or the14GiB resource policy.

## 3. Preparation and fail-closed provenance

[Recovery preparation](../src/diffusionfas/spatial_recovery.py) accepts only
the known origin execution SHA
`866233a31917bae7d73e943a533a72916edb15383ad13bd1af675259fc2c896f`
and its exact original approval SHA. It verifies:

- the original GPU-guard-failed terminal/supervision binding and absence of
  completion/aggregate/verification claims;
- the unchanged historical scientific/model code, runtime, original
  preparation, source repository and all original file commitments;
- exactly six completed checkpoints, original training/calibrator start
  receipts, checkpoint/trace hashes, full source-selection histories,
  patience, finite tensors and optimizer-step counts;
- paired addressed-trace prefixes across all five inherited OULU-NPU arms;
- the interrupted CASIA-FASD/uniform inventory, explicitly excluded.

Only the controller, verifier, CLI and their tests may differ from the
origin code; the new recovery module/tests are explicit additions. The
model, augmentation, preprocessing and statistical implementation remain
byte-bound to the original plan. Preparation reads metadata/checkpoints
on CPU and hashes crop files; it performs **no PAD crop decode, model
inference, calibration or scientific training**.

At authorized execution, inherited checkpoint/trace files are copied
byte-for-byte into the new root and receive `reused.json`, never a new
`started.json` training receipt. The worker strictly loads their state and
recomputes the selected source-validation loss before recalibration.
The final verifier requires exactly14 new training starts,6 inherited
receipts,20 new calibration starts and20 checkpoint replays. It reconstructs
all original metrics and paired intervals and rejects changed provenance,
bytes, decisions or start inventories. CPU preparation alone is not a
claim that neural replay parity has passed.

## 4. Separate authority and resource envelope

Proposed new attempt:120h shared from its approval-command entry through
checking, copying, training, calibration, analysis, separate replay and
closure;8GiB whole-tree RSS;14GiB **total-device** GPU;16GiB owned outputs.
The old failed attempt's elapsed time remains disclosed separately.
Original logs, checkpoints, receipts and failure status are not overwritten.
The original full20-fit81.517h planning estimate is retained conservatively,
not a new measurement or guaranteed runtime for14 fits.

The GPU has16,376MiB total capacity.14GiB leaves2,040MiB nominal capacity
outside the guard. Increasing the guard is not a guarantee against future
contention or CUDA OOM; no other process may be stopped to make room.
The same bounded supervisor, closure reserves, parent checks and one-attempt
guards still apply. A failed recovery cannot silently retry or resume.

Metadata-only commands:

```bash
conda run --no-capture-output -n fas python scripts/spatial_dino.py recover-seal \
  --origin-plan /private/original-run/plan.json \
  --private-root /private/new-recovery
conda run --no-capture-output -n fas python scripts/spatial_dino.py check \
  --plan /private/new-recovery/plan.json --plan-sha256 EXACT_RECOVERY_SHA
```

The existing `approve`/`run` commands accept the recovery plan only with
the new exact answer
`approve_exact_spatial_dino_recovery_6_reused_14_new_20_calibrators_14GiB_120h_no_retry`.
The original20-fit approval answer is rejected for recovery. Show the real
sealed recovery SHA and its full scope to the owner before issuing approval
or launching; the preparation decision is not that approval.

## 5. Validation

59 related tests passed:28 spatial/controller/recovery tests in28.508s
and31 DINO/clip regressions in18.763s. Six changed Python files parsed.
No new dependency, environment, download or scientific launch was used.

The tiny CPU integration test first runs the original20-model synthetic
pipeline, then reuses six synthetic checkpoints and checks that exactly14
new fits execute. All20 synthetic models and calibrators are reconstructed
with60 cases and132 intervals. A spurious new training-start receipt for an
inherited fit is rejected. Its verifier subprocess is intercepted in the
test, not falsely described as a real separate-process GPU replay.

Unit tests also reject partial histories, nonfinite tensors, checkpoint
drift, scientific-code/input changes and use of the original approval or
12GiB budget for the recovery plan. Prior full-DINO independent-process
synthetic replay evidence is historical and does not substitute for the
pending scientific replay of inherited checkpoints.

The sanitized [recovery preparation receipt](../results/spatial-dino-recovery-preparation-v1.json)
records the sealed metadata outcome. No anatomical-performance or successful
scientific-completion claim follows from recovery preparation.

Recovery execution-plan SHA:
`41e5162ff9b44cc60de20509806291b228d4890751a3f96025d6599a6e9626db`.
Metadata preparation receipt SHA:
`0f72dd990e3913d612bd9857775ee3b9b19c60669ad702d0d9fd127f70eafea9`.
The plan binds15,009 files. All six real inherited checkpoints passed the
CPU checks above; GPU source-validation and final reconstruction remain
pending a new exact owner approval.

## 6. Exact approval and unsuccessful recovery launch

The owner selected `approve_exact_recovery_plan` for the SHA above.
Approval receipt SHA:
`b50489a8afcb1636c6c428a3546c1ad0144246bb6f7beae6a0715ca35d8c3f4c`.
The detached supervised command was started, but did not pass readiness.

After100.7069s of supervision the total-device GPU guard terminated the
worker at15,358,492,672 bytes (14,647MiB), above14GiB (14,336MiB).
The log was empty; there were zero new training starts, inherited-fit
receipts or calibration starts. The code writes progress/start markers
before entering any model fit/import; this termination occurred during
the preceding bound-input checks. Batch32 was not exercised in this attempt.
It is not a recorded CUDA OOM.

Afterwards supervisor530742 and worker530750 were both absent. A device
snapshot still showed14,917MiB used, only1,129MiB free. That post-termination
occupancy is outside the terminated recovery, although it does not identify
the responsible process or reconstruct the failure-instant allocation.
The GPU was shared and its occupancy changed: the pre-approval snapshot
had shown5,528MiB used. No other process was stopped.

All six original completed checkpoints remain intact. Neither attempt is
successful, and no new training or statistical result is claimed.
Do not silently retry or raise the cap again. A future separately approved
attempt needs adequate GPU headroom rather than another blind launch under
the same occupied-device conditions.

## 7. Renewed owner approval and verified v2 launch

After the GPU watcher observed at least12GiB free for three minutes, the
owner explicitly instructed `ok chạy đi`. A fresh immutable recovery-v2 root
was sealed and approved; both failed roots remain unchanged. Identical
plan content retains SHA
`41e5162ff9b44cc60de20509806291b228d4890751a3f96025d6599a6e9626db`;
the new approval SHA is
`9009237e449edb7c827cf6712ce524090ef4ea70b2225fe337a6c7e0a94c27b0`.
No model code, recipe, reuse inventory or resource cap was changed.

The pre-seal device snapshot showed13,730MiB free. After persistent launch,
supervisor535532 and worker535560 were verified alive with the correct
parent relation. The worker completed real OULU-NPU/reference inherited
checkpoint loading, source-validation replay and new source calibration,
logging `SPATIAL_REUSE_COMPLETE OULU-NPU reference 44`. This is a readiness
snapshot, not a final result. All20 final checkpoint replays and complete
60-case/132-interval reconstruction remain required.
