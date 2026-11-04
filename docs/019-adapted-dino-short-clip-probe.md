# Adapted DINO short-clip probe: controlled implementation

Date: 2026-11-04.
Status: **owner approved the exact plan; persistent supervised scientific
execution started.** The first real native replay, tracked crop and48
adapted-encoder image forwards were observed. This is not a completed comparison
or target-performance result. The preparation/zero-start statements below are
the preserved pre-approval history, not live status.

The owner requested a two-second clip experiment after the
[negative regional study](018-regional-dino-completed-results.md), and explicitly
selected tracked/shared-failure preprocessing and a separately trained shuffled
order control. This is a new development pilot, not an automatic continuation
of either the regional experiment or the original full-backbone adaptation.

## 1. Question and fixed comparison

Does chronological information improve detection beyond simply observing more
images, on a backbone already adapted to source FAS data?

| Arm | Input and head | Trainable parameters |
|---|---|---:|
| Single | Slot8 of the same accepted16-frame clip, linear classifier |769|
| Mean | Mean of16 frozen CLS features, linear classifier |769|
| Temporal: primary candidate | Mean CLS plus a residual-sequence projection/GRU state |31,745|
| Shuffled: matched capacity control | Identical temporal architecture and initial parameters, shuffled inputs in every role |31,745|

The temporal branch projects768-dimensional, clip-mean-centered CLS features
to32 dimensions, applies tanh and a32-hidden-unit GRU, then concatenates the final
state with the original mean CLS feature before classification. No backbone
gradient, patch reconstruction, additional attention architecture or new loss
is introduced.

The mean baseline is the primary reference; the separately trained shuffled
branch controls temporal-head capacity. A temporal-minus-mean gain alone cannot
establish that chronology helped.

One additional **diagnostic**, not a fifth training arm, feeds shuffled target
features through the chronological model using its unchanged calibrator and
source threshold. A diagnostic degradation alone is not proof of useful
temporal reasoning: shuffled input can also be a distribution shift.

## 2. Source-only checkpoint and data contracts

Use the four completed paper329 **reference**, not reference-plus-tail,
checkpoints, seed20260917. Each was fitted and selected using its own three
source datasets, excluding its named target. The preparation binds the original
completion, independent verification, operating records and checkpoint digests.
This does not claim an exact reproduction of the external paper on which the
original reference was based.

Retain the original official partitions and source fitting/validation/
calibration separation. Source test partitions cannot enter fitting. All four
targets have already informed development; **none is an untouched confirmation
set**. SiW participant identities remain unknown.

The four adapted encoders remain frozen. Each video's features are computed
only by the encoders whose declared source or target role includes that video.
No features from another held-out fold may be substituted.

The new single-frame baseline is rebuilt within this experiment. It is not
the historical paper329 single-middle-frame prediction: the new frame sampling,
track smoothing and clip eligibility differ. Do not compare old/new headline
AUROC as if these were a matched frozen-versus-adapted experiment.

## 3. Native video sampling and crop compatibility

For each previously usable original:

1. Read the bound native receipt. Define the native interval using its first
   timestamp and last timestamp plus the median inter-frame gap.
2. Select the middle two-second window,8 uniformly spaced bin centers/second.
   Choose the nearest native frame, breaking ties towards the earlier frame.
   Require16 distinct frames, each within half a sampling period.
3. Stage exactly the accepted archive member, at most128MiB, and verify its
   byte count and SHA256. No unpack-all, alternate release or decoder fallback.
4. Replay the complete canonical video with the existing pinned strict FFmpeg
   helper. Verify every native RGB frame hash and count, not just selected
   frames. Canonical decode/hash disagreement aborts the attempt.
5. Detect the selected16 frames with the existing pinned SCRFD model.
   Reuse the existing interpolation/jump/scale checks and centered three-point
   geometry smoothing. Missing track endpoints or gaps longer than two sampled
   slots are technical failures.
6. Reuse the original `shared_context_crop`: five-point rotation around the eye
   midpoint,1.3 rectangular context and bicubic resize toRGB224.
   Apply the same ImageNet normalization as the adapted reference.

This preserves the **method** of the backbone's crop contract, not the exact old
single-frame pixels. Constant-zero rotation borders and rectangle-to-square
distortion remain possible. This is not a padding-free MAT/CelebA crop.
There is no claim of reliable pulse estimation from two seconds.

All historical31 failures remain nonacceptances. Any additional metadata,
tracking or supported-crop failure applies to **all four arms and the
diagnostic**. Keep the original population in FA/BFA denominators; compute AUROC
only on usable clips. Do not invent scores or features for failures.

Before any pixel access, audit historical and all newly selected native RGB
hashes for role/domain collisions. A new cross-boundary collision blocks the
plan instead of silently reallocating or dropping data. Bootstrap groups
connect known subjects/groups and any selected exact-frame overlap; SiW
grouping is still not participant-disjoint certification.

## 4. Shuffle, fitting and operation

Shuffle permutations depend only on the fixed seed, fold target and original
key, never on labels, predictions or batching. Every original gets a fixed
nonidentity permutation, used consistently in fitting, validation, calibration
and evaluation for the shuffled arm. An identity draw is deterministically
rotated by one slot. The chronological and shuffled heads start identically.

All heads use200 full-batch FP64 CPU AdamW epochs, LR0.001, weight decay0.05
for matrix parameters and zero decay for biases. The loss is equally weighted
over source domains and classes. Select the earliest strictly best source
validation BCE; no target selection, shortened success, early-stop change or
post-selection refit.

Fit one positive-affine source-only calibrator per arm/fold. Retain the exact
historical source attack order-statistic threshold with strict acceptance
`score < threshold`; ties are rejected. Primary FA bound5%;1% and10% are
declared sensitivities. Target-label oracles diagnose stored scores only.

There are16 head fits and16 calibrators across four targets at one seed:
60 operating cases including the inference-only diagnostic, and132 primary
conditional interval records with2,000 paired group-bootstrap draws each.
Fixed contrasts are mean-minus-single, temporal-minus-mean,
shuffled-minus-mean, temporal-minus-shuffled, temporal-minus-single and
inference-shuffle-minus-temporal.

An independent CPU process replays saved heads on saved features, checks
initialization and source checkpoint selection, independently reconstructs
source thresholds, original-denominator counts, sklearn AUROC, oracle curves
and vectorized bootstrap intervals. This is **saved-feature numerical
reconstruction**, not a second training or full-image encoder extraction.

## 5. Engineering validation and execution boundary

The focused53-test run passed, including:

- Actual original FAS crop helper on synthetic RGB frames.
- Unique timestamp sampling and batch-independent shuffle addressing.
- Matched initial parameters, real Torch gradients and saved-head replay.
- The complete synthetic four-fold producer:16 heads/calibrators,60 cases
  and132 intervals, independently reconstructed.
- Existing regional numerical reconstruction and source-staging regressions.
- Actual pinned native FFmpeg replay on a synthetic video, and whole-controller
  process-tree RSS accounting.

The complete synthetic fixture reduces epochs to two in its isolated test
process; it is not evidence that200 PAD epochs ran. No PAD encoder forward,
new PAD decode or scientific fitting is authorized by those tests.
The Pylance endpoint failed initialization; no editor type-check success is
claimed.

A bounded real-checkpoint **CPU engineering smoke** also passed: four local
adapted checkpoints, two generated-RGB224 forwards each, exact CLS replay,
shape1x768, zero trainable encoder parameters and zero normalization discrepancy
against the adapted reference's tensor contract. Whole supervised wall14.02s,
sampled whole-tree RSS1,454,895,104 bytes; no PAD pixels, new fit or GPU use.
Nine new/touched Python files parsed successfully. The earlier metadata-only
preparation stopped before writing a plan because the installed ONNX Runtime
module is not registered under the `onnxruntime` distribution name. Recording
the actual imported module version fixed this without installing anything;
this was not a scientific attempt.

The [sanitized preparation evidence](../results/clip-dino-preparation-v1.json)
binds the final exact-plan SHA256:

`7bc62b991dee8f8394dcebed3c44909505a215f80349c9dd7c7c2788f5739238`

Final metadata preparation took152.636s. Independent read-only revalidation
passed. There are14,964 bound input files and13 archive identities.

| Inventory | Maximum before additional tracking failures |
|---|---:|
| Original videos, all roles |7,505|
| Metadata-eligible videos |7,376|
| Full canonical native frames to replay |1,147,357|
| Selected-frame detector calls |118,016|
| Production adapted-encoder image forwards |261,600|
| Synthetic production replay forwards |8|
| New heads / source calibrators |16 /16|

The129 metadata failures consist of31 historical failures,97 videos shorter
than two seconds and one native sampling-tolerance failure. All remain in
original denominators in every arm.

| Held-out target | Original target videos | Metadata-eligible target clips | Maximum fold encoder frames, all roles |
|---|---:|---:|---:|
| OULU-NPU |1,800|1,800|50,224|
| CASIA-FASD |360|353|73,600|
| MSU-MFSD |160|156|72,432|
| SiW-Mv2 |623|580|65,344|
| Total |2,943|2,889|261,600|

The conservative feature/staging/audit estimate, including a512MiB operational
reserve, is1,354,219,520 bytes (about1.26GiB). The separate execution decision
must bind this final plan SHA256, rather than this document alone.

Proposed shared ceilings:24h nonresettable wall time including input checking,
separate reconstruction and final closure;8GiB controller/worker/verifier-tree
RSS;8GiB conservative total-device GPU use
(including other jobs);4GiB new owned outputs. These are **caps, not predicted
completion times**. One attempt, no retry, automatic extension, new downloads,
environment changes, additional seeds or automatic follow-on campaign.

Only features, geometry and audit/model records are retained, not an18GiB
image-clip cache. Each staged original is deleted after successful processing;
partial failure evidence remains private and counted against the cap.
All FAS source/checkpoints/data stay read-only.

## 6. Readout and stop decision

Retain every target and all controls, including failures and unfavorable
results. Report both AUROC and actual source-policy FA/BFA, not just a
target-label oracle or a favorable dataset.

If chronology fails to improve over averaging and the matched shuffled
control, stop this construction. If apparent gains require unsafe attack
acceptance, do not call them equal-security improvements. Even favorable
one-seed results only justify a new proposal for repeat seeds and independent
confirmation; they are not proof of novelty, competitiveness or Q3 suitability.
No expansion is executable automatically.

CLI: `scripts/clip_dino.py --help`. Private plan/approval/attempt/model/score
records stay outside Git. Only sanitized preparation and completed aggregate
summaries may be published.

## 7. Exact owner approval and verified scientific start

After publication of implementation commit `7a1acd2`, the owner selected
`approve_exact_plan` for the final SHA256 and all declared ceilings.
The [aggregate-safe start receipt](../results/clip-dino-start-v1.json) binds
that decision and the private immutable start/device receipts.

The supervisor runs persistently, independent of the chat turn. Readiness
verification observed both recorded processes alive and actual progress:
one real canonical video replay, one tracked16-frame RGB224 clip, and48 image
forwards across the three eligible fold-specific encoders. No head-training
completion, final metric or improvement is claimed from this observation.
Only the approved one-seed inventory can execute; no additional attempt or
seed is authorized. Subsequent progress is in the private worker log.
