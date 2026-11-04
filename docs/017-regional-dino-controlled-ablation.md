# Regional DINO: controlled implementation and execution protocol

Date: 2026-11-04.
Status: implementation validated on synthetic data; real scientific execution
requires exact-plan owner approval. No improvement result is claimed here.

## Hypothesis and controls

Global CLS/mean-patch pooling may dilute localized PAD evidence. Source-balanced
mean BCE may still favor source-specific cues. These are hypotheses, not
established explanations of the earlier results.

The implementation freezes one official DINOv2-Reg ViT-B/14 encoder and crosses
three representations with two linear-head objectives:

| Representation | Equal-source/equal-class ERM | GroupDRO |
|---|---|---|
| Existing CLS + mean image patches, 1,536 dimensions | Reference | Objective-only ablation |
| Global + left eye/right eye/nose/mouth means, 4,608 dimensions | Regional-only ablation | Primary candidate |
| Global + four spatial-control means, 4,608 dimensions | Capacity/location control | Capacity/location control |

This full factorial is **six arms**, not four. Four targets and three existing
seeds give **72 head fits and 72 monotone calibrator fits**, without fitting
the encoder. There is no mixture/region/seed selection on target results.

The spatial controls use fixed quarter-image anchors. Each control pools
exactly as many patch centers as its corresponding anatomical region, with
stable spatial-distance ties. Thus anatomical and control arms match feature
dimensions, head parameter counts, patch counts and optimizer settings.
They do not have identical semantic content or identical shapes.

Anatomical ellipses reuse the previously fixed eye-distance radii from the
[MAT input policy](../src/diffusionfas/mat_input.py), but do not mask pixels
or perform inpainting. ROI constants are not tuned on this study's scores.
Normalized image patch tokens exclude CLS and the four register tokens.

## Input and population contract

Reuse the exact accepted single earlier-middle-frame RGB518 crops, original
five landmarks, official source-role partitions and three seeds from the
completed FAS comparison. Transfer landmarks through the saved rotation,
integer crop and OpenCV half-pixel resize mapping. No videos, new detector,
crop correction, crop resizing fallback, download, or new dataset is used.

Historical rectangle-to-square resizing and the constant-zero rotation border
remain unchanged. This is not a new all-crop padding-free/native-camera-support
audit, and does not inherit that claim from the separate 22-clip diagnostic.

Metadata-only checking found five of the original 7,474 usable observations
with unsupported anatomical regions: four out-of-crop ellipses and one
empty patch-center region. The owner selected:

> Shared technical failure for all six arms, with original denominators retained.

Consequently:

- 7,505 active originals remain; **7,469 are usable in this new matched study**.
- The five added failures are two SiW-Mv2 source-train bona-fide originals and
  three SiW-Mv2 target-test attacks.
- Original detector failures remain failures. No fabricated score is assigned.
- The new usable cohort is not identical to paper324's 7,474 observations.
  Compare the six contemporary matched arms, not old and new AUROCs as if
  the usable populations were unchanged.
- Malformed/tampered metadata aborts; only explicitly identified regional
  support failures receive the owner-approved shared-failure disposition.

All four datasets have already informed development. SiW participant
identities remain unknown; its grouping is video/exact-frame based, not
verified subject independence. This is not untouched confirmation.

## Heads and evaluation

All arms use 200 full-batch FP64 AdamW epochs, LR0.001, weight decay0.05
excluding bias, seeded normal initialization and no feature standardization.
Checkpoint selection is the earliest strictly minimum source-validation
equal-source/equal-class BCE; no post-selection refit.

GroupDRO uses three source-domain losses, each equally averaging its two
class-conditional BCE means. A log-space exponentiated adversary update with
fixed step0.1 precedes each AdamW update. Initialization, validation criterion,
regularization and training opportunity otherwise match ERM. A zero-step
regression reproduces the original FAS ERM optimizer/selection numerically.

GroupDRO is established prior art:
[Sagawa et al., 2019/2020](https://arxiv.org/abs/1911.08731).
Regional pooling also has prior art. This implementation does not claim
algorithmic novelty or that this combination is publication-worthy without
robust results and a closer PAD-specific literature comparison.

Source calibration and threshold/metric/oracle primitives reuse the read-only
FAS implementation. Fit the calibrator only on source calibration observations.
Use the minimum per-source safe attack order statistic, strict acceptance
`score < threshold`, and conservative successful-calibration attack budgets.

Report all targets/seeds/arms at primary FA5% and sensitivity FA1%/10%:

- FA/BFA and integer acceptances on all originals; failures are nonacceptances.
- AUROC and usable-only errors on successful observations.
- Fixed-score oracle headroom, explicitly using evaluation labels and never
  used to choose the model, region, checkpoint, threshold, or hyperparameter.
- 2,000 paired grouped bootstrap replicates at the primary operating point,
  yielding 540 point/difference interval records across 12 contexts. Fixed
  contrasts include anatomical versus spatial controls under each objective,
  and DRO versus ERM within each regional representation, not just differences
  from the global baseline.
  Subjects are used where known; SiW uses video/exact-frame components.
  These condition on fitted heads/source thresholds, not future-domain risk.

The separate CPU verifier reconstructs saved-feature probabilities using
an independent dot-product reduction, checks exact acceptance decisions,
source thresholds, sklearn AUROC, all 216 operating cases, grouped-curve
oracle checks, and independently vectorized bootstrap intervals. Its
predeclared numerical tolerance is 1e-10; a decision mismatch always fails.
It does not repeat encoder extraction or independently replicate training.

## Execution boundary

One attempt only, no automatic retry, no automatic extension:

- Shared ceiling: **4 hours**, including input verification, extraction,
  72 heads/calibrators, evaluation and separate-process reconstruction.
- Process-tree RSS: **8 GiB**.
- Conservative total-device GPU use: **8 GiB**; pre-existing occupancy can
  block/stop the run without touching other users' processes.
- New owned output: **4 GiB**.
- Encoder forwards: **7,469 real images + 2 synthetic replay forwards**.
- Private outputs contain identities, features, scores and coefficients.
  Only aggregate-safe reporting may be published.

A plan does not authorize execution. The approval JSON must explicitly bind
the plan SHA256, scope and budget with `approved=true` and
`no_automatic_retry=true`. Existing diffusion/MAT/DINO approvals do not apply.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n fas \
  python scripts/regional_dino.py plan \
  --fas-root /path/to/read-only/fas \
  --private-root /path/to/new/private/regional-study

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n fas \
  python scripts/regional_dino.py run \
  --plan /path/to/private/regional-study/plan.json \
  --approval /path/to/private/exact-owner-approval.json
```

The run refuses changed code/runtime/inputs, an existing attempt, missing or
changed approval, or an occupied device above the cap. A worker stops if its
supervisor disappears. Failed supervision or failed reconstruction is not
reported as a completed scientific result.

## Validation and scientific decision

The focused suite currently passes **25 tests**, including 12 new tests:
pixel-center transforms, matched control support, global-prefix preservation,
register exclusion, GroupDRO updates, zero-step ERM parity, failure retention,
approval/tampering guards, and a complete synthetic 72-head output fixture
with independent reconstruction of 216 cases/540 intervals.
The synthetic fixture is not 72 trained PAD heads.

Pylance transport was unavailable; no editor type-check success is claimed.
The actual conda tests and CLI help succeeded. Source FAS remains read-only.

An additional actual local-weight CPU smoke test completed two synthetic
RGB518 forwards in 2.92 seconds, with exact CLS/patch replay, the expected
1,536/4,608 feature dimensions and zero trainable encoder parameters.
No PAD image was decoded. The smallest eligible ROIs contain only one eye/
nose patch or two mouth patches. These are contextual ViT tokens, not isolated
pixel-texture measurements; small-region and global-attention limitations
must remain explicit.

After an approved completed run, report every arm and seed. Distinguish:

1. Operational improvement under the source-derived policy.
2. Score/oracle improvement without a recovered operational policy.
3. Target-specific tradeoffs or no stable benefit.

No automatic parameter sweep or next model campaign follows a disappointing
result. No comparison to literature numbers with mismatched protocols, Q3
acceptance claim, or modern-baseline superiority is authorized by this probe.
