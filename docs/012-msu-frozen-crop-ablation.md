# 012 - Matched frozen-DDPM crop A/B on eligible MSU validation originals

## Approval, fixed protocol and limits

The user accepted the real-pixel [preview 011](011-msu-real-pixel-crop-review.md)
and explicitly approved one bounded A/B attempt on **22 eligible MSU validation
originals: six bonafide and 16 attacks**. No fitting pixels, calibration, target
pixels, training or threshold fitting were authorized.

The two excluded MSU attacks remain excluded: one fails the unchanged square
in the locked middle window, one has a prior native decode failure. This is
a geometry-filtered development subset, **not the full MSU benchmark**.
All 18 eligible fitting bonafides remain metadata-only.

Fixed before scoring:

- The same lower-middle T16, native frame addresses and cached smoothed
  landmarks for both arms. No alternate-window search, detector rerun,
  augmentation or replacement originals.
- A: the original legacy crop, regenerated and required to match the old cache
  exactly. B: the visually accepted oriented square, with all source bounds and
  actual bilinear support checked; no padding, shifting, shrinking or stretching.
- One frozen local `google/ddpm-celebahq-256` backbone, no adapter. The local
  bundle and source/decoder lineage remain pinned. FP32, no AMP or TF32,
  backbone microbatch four frames.
- Alpha-bars `[0.9, 0.5, 0.1]`, two draws per level, seed `20261103`.
  The original key and frame-slot addresses are identical across arms, making
  each noise draw identical. The float64 mean of the three observed profile
  values is the score; **higher means attack**, without sign reversal.
- Recompute both arms, including the legacy baseline, for all 22 originals.
  No reused scores in the actual A/B computation.
- AUROC and paired bootstrap with 2,000 resamples of the three declared subject
  groups. These few groups make uncertainty estimates coarse and fragile.
- Shared cap: **1,800 seconds / 8 GiB worker-tree RSS / 8 GiB device-wide GPU
  memory / 1 GiB new output**. No automatic retry.

[Implementation](../src/diffusionfas/crop_ablation.py),
[bounded CLI](../scripts/crop_ablation.py),
[targeted tests](../tests/test_crop_ablation.py).
Default scientific CLI/config permissions remain unchanged and closed.

## Preserved preflight failure and approved new attempt

The first attempt stopped in the parent preflight, before launching the
supervisor/worker, staging media or scoring. JSON lists for timestamps and
frame hashes were compared directly with dataclass tuples, incorrectly
reporting a changed cohort. Canonical contents of all 22 entries were identical.

The fix compares normalized `Video` objects; a regression test accepts the
JSON round trip but still rejects genuine metadata changes. The user explicitly
approved a new attempt after this explanation. No silent retry occurred.

[Failed preflight evidence](../results/msu-crop-ablation-v1.json), plan SHA:
`da476df646c2bf94b20904a8071f053f61248829016e54f852552eed675d7d68`.

New completed plan SHA:
`d27ad239c51dd9078e4f2e819ee35e329796e1ab52ce1161648282a42d87319a`.
Both immutable private roots and their approval/failure evidence are retained.

## Results

[Aggregate and independent postflight](../results/msu-crop-ablation-v2.json):

| Crop | Usable bonafide / selected | Usable attacks / selected | AUROC |
|---|---:|---:|---:|
| Legacy | 6 / 6 | 16 / 16 | **0.687500** |
| Real-pixel square | 6 / 6 | 16 / 16 | **0.697917** |

- Paired difference, new minus legacy: **+0.010417 AUROC**, about **1.04
  percentage points**, equivalent to one net additional correctly ranked
  attack/bonafide pair out of 96.
- Paired subject-bootstrap 95% percentile interval for the difference:
  **[0.000000, +0.083333]**. All 2,000 replicates were usable; only **three
  declared subject groups** exist. Zero is included, and the interval must not
  be treated as strong evidence of improvement or generalization.
- Mean bonafide residual: **0.01042564 -> 0.00892337**, a descriptive **14.41%**
  reduction. Mean attack residual also falls: **0.01331269 -> 0.01188794**,
  **10.70%**. Better reconstruction/noise prediction does not by itself imply
  better spoof separation.
- **22/22 crop pairs and 44/44 profiles** completed, no technical failures
  or replacement examples. No thresholds, normalization or score tuning.

The old full diagnostic's MSU AUROC must not be used as the A arm: its usable
population differs. The matched legacy value above is the correct comparator.

### Resources and input integrity

- **134.266 seconds** wall time, within the 30-minute cap.
- Sampled peak worker-tree RSS **1,589,710,848 bytes**.
- Sampled device-wide GPU peak **6,491,734,016 bytes**, including other usage.
- Sampled owned-output peak **205,659,463 bytes**.
- Model peak allocated/reserved memory: **1,884,642,816 / 2,640,314,368 bytes**.
- Strict native decoder verified all **6,028 native RGB frames** across the
  selected originals. All **352 observed middle-window frames** were
  regenerated for both crops. Every legacy crop matched the pinned cache.
- All 352 new crops passed square bounds and bilinear support checks;
  minimum recorded square margin **9.872 source pixels**, padding zero.
- Frozen backbone hash before/after matches the previous pilot. No staged
  raw media remains; caches, identities and individual scores remain private.

## Validation and independent verification

**33 targeted synthetic crop tests passed.** They include exact cohort/split
guards, fixed T16 extraction, normalization equivalence with `ClipCache`,
actual reproducible matched noise draws, paired AUROC direction, genuine
metadata-drift rejection after JSON round trip, and the complete 22-original
worker with an injected failure retained in both arms without replacement.
Existing strict/historical preview and metadata-audit tests also pass.
No new full-suite or editor type-check result is claimed.

Independent CPU-only postflight verified all 22 cache-pair hashes, all 44
profile-to-score calculations, native frame identities/windows and receipt
bindings. Legacy input arrays equal the old cache slices exactly; all 22
newly recomputed legacy profiles equal their prior diagnostic profiles with
**maximum absolute difference zero**.
`sklearn` reproduces both AUROCs. A separate weighted pairwise-win bootstrap
implementation reproduces all 2,000-resample percentile intervals, rather
than calling the experiment's rank-based bootstrap again.

## Interpretation and next gate

The preprocessing objection was valid to test: the accepted square improves
bonafide residual fit, but **this small MSU experiment shows only a small
separation change**, not a decisive rescue of the raw-residual approach.
It does not prove that correctly preprocessed diffusion training cannot work;
no adapter was retrained on the new crops.

Keep the real-pixel-only policy. The next useful evidence would be the same
preregistered frozen A/B on geometry-qualified CASIA and SiW validation
originals, before longer training or a model-family pivot. Those are subsets,
not entire certified datasets, and retain the selection-bias caveats from
audit 010. Their decoding/scoring requires a separate protocol and budget;
this run does not authorize them. No further experiment is running.

The visually accepted preview server was stopped and its token state and
temporary script removed. The historical preview HTML remains private.
