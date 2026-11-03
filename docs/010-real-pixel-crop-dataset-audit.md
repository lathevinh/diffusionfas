# 010 - Dataset coverage for a real-pixel-only CelebA-HQ-style square

## Decision from visual review

The user rejected reflection as a way to manufacture missing head/context
pixels: the intended camera can instead be positioned to capture the required
field of view. The reflected [preview 009](009-celebahq-geometry-visual-review.md)
is therefore historical evidence, **not an approved input policy**.
Its temporary review server has been stopped. No DDPM ablation was run.

All further candidate data must support the unchanged square on real source
pixels. Do not shrink, shift, stretch, reflect, zero-pad or generate missing
context merely to make a sample pass.

## Metadata-only audit

[Audit implementation](../src/diffusionfas/crop_fit_audit.py) and
[bounded runner](../scripts/crop_fit_audit.py) reuse the versioned square
geometry from 009. Each oriented square must have all four corners inside the
closed native pixel-center rectangle `[0,width-1] x [0,height-1]`.
Since the square is convex, this conservatively keeps its entire footprint
inside the real frame, including bilinear sampling support.

This is a geometry criterion for the chosen CelebA-HQ-style reference, **not**
a claim that the HF checkpoint has a hard crop specification or that passing
it guarantees good PAD performance. Exact weight-training provenance remains
limited as described in 009.

Scope:

- All 491 original source fitting/validation bonafides plus 160 source-validation
  attacks: **651 originals** across CASIA-FASD, MSU-MFSD and SiW-Mv2.
- **618** have previously prepared, hash-bound caches; **33** existing technical
  failures remain geometrically unassessed, not relabeled as crop failures.
- Inspect cached smoothed landmarks, native frame orders/timestamps and accepted
  receipt dimensions; verify each cache hash. Do not decode image arrays,
  video, calibration/target pixels or run any model.
- **Fitting primary:** every frame on the existing full-interval 8-FPS cache grid
  must fit, preserving unrestricted existing training-window sampling.
- **Validation primary:** all 16 frames of the already locked lower-middle
  two-second window must fit. Do not search for an easier validation window.
- Also count full-grid and any-valid-window coverage as diagnostics only.
  Neither alternative-window counts nor dataset selection authorize a run.

Plan SHA-256:
`07295e678961390f85e097ebfaff8291ade02e9bd5fe130dafa423e259289c69`.
The serial audit was capped at 600 seconds / 2 GiB worker-tree RSS / 64 MiB
new output. It completed in **17.619 seconds**, sampled peak RSS about
**448 MiB**, with zero GPU/model/video/image-array decoding.

## Results with original denominators

[Aggregate-only evidence](../results/crop-fit-audit-v1.json):

| Dataset | Fitting bonafide: full 8-FPS grid fits / originals | Validation bonafide: fixed T16 fits / originals | Validation attack: fixed T16 fits / originals |
|---|---:|---:|---:|
| CASIA-FASD | 23 / 36 | 7 / 12 | 18 / 36 |
| MSU-MFSD | 18 / 18 | 6 / 6 | 16 / 18 |
| SiW-Mv2 | 200 / 314 | 66 / 105 | 62 / 106 |
| **Total** | **241 / 368** | **79 / 123** | **96 / 160** |

Original denominators include unknown pre-existing failures. Among assessed
validation attacks MSU passes **16/17**, with one actual square-boundary failure;
the eighteenth original is an existing decode failure.

| Dataset/role/class | Assessed caches | Existing failures, geometry unknown | Full-grid fits | Fixed-middle fits | Any valid T16 exists |
|---|---:|---:|---:|---:|---:|
| CASIA fitting bonafide | 36 | 0 | 23 | 23 | 23 |
| CASIA validation bonafide | 12 | 0 | 7 | 7 | 7 |
| CASIA validation attack | 36 | 0 | 18 | 18 | 20 |
| MSU fitting bonafide | 18 | 0 | 18 | 18 | 18 |
| MSU validation bonafide | 6 | 0 | 6 | 6 | 6 |
| MSU validation attack | 17 | 1 | 16 | 16 | 17 |
| SiW fitting bonafide | 305 | 9 | 200 | 215 | 236 |
| SiW validation bonafide | 99 | 6 | 61 | 66 | 70 |
| SiW validation attack | 89 | 17 | 54 | 62 | 70 |

The gap between fixed-middle and any-window coverage is intentional. For
example, MSU's geometry-failing validation attack has another valid window;
using it would change the locked evaluation window and is **not authorized**.
Likewise, fitting's full-grid gate is stricter than accepting one convenient
window in an otherwise inadequately framed video.

## Interpretation and selection gate

**No complete dataset is certified as 100% suitable under these source-fold
denominators.** MSU is the strongest candidate for a small no-padding sanity
check: all 18 fitting bonafides and all six validation bonafides pass; 16 attacks
pass. This is a proposed **geometry-qualified subset**, not the complete MSU
benchmark. Eighteen fitting bonafides do not justify training a diffusion model
from scratch, and validation has only three declared subject groups.

CASIA and SiW contain usable subsets, but neither dataset can be accepted as a
whole. Keeping all verified candidates would mean 241 fitting bonafides and
175 validation originals, versus original denominators of 368 and 283.
No such selection has been performed by this audit.

The user must distinguish:

1. Requiring an entire dataset to pass: none currently qualifies.
2. Allowing explicitly geometry-qualified originals: MSU-first or a declared
   subset of the three source datasets is possible.
3. Collecting additional data with the intended camera framing.

Geometry filtering changes the population and may correlate with attack class,
capture condition, resolution, pose and subject. Preserve exclusion counts and
original IDs privately, disclose coverage by class/domain, and never present
filtered performance as the original full-dataset benchmark.
Any A/B evaluation must recompute the old baseline on the **same eligible
originals/windows**. Do not compare a newly filtered AUROC directly with the
previous 283-original diagnostic or its macro score.

Dataset selection, new crops, model scoring, training and target access remain
unapproved. The next actual image run still needs its own finite budget and
visual review of real-pixel-only crops.

## Validation and independent postflight

[Validation v14](../results/validation-v14.json): **105 tests passed**, including
eight new tests for real-pixel bounds, all-frame T16 windows, exact native
dimensions, fixed-middle versus alternate windows, unknown/invalid geometry
and role-preserving populations. Metadata-only tests deliberately use a cache
with an unreadable object-valued `frames` member and prohibit video/raster
calls, proving the auditor does not load image arrays.

Postflight verified all **618 cache hashes** and **33,324 cached-frame**
decisions independently: vectorized square extents instead of enumerated
corners, and prefix sums instead of convolution for T16 window counts.
All bounds/window counts agreed. There were no invalid-landmark frames.
Zero decoding/scoring counters and all original denominators were checked.
Only aggregate evidence is public; per-original eligibility remains private.
