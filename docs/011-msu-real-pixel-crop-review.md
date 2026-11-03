# 011 - MSU real-pixel-only preview, paused for visual review

## Approval and scope

Following [audit 010](010-real-pixel-crop-dataset-audit.md), the user approved
an MSU-first continuation and then explicitly approved **preview only**:
six validation videos, three bonafide and three attacks, three frames each;
CPU only, at most 20 minutes / 8 GiB worker-tree RSS / 512 MiB new output.
Stop after the HTML is generated. This does not authorize DDPM scoring,
training, fitting pixels, calibration or target access.

The private plan locks the geometry-qualified MSU population:
18 fitting bonafides, six validation bonafides and 16 validation attacks.
Fitting eligibility requires the full cached grid; validation retains the
previously fixed lower-middle T16. Two attacks remain excluded: one lacks
the required context in that fixed window, one has a prior decode failure.
An alternative passing window is not a rescue.

The six preview originals are seeded random samples, three per class, drawn
only from eligible validation originals. Slots remain 0, 7 and 15 of the
fixed T16, with unchanged cached smoothed landmarks and native frame addresses.
The existing seed is a fixed sampling identifier, not an execution date.
Population membership, exclusions, original IDs and images stay private.
No fitting pixels were decoded.

## Strict renderer and review

[The new renderer](../src/diffusionfas/celebahq_geometry.py) uses the unchanged
eye/mouth-oriented square. It rejects any negative corner margin against
`[0,width-1] x [0,height-1]` before rasterizing. It also checks the actual
OpenCV bilinear support mask at 1024 resolution: every destination sample
must have support exactly one before RGB sampling. The constant-border
sentinel is used only to detect invalid support; no accepted output uses
padding. Area downsampling produces the final 256-square image.
There is no reflection, shifting, shrinking, stretching, learned restoration
or synthetic context in the new crop.

The old reflected renderer remains available solely to reproduce historical
preview 009; `--real-pixel-audit-root` selects the distinct strict renderer
and binds the completed audit and its per-original results. The historical
default is not an approved policy for future scientific runs.

Each HTML row contains an annotated original thumbnail, the legacy input and
the new square. Legacy images must match the pinned old cache exactly.
Original pixels are recovered with the existing strict MSU FFmpeg helper;
every native RGB frame in each selected original is hash-verified.
There is no detector rerun or substitution after a failure.

## Results

[Public aggregate only](../results/msu-real-pixel-preview-v1.json):

- **6/6 videos, 18/18 sampled frames, 54 embedded PNGs**.
- **16.237 seconds**, sampled peak worker-tree RSS **500,838,400 bytes**,
  sampled owned output peak **87,941,575 bytes**, all within approval.
- Worker verified **1,732 full native RGB frames**; all 18 legacy crop
  comparisons matched the old cache exactly.
- All 18 new crops have **zero padding**, independently checked minimum
  square margin **22.865 native pixels**.
- Independent postflight verified all 54 image hashes, decoded every PNG,
  checked 256-square model inputs and embedded-HTML membership, recomputed
  oriented-square extents and checked original-frame lineage.
- No staged raw media remains. No model was loaded or scored.

Plan SHA-256:
`bc4014dd7caeb8baa49420c27ae6091f8e562a4417ad3333646c8f6248bc8e41`.
HTML SHA-256:
`febacc6f020ca004abf056d7976716fcc6d2b1b97c8c720e2036b7aff0896743`.
The 14,898,835-byte self-contained HTML and its images remain outside Git.

## Validation and next gate

**25 targeted synthetic tests passed**, covering preview and metadata audit.
New tests check in-bounds equivalence with the reference renderer, rotated
geometry, exact closed bounds, negative margins without epsilon, actual
interpolation support rejection, deterministic MSU selection, fixed-window
exclusions and the complete six-video strict-preview worker.
Historical preview behavior and failure visibility remain tested.
Pylance was unavailable (MCP transport initialization failed); these are
runtime/synthetic-test facts, not an editor type-check claim.

For laptop review over SSH, the user separately authorized a token-protected
server on `127.0.0.1:8765`, expiring automatically after 60 minutes. Only the
exact token path may serve the SHA-bound HTML; no directory listing, arbitrary
files or external image requests. Forward port 8765 through VS Code and use
the laptop browser. Token and server state must never enter Git.

**The user subsequently accepted these crops.** The temporary server has been
stopped and its token state removed. [A separately approved matched frozen
A/B](012-msu-frozen-crop-ablation.md) has completed; automatic continuation
remains disabled. The original preview completion receipt retains its historical
pending state rather than being rewritten.
A separate bounded frozen-DDPM A/B protocol must use
the same 22 eligible validation originals, fixed windows, score direction and
noise draws for both crops. Recompute the legacy baseline on that population;
do not compare directly with the old differently populated MSU AUROC.
The small, geometry-filtered subset is not the full MSU benchmark and cannot
establish generalization. No additional training or threshold fitting is
authorized by this preview.
