# 013 - Native-512 MAT face inpainting, paused for visual review

## Model decision and provenance limits

The user approved a pretrained face-inpainting pilot, requiring model-compatible
crops and no synthetic context. The author-maintained
[MAT repository](https://github.com/fenglinglwb/MAT/tree/d273d891ecdad2e1df106516423a75bc45b2d800)
documents CelebA-HQ-256 and CelebA-HQ-512 models. Its linked official weight
folder returned HTTP 404 during preparation.

The user then explicitly approved the third-party
[MAT-CelebA-HQ-512 safetensors mirror](https://huggingface.co/not-lain/MAT-CelebA-HQ_512).
**Equivalence to the author's weights is unverified.** The mirror card lacks
conversion/training provenance; neither strict state loading nor a valid
content hash proves that provenance. This is a research-only exploratory
pilot, not an official-model reproduction or a deployment certification.

Pinned inputs:

- Official implementation revision:
  `d273d891ecdad2e1df106516423a75bc45b2d800`.
- Mirror revision: `b9dd007aea6d7c9e53818f2d11b071b1be5bf48c`.
- Weight SHA-256:
  `b47eb8ff60bc25131b531eab96cf8f8d4e85a86da688c7595ea50003df93d8bb`.
- Weight size: **253,303,072 bytes**, safetensors only; no pickle loading.
- Architecture: RGB 512, `z_dim=w_dim=512`, `c_dim=0`.
  The state dict loaded strictly into the pinned author's implementation.
- Upstream license is CC BY-NC 4.0 with research-only guidance. The mirror has
  limited licensing/provenance information. No upstream code or weights are
  redistributed here; all such assets remain private. This is not legal
  certification or permission for commercial use.

## Input compatibility, not just matching image dimensions

The official generator consumes already prepared RGB face images; it does not
perform face detection/alignment. Its dataset loader consumes square
CelebA-HQ images, while its inference script specifies normalization and
known-mask polarity.

[Input implementation](../src/diffusionfas/mat_input.py):

1. Recover original native RGB pixels with the pinned strict MSU decoder.
2. Reuse the unchanged CelebA-HQ-style eye/mouth-oriented square and smoothed
   native landmarks, not a tightly cropped/stretched FAS face rectangle.
3. Reject the square if it leaves the real frame; verify actual bilinear
   support at render resolution 1024 before RGB resampling.
4. Area-downsample directly to **512 x 512**. Do **not** upscale a 256 cache.
   The native square itself may have less than 512 pixels of detail;
   interpolation cannot create new detail.
5. Normalize RGB float32 using **`image / 127.5 - 1`**.
6. Supply a float32 binary mask, **1 = known, 0 = hole**, and explicitly zero
   the normalized image inside the hole before calling the model.

The geometry is shared with the accepted real-pixel reference, but this is
**not a claim of bit-exact MAT training preprocessing**. The official training
data construction and mirror provenance do not establish that equivalence.
The new HTML is a separate visual gate for this native-512 input and masks.
Historical 256 renderers and immutable experiment bindings remain unchanged.

Four separate ellipses are generated from transformed landmarks: left eye,
right eye, nose and mouth. Their axes follow the eye direction; radii are
fixed eye-distance multiples listed in the public policy. Masks cannot be
clipped or shifted to rescue an invalid region. They were not tuned against
attack scores. Observed hole areas are **0.8812% to 1.8997%** of the image:
these small anatomical holes may differ from MAT's large/random-hole training.

MAT's final output composites the known input outside the hole. Consequently
whole-image error would be diluted by copied pixels. The preview measures
**mean squared RGB error only inside the hole**, in normalized units. Heatmaps
are black outside the hole and use the same fixed color scale. The displayed
masked input uses gray 128; the actual missing normalized tensor is exactly
zero. No output blending/beautification is added by this wrapper.

## Scope, budgets and controlled randomness

Approved scope: the same six geometry-qualified MSU validation originals
from preview 011, three bonafide and three attacks; slots 0, 7 and 15 of each
fixed middle T16. **18 frames x 4 independent masks = 72 reconstructions**.
No new examples or alternate windows were selected.

Budget: **1,800 seconds / 8 GiB worker-tree RSS / 8 GiB device-wide GPU /
4 GiB new output**, public code/weights at most 2 GiB. CPU/GPU work is bounded
by the existing supervisor. It counts the private pilot and upstream source
roots conservatively. No package was installed, upgraded or removed.
Existing `fas` provides PyTorch 2.7.1, timm 1.0.30 and safetensors 0.5.3;
the MAT optional dependency is declared separately from the base pipeline.

The model stays frozen. One addressed latent draw per original/frame/mask,
`truncation_psi=1`, `noise_mode=const`. Importantly, the author's network still
uses `F.dropout(..., training=True)` inside `eval()`: the wrapper seeds and
restores PyTorch RNG as well as fixing the latent.

The current backend explicitly enables deterministic algorithms and cuDNN,
sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disables TF32/AMP and uses the author's
documented PyTorch reference bias/filter operators on CUDA. It does not
compile custom extensions or silently fall back after compilation failures.
These are intentional runtime choices, not architectural changes.

[Pilot implementation](../src/diffusionfas/mat_preview.py) and
[bounded CLI](../scripts/mat_preview.py).

For a separately approved new attempt with the same pinned inputs:

```bash
conda run --no-capture-output -n fas python scripts/mat_preview.py plan \
  --preview-root /private/msu-real-pixel-preview-v1 \
  --upstream-root /private/pinned-MAT-source \
  --private-root /private/new-mat-attempt

conda run --no-capture-output -n fas python scripts/mat_preview.py run \
  --plan /private/new-mat-attempt/plan.json
```

`--reuse-acquisition-root` on `plan` can bind already acquired, hash-verified
weights instead of downloading again. A new root does not itself grant
scientific authority or permit an automatic retry.

## Preserved initial failure

[First-attempt evidence](../results/mat-preview-v1.json), plan
`a6d909bf8ca07225ddb945f9864014b2176dab58a4951e5183ce97fb9164c57d`.

Weights downloaded and loaded strictly, but the synthetic replay check failed
before any PAD original was decoded: repeated outputs differed by at most
**8.702278e-6**. The backend had not enforced deterministic kernels. In
addition, an upstream nested `impl="cuda"` argument overrode a partial-function
default intended to select reference operators, causing failed JIT attempts
and upstream fallback warnings.

The fixes enforce the keyword at the wrapper boundary and deterministic
backend configuration. Exact replay/no-leakage checks were **not relaxed**.
Regression tests cover both conditions. After the explanation, the user
accepted the explicit continuation form. The new attempt reuses the already
SHA-verified weights; it does not download them again. Both private attempts
and the original failed evidence are retained; there was no automatic retry.

## Completed preview

[Second-attempt aggregate and postflight](../results/mat-preview-v2.json),
plan `39c1df3b2f7dfe3c2ef26dcd7562b6dc9e6400637a648ddca46fa1b904c7a87f`:

- **6/6 videos, 18/18 frames, 72/72 reconstructions**, no technical failures.
- **234** lossless 512-square PNGs embedded in the private HTML.
- **31.907 seconds** wall time.
- Sampled peak RSS **3,661,131,776 bytes**; device-wide GPU peak
  **5,515,509,760 bytes**; owned disk including source **131,144,648 bytes**.
- Model peak allocated/reserved memory:
  **1,348,788,224 / 1,656,750,080 bytes**.
- Three actual synthetic GPU forwards verify exact same-seed replay and exact
  invariance when the hidden ground truth is changed: both max errors **zero**.
- Frozen model hash before/after matches. No staged raw media remains.
- No CUDA-extension setup/fallback messages in the completed worker log.

Independent CPU-only postflight decoded/hash-checked all 234 PNGs and their
HTML embeddings, recomputed all 72 anatomical masks and mask hashes, checked
pixel preservation outside holes and residual-map support. It recomputed
square bounds from native landmark extents and checked all frame lineage:
minimum square margin **22.865 native pixels**. Worker verified the entire
**1,732-frame native RGB sequences**, not just the 18 selected frames.

The displayed MSE uses unquantized model outputs. Postflight does not claim
to reconstruct that exact float MSE from clipped/quantized PNGs; denominator
and support semantics are separately covered by synthetic tests.

**10 targeted MAT tests passed**: native-512 geometry and no 256-cache upscale,
out-of-frame rejection, mask polarity/area/location, normalization and hidden
content removal, masked-only MSE denominator, addressed RNG, operator override,
deterministic backend configuration, authority guards and bound HTML images.
No new full-suite or editor type-check result is claimed.

## Review gate and laptop access

**Stop here: human review is pending. No AUROC, training, threshold fitting,
fitting/calibration/target pixels or automatic continuation.**

Inspect whether the square resembles the expected aligned face framing,
whether each hole covers its intended structure, and whether reconstructions
show border seams, excessive smoothing or identity/expression changes.
Good-looking reconstructions alone do not demonstrate spoof detection.
Only after acceptance should a separate bounded scoring protocol be approved.

The user separately approved a token-protected `127.0.0.1:8765` server for
60 minutes. Forward port 8765 in VS Code and open the token URL on the laptop.
It serves only the exact immutable SHA-bound HTML, with no directory listing
or external resources. Token state stays private.

HTML SHA-256:
`e1e99a1df43c9946c21cd7f994a0480bddda279469ce8afe4a707000166d3073`.
The 69,813,567-byte HTML, images, individual errors and original IDs never
enter Git or any public inference service.
