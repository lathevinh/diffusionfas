# Full-cohort paired crops: verified preparation, not new training

The owner continued from the sixteen-frame preview and explicitly selected
`approve_full_cohort_cpu_preparation`: CPU only, at most60 minutes,
4GiB whole-process-tree RSS and4GiB new output. Only original saved native
PNG frames were used. There was no new video decoding, detector, download,
GPU inference, calibration or model fitting.

The corrected preparation completed in861.875s with no guard failure.
Sampled whole-tree peak RSS was1,585,299,456 bytes; total output was
534,463,639 bytes. The square PNG payloads occupy510,993,491 bytes.
Aggregate evidence is in [the public receipt](../results/spatial-dino-fullcrop-preparation-v1.json).
Images, native frame paths, subject identities and per-original manifests
remain private and outside Git. The source FAS repository is unchanged.

## 1. Exact pixel and eligibility contract

[The preparation module](../src/diffusionfas/spatial_crop_preparation.py)
reuses [the preview renderer](../src/diffusionfas/spatial_crop_preview.py).
For each historically usable original it recomputes and checks the pinned
expanded-square metadata. The selected square keeps the original rotation
and center and extends the shorter dimension to the longer one. It is
rendered at native crop size, then bicubic-resized to224x224.

Both legacy and square interpolation footprints must be entirely inside
real native pixels, including the two-pixel cubic radius and1/32-pixel
OpenCV quantization margin. There is no padding, shifting, shrinking,
alternative-frame search or replacement of failed originals. The intervention
adds context along the shorter axis and changes resampling; it is **not a
pure stretch-only ablation**.

All6,231 accepted native PNGs passed payload hash, canonical native RGB
hash and shape checks. All6,231 regenerated legacy crops matched the
historical RGB224 pixels exactly. All6,231 square PNGs passed encode/decode
pixel roundtrip checks. Crop-specific landmarks were checked against both
pixel-center transforms, and12,462 positive dense/patch priors were computed
and validated. Priors are defined by the existing recipe; this stage did
not run or claim augmentation/PDA transport in a new training controller.

The common mask contains **6,231 usable /7,505 originals**. Shared failures
are31 historical preprocessing failures,2 historical spatial-landmark
failures and1,241 crop-support exclusions. The actual quantization guard
excluded zero additional pairs beyond the earlier strict metadata mask.
All exclusions remain in both crop systems' original denominators.

| Dataset, all original roles | Attack usable/original | Bona fide usable/original |
|---|---:|---:|
| OULU-NPU |3377/3960|909/990|
| CASIA-FASD |268/450|99/150|
| MSU-MFSD |188/210|69/70|
| SiW-Mv2 |697/890|624/785|

## 2. Unchanged source roles and held-out targets

Original fitting/validation/calibration/evaluation assignments were retained,
not repartitioned. For every fold, the surviving source pools still contain
three non-target datasets and both classes in every required domain/role
stratum. The existing subject/group isolation check was rerun on the common
pools. Original evaluation component identifiers are preserved.

| Held-out target | Fitting | Validation | Calibration | Common target/original |
|---|---:|---:|---:|---:|
| OULU-NPU |653|215|209|1566/1800|
| CASIA-FASD |2107|805|738|220/360|
| MSU-MFSD |2133|810|744|147/160|
| SiW-Mv2 |1695|669|613|501/623|

| Held-out target | Attack usable/original | Bona fide usable/original | System BFA ceiling |
|---|---:|---:|---:|
| OULU-NPU |1234/1440|332/360|92.222%|
| CASIA-FASD |160/270|60/90|66.667%|
| MSU-MFSD |108/120|39/40|97.500%|
| SiW-Mv2 |287/362|214/261|81.992%|

The last column assumes every usable bona fide target is accepted and every
excluded bona fide target remains a nonacceptance. It is a preprocessing
ceiling, not a measured classifier BFA or an ISO/iBeta pass criterion.
Future results must report both conditional common-cohort performance and
original-denominator system performance. Compare old and square crops on
the same mask; comparing a restricted square cohort directly with the old
full-source result table would confound the intervention.

## 3. Independent reconstruction

A separate verification checked **all27,483 bound file hashes**, every saved
square RGB pixel hash, all7,505 original mask records, all16 context/role joins,
all four target class-denominator tables and the exact24-job inventory.
It independently derived native-size rendering matrices from the saved
RGB224 matrices using the resize pixel-center convention, verified all6,231
native source quadrilaterals and footprint margins, and checked isotropic
matrix singular values. It decoded only the newly produced square PNGs,
not another set of native frames.

This verification completed in103.068s, sampled verifier RSS815,042,560
bytes. The successful preparation plus final verification took964.944s.
The immutable private preparation, summary, supervisor and verification
hashes are recorded in the public receipt.

53 focused tests passed, covering schema variants, canonical pixel
commitments, exact legacy parity, square PNG immutability, real-pixel support,
source/target isolation, class-stratum loss and the matched24-job inventory,
plus the existing crop/oracle/MAT regressions. Four changed Python files
were AST-parsed. Pylance could not locate these external-repository files;
no editor type-check success is claimed. No dependency was installed.

## 4. Preserved first attempt

The initial full-cohort attempt stopped after163 partial square PNGs because
an older native-asset schema omitted optional `RGB_sha256` metadata. It
did not complete and its immutable root, log and failure receipt remain.

The loader now always verifies the PNG payload and the canonical
`selected_frame` RGB hash/shape. Optional redundant RGB/shape fields are
also required to match when present. Missing optional fields do **not**
disable canonical pixel checks. A regression covers both schemas, rejects
contradictory optional metadata and rejects changed canonical native pixels.
The same common-cohort geometry was used in the separately approved corrected
root; no sample or pixel-support requirement was relaxed to force success.

The independent verification command initially had a missing `PYTHONPATH`,
then demanded a native render-matrix field absent from the older saved
metadata schema. Neither attempt produced a success receipt. The final
verification derives that matrix independently from the existing pinned
RGB224 transform instead of requiring an unrecorded field.

## 5. Next scientific stage remains unexecuted

The fixed proposal is4 held-out targets x2 crops x3 arms
(`reference`, `content`, `anatomical`) = **24 new DINO fits and24 new source
calibrators**. Both crops must train on this exact common source cohort from
the generic checkpoint. No historical full-source adapted checkpoint can
replace the12 matched legacy fits. The frozen inventory has248,400 maximum
optimizer steps at batch32/max200 epochs; this is not a training wall-time
or GPU measurement.

Still required before fitting:

Follow-up: the controller, augmentation/PDA tests, synthetic GPU probe and
exact execution seal have now completed; see
[the paired-crop controller readiness report](028-spatial-dino-paired-crop-controller.md).
The plan remains unapproved and unattempted. This does not change the
preparation-only authority or any preceding scientific results.

- Implement the two-crop controller and loaders with explicit crop-specific
  landmark/prior inputs; do not silently reuse the historical geometry setting.
- Preserve the source recipe, RGB/label augmentation and prior transport
  through augmentation/PDA; test matched sampling/RNG and checkpoint replay.
- Wire both denominator views, paired statistical comparisons and complete
  independent checkpoint/calibrator reconstruction.
- Run a separately bounded synthetic GPU resource probe, seal current code,
  inputs, cohort and execution budget, and obtain exact execution approval.

The private preparation is an immutable input/cohort commitment, **not an
executable sealed scientific plan**. Its scope does not authorize training.
No learned attention maps, extra seeds, permuted control fits, temporal model,
train-two/calibrate-third protocol or threshold redesign were added. Previously
observed development targets, one proposed seed and unresolved SiW participant
dependence remain limitations. The negative anatomical result and weak SiW
low-FA separation from the preceding study have not been overturned.
