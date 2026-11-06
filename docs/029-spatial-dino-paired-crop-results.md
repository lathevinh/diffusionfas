# Completed paired-crop study: mixed preprocessing result, not robust PAD improvement

The exactly approved24-fit run completed successfully. Its immutable plan
SHA is `2df70a86325cf2f06baf20709773d42177d39708721981d4c218d0e08ce2fe86`.
Completion/terminal receipts bind the final aggregate, supervision and
independent reconstruction. [The sanitized completed evidence](../results/spatial-crop-completed-v1.json)
contains both reporting views, actual fit inventory, primary target metrics,
selected declared paired intervals and final receipt hashes. Private
images, original identities, individual scores and checkpoints remain outside Git.

**Main finding:** square cropping improves macro AUROC for the CLS reference,
but reduces it for content and anatomical pooling. Anatomical pooling does
not establish a stable advantage. Low-FA source-threshold transfer remains
poor, particularly on SiW-Mv2. These data do not establish an overall robust
PAD improvement, a publication-ready novel method or an ISO/iBeta pass.

## 1. Verified execution and scope

- Four held-out targets x two crops x three arms:24 new full-DINO fits,
  zero historical adapted-checkpoint reuse and24 new source calibrators.
- Same generic initialization, seed20260917, source role assignments,
  optimizer/loss/augmentation/PDA recipe and common6,231/7,505 mask.
- Explicit crop-specific RGB/landmark/prior inputs and checkpoint
  target/crop/cohort/role commitments; no legacy fallback for square inputs.
-24 checkpoints replayed in the separate production verification process,
  with selected source-validation losses recomputed. Maximum absolute
  saved-probability replay error was **0.0**.
-144 exact operating cases and360 interval records; all360 intervals had
  2,000 valid bootstrap repetitions. Verification performed no new fits.
-54,228 actual optimizer steps and1,040 completed fit epochs in total;
  the sealed maximum-epoch inventory was248,400 steps.

Supervisor wall time was24,395.898s, approximately **6h47m**. This includes
worker input checks, fitting, analysis, checkpoint replay and closure; approval
metadata checks also consume the shared boot-bound approval clock.
Total-device GPU peak was10,078,912,512 bytes, whole-tree RSS peak
6,355,369,984 bytes and new output8,542,305,514 bytes. No guard failed.
There was no retry, resume, new video decode, new detector or new download.

The model implementation is shared between training and replay; verification
is not an independently authored neural implementation. Counts, thresholds,
calibrated probabilities and bootstrap reconstruction use the existing
independent statistical helpers. Calibration coefficients are reconstructed
into probabilities, not independently re-optimized by a second solver.

## 2. Primary macro results

Percentages below use source-selected5% policy thresholds. Each macro is an
equal average over four targets. AUROC is on common usable originals and
is therefore the same point value in both views. FA/BFA use their respective
original or common denominators.

| System | AUROC usable | FA original | BFA original | FA common | BFA common |
|---|---:|---:|---:|---:|---:|
| legacy reference |92.3779|24.2566|71.2524|29.5409|82.0970|
| legacy content |95.0979|20.1749|77.1384|24.8801|90.5470|
| legacy anatomical |94.6872|20.6942|75.1509|25.5238|87.8641|
| square reference |94.9745|23.6390|68.2495|29.1409|79.1236|
| square content |93.2325|20.2277|66.6092|24.9404|76.2454|
| square anatomical |94.2807|21.5872|68.9775|26.4589|78.8695|

Square-minus-legacy macro AUROC differences are **+2.5966pp reference,
-1.8654pp content and-0.4066pp anatomical**. These are point differences,
not cross-seed confidence intervals. Best macro AUROC is legacy content,
95.0979%, but its original-denominator FA is20.1749%, not5%.

The paired legacy systems here were retrained on the restricted common
cohort. Do not compare these crop effects directly against the earlier
full-source20-fit table: changing source/target support changes the experiment.
Square cropping also adds context and changes sampling geometry; this does
not isolate stretching from background/context effects.

## 3. Primary per-target system points

Percentages, **original denominators**, same source-only5% policy. The internal
joint goal is FA<=5% and BFA>=50%; it is not an ISO/iBeta criterion.

| Held-out target | System | AUROC usable | FA | BFA | Joint goal |
|---|---|---:|---:|---:|---|
| OULU |legacy reference|92.5785|23.4722|86.9444|no|
| OULU |legacy content|97.2174|10.7639|87.2222|no|
| OULU |legacy anatomical|96.3543|12.5694|86.9444|no|
| OULU |square reference|97.3485|24.6528|91.6667|no|
| OULU |square content|97.0839|12.3611|89.7222|no|
| OULU |square anatomical|97.0922|23.3333|91.3889|no|
| CASIA |legacy reference|99.8646|0.0000|32.2222|no|
| CASIA |legacy content|100.0000|0.0000|52.2222|yes|
| CASIA |legacy anatomical|100.0000|0.0000|46.6667|no|
| CASIA |square reference|99.5625|0.0000|32.2222|no|
| CASIA |square content|99.0938|0.0000|22.2222|no|
| CASIA |square anatomical|99.9688|0.0000|24.4444|no|
| MSU |legacy reference|94.4444|13.3333|85.0000|no|
| MSU |legacy content|96.9373|8.3333|87.5000|no|
| MSU |legacy anatomical|96.6999|7.5000|85.0000|no|
| MSU |square reference|96.0589|2.5000|67.5000|yes|
| MSU |square content|93.3761|7.5000|72.5000|no|
| MSU |square anatomical|96.1776|5.8333|80.0000|no|
| SiW |legacy reference|82.6240|60.2210|80.8429|no|
| SiW |legacy content|86.2369|61.6022|81.6092|no|
| SiW |legacy anatomical|85.6947|62.7072|81.9923|no|
| SiW |square reference|86.9281|67.4033|81.6092|no|
| SiW |square content|83.3762|61.0497|81.9923|no|
| SiW |square anatomical|83.8842|57.1823|80.0766|no|

Only CASIA/legacy content and MSU/square reference meet the original-view
joint goal. Square reference has a useful local operating result on MSU,
not a four-domain success. No system meets the joint goal on OULU or SiW.
At1%, no system meets the original-view joint goal on any target.

The1,274 shared failures remain nonacceptances in both crop systems.
For excluded bona fide records this lowers original BFA; it also lowers
original FA relative to conditional FA. Neither reduction can be hidden
as a classifier improvement. CASIA's original BFA ceiling is66.667%, SiW's
81.992%. Conditional metrics are available in the public receipt, separately
labelled; they do not replace original-system reporting.

## 4. Paired uncertainty supports a mixed effect

Declared square-minus-legacy **AUROC difference**95% intervals, percentage
points, original-view component bootstrap:

| Target | Reference | Content | Anatomical |
|---|---|---|---|
| OULU |[3.7271,5.7000]|[-0.7288,0.6607]|[-0.0351,1.4605]|
| CASIA |[-0.9296,0.1626]|[-1.9111,-0.2666]|[-0.1274,0.0000]|
| MSU |[-0.4673,4.4757]|[-5.6536,-1.6203]|[-2.2174,1.4440]|
| SiW |[1.5917,7.1749]|[-5.0587,-0.7554]|[-3.6220,-0.0754]|

Reference gains are conditionally supported on OULU and SiW. Content
degrades on CASIA, MSU and SiW; anatomical degrades on SiW. Thus the crop
change is not universally beneficial to this architecture.

Within square crops, anatomical-minus-content AUROC intervals are positive
on CASIA[0.2666,1.8250] and MSU[0.9743,4.8050], but include zero on
OULU[-0.6957,0.7162] and SiW[-1.1174,2.2270]. This local advantage over
degraded square content does not establish anatomical superiority over
the matched legacy content baseline or across domains.

Intervals condition on one training seed and the chosen component structure,
not seed variance or a new-dataset population. These multiple comparisons
are unadjusted; they should not be presented as universal guarantees.
There is no declared cross-seed or macro-domain confidence interval.

## 5. Threshold transfer and low-FA separation remain distinct

The saved target-label oracle at5% is diagnostic, not a deployable threshold.
For SiW, its **maximum original BFA** is:

| System | Oracle max BFA at5% |
|---|---:|
| legacy reference |32.567%|
| legacy content |40.613%|
| legacy anatomical |40.230%|
| square reference |34.866%|
| square content |31.801%|
| square anatomical |34.866%|

All are below50%, even after optimizing a threshold with target labels.
Thus SiW's problem is not only source threshold transfer: the required
low-FA separation is inadequate under this system/cohort.

OULU oracle BFA ranges54.444-80.833% and MSU67.500-85.000%, so several
systems have target-label empirical headroom there despite unsafe transferred
thresholds. CASIA oracles mostly reach the66.667% preprocessing ceiling,
while some source-selected policies reject too many bona fide samples.
This motivates separating calibration-transfer diagnosis from feature
separation, not fitting target thresholds to claim generalization.

## 6. Interpretation and next-decision boundaries

- Retain legacy content as the strongest matched macro-AUROC comparator,
  and square reference as a useful preprocessing control/local MSU result.
- Do not claim "aspect preservation improves PAD generally" or
  "anatomical prior improves domain generalization" from these results.
- No attention/learned-weight maps were computed. Geometry overlays and
  a positive interval on two domains do not establish anatomical causality.
- Do not add calibration, temporal modeling, new masks or extra seeds to
  this completed seal. Any follow-up requires a new bounded protocol/approval.

Targets were already development-exposed, SiW participant independence
remains unresolved and the protocol substitutes SiW-Mv2 for Replay-Attack;
it is not standard MCIO. One seed and conditional intervals do not establish
external benchmark superiority. AUROC near95% together with these unsafe
FA values is not evidence of ISO30107-3 or iBeta Level1 compliance.
