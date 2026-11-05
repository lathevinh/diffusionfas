# Spatial DINO: completed five-arm study, negative anatomical result

The [approved recovery](024-spatial-dino-recovery.md) completed successfully.
All6 inherited checkpoints and14 new fits were processed,20 new source
calibrators fitted, and all20 checkpoints replayed in the separate verifier.
It reconstructed60 operating cases and132 conditional intervals, recomputed
selected source-validation losses and obtained maximum probability replay
error0.0. No fit was performed by the verifier.
The [sanitized receipt](../results/spatial-dino-completed-v1.json) binds the
private aggregate, result, verification and supervision hashes.

**The experiment completed, but the primary anatomical hypothesis was not
supported.** Do not confuse engineering completion with a successful method.

## 1. Matched primary results

All numbers below are percentages, rounded to four decimals.
AUROC uses usable originals. FA/BFA retain the original denominators,
including shared technical failures as nonacceptances. The primary policy
selects a strict threshold from source calibration at alpha5%;5% is a
source constraint, not a guarantee of5% target FA.

| Arm | Macro AUROC | Macro FA | Macro BFA | Targets FA <=5% | Targets also BFA >=50% |
|---|---:|---:|---:|---:|---:|
| reference |94.3353|18.6431|85.5268|1/4|1/4|
| uniform |93.4967|25.9944|80.4071|1/4|0/4|
| content |95.0090|22.0731|86.4224|1/4|1/4|
| anatomical |93.8635|24.9967|90.3113|1/4|1/4|
| permuted |95.1190|21.0483|89.8851|1/4|1/4|

Macro is an equal-target descriptive mean, not pooled AUROC. There is no
macro significance claim. Anatomical minus reference AUROC is about
-0.4718 percentage points; anatomical minus content is-1.1455 points and
minus permuted is-1.2555 points. Higher anatomical BFA accompanies higher
FA and must not be presented as an equal-security improvement.

| Target held out | Arm | AUROC | FA | BFA |
|---|---|---:|---:|---:|
| OULU-NPU |reference|95.4691|9.3056|82.7778|
| OULU-NPU |uniform|93.8247|11.5972|81.9444|
| OULU-NPU |content|96.7498|6.7361|87.2222|
| OULU-NPU |anatomical|96.4468|13.8889|92.7778|
| OULU-NPU |permuted|95.7238|7.2917|84.1667|
| CASIA-FASD |reference|99.8855|0.0000|77.7778|
| CASIA-FASD |uniform|99.1690|0.0000|43.3333|
| CASIA-FASD |content|99.6947|0.0000|70.0000|
| CASIA-FASD |anatomical|99.8516|0.3704|90.0000|
| CASIA-FASD |permuted|99.8558|0.0000|85.5556|
| MSU-MFSD |reference|94.6983|13.3333|85.0000|
| MSU-MFSD |uniform|95.4957|20.8333|97.5000|
| MSU-MFSD |content|95.6466|11.6667|90.0000|
| MSU-MFSD |anatomical|92.9095|16.6667|80.0000|
| MSU-MFSD |permuted|95.8621|19.1667|92.5000|
| SiW-Mv2 |reference|87.2884|51.9337|96.5517|
| SiW-Mv2 |uniform|85.4974|71.5470|98.8506|
| SiW-Mv2 |content|87.9448|69.8895|98.4674|
| SiW-Mv2 |anatomical|86.2462|69.0608|98.4674|
| SiW-Mv2 |permuted|89.0342|57.7348|97.3180|

Target names identify held-out datasets: the other three datasets supply
fitting, validation and calibration. No target-label model/threshold
selection was introduced. CASIA alone satisfies the5% FA condition for
each arm; uniform's CASIA BFA is below50%.

## 2. What the attribution controls show

Paired component-bootstrap AUROC difference95% intervals below are in
percentage points, conditional on this seed and data. There are2,000
replicates per interval. They are not multiplicity-adjusted tests or
training-seed confidence intervals.

| Target | Anatomical minus content | Anatomical minus permuted |
|---|---:|---:|
| OULU-NPU |[-0.6995,+0.1736]|[-0.0886,+1.5012]|
| CASIA-FASD |[-0.0377,+0.4219]|[-0.1700,+0.1887]|
| MSU-MFSD |[-5.1496,-0.7826]|[-5.3632,-0.4873]|
| SiW-Mv2 |[-2.9514,-0.4342]|[-4.1699,-1.4888]|

Anatomical improves OULU AUROC over reference by about0.978 points
(conditional95% interval[+0.5556,+1.3885]), but does not establish an
advantage over content or permuted there. On MSU and SiW it is worse than
both controls, with the displayed conditional intervals below zero.
An isolated OULU/reference win or CASIA BFA gain is not evidence that the
correct anatomical placement generalizes better overall.

The permuted control also destroys prior smoothness; this is not a proof
that anatomy can never help. It is a negative result for this fixed
soft-prior, gate, crop and training recipe. Content/permuted point estimates
must not be rebranded as a validated new winner selected on these targets.

## 3. Execution and limitations

Supervised recovery wall time was22,608.517s (6.28h), with no guard failure.
Sampled peak total-device GPU use was12,880,707,584 bytes, tree RSS
6,246,338,560 bytes and owned outputs7,147,590,075 bytes, all within the
approved14GiB/8GiB/16GiB caps. Both earlier failed attempts remain recorded;
their elapsed time is not included in this recovery wall time.

All five arms use the same historical single-middle-frame RGB224 crops;
direct resizing of nonsquare crops can distort aspect ratio. This matched
study did not test corrected aspect-preserving DINO preprocessing. The
earlier [frozen-DDPM crop A/B](012-msu-frozen-crop-ablation.md) on22 MSU
originals does not answer that DINO question. Future crop comparisons must
separate preprocessing and anatomy effects, not change only the proposed arm.

There is one seed, all four targets are development-exposed, SiW participant
dependence is incompletely known, and SiW replaces Replay-Attack, so this
is not standard MCIO. Neural replay uses the same model implementation in
another process, with independent statistical reconstruction. Metrics are
not external SOTA parity, publication novelty, or ISO/iBeta test evidence.

No automatic follow-up training, new seed, threshold-policy change or
crop ablation follows from completion. Such work needs a separate scope
and authorization.
