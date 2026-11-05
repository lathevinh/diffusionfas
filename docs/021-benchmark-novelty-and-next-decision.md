# Cross-domain FAS: benchmark, novelty and next-experiment decision

Date checked: 2026-11-05.
Scope: five primary research works, public author resources and the completed
local protocols. **Research audit only: no new PAD decode, fitting, scoring,
downloaded checkpoint, dependency change or scientific experiment.**

## 1. Decision

**NO-GO for claiming a new successful detector from the current results.
Conditional GO for a narrowly defined empirical operating-policy study.**

Do not begin another architectural sweep or claim that source calibration,
linear probing, DINO registers, fine-tuning, temporal pooling or low-error
evaluation is new. The recent work below already covers major parts of that
space. Neither journal quartile nor a high AUROC establishes a contribution.

The immediate priority is a **protocol bridge and attribution design**, not a
new loss. Our four-dataset LODO is not standard MCIO/MICO: SiW-Mv2 replaces
Replay-Attack, source partition rules differ, and video/crop/threshold contracts
also differ. No paper in this audit supplies a directly comparable published
number for our exact experiment.

The potentially defensible question is:

> On exactly the same video scores and original population, when do good
> cross-domain rankings fail to deliver a useful source-selected operating
> policy, and how much of that failure is threshold transfer rather than
> fixed-score low-FA separation or acquisition failure?

This is a **candidate empirical contribution**, not established novelty.
Older working-point work and the precise threshold pipeline overlap still
need resolution before any "first" claim; such a claim is not recommended.

## 2. Five-work protocol map

MCIO/MICO here means MSU-MFSD, CASIA-FASD, **Idiap Replay-Attack**, OULU-NPU.
Dataset abbreviations alone do not specify official split versions.

| Work and verified source | Model / protocol | Input / evaluation verified | Relation to our latest clip result |
|---|---|---|---|
| [FLIP, ICCV2023][flip-paper] | CLIP fine-tuning, language grounding and multimodal contrastive variants; MCIO LODO, WCS LODO and single-source transfer; reports five seeds | Author MCIO instructions use MTCNN, RGB224 and two selected frames/video. Public evaluator averages frame probabilities by video; computes AUC, EER-derived HTER and TPR@FPR1% | Strong external reference family, but not a same-input rerun. Two frames/probability averaging is not our sixteen-frame feature averaging. Select the no-target-shot arm; do not mix the five-shot variant into DG. |
| [CFPL-FAS, CVPR2024][cfpl-cvf] and [full text][cfpl-paper] | CLIP content/style Q-Formers, prompt-text matching, style diversification and prompt modulation; MCIO and RGB S/CeFA/WMCA LODO | RGB224, random resized crop/flip,500 epochs; HTER, AUC, TPR@FPR1% | Closely related to learned domain/style conditioning. Exact released CFPL recipe, video aggregation and threshold population were not verified. Do not equate any author-repository CLIP trainer with CFPL. |
| [FAS-Aug + SARE,2024 primary preprint][fasaug-paper] | Physics-inspired recapture synthesis plus spoofing-risk equalization; MICO, MICY and MICO+SY | Paper reports HTER/AUC; author code documents MTCNN extraction and MICO training lists. MICO+SY adds extra bona-fide data, not the same population as MICO | Our adapted-reference study already reconstructs FAS-Aug concepts, with disclosed departures. "Add physics augmentation" or "equalize domain risks" alone is not a new contribution. Exact threshold and frame aggregation remain unverified here. |
| [Vision-foundation-model benchmark, CVPRW2026][vfm-paper] / [author project][vfm-project] |15 backbones; full-layer adapted DINOv2-Reg B/14 with FAS-Aug/PDA/APL; MICO LODO plus two-source LSD | Paper section4: dlib crops, RGB224, five fixed-interval sampled frames; HTER/AUC. The exact threshold data population is not specified in the inspected metric section | Closest vision-only architectural reference. Our paper329 is an independent reconstruction, not author-exact reproduction. Our current score does not establish parity or superiority. |
| [Unified linear-probing benchmark, IJCB2026 according to author arXiv record][probe-paper] |24 frozen encoders; **single-source cross-dataset**, explicitly not multi-source LODO/LSD | MTCNN, up to20 evenly spaced frames; family-specific processing. Reports ACER with development-EER threshold applied to test, not just AUROC | Directly overlaps frozen-DINO/linear-probe empirical positioning. It is not a four-fold LODO leaderboard. Its explicit dev-threshold evaluation rules out claiming that prior work uniformly ignores transferred thresholds. |

Primary text locations: FLIP section4.1 and author preprocessing/evaluator;
CFPL section4.1; FAS-Aug section4.1; VFM sections3.2 and4.1-4.2;
linear-probing benchmark sections4.1-4.2. Headline benchmark numbers are
deliberately not combined with ours into a ranking table.

### Critical threshold distinction, established from actual author code

At FLIP revision `4f95def259e135a0cbaff1d770f559ca739c4c9f`,
[`utils/evaluate.py`][flip-eval] documents evaluation on target data. It first
averages predictions by video, then calls `get_EER_states(prob_list, label_list)`
and `get_HTER_at_thr` using **the same evaluation arrays**.
Its ROC operating diagnostic is also derived from those labels.
[`utils/statistic.py`][flip-statistic] confirms a scanned EER threshold.

This establishes the evaluator's information use, not every author's training
run or reported checkpoint-selection history. It is not an accusation that the
paper violates its stated convention. It does mean that this HTER is not our
source-only5%-FA policy result. Do not call these equal-security results or
interchange target-label EER/ROC diagnostics with deployable source thresholds.
Verify label polarity, aggregation and ties before aligning APCER/FPR terms.

For the unified probing work, dev-to-test threshold transfer is explicit, but
the inspected passage does not completely resolve which corpus supplies dev
for every off-diagonal entry. Preserve that uncertainty; do not call it
target-test EER. For CFPL, FAS-Aug and VFM, this audit does not independently
establish the full author threshold implementation.

## 3. Code, checkpoint and reuse gate

Public visibility is not an affirmative software/data/checkpoint license.
No author implementation was copied into our repository in this audit.

| Resource inspected | Verified availability | Remaining execution/reuse restriction |
|---|---|---|
| [FLIP author repository][flip-code], revision `4f95def259e135a0cbaff1d770f559ca739c4c9f` | Actual evaluator, preprocessing instructions and [model-zoo links][flip-zoo] | Root license API returned404; no affirmative project license verified. Checkpoint links were observed, not downloaded or tested; their license/availability and training-data scope remain unverified. |
| [CFPL-associated author benchmark][cfpl-code], revision `10321b910be8bef7a4ff4d47169331b20d69ce99` | Author README cites CFPL. Tree has CLIP/CoOp/ICPE trainers; `icpe.py` explicitly identifies a different TPAMI method | No CFPL-named trainer was found; this is not proof that no equivalent implementation exists. Exact CQF/SQF/PTM/DSP release not verified. Root license API404; bundled Dassl license does not license all author code. |
| [FAS-Aug author implementation][fasaug-code], revision `0da1dd79bad00e225b8cb6977c7f3f06ee8f7517` | Readme has runnable training/preprocessing instructions. Paper-linked underscore repo redirects to implementation repos | No license file in inspected tree; root license API404. Author README explicitly does not redistribute datasets. The second linked implementation repo's license API also returned404. |
| [VFM author project][vfm-project] | Paper, recipe and result tables verified | No executable author package/checkpoint with verified reuse rights established from the inspected project page. Existing local independent reconstruction is usable only under its own provenance. |
| [Unified probing paper][probe-paper] | Primary protocol and results verified | A matching author implementation/checkpoint release and license were not established in this audit. Do not invent a release URL or claim exact reproduction. |

An API404 alone is not a legal determination or proof that no license exists
anywhere. These are **unresolved permission gates**, not an invitation to copy
unlicensed code. Independent implementation of published concepts must still
disclose recipe departures, respect asset/data rights and avoid author-exact
claims.

## 4. What our evidence currently supports

The [completed clip study](020-adapted-dino-short-clip-results.md) has macro
AUROC94.850% single,95.137% mean16,95.091% temporal and95.175% trained shuffled.
Each arm meets source-policy FA<=5%, BFA>=50% only on CASIA. One seed,
development-exposed targets, unknown SiW participant dependence and shared clip
failures constrain interpretation.

The [regional experiment](018-regional-dino-completed-results.md) used
pretrained, not PAD-fine-tuned, RGB518 DINO; three seeds and a different usable
population. It cannot isolate the effect of adaptation against the clip table.
The [older publication audit](016-submission-readiness-audit.md) remains useful
historical framing, not evidence that the novelty gap has now been closed.

Neither a95% macro AUROC on our custom LODO nor failure of the GRU construction
demonstrates competitiveness with MCIO paper tables. Likewise, our strict
source policy cannot be called worse than target-label EER HTER without
separating their different information budgets.

## 5. A calibration shortcut that cannot solve this policy

Our current source calibrator is a single common positive-affine sigmoid of
the head logit, and the threshold is a source attack order statistic, taking
the minimum source upper bound. For strictly increasing `g`:

`Q_k(g(scores)) = g(Q_k(scores))`

`min_d g(Q_d) = g(min_d Q_d)`

`g(test_score) < g(threshold) <=> test_score < threshold`.

Therefore, a common temperature/Platt remapping followed by the same quantile
threshold rule **cannot change decisions in exact arithmetic**, nor AUROC.
Saturation/finite-precision ties can change numerical decisions, not establish
a scientifically useful generalization mechanism. This was checked on
synthetic scores against the actual historical threshold helper, without PAD
access or scientific fitting.

Changing the threshold-selection rule, using sample-dependent scores or
learning a different representation is a genuinely different intervention.
Domain-conditioned remapping may change global ordering, but it is already
adjacent to substantial calibration/DG literature and is not declared novel.
Using target samples, even without labels, changes the information budget and
must be disclosed as adaptation rather than silently called source-only DG.

## 6. Next bounded design and contribution gate

### A. Protocol bridge before a method claim

Freeze two distinct evaluation tracks; never pool their tables:

1. **Literature-compatible MCIO track:** verify authorized Replay-Attack access,
   official split versions, each source-data policy, detector/frame aggregation,
   external pretraining and the comparator's threshold convention. Replay
   access is not assumed or requested/downloaded by this audit.
2. **Our operational track:** existing original-denominator source-only
   policy, same locked population, all failures retained, strict ties and
   FA/BFA plus usable-only AUROC. SiW-Mv2 stays explicitly a different benchmark.

Select the existing source-only adapted DINO reference as the internal strong
baseline. Select FLIP-V as the first external-family reconstruction target:
its vision-only architecture is simpler to isolate than prompt-based systems,
and its public preprocessing/evaluator allow a concrete protocol audit.
It is **not the strongest FLIP variant or a sufficient sole SOTA comparator**;
FLIP-MCL and CFPL remain comparison targets after release/rights/budget gates.
No external run is authorized here.

### B. Test one empirical hypothesis on fixed scores

**H1:** On some matched target/model cases, a sample-feasible low-FA operating
point exists, but source-only policies fail to recover it; on others, the
stored ranking itself cannot meet the chosen utility.

Proposed contrasts, all on the same scores and labels with fixed populations:

- Current minimum-source attack-quantile policy.
- Pooled-source attack quantile: an intentionally weaker source-security
  contract, reported separately rather than advertised as equal protection.
- Source-development EER threshold: a conventional utility/security trade-off,
  not assumed to satisfy5% FA.
- Target-label EER and low-FA ROC/oracle: diagnostics only.

Report the security/utility distinction, not just HTER or a sign-ambiguous
"oracle minus actual BFA." If actual FA is unsafe, their BFA values are at
different security levels. Use integer acceptance budgets, tie handling,
failure accounting and subject/component paired uncertainty.

This analysis requires a separately specified execution/approval boundary if
new scientific outputs are produced. No such output was computed here.
It is an empirical design, **not an invented algorithm**.

### C. Explicit go/stop criteria

- **Empirical GO, conditional:** repeatable matched evidence across more than
  one model family and training seed; all targets retained; threshold
  information budgets audited; precise distinction from prior working-point
  studies; confirmation data/analysis locked before viewing. Otherwise do not
  claim a robust general finding from selected examples.
- **Method NO-GO now:** no current new method passed. For a future method,
  predeclare at least two operationally improved targets, no newly unsafe
  target, at least one additional joint-feasible target, and no BFA regression
  above5 percentage points on another target; require paired uncertainty and
  repeat seeds before an improvement claim. These are proposed research
  acceptance gates, not an industry standard or a journal guarantee.
- **Stop:** do not expand GRU/anatomical/GroupDRO, sweep parameters on these
  targets, or sell common temperature scaling as quantile-policy improvement.
  If contribution remains indistinguishable from prior work, do not submit
  merely because a journal is Q3.

The closest older overlap remains
[Costa-Pazo et al., IET Biometrics2021][working-paper] and its
[author GRAD-GPAD framework][working-code]. Its primary publisher full-text
endpoint returned403 today; the author repository confirms a pre-existing
realistic generalization evaluation framework, but does not settle exact
source-policy/oracle overlap. The bounded search is not proof of absence of
other threshold/conformal/robust-risk work.

**Bottom line:** commit to an empirical question only if the matched protocol
bridge and prior-art distinction survive inspection. There is no substantiated
new detector contribution or Q3 acceptance promise at this point.

[flip-paper]: https://arxiv.org/abs/2309.16649
[flip-code]: https://github.com/koushiksrivats/FLIP/tree/4f95def259e135a0cbaff1d770f559ca739c4c9f
[flip-eval]: https://github.com/koushiksrivats/FLIP/blob/4f95def259e135a0cbaff1d770f559ca739c4c9f/utils/evaluate.py
[flip-statistic]: https://github.com/koushiksrivats/FLIP/blob/4f95def259e135a0cbaff1d770f559ca739c4c9f/utils/statistic.py
[flip-zoo]: https://github.com/koushiksrivats/FLIP/blob/4f95def259e135a0cbaff1d770f559ca739c4c9f/docs/model_zoo.md
[cfpl-cvf]: https://openaccess.thecvf.com/content/CVPR2024/html/Liu_CFPL-FAS_Class_Free_Prompt_Learning_for_Generalizable_Face_Anti-spoofing_CVPR_2024_paper.html
[cfpl-paper]: https://arxiv.org/html/2403.14333v1
[cfpl-code]: https://github.com/liuajian/CASIA-FAS-Benchmark/tree/10321b910be8bef7a4ff4d47169331b20d69ce99
[fasaug-paper]: https://arxiv.org/html/2409.03501v1
[fasaug-code]: https://github.com/CeceliaSoh/FAS-Aug/tree/0da1dd79bad00e225b8cb6977c7f3f06ee8f7517
[vfm-paper]: https://arxiv.org/html/2604.19196v2
[vfm-project]: https://gsisaoki.github.io/FAS-VFMbenchmark-CVPRW2026/
[probe-paper]: https://arxiv.org/html/2607.26993v2
[working-paper]: https://doi.org/10.1049/bme2.12049
[working-code]: https://github.com/acostapazo/gradgpad
