# Submission-readiness audit: frozen-DINO operating-point transfer

Date: 2026-11-04.

## 1. Decision

**GO for a focused manuscript revision; NOT ready to submit now.**

There is a defensible empirical-study direction, but not a demonstrated new
DINO method or a guaranteed Q3 publication. Lead with the completed
four-dataset comparison of transferred operating policies and fixed-score
oracle headroom. Do not lead with the 22-clip inpainting comparison.

No additional experiments are required to finish that bounded empirical-study
framing. A new adaptation algorithm, untouched confirmation, demonstrated
fusion benefit, or deployment-security claim would require separately
specified evidence and approval. This audit authorizes none of them.

Confidence is high in the public arithmetic and the identified writing
repairs, moderate in the proposed positioning, and insufficient to certify
institutional Q3 eligibility. Acceptance remains an editorial/reviewer decision.

### Audit boundary

- Read-only source revision:
  `707e3ddf6103835c55e937cbbd84081d91f188c7` in `lathevinh/fas`.
- Main evidence: [manuscript][manuscript], [completed comparison][comparison],
  [public cases][cases], [original protocol][protocol],
  [completed-package review][review325], and [prior-work review][review320].
- Independently recounted all 144 public operating-case rows, including the
  48 primary cases and limiting-source counts. Read the entire manuscript.
- This was not a replay of private training, raw predictions, bootstrap
  draws, or dataset membership. Public aggregate checks are not independent
  scientific replication.
- No source-checkout edits, media access, fitting, recalibration, threshold
  changes, new benchmark runs, submission, or payment were performed.

## 2. What the paper can contribute

A suitable contribution statement is:

> On a declared four-dataset developmental protocol, we compare actual
> source-derived operating policies with evaluation-label, fixed-score
> low-FA oracle headroom, separating failures of the available policy from
> sample-specific score limitations, across frozen foundation-model systems
> and a matched-input trained specialist.

The value is the controlled empirical diagnosis and its counterexamples,
not the mathematical existence of a constrained ROC point. Frozen encoders
still have **source-supervised heads and calibration**; this is not
bonafide-only, zero-shot, or training-free PAD.

### Closest-work assessment

| Work and primary source | Established overlap | Defensible distinction / restriction |
|---|---|---|
| Lorenz, George and Marcel, [Unified Linear-Probing Benchmark](https://arxiv.org/abs/2607.26993), accepted IJCB2026 according to its arXiv page | 24 frozen encoders; linear probing for PAD; MCIO and operating metrics | Frozen-DINO PAD and linear probing are not novelty. Position the paired transferred-policy/oracle diagnosis, not encoder discovery. |
| Feng et al., [Benchmarking Vision Foundation Models for Domain-Generalizable FAS](https://arxiv.org/abs/2604.19196), CVPRW2026 according to its arXiv page | 15 models, register DINOv2, MICO/LSD cross-domain protocols, FAS-specific augmentation/loss | Register DINO and multi-source domain generalization are already studied. Our declared populations/recipes differ; do not make unmatched leaderboard superiority claims. |
| Lorenz, George and Marcel, [LoRA-based Adaptation Alone Is Not Enough](https://arxiv.org/abs/2608.09633), arXiv v2 | 32 foundation models, zero-shot and LoRA; strong intra-/cross-dataset contrast | Generalization failures and the limits of light adaptation are not new observations. Do not assume our specialist causally isolates frozen versus adapted representations. |
| Costa-Pazo et al., [Comprehensive evaluation of the generalisation problem](https://doi.org/10.1049/bme2.12049), IET Biometrics 2021; [author framework](https://github.com/acostapazo/gradgpad) | Generalization protocols, constrained working points and working-point curves; the prior local review verified APCER at BPCER10% in full text | Looking beyond AUROC is not itself novel. Exact overlap with our source-threshold/same-FA oracle pipeline remains unresolved. Do not infer absence from an abstract or repository README. |
| [DINO-VPT](https://arxiv.org/abs/2607.20900), accepted IJCB2026 according to its arXiv page | DINO-based physical/digital FAS with hierarchical visual prompting | A learned prompt/routing method, not evidence that this paper introduces a new DINO technique. Relevant when delimiting method claims, not a numerically matched baseline. |

The Feng and LoRA primary pages were rechecked during this audit. The
Costa-Pazo publisher endpoint now returned HTTP403; the accessible author
repository confirms the framework and citation but does not settle exact
threshold-pipeline overlap. The earlier full-text review remains provenance,
not a claim that the full article was freshly inspected today.

**Positioning action:** write a precise task/protocol/decision-rule comparison
with Costa-Pazo and the recent foundation-model papers. Avoid "first",
"novel constrained ROC", and claims that prior work only reports AUROC.
This is a bounded literature-writing task, not a reason to reopen an
indefinite model campaign. If novelty is asserted for the exact pipeline,
resolve the full-text overlap before making that assertion.

## 3. Evidence and claim map

The primary study has four targets, three seeds and four systems:
48 system/seed/target cases, not 48 independent datasets. It uses one
accepted earlier-middle frame per original video.

Population accounting is 7,505 active originals / 7,474 usable across roles;
target evaluation is 2,943 originals / 2,929 usable, including 14 failures.
Do not advertise all 7,505 as test examples.

| Endpoint | Recounted public result | Permitted interpretation |
|---|---:|---|
| Primary FA <=5%, BFA >=50% | 0/48 actual source policies meet both | This particular available source-threshold rule fails the declared joint utility. |
| Primary source-policy FA constraint alone | 15/48 | Being FA-safe does not imply useful bona-fide acceptance. |
| Same-FA evaluation-label oracle joint utility | 21/48 | Some stored scores admit a useful hindsight threshold; no target-label-free recovery is demonstrated. |
| Primary oracle infeasible | 27/48 | Threshold movement alone cannot meet this sample-specific utility for those fixed scalar scores. Not all PAD methods are impossible. |
| Oracle feasibility by target | CASIA 11/12; MSU 7/12; OULU 3/12; SiW 0/12 | Heterogeneous target/model evidence, not universal domain behavior. |
| 1% sensitivity | Actual-policy utility 0/48; oracle utility 12/48 | A sensitivity endpoint, not a replacement primary objective. |
| 10% sensitivity | Actual-policy utility 10/48; oracle utility 42/48 | Relaxing security changes the task; it does not rescue the primary claim. |

The 50% BFA floor is an explicitly chosen illustrative utility criterion,
not an industry deployment standard. Preserve usable-only errors, original
denominators, failures, integer budgets, ties, all seeds and unfavorable cases.

Two useful manuscript examples already exist:

- CASIA DINO-Reg: mean AUROC 99.94%, actual-policy FA 0%, BFA 39.26%,
  oracle BFA 98.89%. Excellent ranking need not yield a useful transferred
  threshold.
- MSU two-DINO: mean AUROC 92.67%, actual-policy FA 22.78%, BFA 92.50%,
  oracle BFA 68.33%. The policy is unsafe at the primary bound; subtracting
  its BFA from the constrained oracle's BFA is not a same-security gain.

The specialist gives OULU oracle feasibility in all three seeds while the
tested frozen systems give none, but its actual OULU policies still violate
the FA constraint. Preserve this counterexample rather than claiming that
OULU, adaptation, or specialist learning is universally ineffective.

### The 22-clip results are secondary, if included at all

[The matched comparison](015-matched-msu-dino-mat-comparison.md) contains
22 MSU validation originals, three subject groups and T16 aggregation.
Two-DINO mean AUROC 0.9826388889 exceeds MAT 0.75 on that small selected cohort.
It is **not** an improvement from the earlier 0.9267 benchmark on the same
population, nor untouched confirmation. Cohort, aggregation and model-specific
crop contracts differ. Source-supervised DINO and frozen generative MAT also
have different supervision histories.

MAT's distinct ordering of one or two attack/bona-fide pairs per seed does
not demonstrate fusion utility, false-accept rescue, or a deployable policy.
Keep this outside the main paper unless needed as an explicitly exploratory
supplement. Do not replace the completed primary experiment with it.

## 4. Ranked manuscript repairs

### P0: complete before submission

1. **Lead with the completed primary comparison, not project history.**
   [Current historical methods/results][history] precede the stronger official
   comparison. Move 309/314/316 development details into a clearly separated
   supplement. Historical official-test observations in source roles cannot
   be pooled with the later source-train/development-only experiment.
   Keep the existing primary side-by-side table and low-FA figure; promote
   them rather than commissioning another comparison.

2. **Make the central rule reproducible in the paper.**
   [Section4.4][threshold-section] states a constraint, but does not explicitly
   identify the minimum per-source safe order statistic. For nonempty source
   attack score sets, let `a_d,(j)` be the ascending successful calibration
   scores, `n_d` their count, and `k_d=floor(alpha*n_d)`. State:

   ```text
   t_d = a_d,(k_d + 1)
   t   = min_d t_d
   accept iff score < t
   ```

   This is the stored rule, not a proposed alternative policy. Describe
   equality/ties, small-support budgets and any implementation boundary
   sentinel explicitly. Calibration failures do not enlarge successful-image
   budgets; target end-to-end denominators use originals. Avoid interpolated
   quantiles or presenting "choose any safe threshold" as equivalent.

3. **Provide a standalone completed-methods supplement.**
   [Sections4.1-4.2][model-section] contain the frozen-head recipe but defer
   most specialist details to the original protocol. That JSON deliberately
   retains its pre-execution status, `scientific_execution_authorized=false`,
   and pending initialization fields. Preserve that sealed historical record;
   do not present it alone as the final execution receipt or overwrite it.
   Add a completed-methods account bound to the completed receipts.

   Existing specifications supply the following table; reconcile each entry
   with completed execution evidence when preparing the final supplement:

   | Component | Existing specified recipe to expose |
   |---|---|
   | Frozen heads | 200 epochs, FP64 full-batch AdamW; equal-source/equal-class BCE; DINO LR0.001, CLIP LR0.01; decay0.05 except bias; minimum validation BCE / earliest tie; no refit |
   | Specialist | DenseNet161 ImageNet initialization, first two dense blocks; pixel/global heads; matched-input architecture retraining, not exact original DeepPixBiS reproduction |
   | Specialist optimizer | 50 epochs, batch16, Adam LR0.0001, betas(0.9,0.999), epsilon1e-8, decay1e-6; constant LR |
   | Specialist loss/sampling | Equal pixel/global BCE weighting; equal-source/equal-class replacement sampling; horizontal flip p0.5 during fitting only |
   | Selection/numerics | Lowest source-validation loss, earliest tie, no refit; FP32 without AMP/TF32; deterministic algorithms |
   | Scoring/calibration | One minus mean sigmoid pixel probability; clipped-logit calibration input; positive-affine sigmoid; fixed 50/50 calibrated two-DINO mean |

   Include actual checkpoint/initialization provenance, software, input sizes,
   normalization, feature pooling and preprocessing/border/resize contracts.
   Do not imply that different encoder resolutions or historical crops are
   identical to MAT's real-pixel square crop. This audit did not independently
   verify every primary-study crop. The 22-clip native crop check does not
   retroactively validate all primary inputs.

4. **Tighten novelty, interpretation and comparison scope.**
   Use the contribution in Section2 and the existing qualified conclusion.
   Neither source-threshold failure nor oracle infeasibility establishes
   failure of all source-only policies. No claim of one-class PAD,
   untouched testing, causal architecture superiority, state of the art,
   clinical/security certification, or automatic target calibration follows.
   An unsafe-policy versus constrained-oracle BFA difference is not a
   calibration loss.

5. **Finish submission metadata and venue eligibility.**
   Complete author-approved authorship/affiliations, declarations, data/code
   availability, dataset-license/ethics wording, funding/conflicts and
   venue-formatted references. Do not invent missing information.
   References already correct the prior Lorenz/Feng attribution errors;
   do not request that repair again. Confirm current publication metadata
   and replace repository report numbers as scientific explanations.
   Lock the journal only after the ranking/cost gates in Section6.

### P1: important improvements using existing evidence

- Add a compact limiting-source summary rather than directing every reader
  to the case CSV:

  | Target: 12 primary cases each | Source determining the minimum threshold |
  |---|---|
  | OULU | MSU8; SiW4 |
  | CASIA | OULU6; MSU3; SiW3 |
  | MSU | SiW8; OULU4 |
  | SiW | MSU6; OULU6 |

  These are numerical bottlenecks of this rule, not causal explanations of
  domain shift or proof that a named source should be removed.

- Keep all three seeds and the existing conditional paired intervals in
  the supplement. No pooled 48-case significance test, seed-as-subject
  inference, oracle confidence intervals, or population-of-domains guarantee
  should be invented. Zero empirical FA with a degenerate bootstrap interval
  is not an upper bound on unseen attack risk.
- State the local SiW-Mv2 Protocol-I intersection and unknown participant
  identities prominently. Video/exact-frame grouping is not subject isolation.
  Preserve the "all four datasets informed development" limitation.
- Move attempt/recovery bookkeeping out of the main narrative. Retain the
  measured compute accounting in the supplement, but do not compare a cached
  fitting wall time with full encoder training/inference as if end-to-end
  efficiency were matched.
- Explain the practical distinction between policy-placement headroom and
  insufficient low-FA score separation, without promising an untested remedy.

### Suggested standalone organization

1. Introduction: operational question, bounded contributions, no new model.
2. Related work: foundation-model PAD and prior working-point evaluation.
3. Protocol/methods: populations, supervision, inputs, models, source rule,
   original-denominator errors, oracle and conditional uncertainty.
4. Results: primary paired table/low-FA figure, counterexamples, all seeds
   and declared sensitivity.
5. Discussion/limitations: deployment interpretation, data exposure,
   SiW dependence, unmatched compute, oracle limitations.
6. Conclusion.

Supplement: completed recipes/provenance, existing per-case tables and
intervals, exclusions/failures, clearly separate historical development.

## 5. Conditional journal shortlist

**This is not a certified list of institutional Q3 journals.** SCImago/SJR,
Scopus CiteScore quartiles and Clarivate JCR quartiles are different.
An index listing or a journal's best quartile does not establish a required
category/year quartile. Institutional rules may also depend on submission,
acceptance or publication year.

Evidence labels:

- **A:** substantive current primary page retrieved directly.
- **B:** primary-provider text visible through search indexing, but the
  provider page could not be inspected directly.
- **U:** not verified; no price/rank guarantee.

| Priority / journal | Scope and article type | Mandatory versus optional cost | Ranking evidence and decision |
|---|---|---|---|
| **1. International Journal of Biometrics (IJBM), Inderscience** | **A:** human authentication/verification and face biometrics; original empirical paper fits in principle. Full original article, not a brief/research note. Double-blind review. | **A:** standard non-OA publication has no publication charge. General hybrid OA policy lists **EUR2,200 for 2026**, optional after acceptance, subject to journal-specific exceptions/current terms. Not a guaranteed future-year quote; taxes not established. | **B:** indexed SCImago source17700155303 reports SJR2025 0.264, best Q3. An indexed category row explicitly reports **Electrical and Electronic Engineering, 2025, Q3**. Direct profile returned403; other category rows/JCR not certified. **Best scope/cost candidate, conditional on institutional ranking acceptance.** |
| **2. International Journal of Pattern Recognition and Artificial Intelligence (IJPRAI), World Scientific** | Pattern-recognition topic is plausible, but **U** for current full aims/scope and article-type requirements. The indexed official typesetting template is not a scope/fee policy. | **U:** title-specific mandatory page/publication charges and optional OA price not verified. Do not treat it as free or import the publisher's generic USD1,500 OA example. | **B:** indexed SCImago source24310 reports SJR2025 0.255, best Q3. Exact current category-year rows not verified; graph labels/secondary listings do not suffice. **Reserve only until current scope, costs and category are confirmed.** |
| **3. Journal of Electronic Imaging (JEI), SPIE/IS&T** | **B:** indexed publisher/co-publisher descriptions concern image/imaging research; PAD could fit if the imaging insight is clear. **U** for current complete article-type/scope requirements. | **U:** no current title-specific mandatory/optional charge schedule verified. A 2013 SPIE announcement is not a 2026 fee quote. IS&T's JIST charge form is for another journal and must not be used. | **B:** indexed SCImago source25978 reports SJR2025 0.249, best Q3; category-specific status unverified. Publisher pages returned a challenge and the co-publisher page403. **Conditional reserve, not a locked destination.** |

### Current primary venue sources

- IJBM [journal/scope/submission essentials](https://www.inderscience.com/jhome.php?jcode=ijbm),
  [author guidelines](https://www.inderscience.com/mobile/inauthors/index.php?pid=70),
  [OA policy and dated rate](https://www.inderscience.com/mobile/inauthors/index.php?pid=75).
- SCImago: [IJBM](https://www.scimagojr.com/journalsearch.php?q=17700155303&tip=sid),
  [IJPRAI](https://www.scimagojr.com/journalsearch.php?q=24310&tip=sid),
  [JEI](https://www.scimagojr.com/journalsearch.php?q=25978&tip=sid).
  Direct retrieval failed; the best-quartile/category observations above
  are explicitly indexed-primary evidence, not inspected exports.
- IJPRAI [publisher home](https://www.worldscientific.com/worldscinet/ijprai)
  and [submission guidelines](https://www.worldscientific.com/page/ijprai/submission-guidelines)
  returned403. Its [indexed official template](https://worldscientific.com/sda/1037/ws-ijprai.pdf)
  does not certify publication charges.
- JEI [SPIE page](https://spie.org/publications/journals/journal-of-electronic-imaging),
  [digital-library page](https://proceedings.spiedigitallibrary.org/journals/journal-of-electronic-imaging),
  and [IS&T co-publisher page](https://www.imaging.org/IST/IST/Publications/Journal_of_Electronic_Imaging.aspx).
  Legitimate alternative public endpoints were attempted; no access controls
  were bypassed.

IJBM additionally requires an appropriate AI-use declaration when AI assists
article development/writing, and makes authors responsible for verification.
Follow its double-blind checklist, final author approval and requested
expert-nomination rules. This report does not supply or submit personal
metadata, nominate reviewers, or select paid OA.

## 6. Submission gates and next action

Proceed in this order:

1. **Editorial revision:** implement the P0 scientific-writing repairs against
   the unchanged completed evidence. No new training/scoring is needed.
2. **Human approval:** authors verify every scientific claim, references,
   authorship and declarations. A revision is not journal acceptance.
3. **Ranking decision:** establish whether the institution requires SJR,
   CiteScore or JCR, which category, and which reference year. Obtain an
   inspectable official record/export for the exact ISSN/category/year.
   The present search-indexed evidence alone does not close this gate.
4. **Venue lock:** prefer IJBM if its ranking satisfies that rule; verify
   current article requirements and non-OA/OA terms. If not, verify a reserve
   venue's scope, title-specific costs and ranking before selecting it.
5. **Submission package:** format and anonymize, check links/tables and
   supplements, obtain final author approval. Submit to one journal only;
   no submission or payment is implied by this audit.

**Bottom line:** finish a focused empirical paper about operating-point
transfer and score-specific headroom. Do not run another DINO/MAT/DDPM
campaign merely to make the headline AUROC larger. Do not promise Q3
eligibility or acceptance before the remaining venue and editorial gates.

[manuscript]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/paper319/manuscript.md
[comparison]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/324-completed-comparison-results-and-paper-update.md
[cases]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/results/phase1/paper324-cases-v1.csv
[protocol]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/paper319/supplementary-protocol.json
[review325]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/325-review-doc324-completed-results-and-paper319.md
[review320]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/320-review-doc319-and-paper319-preparation-and-manuscript.md
[history]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/paper319/manuscript.md#L239
[threshold-section]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/paper319/manuscript.md#L307
[model-section]: https://github.com/lathevinh/fas/blob/707e3ddf6103835c55e937cbbd84081d91f188c7/docs/paper319/manuscript.md#L255
