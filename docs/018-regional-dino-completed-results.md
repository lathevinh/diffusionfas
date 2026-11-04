# Regional DINO + GroupDRO: completed controlled results

Date: 2026-11-04.
Status: **completed and reconstructed; NO-GO for the primary candidate.**

The approved [six-arm protocol](017-regional-dino-controlled-ablation.md)
completed all 72 heads, 72 calibrators, four held-out targets and three seeds.
**Anatomical pooling plus GroupDRO did not improve generalization over the
contemporary global DINO-Reg reference.** No automatic retry, parameter sweep,
or additional model campaign follows this result.

Full aggregate-safe evidence, including all 216 operating cases and 540
conditional interval records: [results](../results/regional-dino-v1.json).
The [preparation receipt](../results/regional-dino-preparation-v1.json)
is the historical pre-approval snapshot, not the current execution status.

## 1. Scope and provenance

- Pre-run implementation commit: `f42093f`.
- Read-only FAS revision: `707e3ddf6103835c55e937cbbd84081d91f188c7`.
- Exact approved plan:
  `8e6d56d90bc2541f6ac3a60f1fb6443d187dc19d3d63b0d43cd485e0b5e53919`.
- Owner-approval receipt:
  `df1a43991f42cbf5df4b20f88dcb5ffb067f0a794f7b9b8909c71390a740eca1`.
- Private result commitment:
  `98e4cc7aaedb3db600cc0d805701b3c43ae8072f86bcbaa9c0814e6a290d28e0`.
- Private aggregate commitment:
  `dcac8e11fb41379f75cd624917dcdf91572a94cc8e81d41dc42032e354991786`.

No encoder training, video decode, new detector, crop alteration, download,
target-based model selection, or scientific retry occurred.
Encoder extraction used 7,469 existing RGB518 crops plus two synthetic
replay forwards. The separate earlier CPU engineering smoke used two more
synthetic forwards, not PAD images.

### Matched population, not an unchanged historical benchmark

There are 7,505 active originals and 7,469 usable observations across all
roles. Target evaluation contains 2,943 originals / 2,926 usable, including
17 technical failures.

The owner-approved geometry rule adds five shared failures to the original
31: two SiW-Mv2 source-train bona-fide originals and three SiW-Mv2 target-test
attacks. All six arms retain those originals as nonacceptances. No missing
score is inserted into AUROC.

Thus the new global reference can differ from paper324 even though every
re-extracted global feature matches its old cache exactly: source fitting/
validation/calibration eligibility and the SiW usable target cohort changed.
Use **within-study matched comparisons**, not old-versus-new headline gains.

## 2. All primary AUROC results

Percentages below are equal means over the three fixed seeds, on usable
observations. Every target and arm is retained.

| Target | Global ERM | Global DRO | Anatomical ERM | Anatomical DRO: primary | Spatial ERM | Spatial DRO |
|---|---:|---:|---:|---:|---:|---:|
| OULU-NPU | 87.74 | 86.84 | 85.71 | 84.97 | 87.67 | 86.67 |
| CASIA-FASD | 99.94 | 99.96 | 99.81 | 99.81 | 99.82 | 99.83 |
| MSU-MFSD | 87.61 | 88.99 | 81.17 | 80.70 | 86.57 | 87.33 |
| SiW-Mv2 | 87.94 | 85.55 | 85.08 | 82.94 | 81.20 | 79.90 |
| Equal-target descriptive macro | 90.81 | 90.33 | 87.94 | 87.11 | 88.81 | 88.43 |

Primary-candidate AUROC changes from global ERM are approximately:

- OULU: **-2.76 percentage points**.
- CASIA: **-0.13 points**.
- MSU: **-6.91 points**.
- SiW: **-5.00 points**.
- Equal-target macro: **-3.70 points**.

For OULU, MSU and SiW, candidate-minus-reference conditional paired AUROC
intervals are below zero in each seed. CASIA intervals include zero.
These are unadjusted, conditional descriptive intervals, not a population
of domains test or nine independent confirmations.

Anatomical DRO versus the matched spatial DRO control is also heterogeneous:
it is worse on MSU and better on SiW in all three conditional AUROC interval
comparisons, while both remain worse than global ERM there. This does not
support a universal anatomical-region advantage.

## 3. Actual source-policy operation at FA <=5%

All rates below use original denominators, with technical failures retained.
The thresholds are derived from source calibration, not target labels.

| Target | Global ERM FA | Global ERM BFA | Anatomical DRO FA | Anatomical DRO BFA |
|---|---:|---:|---:|---:|
| OULU-NPU | 36.20 | 91.94 | 56.64 | 95.93 |
| CASIA-FASD | 0.00 | 40.00 | 0.00 | 40.74 |
| MSU-MFSD | 41.67 | 90.00 | 27.50 | 75.00 |
| SiW-Mv2 | 9.67 | 58.24 | 19.61 | 65.26 |

**Both global ERM and the primary candidate meet the joint FA<=5%,
BFA>=50% utility in 0/12 target/seed cases.** Each is FA-safe in 3/12 cases,
all on CASIA, where bona-fide acceptance remains below 50%.

Higher BFA on OULU/SiW is not a same-security improvement: attack acceptance
also rises, well above the required bound. MSU attack acceptance falls but
still remains unsafe, and bona-fide acceptance falls as well.

### Every arm's utility and hindsight headroom

| Arm | FA-safe cases /12 | Actual joint utility /12 | Oracle joint utility /12 |
|---|---:|---:|---:|
| Global ERM | 3 | 0 | 3 |
| Global DRO | 3 | 0 | 6 |
| Anatomical ERM | 3 | 3 | 3 |
| Anatomical DRO | 3 | 0 | 3 |
| Spatial ERM | 3 | 3 | 4 |
| Spatial DRO | 3 | 0 | 3 |

The limited favorable result is **anatomical ERM on CASIA**: mean FA0%,
BFA51.85%, versus the reference's FA0%, BFA40%. It reaches utility in all
three seeds. However, spatial-control ERM also reaches utility in all three
seeds, with mean FA0%, BFA50%. The anatomical arm loses AUROC substantially
on other targets. This is not stable generalization improvement and does
not isolate anatomy as the cause of the CASIA policy gain.

Global DRO increases MSU mean AUROC from87.61% to88.99%, and increases
oracle feasibility there, but its actual MSU FA remains34.17%. It worsens
other targets and does not recover the primary operational utility.
It must not be selected post hoc as a newly successful universal method.

### Declared sensitivity, without changing the primary endpoint

| Arm | FA bound | FA-safe /12 | Actual joint utility /12 | Oracle joint utility /12 |
|---|---:|---:|---:|---:|
| Global ERM | 1% | 3 | 0 | 3 |
| Global ERM | 5% | 3 | 0 | 3 |
| Global ERM | 10% | 3 | 3 | 12 |
| Anatomical DRO | 1% | 6 | 0 | 3 |
| Anatomical DRO | 5% | 3 | 0 | 3 |
| Anatomical DRO | 10% | 3 | 3 | 6 |

The complete results retain sensitivity cases for the other four arms too.
Oracle thresholds use target labels and diagnose only the fixed stored
scores. Their feasibility is not a selectable source-only policy.

## 4. Verification and resources

The separate CPU process reconstructed:

- All 12 contexts, 216 distinct operating cases and 540 distinct primary
  interval records, including the predefined matched ablation contrasts.
- Saved-feature probabilities using an independent dot-product reduction:
  maximum discrepancy **2.4980018054e-15**, below the predeclared1e-10 tolerance.
- Exact acceptance decisions, independently sorted source thresholds,
  original/usable counts, sklearn AUROC and grouped-curve oracle checks.
- Bootstrap intervals using an independent vectorized group-count/ranking
  calculation. All540 records retained2,000/2,000 valid replicates.

Global features matched the historical cache with **maximum difference0.0**.
Synthetic CLS/patch token replay was exact; encoder state remained unchanged.
This is saved-feature/checkpoint-output reconstruction, **not independent
replication of training or a second full encoder extraction**.

Resampling groups: OULU20 known subjects, CASIA30, MSU20; SiW623
video/exact-frame components. Unknown SiW participant dependence and prior
development exposure remain limitations. No untouched confirmation or
certified future security rate is claimed.

| Resource | Measured | Approved ceiling |
|---|---:|---:|
| Whole supervised attempt, including reconstruction | 370.295s, about6.17min | 4h |
| Sampled process-tree RSS | 3,164,905,472 bytes, about2.95GiB | 8GiB |
| Sampled total-device GPU use | 3,431,989,248 bytes, about3.20GiB | 8GiB |
| Owned outputs | 337,236,639 bytes, about321.6MiB | 4GiB |

Supervisor return code0, no guard failure, one attempt, no retry.
The 25 focused implementation tests passed before execution, including
zero-step DRO/legacy-ERM parity and synthetic full-schema reconstruction.
Pylance transport was unavailable; no editor type-check success is asserted.

## 5. Scientific disposition

**Close the current anatomical-pooling + GroupDRO candidate as a negative
controlled result.** It does not provide the stable detection improvement
needed for the proposed method-paper argument. Do not select only CASIA,
drop unfavorable targets/seeds, change the primary FA bound, or present
hindsight thresholds as a deployable fix.

This does not show that DINO, regional information, or all robust training
methods are ineffective. It shows that this fixed, tested construction
does not beat its contemporary baseline consistently.

Any next method campaign needs a distinct, evidence-backed hypothesis,
closer prior-art comparison and a new explicit scope/budget decision.
No parameter sweep or more complex model is launched automatically.
The separate pre-existing FAS fine-tuning study was not modified or rerun.
Private identities, crop files, features, scores and coefficients remain
outside Git; only aggregate-safe evidence is published.
