# Adapted DINO short-clip probe: completed results

Date: 2026-11-05.
Status: **completed and separately reconstructed; no demonstrated chronological
benefit over averaging. Stop this tested construction; no automatic expansion.**

The approved [protocol](019-adapted-dino-short-clip-probe.md) completed four
targets at seed20260917:16 heads and16 source-only calibrators. Full
aggregate-safe [results](../results/clip-dino-v1.json) retain all60 operating
cases and132 conditional interval records, including the inference-only shuffle
diagnostic. No new scientific run was needed to produce this report.

## 1. Matched AUROC

Percentages on usable clips; higher is better.

| Target | Single | Mean16 | Temporal | Trained shuffled |
|---|---:|---:|---:|---:|
| OULU-NPU |96.438|96.545|96.572|96.273|
| CASIA-FASD |99.923|99.936|99.944|99.953|
| MSU-MFSD |96.638|96.940|96.659|96.552|
| SiW-Mv2 |86.401|87.126|87.188|87.923|
| Descriptive equal-target macro |94.850|95.137|95.091|95.175|

Mean16 minus single gives a descriptive macro gain of0.287 percentage points.
Temporal minus mean16 gives **-0.046 points**. All four conditional paired95%
AUROC intervals for temporal-minus-mean include zero; so do all four
mean-minus-single intervals. These are one-seed, unadjusted, conditional
group-bootstrap descriptions, not proof of equivalence or population-domain
generalization.

Temporal beats the trained shuffled control on OULU, including a positive
conditional interval, but not consistently across targets: shuffled is better
on CASIA and SiW by point estimate. That isolated OULU contrast does not establish
a stable chronological gain.

With the chronological head fixed, inference shuffling yields macro AUROC
95.298%, versus95.091% in chronological order. It retains the original
chronological calibrator and policies and is **diagnostic only**, not a new
selectable system. The lack of a consistent order disadvantage further fails
to support the intended temporal mechanism.

## 2. Actual source-derived policy at the5% FA bound

Percentages below use original denominators, including shared failures.
Higher BFA accompanied by unsafe FA is not an equal-security improvement.

| Target | Mean16 FA | Mean16 BFA | Temporal FA | Temporal BFA |
|---|---:|---:|---:|---:|
| OULU-NPU |16.667|93.889|17.153|94.167|
| CASIA-FASD |1.852|96.667|1.111|95.556|
| MSU-MFSD |15.833|92.500|16.667|92.500|
| SiW-Mv2 |66.298|95.019|61.602|95.019|

Every trained arm meets joint FA<=5%, BFA>=50% in **1/4 targets**, CASIA only.
The inference-only diagnostic also passes only CASIA. Chronological training
does not add an operationally feasible target.

Temporal's SiW FA reduction from66.298% to61.602% remains far above5%.
Its target-label oracle maximum BFA at5% FA is45.977%, also below the50%
utility requirement. Oracle operation is not deployable source-only operation.
The complete export retains1%/10% sensitivities and all oracle diagnostics;
neither is used to replace the primary result.

## 3. Coverage, verification and resource use

- Original population:7,505; usable clips:7,367; shared failures:138.
- Failures:31 historical,97 insufficient duration, one native sampling-tolerance
  failure, four track jumps, one scale failure, three missing endpoints and one
  excessive tracking gap. No original was removed from FA/BFA denominators.
- Target originals:2,943; usable target clips:2,886:
  OULU1,800, CASIA353, MSU156, SiW577.
- Full canonical replay verified1,147,357 native frames.
- Actual production frozen-encoder image forwards:261,264, plus8 synthetic
  replay forwards. Encoders were not trained.
- All16 heads completed the fixed200 epochs. Selected epochs vary according
  to source validation; selection is not evidence of early termination.
- Separate CPU saved-feature reconstruction passed all60 cases and132
  interval records; every record retained2,000/2,000 valid replicates.
- Maximum reconstructed probability absolute error:0.0, tolerance1e-10.
- 53 focused engineering tests and the four-real-checkpoint synthetic CPU smoke
  passed before execution. Pylance remained unavailable.

| Resource | Measured | Approved ceiling |
|---|---:|---:|
| Supervised attempt, including reconstruction |38,191.804s, about10h36m32s|24h shared clock|
| Sampled whole-controller-tree RSS |6.733GiB|8GiB|
| Sampled total-device GPU use |4.586GiB|8GiB|
| Sampled owned output |0.966GiB|4GiB|

Return code0, no guard failure, one attempt, no retry. A bound completion
receipt also checks final wall/output closure. No experiment process remains.

## 4. Provenance and conclusion

- Implementation commit:`7a1acd2`; approved start publication:`16a8d2b`.
- Read-only FAS revision:`707e3ddf6103835c55e937cbbd84081d91f188c7`.
- Exact approved plan:
  `7bc62b991dee8f8394dcebed3c44909505a215f80349c9dd7c7c2788f5739238`.
- Private aggregate SHA256:
  `10c059bcc195924dcf77977771a92515e904abcf985d2a8b2679d1f094b5289d`.
- Private result SHA256:
  `df80f286d8721ba88614364334a897ca292970ed0f90ea7a78851191a8e963a3`.

The within-study single-frame baseline, not an earlier different crop/cohort,
is the valid comparator. These targets were already development-exposed and
there is only one training seed; unknown SiW participant dependence remains.
Reconstruction checks saved features and heads, not training replication or
a repeated full encoder extraction.

**The adapted backbone has useful ranking performance, but this frozen-CLS
projection/GRU does not demonstrate improved chronological detection or recover
safe cross-domain operation.** Do not expand seeds, tune on these targets or
present this as a successful temporal method/Q3 contribution. This rejects the
tested representation/head construction, not all video-based PAD.
