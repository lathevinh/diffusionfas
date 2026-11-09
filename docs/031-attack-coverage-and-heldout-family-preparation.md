# Attack coverage audit and held-out-family protocol preparation

## 1. Completed scope

This prepares a coverage matrix and **non-executable candidate inventories**.
It does not train models, assign new historical roles, change crop eligibility,
load pixels, decode media, parse prediction arrays or fit calibration.
Existing context payloads are hashed only to verify inherited receipt bindings.
The [previous attack diagnosis](030-siw-saved-score-attack-diagnosis.md)
remains the source of the saved-score findings.

The bounded CPU worker completed in3.005 seconds, with841,105,408 bytes sampled
process-tree RSS and2,738,328 bytes sampled private output, within30 minutes,
2GiB RSS and64MiB output. There was no guard failure or retry.

Independent reconstruction checked7505 original metadata joins, rebuilt1814
group/RGB components using separate union-find code, recounted all60 matrix
cells, four domain-feasibility folds and five candidate manifests/hashes.
Fourteen focused tests passed; three new Python files passed AST parsing.
No editor/Pylance type-check success is claimed.

Deliverables:

- [Sanitized aggregate evidence](../results/coverage-protocol-preparation-v1.json).
- [Metadata preparation implementation](../src/diffusionfas/coverage_protocol.py).
- [Bounded CLI](../scripts/coverage_protocol.py).
- [Synthetic invariant tests](../tests/test_coverage_protocol.py).

Individual keys, subject IDs, component IDs and candidate membership lists
remain private. The public evidence exposes only counts, hashes, limitations
and non-executable protocol metadata.

## 2. Dataset, subjects and preprocessing support

All roles and the paired-crop common mask are inherited unchanged.

| Dataset | Originals | Common usable | Excluded | Known subjects | Unknown-subject originals | Components |
|---|---:|---:|---:|---:|---:|---:|
| OULU-NPU |4950|4286|664|55|0|55|
| CASIA-FASD |600|367|233|50|0|50|
| MSU-MFSD |280|257|23|35|0|35|
| SiW-Mv2 |1675|1321|354|0|1675|1674|
| Total |7505|6231|1274|not pooled|1675|1814|

Subject counts refer to registered dataset-local namespaces, not globally
verified human identities. Subject counts in different family/role cells
cannot be added: the same subject may contribute multiple attack families.
Zero known SiW subjects does not mean zero participants.

OULU, CASIA and MSU subject components contain both bona fide and print/replay.
The140 components across these three datasets are mixed-class and mixed-family.
Thus component-closed family removal there can remove bona fide and the other
family too; naive reuse of subject-group removal is not a clean intervention.

SiW has1673 singleton components and one two-original component. None is
mixed-class or mixed-family. Component-closed family removal in the prepared
SiW candidates has **zero collateral bona fide/other-family removals**.
This only concerns registered video/group/exact-frame components, not people.

The inherited audit found no available cross-role video, group, selected-RGB
or checked full-media hash intersections. It was not repeated by reading
transaction files in this preparation. SiW subject independence, near-duplicate
content, re-encodings and physical instrument independence remain unresolved.

## 3. What the current data can identify

The full60-cell evidence matrix includes dataset, role, class, family,
original/common/excluded support, known/unknown subject support and component
counts. All accepted attack-family labels are known.

| Held-out dataset | Mapped target families | Usable fitting-family coverage |
|---|---|---|
| OULU |print, replay|Both covered by other datasets|
| CASIA |print, replay|Both covered by other datasets|
| MSU |print, replay|Both covered by other datasets|
| SiW |print, replay, makeup, mask, partial|Only print/replay covered|

This makes current leave-SiW-domain-out evidence a mixture of domain shift and
family-coverage shift. Makeup/mask/partial are available **only in SiW** within
this accepted cohort. There is no existing other-domain source pool supplying
those families while preserving a strict held-out SiW domain.

These are the existing project's normalized family labels, not exact PAI-device
identity or a provider-certified ontology. Different project families can
share materials or physical properties.

Three distinct claims must not be conflated:

1. **Unseen-domain with mapped-family coverage:** existing OULU/CASIA/MSU
   target folds offer this development setting; devices/subjects are not
   automatically matched by the word "covered."
2. **Within-domain unseen mapped family:** SiW train versus SiW test permits
   candidate family withholding at the video/group level, subject to the
   identity limitation below.
3. **Joint unseen-domain and added family coverage:** not identifiable from a
   coverage-addition comparison using this cohort alone. Adding SiW train
   changes the domain exposure and the estimand.

## 4. Prepared within-SiW candidates

All five families are enumerated, not selected by their saved target
performance. This does not erase prior target exposure; it is not a fresh
preregistration or unbiased benchmark.

For each held-out family H:

- Retain SiW's accepted fitting/validation/calibration/evaluation assignments.
- Remove entire H-containing components from baseline fitting, validation and
  calibration. Do not recover technical or crop-support failures.
- Keep unknown-family non-evaluation attacks out and flag any such uncertainty.
- Give both future arms the same H-free validation/calibration pools.
- Provide a separate eligible pool of usable H fitting attacks for a future
  coverage comparator; it is **not an additive training arm**.
- Retain the same full623-original/501-common target in both arms. Show H-only
  and seen-family diagnostic slices without replacing the full target.

The fixed usable bona fide pools are253 fitting,79 validation,78 calibration
and214 target, for every candidate. Candidate support:

| H | Baseline fitting attacks N | Eligible H fitting attacks | H-free validation attacks | H-free calibration attacks | Target H original/common | Target H5% error budget |
|---|---:|---:|---:|---:|---:|---:|
| Makeup |212|36|74|65|56/42|2|
| Mask |188|60|67|62|83/69|4|
| Partial |145|103|46|40|128/122|6|
| Print |217|31|74|69|56/30|2|
| Replay |230|18|79|72|39/24|1|

Counts are available pools, not selected training batches, unique independent
participants or sealed sample-size choices. Family-specific integer budgets
are descriptive empirical support, not guarantees. Only18 usable replay fitting
attacks illustrates the limited diversity for some interventions.

### Matched coverage comparison required before training

Keep the legacy-content architecture, legacy crop, loss, optimizer and
augmentation recipe as reference; do not introduce a new loss simultaneously.
The old three-source sampler needs a separately tested single-domain adapter.

For a selected candidate, construct a new immutable pair manifest:

- **A, family-unseen:** N usable non-H fitting attacks.
- **B, family-covered:** N attacks total, replacing k non-H attacks with k
  eligible H attacks, rather than adding H on top of A.
- Choose k and a score-independent component-preserving matching rule from
  fitting metadata only. Seal the exact A/B identities, affected seen-family
  proportions and replacement components before any new scores are computed.
- Match fitting bona fide identities, validation/calibration/evaluation pools,
  initialization, batch/class balance, augmentation random streams where
  applicable and optimizer-step budget. Document donor/pixel divergence caused
  by different attack identities instead of claiming identical inputs.

The current inventories **do not yet select k or seal the replacement
manifest**. Different family composition is the intended intervention; finite
sample identity/composition effects still prevent a universal causal claim.
Do not compare N versus N+H and call it a coverage-only effect.

Use the same checkpoint-selection rule on the shared H-free validation pool.
Each newly trained model must obtain its own calibrator and source quantile
threshold from the same shared H-free calibration identities; numeric
calibrators/thresholds need not equal each other or the old cross-domain ones.
Do not reuse a threshold fitted to a different checkpoint or tune using H test.

Report full-target FA/BFA at the declared1%,5%,10% calibration policies,
original and common denominators, and paired A/B changes. H-only and
seen-family slices are secondary diagnostics. Failure to improve the full
target cannot be hidden behind a favorable H-only result. A declared5% FA,
50% BFA internal utility goal is not ISO/iBeta certification.

Fixed steps, checkpoint-selection behavior, training seeds, bootstrap grouping
and inference/replay tolerances must be declared in the new controller seal.
Use paired training seeds if making claims beyond one-run feasibility;
current metadata inventories authorize no seeds, fits or GPU hours.

## 5. Execution gates and next decision

Preparation is complete; **training readiness is false for all five candidates**.

1. **Identity gate:** all active SiW originals lack subject IDs. For a credible
   subject-independent claim, obtain an authoritative participant mapping and
   rebuild/audit splits under a separately approved protocol. Do not infer
   subjects from numbered video tokens. If mapping cannot be obtained, proceed
   only after explicit acceptance of a video/group-level development diagnostic,
   never a subject-disjoint publication claim.
2. **Estimand gate:** explicitly choose within-SiW family coverage if using these
   candidates. For a strict unseen-domain claim with makeup/mask/partial source
   coverage, suitable independent source data is missing from the current cohort.
   Acquisition/access changes need their own authority and provenance audit.
3. **Matching/engineering gate:** seal k and component-preserving A/B manifests,
   implement/test the one-domain sampler, paired fixed-step schedule and
   synthetic checkpoint/replay checks. The current sampler requires exactly
   three sources and must not silently fall back.
4. **Budget/approval gate:** select the bounded experiment, cap fits/resources,
   seal current code/inputs/runtime and obtain fresh explicit training approval.
   Five candidate pairs are not ten authorized fits.

**Recommended next action:** resolve whether authoritative SiW participant
metadata can be obtained before investing in a new GPU study. If not, the
bounded within-SiW video/group-level coverage diagnostic is technically
preparable, but its weaker scientific claim must be accepted explicitly.
Do not claim another crop/anatomical/tail-loss grid fixes the identified gap.

For an independently approved metadata-only rerun, use a fresh private root:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n fas python scripts/coverage_protocol.py \
  --diagnosis-root /path/to/private/siw-attack-diagnosis-v2/data \
  --completed-root /path/to/private/spatial-crop-run-v1 \
  --private-root /path/to/private/new-coverage-protocol
```
