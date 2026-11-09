# SiW saved-score attack diagnosis: family coverage dominates current failures

## 1. Scope, completion and verification

This is a **CPU-only diagnosis of the completed24-fit crop study**, not a new
experiment. Existing scores, source calibrators, thresholds, original roles
and the shared crop mask are unchanged. Full SiW remains primary.

The corrected run completed in47.063 seconds, with849,297,408 bytes sampled
process-tree RSS and1,396,633 bytes sampled private output. Caps were30 minutes,
2GiB RSS and64MiB output. No model loading, inference, image/video decoding,
training, calibration fitting, downloads or new bootstraps occurred.

The first attempt stopped after24.433 seconds because the diagnostic compared
an all7505-row role-component map to the2943-row **target-bootstrap** commitment.
All2943 shared values agreed; the key scopes differed. The contract was fixed,
the failed root retained, and the owner explicitly approved one corrected run
into a fresh root. There was no resource-guard failure or automatic retry.

Independent reconstruction checked:

- All7505 metadata/role/eligibility joins and all completed input receipts.
- All36 global operating cases: six systems, three alphas, two denominators.
- All378 attack-slice counts and target-label oracles, using independent
  strict-threshold insertion counts rather than the diagnostic oracle helper.
- Existing source-selected thresholds and source calibration acceptance.
- Video, subject, group, selected-RGB and checked full-media hash intersections
  between roles, without reading media payloads.

Seventeen focused tests passed, including order/length/probability joins,
unknown taxonomy, retained failures, strict ties, small-group budgets and the
target-only bootstrap-map regression. Python syntax was AST-validated; no
external-workspace Pylance type-check success is claimed.

See [aggregate evidence](../results/siw-attack-diagnosis-v2.json),
[implementation](../src/diffusionfas/siw_attack_diagnosis.py),
[bounded CLI](../scripts/siw_attack_diagnosis.py) and
[tests](../tests/test_siw_attack_diagnosis.py). The public export contains no
video/subject IDs, images, checkpoints, individual scores or diagnostic
threshold values.

## 2. Taxonomy provenance and actual source coverage

The integrated manifest omitted attack-family/type fields. They were restored
by validated original-identity joins, not guessed from unverified filename
patterns:

- SiW: hash-bound metadata inventory and frozen Protocol-I intersection
  membership, checked against the existing metadata adapter.
- MSU: hash-bound native metadata inventory.
- OULU: existing official video-ID metadata parser.
- CASIA: existing pinned Bob-reference media-path parser, including class,
  subject namespace and split checks.

There are **zero unknown attack-family labels** in the SiW target or usable
fitting attacks. The project mapping contains14 SiW attack types grouped into
print, replay, makeup, mask and partial.

For the held-out SiW fold, fitting uses CASIA+MSU+OULU. All usable source
fitting attacks map to **print/replay only**:

| Source | Print original/common | Replay original/common |
|---|---:|---:|
| CASIA |72/42|36/24|
| MSU |18/16|36/29|
| OULU |720/625|720/591|
| Total |810/683|792/644|

Validation and calibration also contain only those two mapped families.
Makeup/mask/partial are absent from this fold's usable fitting attacks.

**These are project-normalized family labels, not a provider-certified PAI
ontology.** "Source-family-covered" does not prove identical devices, materials,
subjects or capture conditions. "Source-family-unseen" means absent under this
mapping in usable fitting data; it is not a claim of physically unprecedented
PAIs. In particular, a paper mask and a cut/warped photograph may share physical
properties despite different mapped families.

## 3. Full denominator and shared preprocessing failures

The unchanged target contains623 originals:362 attacks and261 bona fide.
Common usable support is501:287 attacks and214 bona fide. There are122
shared nonacceptances: four historical failures, one geometry failure and117
real-pixel support exclusions. They are not relabelled or recovered.

| Attack coverage | Original attacks | Common attacks | Excluded | Exclusion rate |
|---|---:|---:|---:|---:|
| Source-family-covered: print/replay |95|54|41|43.158%|
| Source-family-unseen: makeup/mask/partial |267|233|34|12.734%|
| All attacks |362|287|75|20.718%|

Unseen mapped families comprise73.757% of original attacks and81.185% of
usable attacks. The markedly higher print/replay exclusion rate prevents a
clean causal comparison between attack families. Original and common views
must both be shown; exclusions reduce original FA through nonacceptance, not
better score separation.

## 4. Primary5% source policy: errors concentrate on absent families

Percentages use the fixed transferred source threshold atalpha=1/20.
"Unseen share" divides unseen-family false accepts by all target false accepts.

| System | Covered FA original/common | Unseen FA original/common | Unseen share of false accepts |
|---|---:|---:|---:|
| legacy reference |3.158/5.556|80.524/92.275|98.624%|
| legacy content |3.158/5.556|82.397/94.421|98.655%|
| legacy anatomical |3.158/5.556|83.895/96.137|98.678%|
| square reference |14.737/25.926|86.142/98.712|94.262%|
| square content |2.105/3.704|82.022/93.991|99.095%|
| square anatomical |0.000/0.000|77.528/88.841|100.000%|

For **legacy content**, full-target FA is223/362=61.602%:
**220/223 false accepts** are makeup/mask/partial, versus3/95 on print/replay.
Its original BFA is213/261=81.609%; conditional BFA is213/214=99.533%.
This is not a safe full-target system.

The covered subset is not uniformly solved: legacy content replay alone is
2/39=5.128% original FA, and combined covered common FA is3/54=5.556%.
Square reference also fails covered attacks substantially. Thus the finding
is **dominant absent-family failure**, not "all known attacks are solved."

### Legacy-content family breakdown

All family oracles below share the same261 original bona fide samples.

| Family | Original/common attacks | Accepted attacks | FA original/common | Oracle original BFA at5% |
|---|---:|---:|---:|---:|
| Print |56/30|1|1.786/3.333|81.609%|
| Replay |39/24|2|5.128/8.333|81.226%|
| Makeup |56/42|39|69.643/92.857|46.743%|
| Mask |83/69|63|75.904/91.304|65.517%|
| Partial |128/122|118|92.188/96.721|26.437%|

### Legacy-content recorded attack types

| Type | Original/common attacks | Accepted attacks | Original FA |
|---|---:|---:|---:|
| Paper |56/30|1|1.786%|
| Replay |39/24|2|5.128%|
| Makeup_Co |24/18|18|75.000%|
| Makeup_Im |22/17|14|63.636%|
| Makeup_Ob |10/7|7|70.000%|
| Mask_Half |20/16|12|60.000%|
| Mask_Mann |14/12|12|85.714%|
| Mask_Paper |6/5|5|83.333%|
| Mask_Silicone |8/5|5|62.500%|
| Mask_Trans |35/31|29|82.857%|
| Partial_Eye |23/23|21|91.304%|
| Partial_Funnyeye |64/63|62|96.875%|
| Partial_Mouth |12/11|10|83.333%|
| Partial_Paperglass |29/25|25|86.207%|

These are descriptive counts, not attack-type population guarantees.
Several types have fewer than20 originals; their empirical5% integer error
budget is zero. No new group confidence intervals or seed repeats were run.

## 5. Why threshold retuning alone is not the next fix

At5%, legacy content source calibration FA is0/22 on CASIA,0/18 on MSU
and22/443=4.966% on OULU, using usable source calibration denominators.
The transferred source policy does not expose these absent target families.

Even the stricter1% source policy still yields48.895% full-target original FA
and79.693% original BFA. At10%, these are64.088% and81.609%. These are
reconstructions of declared thresholds, not new threshold searches.

The target-label5% oracle maximum original BFA is:

- Full SiW:40.613%.
- Covered print/replay subset:81.609%.
- Unseen makeup/mask/partial subset:30.268%.

Oracles use target labels and are **nondeployable diagnostics**. Each slice
has its own integer attack-error budget and may choose a different threshold;
they cannot be averaged or assembled into a deployable policy. The full-target
oracle remains below the internal50% BFA utility goal, consistent with
insufficient low-FA separation rather than a calibration-only problem.

## 6. Independence audit: what is and is not established

Across the accepted four-role partitions, no checked shared video token,
group, selected-frame RGB hash, or historically successful full-media hash
crossed a role boundary. Source preparation's fixed group/content components
and target-bootstrap commitments were reconstructed.

Known subject identifiers do not cross roles on OULU/CASIA/MSU under their
registered namespaces. This is not a new certification of provider identity
semantics or cross-dataset participant independence.

**All1675 active SiW originals lack subject IDs**, including all623 target
originals. Video tokens must not be interpreted as subjects. SiW remains
video/group-disjoint under available metadata, not verified subject-disjoint.
Its sensor/session/environment/instrument metadata is also insufficient to
attribute the observed errors to particular cameras or lighting conditions.

The full-media hash audit checked7474 historical detector-success records.
The remaining31 originals were not checked for full-media hashes; the public
field is `full_media_hash_unchecked_originals`, not proof that their underlying
records lack hashes. The audit does not detect near-duplicates, altered
re-encodings, shared physical attack instruments or repeated people without
subject metadata. Exact hash isolation alone cannot establish those properties.

## 7. Next decision

1. **Pause crop/anatomical grids and a blind tail-loss run.** Across all six
   systems94.262-100% of false accepts come from the absent mapped families.
   A source-only objective change might help, but this evidence does not
   justify expecting more print/replay training to solve makeup/mask/partial.
2. Frame the next controlled study around **attack coverage and genuinely
   declared held-out families**, retaining full SiW as the current development
   result. Covered-only subsets are diagnostics, not an easier replacement
   benchmark or an ISO pass claim.
3. Before training, specify whether the next claim is unseen-domain,
   unseen-attack or both, and find admissible source coverage/identity metadata.
   Adding SiW training attacks changes the leave-SiW-domain-out estimand; do not
   do it silently or call the resulting SiW test unseen-domain.
4. A new family-held-out protocol may be prepared from existing labelled
   metadata, but its exact cohort, role independence, one selected objective,
   matched baseline and training budget need a separate approval. Targets are
   already development-exposed and SiW subjects remain unresolved.

There is no new GPU authorization, model-selection claim, causal explanation,
ISO30107-3/iBeta pass claim, or unbiased final-test result here.

The subsequent [coverage/protocol preparation](031-attack-coverage-and-heldout-family-preparation.md)
has now verified a60-cell metadata matrix and prepared five non-executable
within-SiW family candidate inventories. Training remains blocked on the
identity/estimand decision, matched replacement manifest, sampler adaptation
and a fresh bounded approval.

For a separately approved metadata-only rerun, use a fresh private root:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n fas python scripts/siw_attack_diagnosis.py \
  --completed-root /path/to/private/spatial-crop-run-v1 \
  --private-root /path/to/private/new-siw-attack-diagnosis
```
