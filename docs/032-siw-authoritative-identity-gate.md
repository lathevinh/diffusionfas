# SiW identity gate: official subject language is not a recovered identity map

## 1. Decision reached

The bounded public-documentation review on2026-11-09 **did not recover an
authoritative video-to-participant mapping** for the accepted SiW-Mv2 cohort.
This is a scoped negative finding, not proof that the dataset steward has no
such mapping.

The official project describes train/test partitioning in terms of subjects.
That is relevant provider evidence and must not be ignored. However, the
reviewed documentation and code do not supply a machine-checkable link from
each accepted original video to a stable participant identity across videos,
attack types and live/spoof recordings.

Consequently, retain both statements:

- The provider describes its protocols in subject terms.
- Our accepted cohort's1675 active originals still have unknown subject IDs;
  we have not independently verified participant-disjoint roles.

Do not infer subjects from numeric video suffixes, equate a video folder with
a unique person, infer identities by face recognition, or replace unknown
subject IDs with generated values. No source manifest or historical split was
changed by this review.

## 2. Reviewed evidence and its limits

### A. Official project protocol description

[Official root README at the pinned revision](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/8667dbcd316b38141729c057adf7517fe0602608/README.md#2-siw-mv2-protocols)
describes:

- Protocol I: known attacks, with live and spoof subjects partitioned into
  train and test.
- Protocol II: one held-out attack type out of14, with a stated live-subject
  train/test allocation.
- Protocol III: attribute/domain partitions.

This establishes the authors' declared protocol intent. It does not expose a
row-level identity table in the reviewed README.

The current root README was also checked for changes relevant to this
question. No recovered participant mapping was identified there.

### B. Official dataset README and existing pinned cache

[Dataset README](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/8667dbcd316b38141729c057adf7517fe0602608/source_SiW_Mv2/README.md)
calls protocol-list entries subjects and notes repeated entries used for
balancing. It also describes raw videos and per-video extracted-frame folders.

The locally cached README remains governed by the existing reference binding
`3168b03e638fe52052de70bbeb221352f6e2418124a2e1bf0af59820b857b5d2`.
Neither the naming convention nor repeated-list weighting is sufficient to
recover participant equivalence across videos. Repeated entries must not be
treated as additional independent subjects.

### C. Loader, configuration and preprocessing semantics

[Pinned configuration](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/8667dbcd316b38141729c057adf7517fe0602608/source_SiW_Mv2/config_siwm.py)
reads protocol token lists and constructs image-folder paths. Its Protocol-II
branch filters the selected attack type from training and enumerates matching
attack folders for evaluation.

[Pinned loader](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/8667dbcd316b38141729c057adf7517fe0602608/source_SiW_Mv2/dataset.py)
works with those folders and their frames. The existing cached loader is bound
by `0003fcb00cb08da5ea61cec729fcaedc108cc6ffb72ff977f8c8c3a1d3b97f66`.

[Reviewed preprocessing](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/preprocessing.py)
creates an output folder from the input video's filename stem. The reviewed
code does not provide a separate participant identity-resolution step.
It was read as public source only, not imported or executed.

These observations explain why the published loader can run without a
participant map. They neither prove leakage nor establish its absence.

### D. Targeted public code/issue search

GitHub code search for `subject` in the official repository returned five
indexed files: the root/dataset READMEs, dataset loader, dataset configuration
and multi-domain configuration. The reviewed hits did not yield a SiW-Mv2
video-to-participant map. A separate exact-field search for `subject_id`,
`participant_id` and `identity_mapping` returned zero indexed hits.

These are search-coverage limits, not an exhaustive audit of every artifact or
of access-controlled release material.

[CHELSEA234/Multi-domain-learning-FAS#22](https://github.com/CHELSEA234/Multi-domain-learning-FAS/issues/22)
reports missing live list entries. The retrieved issue was open; the search
response reported zero comments. It provides no maintainer-confirmed identity
mapping and does not authorize changing our frozen population. Its report may
concern a different local release. Existing list/archive intersection and
duplicate dispositions remain unchanged.

The official lab dataset page was located through search, but direct HTTPS
and HTTP retrieval both failed at the transport layer during this review.
Search snippets were not treated as independently retrieved authoritative
identity metadata. The paper supplement and access-controlled download
contents were not retrieved; their possible metadata remains unverified.

## 3. Five-family preparation is not official Protocol II

The [prepared candidates](031-attack-coverage-and-heldout-family-preparation.md)
withhold one of five project-normalized families and retain the accepted
Protocol-I-intersection test population. They also reserve separate
validation/calibration roles inherited from our earlier study.

The official unknown-attack protocol instead distinguishes14 attack types
and uses its own specified population and selection rules. Therefore:

- Do not label our five candidates as official SiW-Mv2 Protocol II.
- Do not compare their results directly to official Protocol-II numbers as
  though populations, held-out units and calibration conventions matched.
- Leave-one-attack-out evaluation already exists in the official project;
  merely proposing family withholding is not a methodological novelty claim.
- Reproducing official Protocol II would require a separate population/role/
  denominator reconciliation, not a flag change in the current controller.

## 4. Information needed from the dataset steward

Request a **pseudonymous** mapping or an equivalent verifiable group manifest;
real names, contact details, face embeddings and personal records are not
needed. This is a draft request only; nothing was sent externally.

Suggested request:

> We are auditing independence for a SiW-Mv2 research protocol. Can you provide
> a pseudonymous video-to-participant equivalence mapping, or an official
> manifest certifying the corresponding grouping and train/test isolation?
> Please clarify whether the same participant can appear in multiple videos,
> attack types and live/spoof recordings, and what identity the mapping denotes
> for impersonation, replay/print and mask presentations. We also need the
> release/protocol version and instructions for reconciling missing list
> entries without assuming numbered video tokens are participant IDs.

For attack presentations, clarify whether an identifier denotes the presenter,
the depicted/source identity, or a target identity being impersonated. Those
are not interchangeable independence units. Accept equivalence-group metadata
only with provenance and a declared semantic meaning.

If supplied, validate coverage, duplicate keys, conflicting identities,
cross-role participant intersections and consistency with the frozen original
population. Any necessary role changes require a new protocol and invalidate
unqualified reuse of old training/calibration results. A hash binds bytes;
provider provenance is still needed to support their meaning.

## 5. Next gate: explicit scientific scope, not automatic GPU work

There are two defensible next paths:

1. **Identity-first:** obtain the authoritative mapping or grouping
   certification, then audit it before constructing subject-independent
   training manifests. No map has been supplied or recovered yet.
2. **Limited development diagnostic:** explicitly accept that identity is
   unresolved, and prepare component-preserving, equal-cardinality A/B
   manifests for a within-SiW video/group-level coverage study. Report that
   limitation prominently; do not call it subject-disjoint, unseen-domain,
   official Protocol II, an unbiased final test, or a certification result.

The second path needs an explicit owner scope decision. Even that decision
would authorize only the stated preparation work, not GPU training. Sampler
adaptation, matched update scheduling, synthetic engineering checks and an
exact resource-bounded scientific approval remain separate gates.

## 6. Execution and publication limits

This documentation-only review involved public web/code/issue queries and
existing text metadata. It performed no model loading, inference, training,
calibration, image/video decoding, dataset download or identity inference.
No private video IDs, participant information, scores or assets were sent to
public search services. The public queries used only dataset/project names
and generic identity/protocol terms.

There are no new scientific results or statistical tests. The related
controller and prepared inventories are unchanged. Documentation-only
publication requires link/format checks, not a rerun of PAD inference.
