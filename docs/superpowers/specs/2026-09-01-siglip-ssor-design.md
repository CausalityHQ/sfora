# SigLIP Seen-Span Occupancy and Restoration Diagnostic Design

Date: 2026-09-01

## Purpose

The seed-17 pooled SigLIP control reaches 97.9813% leave-one-out Recall@1 on
optimization classes but only 94.5375% on class-disjoint clean validation. SSOR
tests one precise explanation for that gap: fine-tuning concentrates the deployed
head on the low-rank simplex spanned by seen-class means, while transferable
fine-grained evidence remains in the orthogonal complement and is underweighted.

SSOR is an optimization-only, claim-ineligible diagnostic and a deployable linear
head repair. It uses no token matching, gallery reranking, evaluation features,
test-time fitting, new supervision, or tower update. A pass authorizes one sealed
clean evaluation of one restored 512-dimensional head; it is not itself a SOTA
claim.

## Inputs and capability boundary

The diagnostic consumes the authenticated seed-17 final checkpoint and one
optimization-only cache containing, in identical example order:

- the trained tower's finite fp32 1152-dimensional attention-pooler outputs;
- the checkpoint's finite fp32 unit-normalized 512-dimensional deployed
  descriptors;
- contiguous int64 labels for exactly the optimization classes; and
- unique example IDs plus the source, checkpoint, cache, feature, label, and
  projection-weight digests.

The dedicated cache producer (not the existing three-role cache builder) restores
the checkpoint through the existing strict control loader and accepts only the
optimization example tuple. It performs one deterministic evaluation pass after
the three-seed control is terminal and rejects every label outside the fixed
optimization class set. It has no clean, burned, official-test, network, or
arbitrary checkpoint capability. The CPU diagnostic accepts a validated
`FeatureSplitAuthority` whose role is exactly `optimization-train` and whose
`official_test_access` is false. The cached descriptors must match a CPU-float64
recomputation of the row-normalized product of cached fp32 pooler features and the
sealed fp32 bias-free head with maximum per-row cosine deviation at most `1e-5`;
a parallel unbound descriptor source is invalid.

## Seen-class span

For one fit-label set, let `z_i in R^512` be the deployed unit descriptor. Compute
one unit-normalized mean per fit class and retain the **uncentered** class-mean
matrix. This keeps the global mean/hub direction inside the seen span rather than
silently treating mean suppression as complement restoration. If there are `C`
fit classes, the matrix must have exactly rank `C` under the fixed deterministic
CPU-float64 relative cutoff `1e-10`; a deficient or ambiguous boundary is invalid.

Let `Q` contain the retained right singular vectors and define the orthogonal
projector

`P = Q^T Q`, where the retained right singular vectors are rows of `Q`.

The executable verifies shape `512x512`, rank and trace `C`, symmetry,
`P^2=P`, finite entries, and operator eigenvalues in `[0,1]` within `1e-10`. It
also proves `P m_c = m_c` for every unit class mean and constructs a deterministic
nonzero probe orthogonal to the means whose projected norm is at most `1e-10`.
It records mean span and complement energies and their classwise medians. These
occupancy statistics are descriptive; only integer retrieval counts decide the
gate.

## Restoration family

For fixed positive `beta`, restore one descriptor by

`z_beta = normalize(P z + beta (I-P) z)`.

The identity comparator is `beta=1`. The only candidate grid and tie order is
`{1.0, 0.5, 0.75, 1.25, 1.5, 2.0}`. Values below one suppress the complement;
values above one restore it; identity-first ties express a valid null. Because
normalization follows the linear map, deployment is exactly one bias-free head

`W_ssor = (P + beta(I-P)) W_control`,

followed by the existing unit normalization. Output width, one-view execution,
cosine retrieval, and every tower parameter remain unchanged.

## Nested class-disjoint selection

Reuse the four outer folds from `build_sfq_fold_schedule`, constructed and sealed
specifically from the 512-dimensional deployed descriptors. For each outer fold:

1. fit the outer projector using only outer-fit labels;
2. derive an authenticated fp32 subset and ordered-ID authority, bijectively
   relabel its classes, and call the same SFQ nearest-class-pair scheduler to form
   three deterministic inner folds; seal each derived authority and schedule;
3. for each candidate beta, fit an inner projector without the inner validation
   labels and score those validation queries against the complete outer-fit
   gallery, so fit classes remain realistic distractors;
4. choose beta by aggregate inner integer hits, then the registered grid order;
5. score identity, every registered beta, and the selected beta for untouched
   outer-validation queries against the complete 49-class optimization gallery.

No outer validation outcome selects its own beta or projector. Every optimization
class appears in exactly one outer validation fold and in exactly one inner
validation fold for each outer fit partition. Lowest-row ties and float64 cosine
scoring reuse the SFQ authority and receive an independent scalar replay.

The single deployment beta must be selected by at least three of four outer
folds. Otherwise the run is a valid non-consensus failure and no head is sealed.
The deployment projector is refit once on all optimization classes. The result
seals every-beta outer scores, fold selections, consensus beta, projector/head
digests, and all integer counts before any clean capability exists.

Inner selection, outer evidence, and final deployment necessarily use projectors
fit on increasing class counts. The experiment treats this as the standard nested
fit-size extrapolation, not as an equality of subspaces. It therefore seals the
complete outer-fold score curve for every registered beta, requires three-fold
agreement on one beta, and records inner, outer, and all-class projector ranks and
complement energies. These are the preregistered transfer checks; no post-result
beta adjustment is permitted.

## Decision

The diagnostic is valid only when every outer and inner projector satisfies the
rank and projector invariants, all evidence is finite and deterministic, and the
vectorized/scalar retrieval counts agree. Mixed-side selections, identity
selection, or lack of three-fold consensus are valid failures, not invalid runs.
The diagnostic reports identity error count and is decision-eligible only with
at least 40 aggregate identity errors. It passes only when:

- nested SSOR aggregate Recall@1 exceeds the identity comparator by at least
  2,000 ppm;
- SSOR strictly beats identity in at least three of four outer folds; and
- no outer fold loses more than 10,000 ppm.
- the consensus beta is non-identity and selected by at least three folds.

A passing result authorizes one separately isolated clean evaluation of the sealed
`W_ssor`. The clean result must beat the paired seed-17 control by at least 5,000
ppm to remain a candidate, but that floor reaches only about 95.04% and is not the
97.4% release target. No beta, rank, fold, threshold, or projector may be revisited
after clean evidence.

A valid failure rejects this exact seen-span restoration family on the trained
SigLIP representation. Complement-only failure means the remaining gap is
direction quality or capacity rather than span occupancy and routes effort to the
already prepared SFQ, depth-readout, RSTA, and semantic-gradient lanes. It does not
reject nonlinear or tower-level transfer methods.

## Cost and integration

The only GPU work is one deterministic seed-17 optimization-cache pass, shared
with other post-control head diagnostics. All span fitting and retrieval scoring
are CPU float64 over roughly 4,100 by 512 matrices and should finish in minutes.
Persistent tensors remain below 256 MiB.

- `src/sfora/siglip_ssor.py`: projector construction, nested selection, scoring,
  deployment head, canonical result, and independent validation.
- `scripts/diagnose_siglip_ssor.py`: strict optimization-only local CLI.
- `tests/test_siglip_ssor.py`: algebra, isolation, nested folds, ties, gates,
  canonical mutation matrix, and deployment equality.
- `tests/test_diagnose_siglip_ssor.py`: cache/checkpoint binding and capability
  refusal.

Implementation is TDD-first and receives focused tests, dependency-complete
Python assurance, scoped Ruff/format/compile checks, and independent read-only
review before the cache pass. Scientific execution remains fenced while the
original three-seed control is active.
