# Pass 138 — Codex Sol fallback review (2026-08-08)

## Scope

This was a read-only fallback consultation using Codex Sol because the Fable/
Claude service was unavailable. It read the current ledger, CIS/SRC
preregistrations, recent failure memos, and the coalition implementation. It
did not edit code or start a GPU job.

## Verdict

No new train-time single-model/single-view metric-learning candidate survived
the repository provenance plus mechanism-level prior-art gates. No additional
GPU run is authorized from this review.

## CIS pre-result validity warning

The running CIS implementation forms a normalized sum of member embeddings and
applies one union-target BCE. Its proxy gradient is therefore shared by all
members (apart from each member's tangent projection). A positive for class
`a` can pull a non-owner member from class `b` toward `a`; this is the
ownership/credit-assignment defect previously noted in Pass 68. Existing CPU
tests establish permutation invariance and finite gradients, but do not test
owner-specific credit assignment. This is a validity warning, not yet a
benchmark verdict: the already-running CIS controller must finish its paired
controls before the implementation and effect are judged.

## Candidate triage (Gate 2)

* Mixed finite-difference coalition supervision is algebraically zero on linear
  proxy logits and otherwise reduces to interpolation/smoothness regularizers
  (Manifold Mixup and related compositional metric learning).
* Transformation-order commutator supervision is augmentation consistency or
  equivariant representation learning; no repository measurement ties it to a
  surviving retrieval deficit.
* Virtual class-shard updates that penalize harmful cross-class changes reduce
  to gradient-alignment/meta-learning families (MLDG, Fish, Fishr).

These are recorded as rejected constructions, not GPU candidates. The review
therefore leaves CIS as the only active experiment and SRC as an unresolved
conditional path requiring an implementation repair and new preregistration.
