# Pass 135 — Codex Sol design review (2026-08-08)

This was a read-only fallback consultation because Fable and Claude were at
their weekly limit. It did not edit files or touch the DGX queue. The review
read the search protocol, the method ledger, the live CIS/SRC specifications,
the CEA failure, and the current coalition implementation.

## Verdict

No new train-time single-model/single-view metric-learning mechanism survived
the repository's provenance and mechanism-level prior-art gates. In
particular, the remaining SRC path is not currently a valid executable of its
own preregistration and must not be queued as-is.

## SRC hard-validity finding

The frozen specification requires both the normalized union-labelled coalition
and every leave-one-out residual:

`L = L_PA + lambda_u * union(bundle) + lambda_r/m * sum_k residual(bundle\\{k})`.

The current dispatcher invokes `L_PA + lambda * residual` only; it never adds
the full union term. At bundle size two, each leave-one-out residual is just the
other member's ordinary single-image target, so the average is algebraically
identical to the existing single-image control (`abs_diff=0` in a scalar CPU
check). The existing three-member test does not detect either defect. Under
the protocol's Gate-0 implementation-match rule, the preregistered SRC
expectation is void until the implementation and tests are repaired and a new
preregistration is written.

## Gate-2 triage

Three strongest constructions considered by the review were rejected before
GPU:

1. Class-shard functional-interference penalties reduce under Taylor expansion
   to class-disjoint meta-learning/gradient-alignment updates (MLDG, Reptile,
   Fish, and the ledger's DML-DC family).
2. Cross-image token bottlenecks reduce to cross-image completion or local-to-
   global distillation/support conditioning (CroCo, CrossTransformers, and
   occluded-re-ID completion), while risking acquisition shortcuts on In-Shop.
3. Ownership-coded coalitions reduce to error-correcting/vector-symbolic
   binding, routed blocks, or codebook supervision; fixed random roles add no
   supervision information and learned roles are already occupied machinery.

Activation, architecture, optimizer, sampling, geometry, and graph/gating
families are likewise covered by the existing ledger or contradicted by the
repository measurements. This is a negative research result, not evidence
that no future method can exist; it means no candidate currently clears the
required provenance and prior-art gates.

The independent review does not change the active CIS experiment. Its four-arm
In-Shop controller remains the only authorized GPU work.
