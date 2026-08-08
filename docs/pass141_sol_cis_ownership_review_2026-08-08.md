# Pass 141 — CIS ownership-credit review (Gate 2: NONE)

This is a read-only Codex Sol review conditioned on new CIS measurements. It
did not authorize implementation or GPU work. Fable and its automatic Claude
fallback were unavailable because of the weekly limit; this is therefore a
repository-grounded Sol review, not a cross-provider consensus.

## New evidence

The In-Shop CIS coalition and paired Proxy Anchor reference each completed 60
epochs. Both reached raw best R@1 `0.9170` at epoch 41. CIS ended at `0.9149`,
while the reference ended at `0.9158` (CIS `-0.09` point on final R@1). The
single-image multi-label control was still running at epoch 21 (`0.9072`), so
it is not a deciding comparison. These are one-seed raw observations; the
required selection-corrected paired analysis is still needed before a
benchmark claim.

## Mechanism audit

For normalized CIS members `z_i=x_i/||x_i||`, coalition
`u=m^-1/2 sum_i z_i`, normalized descriptor `b=u/||u||`, and proxy logits
`q_c=tau p_c^T b`, define `h=tau sum_c (dL/dq_c) p_c`. The coalition gradient
is

```
dL/dz_i = [1/(sqrt(m)||u||)] (I - bb^T) h
```

and the input gradient is the corresponding tangent projection through `z_i`.
Every member receives the same proxy-gradient vector `h`; only the tangent
projection differs. Thus another member's positive proxy can attract a
non-owner. This is a validity warning about the operator, not a claim that one
seed proves CIS universally ineffective.

Two apparent owner-credit escapes were attacked:

1. Leave-one-out logits `d_ic=q_c(S)-q_c(S\\{i})` reduce to `tau p_c^T z_i`
   for an unnormalized linear coalition, exactly ordinary single-image proxy
   supervision. With normalization, differentiating the context recreates
   cross-owner contamination; stopping that path only makes context-conditioned
   scoring of the same per-image proxy atom. This is also covered by SRC, Deep
   Compositional Metric Learning, Metrix, and the ledger's mixed finite-
   difference rejection.
2. Repeated coded coalitions `q_rc=p_c^T sum_i A_ri z_i` decode individual
   proxy scores with `A^dagger`, again recovering ordinary supervision. A
   nonlinear decoder becomes ECOC/vector-symbolic binding, compositional
   embedding, or mixture-of-experts; hypothetical update credit becomes
   gradient-alignment/meta-learning and violates the approximate 1x budget.

The underlying identifiability obstruction is that class labels identify each
image's owner but provide no extra target for cross-image interaction terms.
Owner-separable constructions collapse to existing per-image supervision;
coupled constructions require an occupied routing, compositional, or
meta-learning mechanism.

## Verdict

**NONE at Gate 2.** No new CIS-derived owner-credit candidate is defensible
enough to implement. Do not queue a follow-up GPU method from this review.
Complete the already-running single-image and class-dropout controls and the
formal raw/selection-corrected analysis before deciding the empirical CIS
entry; this Gate-2 result is independent of that benchmark verdict.
