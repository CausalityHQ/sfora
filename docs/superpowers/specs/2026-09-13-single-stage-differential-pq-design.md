# Single-Stage Differential PQ Design

## Question

Can a projection and exact 24-byte product quantizer trained as one system preserve
the local score differences that determine retrieval order, where post-hoc PQ, OPQ,
additive quantization, normalization, and fixed-code correction have failed?

This is a claim-ineligible SOP official-train class-disjoint development experiment.
The official SOP test remains untouched. The implementation is generic over dense
floating embeddings and integer class identities; SOP-specific authority belongs in
the driver, not the library.

## Failure mechanism and hypothesis

The 24-byte code already contains the float method's useful neighbors: exact float
reranking inside its top 32 recovers `0.591233` mAP@R versus the `0.591265` float
result. The deficit is local order. Reconstruction-oriented quantizers allocate bits
against marginal error, while retrieval depends on differences between nearby rows.
For reconstruction errors `e_i = z_i - q(z_i)`, the live quantity is

`E[||(e_i - e_j)||^2]` for teacher-neighbor pairs `(i,j)`, not only
`E[||e_i||^2]`.

The treatment jointly trains the projection and PQ codebooks through the exact hard
ADC forward pass. Its differential arm additionally minimizes neighbor-pair error
differences. If useful teacher directions can be rotated into a quantizable geometry,
the full 768-dimensional input treatment should beat a matched control that can see
only the already-compressed 128-dimensional head output.

## Frozen arms and deployment

All arms start from the same deployed 128-dimensional direct projection and the same
24-byte PQ codebooks. They differ only by projection-update constraint and differential
weight:

| Arm | Projection update | Differential weight |
|---|---:|---:|
| `restricted_rank` | constrained to incumbent-head row space | 0.0 |
| `restricted_differential` | constrained to incumbent-head row space | 0.1 |
| `full_rank` | unconstrained | 0.0 |
| `full_differential` | unconstrained | 0.1 |

All four arms use the same 768-to-128 affine architecture, existing checkpoint,
optimizer, input rows, parameter count, and update schedule. Write the incumbent weight
as `W0`, train a same-shaped displacement `D`, and derive an orthonormal row-space basis
`Q` once from `W0`. Full arms use `W0 + D`; restricted arms use
`W0 + (D Q^T) Q`. Every arm initializes `D=0` and therefore executes the identical
original affine computation at step zero, without reconstructing inputs or `W0` through
a numerically approximate projector. Thereafter the restricted learned displacement
cannot use components in the null space of `W0`. Every arm must emit bit-identical
projected rows and PQ assignments at step zero on the complete fitting set; the receipt
records each code digest and maximum float-row delta.

The database representation remains exactly 24 unsigned bytes per vector with no
per-vector sidecar. Query-time scoring is the existing asymmetric PQ scorer: 24
query tables, 24 indexed lookups, and 24 additions per candidate. Training-only
teacher rows, neighborhood identities, and float reconstructions never enter the
database representation.

## Objective and schedule

Candidate identities are frozen before optimization from fitting classes only: the
teacher top 64, 32 compressed-exclusive rows, and 32 deterministic uniform-tail rows,
for exactly 128 candidates per fitting row. The objective is

`L = L_adc_kl + 0.25 L_float_kl + 0.1 L_reconstruction + gamma L_differential`,

where `L_adc_kl` uses the exact hard-PQ forward score and teacher neighborhood
probabilities at temperature `0.05`; `L_float_kl` keeps the trainable float projection
aligned with the same teacher order; and

`L_differential = mean(|| (z_i - q(z_i)) - (z_j - q(z_j)) ||^2)`

over all 28 unordered pairs among the first eight frozen teacher neighbors per fitting
row. Hard code selection is used in the forward pass. The existing straight-through
boundary supplies gradients to the projection and selected codewords. The differential
residual uses hard decoded codewords directly so gradients reach both the projected
rows and their selected codewords rather than cancelling through an identity
straight-through path.

The pair graph is intentionally between candidate database rows, not between the
fitting query and each neighbor. Deployment is asymmetric: the query remains float,
so its own quantization residual never enters a score. Local ordering error depends on
differences between the two competing database-code residuals. Co-neighbor pairs test
that exact quantity without inventing a quantized-query deployment path.

Use seed 0, 1,000 updates, batch size 256, AdamW, projection learning rate `3e-4`,
codebook learning rate `1e-3`, weight decay `1e-4`, gradient-norm cap `5.0`, and no
checkpoint selection. Candidate identities are never refreshed. Record objective
components, hard-code churn, per-stage utilization/entropy, neighbor differential
error, and exact exhaustive validation rankings.

## Decisions

The matched controls are the frozen float result (`0.5912648481`), PQ24
(`0.5707206723`), PQ32 (`0.5789231844`), and normalized additive diagnostic
(`0.5770697875`).

- Strong go: the preregistered primary `full_differential` arm has mAP@R at least
  `0.5875`, Recall@1 no worse than `0.8310`, and a positive paired class-cluster
  bootstrap lower bound over PQ32.
- Go: the primary `full_differential` arm has mAP@R at least `0.5830`, Recall@1 no
  worse than PQ32 minus `0.001`, and a positive paired lower bound over PQ32. The
  other arms are descriptive/mechanistic controls and are not searched for a passing
  confidence bound.
- Kill: every arm mAP@R at most `0.5771`, or training is nonfinite/collapsed.
- Positive-not-significant: the primary reaches `0.5830` but misses its paired or R1
  gate. Run only the already-frozen replication seeds; do not infer rate failure.
- Ambiguous: the primary is below `0.5830` while at least one arm exceeds `0.5771`.
  Permit no validation-guided recipe change; proceed to the preregistered rate-ceiling
  diagnostic instead.
- Mechanism support requires `full_rank - restricted_rank >= 0.004` as evidence for
  the single-stage information-access hypothesis and
  `full_differential - full_rank >= 0.002` for the differential criterion. The first
  comparison holds architecture, parameter count, initialization, optimizer, and
  output geometry fixed; only the incumbent-null-space information differs. Absolute
  quality can pass while either mechanism claim fails.

One seed is the screen. Only a go runs fixed seeds 1 and 2. No result opens the SOP
official test automatically.

## Failure interpretation and next boundary

If all four arms fail, run a deliberately claim-ineligible leaked-label 192-bit oracle
on the same code family. It trains on the evaluation-class labels solely to measure
the attainable rate ceiling. Below `0.589` mAP@R, float parity is not reachable for
this 24-independent-byte PQ family on this split; the publishable result becomes the
measured rate-quality frontier and failure diagnosis rather than a parity claim.

Do not retry decoder-side norm corrections, fixed-code query networks, OPQ-only
rotations, frozen-code residual shrinkage, or float sidecar reranking.

## Verification and execution

Library tests mutation-lock exact hard-forward loss terms, differential gradients,
finite/type/shape authority, and zero-weight backward compatibility. Driver tests lock
the four arms, step-zero equality, exact wire budget, frozen candidates, decision
thresholds, canonical no-clobber artifacts, and explicit-only CLI. A clean committed
source revision is deployed to the DGX; one monitored process runs at a time, and the
original exit/result is preserved.
