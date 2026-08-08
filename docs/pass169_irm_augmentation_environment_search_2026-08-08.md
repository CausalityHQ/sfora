# Pass 169 — cross-augmentation IRM/Gram invariance (NONE before GPU)

## Motivation

The corrected In-Shop evidence has a four-seed frozen-final baseline of
`0.9153889436 ± 0.0013195712`, a between:local error ratio of `3.05–3.22`, and
an unseen-minus-seen positive-similarity gap of `-0.04968`. An independent
search asked whether class-conditional Gram geometry should be invariant across
augmentation environments, using an IRM-style variance penalty over proxy
gradients/Gram matrices.

## Gates 1–2

Gate 1 is unresolved: the repository has no verified per-augmentation
environment artifact showing that this statistic predicts the transfer gap.
The aggregate error measurements are insufficient provenance for an
environment-invariance operator.

Gate 2 kills the mechanism. Invariant-risk minimization (Arjovsky et al.,
ICLR 2020), Deep Causal Metric Learning (Deng and Zhang, ICML 2022), and
transformed-attention/augmentation-consistency metric learning already enforce
environment-invariant representations or attention. A leave-one-out
cross-fitted class-prototype variant reduces to Prototypical Networks,
ProxyNCA, or cross-batch memory.

## Decision

`NONE` before GPU. This is a protocol-valid negative scoped to
cross-augmentation environment invariance; it is not evidence that all
environment-aware methods are impossible. No implementation or GPU run was
authorized.
