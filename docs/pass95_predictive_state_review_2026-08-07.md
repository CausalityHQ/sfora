# Pass 95 — predictive-state / reservoir candidate (2026-08-07)

## Dead at Gate 2

The repository's stable same-wrong identity overlap (0.6476) suggests that a
descriptor could separate persistent identity evidence from transient
within-class state by predicting a future/augmented state and using the
prediction residual as a training signal.  A reservoir or predictive-coding
head was considered as a train-time-only auxiliary, with the deployed vector
unchanged.

This does not survive the prior-art gate.  Introspective DML already augments
the deployed semantic embedding with an uncertainty state and trains on
semantic ambiguity; Deep Causal Metric Learning already learns invariant
embeddings from intervention/environment structure; and the repository's
self-supervised-ranking precedent explicitly predicts controlled transforms to
preserve intra-class structure.  A fixed reservoir changes the optimizer or
feature parameterization, while a learned predictor is an auxiliary
reconstruction/distillation objective.  Neither is a new similarity-learning
mechanism under the protocol.

No code or GPU run occurred.
