# Cars196 Fixed Local-Scaling External Falsifier

## Prospective boundary

Before reading the Cars outcome, the In-Shop-selected local scale was frozen at
`k=50` with the same score `2*cosine-density50`. The external pass rule was an
absolute Recall@1 gain of at least 0.001 and an exact paired McNemar p-value
below 0.05. No Cars label or metric selected a parameter.

## Result

- Frozen archive:
  `reports/emb/pfml_cars_alpha3_seed0_final.test.npz`
- SHA-256:
  `771ce5c8d3ceb61a200df9c5b34c54324ec92105e338c992a52c0f31ba390283`
- Rows / dimensions: 8,131 / 512.
- Stable full-order raw Recall@1: `0.7927684171688599`.
- Fixed local-scaling Recall@1: `0.805681957938753`.
- Absolute gain: `0.01291354076989304` (+1.291354 percentage points).
- Wrong→right: 364; right→wrong: 259; discordant: 623.
- Exact two-sided McNemar p-value: `2.968313422505538e-05`.
- Decision: **PASS external falsifier**.

The archive's production/independent scalar is `0.7931373754765711`, while its
registered stable-full-order scalar is `0.7927684171688599`. The independent
blockwise `np.argmax` calculation reproduced the stable-full-order value, so the
paired comparison uses that exact tie convention for both raw and corrected
rankings.

## Interpretation

The same fixed local scale improved two datasets and three frozen embedding
pairs, making the density bias a credible cross-dataset geometry defect. It
does not establish novelty: local scaling and CSLS are prior art. The result
instead motivates a new learning question—whether a per-item local-scale
potential can be amortized into the model so raw retrieval gains the correction
without observing the test gallery graph.

Execution was CPU NumPy with CUDA disabled. No checkpoint, embedding, or result
archive was modified.

