# In-Shop published-checkpoint diagnostic 296

Date: 2026-08-03. Written before downloading or evaluating the authors' checkpoint.
This is a reference-fidelity diagnostic, not a candidate-method experiment.

## Trigger

The corrected seed-0 official-recipe run produced raw best-over-training R@1
`0.9039246026164017` (12,852/14,218 queries) and frozen-final R@1
`0.9020959347306231`. The raw value is formally outside the registered
`[0.904, 0.934]` interval. The miss is two queries: because R@1 lies on a
`1/14,218` grid, the first attainable value at or above `0.904` is
`12,854/14,218 = 0.9040652693768463`.

Source comparison against Proxy Anchor revision
`51db57031e38f75c03f69bbdfad1a3233afd9787` has not found a material mismatch in
the official partition, BN-Inception architecture/checkpoint, preprocessing,
GAP+GMP head, proxy loss and initialization, AdamW groups, one-epoch warm-up,
BatchNorm mode, gradient clipping, StepLR, epoch count, or per-epoch test
evaluation. The upstream program fixes random seed 1, whereas the registered
local run deliberately used seed 0; the published `0.919` is a single checkpoint
result without a seed distribution.

## Locked diagnostic

Download only the In-Shop checkpoint linked by the pinned upstream README. Record
its SHA-256. Load its `model_state_dict` into the vendored, hash-pinned
BN-Inception implementation and evaluate the unchanged local official query and
gallery partitions with deterministic reference preprocessing. Persist query and
gallery labels, IDs and normalized embeddings, then independently recompute:

- the upstream strict-negative-rank R@1 (`# negatives with similarity strictly
  greater than the best positive < 1`);
- canonical float64 nearest-neighbour R@1;
- float64 cosine and exact-tie-expected R@1;
- the already locked partition counts, identity/path disjointness and content
  profile.

No training and no candidate implementation are authorized by this diagnostic.

## Predictions and decision rule

The primary-source README reports R@1 `0.919`. Predict the upstream scorer lands
in `[0.917, 0.921]`; the narrow tolerance allows scorer/tie and library-version
effects but not a training-sized discrepancy.

- **Inside `[0.917, 0.921]`:** the local data, architecture, checkpoint loading,
  preprocessing and evaluator are jointly validated. The seed-0 interval failure
  remains a formal failed preregistration, but it is localized to training
  stochasticity/runtime rather than an evaluation-path bug. Do not reinterpret
  the failed interval as passed. Establish the corrected local frozen-final seed-0
  value as the paired control for subsequent same-seed experiments, while reporting
  the published-checkpoint result separately.
- **Outside `[0.917, 0.921]`:** method screening remains blocked. Diagnose the
  checkpoint key mapping, source data bytes/order, preprocessing/library semantics
  and scorer disagreement before any training.

The outcome cannot revise these bounds or justify a novel-method claim.
