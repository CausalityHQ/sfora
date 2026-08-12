# CADR train-only result: KILL

Cosine-Anchored Diagonal Boundary Reweighting failed its prospectively frozen
train-only gate. The official In-Shop query/gallery archives were not opened,
and no Stage-B result was computed.

## Registered execution

- Design commit: `a855f26`, prospectively clarified by `719cf74`.
- Plan commit: `270c883`.
- Executing source commit: `88700c7`.
- Train input SHA-256:
  `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- Result: `reports/generated/inshop_cadr_train_gate.json`, SHA-256
  `e3982085a0d55b2276981a2dc65bb3ccbba427ec71035e43eb444782c58c6258`.
- One CPU process, CUDA hidden, OMP/MKL/OpenBLAS fixed to one thread, exit 0.
- Output is a regular mode-0600, one-link file; no atomic temp remains.

## Outcome

The deterministic split produced 42,316 fit positives, 147,740 fit hard
negatives, 10,617 validation positives, and 35,381 validation hard negatives.
The selected lambda was `0.0001`. The learned weights had mean
`22.2583797431717` and standard deviation `7.887220778449924`, so the fit found
a large nonuniform diagonal signal.

That signal did not transfer even to label-disjoint validation pairs:

| Arm | Balanced validation log loss |
|---|---:|
| CADR | 0.13540867077747953 |
| Platt-calibrated raw cosine | 0.11651901001865922 |
| selected diagonal WCCN | 0.11677725650255413 |

The lambda, positive-mean, and contrast predicates passed. Both required
generalization predicates failed: CADR did not beat raw cosine by 1% and did
not beat WCCN by 0.5%. The registered Stage-A decision is therefore `KILL`.

## Interpretation

The result separates train fit from transferable similarity. Source-class
hard-boundary logistic regression strongly reweighted coordinates, but that
reweighting overfit the source identity boundaries and degraded held-out-label
calibration. This is consistent with the repository's earlier warning that
fitting a metric on encoder-seen identities need not identify open-set
headroom.

CADR is closed under the registered diagonal basis, hard-pair mining, lambda
grid, and objective. It may not be rescued by another lambda, normalization,
pair cap, or nonlinear head on the official pair. Because Stage A failed, the
query/gallery SHA-bound files remained outside the process interface and Stage
B was not run.

Before execution, 9 focused tests passed with Ruff, `py_compile`, and diff
checks. The persisted report passed the production relational validator after
strict reload.
