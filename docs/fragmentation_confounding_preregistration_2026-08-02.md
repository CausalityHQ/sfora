# Fragmentation confounding audit preregistration

Recorded on 2026-08-02 before computing any adjusted fragmentation effect.
This is a CPU-only provenance diagnostic on the already frozen seed-0 In-Shop
epoch-10 training pack.

## Question

The observed exactly class-size-matched disconnected-minus-connected
leave-one-out R@1 difference is **+3.534 points**. Disconnectedness may merely
identify classes that are internally compact in several clumps or far from
foreign classes. If so, fragmentation is not the information-bearing property
and should not motivate a mode-preservation method.

## Locked variables

Use every training identity with at least three examples from
`inshop_pa_epoch10_operating.train.npz`:

- outcome: class-balanced leave-one-out image R@1;
- exposure: whether the symmetrized within-class 1-NN graph is disconnected,
  exactly as in `scripts/measure_spectral_class_connectivity.py`;
- covariates fixed without test data:
  1. exact class size;
  2. mean off-diagonal within-class cosine similarity;
  3. maximum cosine similarity from the class centroid to any foreign class
     centroid.

Assign covariates 2 and 3 to deterministic global quintiles pooled over all
eligible classes, with quantile edges computed by NumPy's default linear
quantile and duplicate edges collapsed; then cross those bins with exact class
size. A cell is the
tuple `(exact size, within-similarity quintile, foreign-centroid quintile)`.
Retain only cells containing both fragmented and connected classes. The
adjusted effect is the weighted mean of cell-level outcome differences, with
weight `min(n_fragmented, n_connected)`. Report retained classes, cells, both
exposure counts, and effective matched weight.

As a continuous check rather than a second decision rule, also report the raw
Pearson correlations among fragmentation, both covariates, and outcome.

## Prediction and falsification

- Prediction: the adjusted fragmented-minus-connected effect remains above
  **+1.0 R@1 point** and retains at least **25%** of eligible classes.
- The fragmentation premise is falsified by adjusted effect **<= 0**, or the
  diagnostic is underpowered and cannot pass if retained coverage is below
  **25%** or either exposure contributes fewer than 30 retained classes.
- An effect in `(0, +1.0]` is attenuated/inconclusive and does not motivate a
  method.

This audit cannot prove causality. A pass only strengthens provenance and still
requires the independent-seed replication and a Gate-2-surviving operator.
