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

## Result

The locked audit **passed** on the frozen seed-0 pack. After crossing exact
class size with the two global quintile bins, 72 cells retained 2,402 / 3,975
eligible classes (**60.43%** coverage): 446 fragmented and 1,956 connected,
with effective matched weight 367. The adjusted disconnected-minus-connected
class-balanced leave-one-out R@1 difference was **+5.875 points**, above the
registered +1.0 prediction and not attenuated from the size-only +3.534.

Fragmentation itself has only Pearson `0.04754` correlation with class R@1 and
is strongly anticorrelated with mean within-class cosine (`-0.52404`); mean
within-class cosine correlates `0.41302` with R@1. Nearest-foreign-centroid
cosine correlates `-0.29337` with R@1. Thus the adjusted result is not a claim
that disconnectedness dominates ordinary compactness or separation. It says
that coarse registered controls for both did not explain away the marker.

Immutable inputs/code:

- embedding pack SHA-256:
  `85e76245603689c824ec3f6aefceb67eee34fb7df94d3a825977a8bd4d139b27`;
- analyzer SHA-256:
  `27f840b91b4d814d6cd8411536f68f2cc9e272fca85e1e3d137b238f04b6a7fa`;
- rendered JSON SHA-256:
  `9f3683abbec6668b12e274ceb6a1a382fa926c27a1cf398e1a689eb3a98d849c`.

The result remains observational, reuses the hypothesis-generating seed, and
uses coarse quintile balance rather than exact continuous balance. It therefore
does not override the independent seed-1/2 kill rule and does not resurrect
candidate 185 or any occupied diversity/subcentre method.
