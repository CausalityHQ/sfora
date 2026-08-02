# Fragmentation partner-exclusion audit preregistration

Recorded on 2026-08-02 before implementing or computing the partner-excluded
outcome. This is a CPU-only audit of the measurement instrument on the already
frozen seed-0 In-Shop epoch-10 training pack; it is not a method candidate.

## Motivation

The original adjusted fragmented-minus-connected leave-one-out R@1 gap was
**+5.875 points**, while fragmentation correlated only **+0.04754** with class
R@1, fragmentation correlated **-0.52404** with within-class mean cosine, and
within-class mean cosine correlated **+0.41302** with R@1. The symmetrized
within-class 1-NN graph and the leave-one-out outcome are computed from the same
nearest-neighbour relation. A class containing tight pairs can therefore be
fragmented and retrieve correctly for the same mechanical reason, without an
independent semantic mode or useful intervention being present.

## Locked input and estimator

Use `inshop_pa_epoch10_operating.train.npz`, SHA-256
`85e76245603689c824ec3f6aefceb67eee34fb7df94d3a825977a8bd4d139b27`.
Reuse unchanged from `scripts/measure_fragmentation_confounding.py`:

- every identity with at least three training images;
- the original symmetrized within-class 1-NN disconnectedness exposure;
- exact class size, global within-class-mean-cosine quintile, and global
  nearest-foreign-centroid-cosine quintile matching cells;
- cells containing both exposures and weight
  `min(n_fragmented, n_connected)`.

Change only the per-image retrieval outcome. For every query, identify its
highest-cosine *same-class* gallery image (excluding itself), delete that one
image from the gallery for that query, and score whether the best remaining
gallery image has the query label. Average this partner-excluded correctness
within each class before applying the locked adjustment. Ties use NumPy
`argmax` first-index order. The exposure and matching covariates remain those
from the original embeddings and are not recomputed after deletion.

Report the adjusted fragmented-minus-connected partner-excluded R@1 gap,
retained coverage and exposure counts, plus the unadjusted class-balanced gap.

## Prediction and decision rule

- Estimator-coupling prediction: the adjusted gap collapses to **<= +1.0 R@1
  point**. This closes the fragmentation marker as provenance for further
  method generation; candidates 185--198 remain dead independently on prior
  art.
- A gap of **>= +3.0 points**, with at least **25%** retained coverage and at
  least 30 retained classes from each exposure, rejects that explanation and
  isolates an association with support redundancy after the trivial support is
  removed. It reopens measurement, not any previously rejected operator.
- A gap in **(+1.0, +3.0) points**, inadequate coverage, or fewer than 30
  retained classes from either exposure is inconclusive and authorizes no
  method.

This audit reuses the hypothesis-generating seed and cannot establish causality.
No candidate or GPU run is authorized by a pass; a surviving relation would
first require independent-seed confirmation and a new Gate-2 audit.

## Result

The estimator-coupling prediction **passed decisively**. With the same 72
matched cells, 2,402 / 3,975 retained classes (**60.43%** coverage), 446
fragmented retained classes, 1,956 connected retained classes, and effective
matched weight 367, the adjusted fragmented-minus-connected
partner-excluded R@1 gap was **-3.910 points**. The unadjusted class-balanced
gap was **-2.240 points** (fragmented **0.84707**, connected **0.86947**).

Removing only the same-class neighbour used most directly by the exposure did
not merely attenuate the original **+5.875-point** adjusted association; it
reversed it. The original marker is therefore not valid provenance for a
mode-preservation intervention. Its stability across optimizer seeds remains a
real fixed-data property, but the favourable outcome association was coupled
to the retrieval estimator's trivial pair support. No additional seed or GPU
work is warranted on this line, and candidates 185--198 remain dead.

Immutable evidence:

- analyzer SHA-256:
  `28f12d8db68b2ed4eb9cf2d51f6890e55078e886da2afbdae4eeeaa11d2dbf1a`;
- result JSON SHA-256:
  `cddec1f14ec41128beac0db6f619d748c65ab2ad8e4c3a2dd795d6c63555f97b`.
