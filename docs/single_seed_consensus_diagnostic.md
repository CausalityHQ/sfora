# Single-seed consensus predictability diagnostic

Date preregistered: 2026-08-01

## Motivation

On CUB, seed-0 HERD reaches 0.6940 R@1, while the five-seed concatenation reaches
0.7350 and an inductive train-fit GPA fold reaches 0.7205 at 512 dimensions. The
unresolved question is whether another seed contributes information absent from
seed 0, or mostly a deterministic calibration of information seed 0 already
contains.

## Frozen diagnostic

Use exactly `herd_tt_seed{0..4}.train.npz` and matching test packs. Fit GPA
rotations on the training split only and form the L2-normalised aligned-mean
training target. Fit one uncentred linear ridge map from seed-0 training
embeddings to that target:

`W = (X'X + lambda I)^-1 X'Y`, where
`lambda = 1e-3 * trace(X'X) / d`.

Freeze `W`, apply it to seed-0 test embeddings, L2-normalise, and compute test
R@1. No labels enter the map, no test statistic enters fitting, and no lambda is
selected after observing retrieval.

Report seed 0, five-seed concat, train-fit GPA target, orthogonal-map control,
and ridge-map R@1.

## Registered interpretation

- **Strong deterministic recovery:** ridge R@1 >= 0.7203 (98% of the known
  0.7350 concat) and at least 1.0 point above seed 0.
- **A useful but incomplete lead:** ridge gains at least 1.0 point over seed 0
  but remains below 0.7203.
- **Falsified:** ridge gains less than 0.5 point over seed 0.
- Anything between +0.5 and +1.0 point is inconclusive and does not justify GPU.

This diagnostic is not a novelty claim. Linear metric post-processing and
ensemble distillation are established. A positive result would only motivate a
new search for an end-to-end mechanism; it would not itself pass Gate 2.
