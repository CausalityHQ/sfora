# Pass 190 — Soft CE-BN hill-climb

## Motivation from the live failure

Hard embedding-head CE-BN is currently running but collapses the train/eval
trajectory (best epoch-5 R@1 `0.6728` versus ordinary Proxy Anchor `0.8333`).
The corrected descriptor CPU probe separates the intervention from the
optimization: blending the untouched normalized descriptor with the
leave-own-class-out normalized descriptor improves monotonically:

| CE-BN blend λ | R@1 |
|---:|---:|
| 0.00 | 0.913701 |
| 0.20 | 0.915389 |
| 0.40 | 0.917780 |
| 0.50 | 0.918906 |
| 0.70 | 0.921789 |
| 1.00 | 0.927275 |

This is a hill-climb diagnostic, not benchmark evidence. It motivates a softer
train-time operation that uses `z=(1-λ)h+λ·CEBN(h)` with λ=`0.70`, preserving a
continuous ordinary path while retaining most of the measured signal.

## Gate 2 audit

Batch Renormalization (Ioffe, NeurIPS 2017,
<https://arxiv.org/abs/1702.03275>) interpolates batch and running statistics to
reduce train/eval minibatch dependence. It does not remove same-label rows from
the moment estimator, and it has no metric-learning positive relation. The
proposed operation is therefore narrower: label-excluded moments at the
embedding head plus an explicit descriptor-path blend. I found no primary work
using that label-excluded blend as supervised metric-learning training.

## Preregistration (after hard CE-BN closeout)

Run one matched corrected In-Shop seed only after Pass183 closes. Prediction:
selection-corrected R@1 delta `≥ +0.30 pt` versus paired Proxy Anchor. Falsify
below `+0.15 pt` or any non-positive raw delta; do not tune λ or run a second
seed after falsification. Report raw and corrected values.

