# Single-model effective-rank diagnostic

Date: 2026-08-01. Registered before reading any rank-sweep result.

## Motivation

The repository tested narrow 128- and 64-dimensional heads during training and
both lost (Proxy Anchor 0.6919 to 0.6584 and 0.6317). That does not answer
whether a normally trained 512-dimensional solution actually uses all 512
directions for retrieval. Existing projection sweeps concern concatenated
multi-model ensembles, not one trained model.

## Fixed diagnostic

Use `herd_tt_seed0.train.npz` only to fit the mean and right singular vectors of
the 512-dimensional training embeddings. Apply that fixed projection to
`herd_tt_seed0.test.npz`, L2-normalize after projection, and compute leave-self-
out test Recall@1 at ranks 8, 16, 32, 64, 128, 256, 384, and 512. Also report
the unprojected normalized test Recall@1 and cumulative training variance.

No test embedding may influence the mean, basis, rank choice, or thresholds.

## Registered interpretation

- **Architecture lead:** the smallest rank within 0.10 Recall@1 point of the
  fitted rank-512 result is at most 128, and fitted rank 512 itself is within
  0.10 point of the unprojected result. This establishes substantial unused
  representational dimension without conflating it with training a narrow head.
- **Strong lead:** the same condition holds at rank 64.
- **Falsified:** rank 128 loses more than 0.10 point, or fitting/centering alone
  changes rank-512 Recall@1 by more than 0.10 point. In that case the apparent
  compression depends on evaluation geometry rather than unused capacity.

A passing diagnostic is not a method result. Before implementation it must face
prior art in PCA/compressed retrieval, nested or Matryoshka embeddings,
low-rank heads, and capacity reallocation. A compression-only head is not novel;
the downstream operator must explain how freed capacity creates new similarity
information at roughly unchanged compute.

## Result

Run CPU-only on the registered CUB HERD seed-0 packs after commit `3cfc3f1`:

| representation | Recall@1 | cumulative train variance |
| --- | ---: | ---: |
| unprojected 512-D | 0.6940 | — |
| fitted PCA rank 512 | 0.6818 | 1.0000 |
| rank 384 | 0.6789 | 0.9867 |
| rank 256 | 0.6794 | 0.9567 |
| rank 128 | 0.6661 | 0.8855 |
| rank 64 | 0.6438 | 0.7796 |

The diagnostic is **falsified**. Fitting and centering alone changes Recall@1 by
-1.215 points, far beyond the registered 0.10-point tolerance. Rank 128 loses
2.785 points versus the unprojected embedding. Variance retained by PCA is not a
proxy for retrieval information here, and the cosine embedding's origin/mean
direction is materially part of its geometry. No bottleneck or capacity-
reallocation candidate follows from this result.
