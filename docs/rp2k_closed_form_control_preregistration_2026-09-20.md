# RP2K closed-form compact-control panel

## Status and scope

This claim-ineligible diagnostic is frozen after the failed RP2K validation
gate and before running any new control.  It uses only the already-consumed
RP2K validation features.  The RP2K test split remains unopened.  Its purpose
is to decide whether the supervised SGD compact head contains useful geometry
beyond a closed-form Fisher projection or asymmetric equal-byte OPQ scoring.

Immutable inputs are feature archive SHA-256
`46cfbd7dea5592de7fcbdfedd7a3b5bc3960160b853d4e484c46ac07b31abbc7`,
learned-head checkpoint SHA-256
`866e942a51de3f15c37aaf4cedcb8600119c834e639445e81ca27f95da089035`,
and throwaway driver SHA-256
`56e920aa789154de76ac17aa0e9181f8213113a659f3042cbd55e0aaf57f22a7`.
The fit/evaluation sets remain 15,266 rows from 1,074 classes and 17,185
rows from 120 disjoint classes respectively.

## Fixed controls

Every compact gallery stores 64 bytes per item.  The panel measures symmetric
and asymmetric-query variants for:

1. the sealed learned int8-64 head;
2. fit-only PCA int8-64;
3. shrinkage Fisher/retrieval whitening: Ledoit-Wolf within-class covariance,
   full whitening, then the top 64 eigenvectors of between-class covariance
   in the whitened space; and
4. fit-only `OPQ64_768,PQ64x8`, scored both decoded-symmetrically and with the
   original float query against the decoded compact gallery.

All transforms are fitted only on the registered train subset.  Evaluation is
the same deterministic, self-excluded UnED mMP@5 and R@1 protocol.  Paired
intervals use class-cluster resampling but weight each sampled class by its
number of queries, so the interval and point estimate are both query-weighted.

## Frozen interpretation

- If Fisher is within `0.002` mMP@5 of learned SGD and does not trail R@1 by
  more than `0.001`, retire SGD metric fitting as a research contribution and
  prefer the closed-form method for further generic replication.
- If learned SGD exceeds Fisher by at least `0.003` mMP@5 with a paired 95%
  lower bound above zero and does not regress R@1 by more than `0.001`, retain
  SGD as a distinct candidate.
- If asymmetric OPQ is within `0.002` mMP@5 of learned symmetric int8-64,
  the current evidence does not establish a material supervised advantage at
  64 bytes.  If learned exceeds it by at least `0.003` with a paired lower
  bound above zero, the supervised advantage survives this control.
- Outcomes between these bands are inconclusive.  No threshold will be
  changed from this result, and no outcome authorizes RP2K test access.

This is an algorithm-selection experiment, not a serving benchmark.  CUDA,
CuTile, or CUDA-Oxide work remains deferred because no arithmetic bottleneck
can change the quality decision.
