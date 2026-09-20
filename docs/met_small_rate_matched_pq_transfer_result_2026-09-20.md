# MET-small rate-matched product-quantizer transfer result

## Result

The fixed PCA80+OPQ24 recipe does not transfer as a quality improvement to
MET-small's separate query/gallery protocol under the public codec's exact
squared-L2 asymmetric-distance contract.

| Codec | bytes/vector | mMP@5 | R@1 | fit seconds | encode seconds | score seconds |
|---|---:|---:|---:|---:|---:|---:|
| PCA80+OPQ24 | 24 | 0.656202 | 0.689922 | **64.15** | 0.708 | 0.096 |
| OPQ24 | 24 | **0.671576** | 0.697674 | 117.36 | **0.585** | 0.013 |
| OPQ32 | 32 | 0.653876 | **0.720930** | 122.94 | 0.604 | **0.012** |

At equal 24-byte storage, PCA80+OPQ24 loses `0.015375` mMP@5 and one
validation-query R@1 outcome (`0.007752`) to OPQ24. Against OPQ32 it gains
`0.002326` mMP@5 but loses four R@1 outcomes (`0.031008`). The frozen
`third_domain_supported` flag is therefore false.

The query-class bootstrap intervals include zero for every contrast. With
only 129 queries and 111 query classes, this is a mechanism falsification, not
a precise estimate of a small effect. It rejects a universal fixed PCA80
allocation; it does not reject rate-matched rank selection learned without
evaluation labels.

The storage tradeoff remains real: PCA80+OPQ24 uses 356,352 shared parameter
bytes versus 3,145,728 for each full-rank OPQ arm, and fits about twice as fast.
Those efficiency gains do not compensate for the failed quality rule.

## Scoring-contract boundary

This experiment ranks hard resident gallery codes with the public library's
exact squared-L2 ADC. Earlier MET OPQ diagnostics decoded codes, renormalized
them, and ranked by cosine. Those historical values remain valid for that
different scorer but must not be compared numerically with this table. This
run compares its three arms internally under one common ADC contract.

The complete query score took `0.012` to `0.096` seconds for all 129 queries,
whereas fitting took `64` to `123` seconds. This run therefore supplies no
evidence that a custom ADC CUDA kernel is the current bottleneck. CuTile or
CUDA-Oxide work remains conditional on an end-to-end profile identifying a
kernel-shaped operation that dominates a named workload.

## Authority

- result:
  `docs/evidence/rank_finished_l14_336_met_small_rate_matched_pq_transfer_v1.json`;
- result SHA-256:
  `31b92ea3ca6bc92311164dcadfaeb42b667104266b602d7c14e4b27c0aad807f`;
- throwaway driver SHA-256:
  `643ad436b36cedea142deece29d487b7d6fcb4b51f5a1cf47b520834f9736aec`;
- exact feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- 38,307 gallery/fit rows, 129 official validation queries, seed 50, 20
  k-means iterations, four OPQ alternations, one Torch fitting thread;
- the official MET test image was not read or scored.

## Next discriminator

The next algorithm experiment should increase statistical power without
opening the test split: a frozen class-disjoint pseudo-query protocol formed
only from multi-row gallery classes, with the 129 shifted official validation
queries retained as a secondary sanity check. It should compare score-aware,
label-free compact representations and use gallery-only teacher-neighborhood
distortion for any rank selection. No validation-driven dimension sweep is
licensed by this failure.
