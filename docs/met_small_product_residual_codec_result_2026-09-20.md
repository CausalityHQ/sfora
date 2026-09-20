# MET-small 64-byte product-residual codec result

Date: 2026-09-20. Status: **STOPPED after the registered promotion and
seed-stability gates failed.** Product-residual quantization is not promoted to
fresh-domain replication or library implementation.

## Authority

- frozen feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- gallery: 35,257 normalized 768-dimensional vectors;
- training-derived, one-per-class pseudoqueries: 3,050;
- item payload: exactly 64 bytes for both codecs;
- OPQ baseline: `OPQ64_768,PQ64x8`;
- candidate: 32 independent 24-dimensional product blocks, two 8-bit
  residual stages per block, beam 16;
- result SHA-256:
  `9617398f0011c3fcaecb04d30b30d23170330a572d3c30c7a836cbabb896415d`;
- mechanism result SHA-256:
  `92a391fee54b5b06a630b4a5778118a9fe6bd1eb9d4b38ad0622492b8636d85d`;
- three-seed result SHA-256:
  `5b7ef2e4ad326a197e467ba584f97c92f50f776c1505d459e005d5fc40fb1b09`.

All results are claim-ineligible development evidence.

## Primary result

| Representation | mMP@5 | R@1 | reconstruction MSE |
| --- | ---: | ---: | ---: |
| float 768-D exact L2 | `0.590541` | `0.523934` | — |
| OPQ64x8 exact decoded L2 | `0.582765` | `0.517705` | `0.000182941` |
| product residual exact decoded L2 | `0.588164` | `0.518033` | `0.000189407` |

The candidate-minus-OPQ mMP@5 delta was `+0.005399`, with paired query
bootstrap 95% interval `[-0.000011, +0.010803]`. The R@1 delta was
`+0.000328`, interval `[-0.006885, +0.007869]`. The preregistered promotion
gate therefore failed. The candidate used 1,572,864 codebook bytes and an
8,388,608-byte exact pair-norm table, versus 786,432 OPQ codebook bytes.

## Mechanism diagnostic

Product residual quantization did preserve the float neighbourhood better at
ranks 2–10. Relative to OPQ, exact-float top-10 overlap improved by
`+0.007049` under decoded L2, 95% interval
`[+0.003475, +0.010721]`, and by `+0.006525` under inner product, interval
`[+0.002426, +0.010623]`. Top-1 fidelity was inconclusive.

The mMP@5 gain under inner-product scoring was only `+0.001989`, while R@1
changed by `-0.001967`. Product residual quantization had smaller directional
and norm error on the float top-10 despite 3.5% worse global MSE. This supports
a real local-rank effect, but not a robust first-neighbour or deployment-quality
gain.

## Seed floor and decision

With the OPQ rotation and baseline fixed, three predetermined product-residual
seeds produced:

- decoded-L2 top-10 overlap deltas from `+0.005049` to `+0.007049`;
- inner-product top-10 overlap deltas from `+0.006525` to `+0.008984`;
- decoded-L2 mMP@5 deltas from `+0.001738` to `+0.009071`;
- inner-product mMP@5 deltas from `-0.000432` to `+0.001989`;
- decoded-L2 top-1 fidelity deltas from `-0.003934` to `+0.001639`.

The frozen fresh-replication gate required every seed to improve top-10 overlap
and mMP@5 under both scorers. It failed because one inner-product mMP@5 delta
was negative. No threshold is relaxed and no seed is selected.

The result is scientifically useful: the residual codec consistently improves
deeper rank fidelity and recovers about 69% of OPQ's mMP@5 gap to float, but it
recovers only about 5% of the R@1 gap. Because the method is established prior
art, its first-neighbour benefit is null, and its shared scoring state is
larger, it is not a publishable quality/performance direction for Sfora.
Fresh-domain replication, custom CUDA/cuTile kernels, and production code are
not warranted. The next candidate must add information unavailable to a
single-vector codec rather than spend more complexity on an already-saturated
64-byte global vector.
