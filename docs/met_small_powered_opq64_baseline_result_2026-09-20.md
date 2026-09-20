# MET-small powered OPQ64x8 baseline result

OPQ64x8 is a strong 64-byte codec but does not meet the frozen near-ceiling
gate on the powered protocol.

## Authenticated result

- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- result SHA-256:
  `4959414886e3dab77b8555c99aa7394d1d1aa77058be63a7d0baeded60ddea63`;
- Faiss 1.12.0, one OMP thread, `OPQ64_768,PQ64x8`;
- 35,257 reduced-gallery fit rows, 3,050 powered pseudoqueries, 129 shifted
  validation queries;
- exactly 64 code bytes per gallery item;
- scientific process exited zero and no duplicate remained.

## Quality

| split | float mMP@5 / R@1 | OPQ64x8 mMP@5 / R@1 | float minus OPQ |
|---|---:|---:|---:|
| powered pseudoqueries | 0.590541 / 0.523934 | 0.582765 / 0.517705 | 0.007776 / 0.006230 |
| shifted validation | 0.695607 / 0.658915 | 0.689664 / 0.627907 | 0.005943 / 0.031008 |

OPQ retains 98.68% of float powered mMP@5 and 98.81% of float powered
Recall@1. Its gallery reconstruction MSE is 0.00018294. The powered Recall@1
gap passes the at-most-0.01 condition, but the mMP@5 gap exceeds the at-most
0.005 condition, so `codec_near_ceiling=false`.

## Runtime and decision

Deterministic CPU fitting took 293.60 seconds. Exhaustive scoring of both
query populations took 1.28 seconds for OPQ and 1.38 seconds for float. These
are diagnostic batch timings, not serving latency measurements.

There is measurable fine-ordering headroom at fixed 64-byte storage, but it is
only 0.0078 mMP@5 on the powered proxy. Fund one narrowly preregistered
rank-aware codec experiment with exact ADC and a matched reconstruction-only
control. Do not fund a custom kernel unless that experiment first clears a
material quality gate and then replicates on an independent domain.
