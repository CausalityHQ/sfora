# MET-small noise-shaped PQ conditional-mean decoder result

## Result

Re-estimating every PQ center as the conditional mean of the gallery rows that
received its noise-shaped assignment improves reconstruction error but does not
recover deployed squared-L2 retrieval quality. The frozen decision is
`recentered_decoder_supported=false`.

| scorer | mean pseudo mMP@5 delta | mean pseudo R@1 delta | paired-query 95% interval | promotion |
|---|---:|---:|---:|---:|
| dot product | **+0.009044** | **+0.002732** | **[+0.004865, +0.013452]** | pass |
| squared L2 | **-0.002896** | **-0.006995** | **[-0.006674, +0.000891]** | fail |
| normalized squared L2 | -0.000175 | **-0.006667** | [-0.003916, +0.003636] | fail |

Squared-L2 mMP@5 deltas were `-0.004667`, `-0.005421`, and `+0.001399`
for seeds 50, 51, and 52. The mean is below zero, the confidence interval
includes zero, only one seed improves, Recall@1 regresses, and the effect does
not exceed conventional-codebook seed spread. It fails every promotion gate.

The control itself behaved as intended. All `2,048` cells were occupied in
every seed. Conditional-mean decoding reduced mean squared reconstruction error
from `0.482785` to `0.449543` for seed 50, `0.482009` to `0.449602` for seed
51, and `0.483458` to `0.449946` for seed 52. Those values remain above the
matched conventional PQ errors near `0.4175`. Better reconstruction under the
changed assignments is therefore insufficient to repair their local ordering.

The underpowered 129-query official-validation split moves in the opposite
direction (`+0.027821` mean squared-L2 mMP@5 and five total R@1 hits across
three seeds), but it was preregistered as descriptive shift evidence and does
not override the 3,050-query powered decision.

## Consequence

The scorer reversal is not explained by either the original center mismatch or
the decoded-vector norm alone. Fixed-codebook ScaNN-compatible noise-shaped
assignment remains closed for the public squared-L2 codec. Do not spend another
experiment on its threshold, block shape, seed, normalization, or decoder.

The next credible experiment is a different mechanism: learn an OPQ-initialized
orthogonal rotation against neighborhood ordering under the exact deployed
hard ADC score, with reconstruction-only and shuffled-neighborhood controls.
That changes the quantization geometry while preserving the 64-byte wire and
query-time lookup count.

## Authority

- committed experiment source: `d4eedb3f`;
- driver SHA-256:
  `c0b320ae0c31fad99318cde42b4d4efacff172e13d8fe91bfd52f0dc73c1b995`;
- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- canonical result:
  `docs/evidence/rank_finished_l14_336_met_small_noise_shaped_pq128x4_recentered_v1.json`;
- result SHA-256:
  `a574d8e2a98a5c003cc1a623be8c802cd6761d1851eaa9acd006319c7aaeaacf`;
- result bytes: `732,112`;
- original DGX process terminal status: `0`;
- external official test remained sealed;
- no custom CUDA, CuTile, or CUDA-Oxide kernel was used.
