# MET-small noise-shaped PQ decoded-norm attribution result

## Result

L2-normalizing decoded reconstructions does not restore the native
noise-shaped-assignment gain. The fixed attribution diagnostic fails its
preregistered promotion gate and closes this route for Sfora's public codec.

| scorer | mean pseudo mMP@5 delta | mean pseudo R@1 delta | query-bootstrap 95% | promotion |
|---|---:|---:|---:|---:|
| raw dot product | **+0.009581** | **+0.002295** | **[+0.005240, +0.013984]** | pass |
| unnormalized squared L2 | **-0.005512** | **-0.015956** | **[-0.010124, -0.000832]** | fail |
| normalized squared L2 | -0.000195 | **-0.012459** | [-0.004171, +0.003902] | fail |

Normalized squared L2 improves over unnormalized squared L2 in mMP@5 but does
not reproduce the raw-dot effect: its mean change is slightly negative, its
confidence interval spans zero, Recall@1 drops materially, only one of three
seeds improves mMP@5, and the effect does not clear isotropic seed variation.

The canonical decision is
`normalization_resolves_scorer_mismatch=false`. The prior dot-product result is
real for this diagnostic representation, but it is not explained by removing
only the decoded norm penalty and it does not transfer to either evaluated
public squared-L2 geometry. No threshold, seed, block, payload, or scorer sweep
is permitted. `generic_supported=false` remains binding.

## Secondary official-validation evidence

Across the three seeds, normalized squared L2 changes official-validation
mMP@5 by mean `+0.013523` and Recall@1 by an exact total of `+1` hit. The
powered pseudo-query result remains negative, and the 129-query secondary split
is too small and seed-variable to reverse that decision.

## Authority

- committed diagnostic/preregistration source:
  `4a2978e785b5eb9528c3d050d6e0147d4b8636b2`;
- driver SHA-256:
  `16ff376a645bd1eb156eaad3afdcb9f20acdf7b6bb7c641a0eed5df1bb6c2203`;
- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- canonical result:
  `docs/evidence/rank_finished_l14_336_met_small_native_noise_shaped_pq128x4_normalization_v1.json`;
- result SHA-256:
  `fb43745a0a1f152cdf4444ae665fce2d176a599cd87b985151178be876677421`;
- result bytes: `487,272`;
- remote process terminal status: `0`;
- external official test remained sealed;
- no custom CUDA, CuTile, or CUDA-Oxide kernel was used.

## Consequence

Close fixed-codebook ScaNN-compatible noise-shaped PQ assignment as a route to
the current 64-byte public scorer. The next experiment must test a different
similarity-learning mechanism rather than tune this codec. It should remain
small, multi-seed, powered by the 3,050-query protocol, and should not receive
production optimization until it demonstrates a scorer-compatible quality
gain.
