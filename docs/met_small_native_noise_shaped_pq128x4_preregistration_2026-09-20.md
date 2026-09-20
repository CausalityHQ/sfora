# MET-small native fixed-codebook noise-shaped PQ128x4 spike

## Question

The external ScaNN screen improved the powered MET proxy at a nominal 64-byte
payload, but it did not establish that Sfora can reproduce the mechanism under
its own codec and scorers. This prospective spike asks one narrow question:
with conventionally trained codebooks held fixed, does ScaNN-compatible
noise-shaped hard assignment improve retrieval over nearest-codeword assignment
at the exact same packed payload?

This is an algorithm-mechanism experiment, not production code or a latency
benchmark. A pass can promote one public-geometry follow-up; it cannot establish
generic support, a product default, or a reason to implement a custom CUDA
kernel.

## Frozen authority and population

- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- archive populations: 38,307 MET-small training rows of width 768 and 129
  official-validation rows;
- powered proxy: the lexicographically first training path from each class,
  giving 3,050 pseudo-queries and a disjoint 35,257-row reduced gallery, exactly
  as frozen in
  `docs/met_small_pseudoquery_protocol_preregistration_2026-09-20.md`;
- all vectors are independently L2-normalized before fitting, encoding, or
  scoring;
- the 129 official-validation queries remain secondary non-inferiority evidence;
  the external official test remains sealed;
- ScaNN source authority: Google Research commit
  `4700efb9afa54286b0e04473ba80a13e8461e25f`, function
  `AhImpl<T>::IndexDatapointNoiseShaped` in
  `scann/scann/hashes/internal/asymmetric_hashing_impl.cc`;
- throwaway driver SHA-256:
  `0c446cfd4d871d90ca113b2915e679b811cc4a13a4b0f1cc94c383b9164c9148`.

The driver self-test differentially compares the batched implementation with an
independent scalar rendition of the cited ScaNN update. Its mutation fixture
must change nearest-codeword assignments from `[1,0,1]` to `[1,0,2]`.

## Frozen geometry and arms

For each seed in `{50,51,52}`, fit one conventional product quantizer on the
same reduced gallery:

- 768 dimensions split into 128 contiguous blocks of width 6;
- 16 float32 centers per block;
- scikit-learn KMeans with the seed above and at most 20 iterations;
- codebooks are frozen before either arm is encoded;
- 128 four-bit assignments are packed low-nibble first into exactly 64 bytes
  per gallery row.

The two arms share those exact codebooks:

- `isotropic`: independently minimize squared residual norm in each block,
  with lowest-center tie breaking;
- `noise_shaped`: start from that same isotropic code, rank blocks by descending
  initial residual norm, and perform at most ten ScaNN coordinate-descent rounds.
  For row `x`, threshold `t=0.2`, and `d=768`, use
  `eta=(t^2/||x||^2)/((1-t^2/||x||^2)/(d-1))`. Candidate residual statistics
  and the residual component parallel to `x/||x||` are accumulated in float64.
  A coordinate update is permitted only when squared parallel residual does not
  increase and total anisotropic cost strictly decreases. Center and block ties
  resolve to the lower ordinal.

No labels, query vectors, or retrieval results may influence codebook fitting or
assignment. Report codebook and packed-code SHA-256 values, exact payload bytes,
code churn, multiplier range, reconstruction diagnostics, and fitting/encoding
times.

## Frozen scoring

Decode the packed gallery transiently and perform exhaustive exact top-five
search with stable lowest-gallery-ordinal ties under both:

1. dot product, the mechanism-fidelity scorer used by the external ScaNN screen;
2. squared L2, Sfora's public retrieval geometry for unnormalized decoded PQ
   reconstructions.

Force highest float32 matmul precision with TF32 disabled. The earlier external
ScaNN screen used quantized LUT16 lookup tables, whereas this native spike uses
decoded float32 exhaustive scores; therefore a native shortfall cannot be
attributed to LUT quantization. Report the native dot-product powered-proxy
mMP@5 delta as a ratio of the external screen's `+0.012174863388` delta, but do
not make that ratio a decision gate.

For every seed, arm, scorer, and split, report mMP@5, Recall@1, per-query
outcomes, and batch scoring time. These timings include unpacking and decoding
and are diagnostic only; they make no serving-latency claim.

## Frozen decision

For each scorer independently, `mechanism_supported=true` only if all of the
following hold:

- mean powered-proxy mMP@5 delta across the three seeds is at least `+0.002`;
- a 10,000-resample query bootstrap over the seed-mean powered-proxy mMP@5
  delta has lower 95% bound greater than zero;
- powered-proxy mMP@5 improves for every seed;
- mean powered-proxy Recall@1 delta is at least `-0.003`.

Inclusive floating lower bounds use an absolute `1e-12` roundoff tolerance.
Official validation is too small to veto three codebook seeds reliably, so it
does not enter `mechanism_supported`. Report its mean mMP@5 delta, exact total
Recall@1 hit-count delta, and the earlier external screen's per-seed
non-inferiority predicate (`mMP@5 >= -0.005`, at most one lost binary hit) as
secondary distribution-shift evidence only.

`seed_floor_cleared=true` only if the mean powered-proxy mMP@5 delta is strictly
larger than the maximum-minus-minimum isotropic powered-proxy mMP@5 across the
three seeds. `promotion_supported` requires both gates.

The native mechanism earns the one permitted public-geometry follow-up only if
dot product has `mechanism_supported=true` and squared L2 has
`promotion_supported=true`. That follow-up is a fixed-codebook encoder
comparison at Sfora's existing public OPQ64x8 geometry; it is not an adaptive
sweep. If dot product has `promotion_supported=true` but squared L2 does not,
the mechanism remains supported while Sfora's current decoded-vector norm term
is the suspected incompatibility; this licenses exactly one fixed
normalization/scorer attribution diagnostic, not the OPQ follow-up. If dot
product itself lacks `mechanism_supported`, the noise-shaped-assignment route
closes.

Regardless of outcome, `generic_supported=false`: this spike uses one dataset,
one representation, one payload, and already-observed development queries.
CUDA, CuTile, and CUDA-Oxide work remains forbidden unless a later end-to-end
profile identifies a named production operation consuming at least half of
runtime with a credible twofold operation-level gain.
