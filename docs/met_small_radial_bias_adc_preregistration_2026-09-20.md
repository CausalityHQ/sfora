# MET-small radial-bias exact-ADC preregistration

## Question

The best observed 64-byte conventional PQ128x4 scorer on the powered MET-small
proxy remains near `0.5704` mMP@5. Noise-shaped assignments reduced
reconstruction error but did not improve the deployed squared-L2 ADC ranking.
This throwaway diagnostic tests one narrower mechanism: whether the decoded
gallery norm is over-penalized relative to its query alignment.

For fixed codebooks and codes, exact ADC ranks by
`||r||^2 - 2 alpha q·r`, ignoring the query-only constant. The deployed scorer
uses `alpha=1`. The single frozen candidate uses `alpha=0.5`, making the norm
and alignment coefficients equal. It changes neither the 64-byte code nor any
stored parameter.

## Frozen protocol

- authenticate feature archive SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- use the established deterministic MET-small pseudoquery split: 3,050
  pseudoqueries and 35,257 reduced-gallery rows;
- normalize all source embeddings;
- fit conventional PQ128x4 independently at seeds 50, 51, and 52, with one
  sklearn Lloyd initialization and at most 20 iterations per six-dimensional
  block;
- encode each gallery once, require nibble-pack roundtrip and exactly 64 bytes
  per row;
- rank through the exact per-block lookup ADC path with deterministic
  lowest-gallery-ordinal ties, first at `alpha=1`, then at `alpha=0.5`;
- report mMP@5 and Recall@1 on the 3,050 powered pseudoqueries as primary;
- report the same metrics on all 129 shifted official validation queries as
  descriptive secondary evidence only;
- run a 10,000-resample PCG64(51,337) paired-query bootstrap over the
  seed-mean pseudoquery mMP@5 delta.

No alpha is selected on either query set. No label tunes the codebooks or
scorer. Exact ADC-versus-explicit-decoding parity, non-unit queries, ties, and
64-byte packing are mutation-locked in the runner self-test.

## Frozen decision

Promote radial-bias correction to one equal-byte OPQ64x8 follow-up only if all
of these hold:

1. mean pseudoquery mMP@5 delta is strictly greater than `+0.0024`, the prior
   conventional-PQ seed-spread scale;
2. the paired-query 95% interval lower bound is strictly positive;
3. mean pseudoquery Recall@1 delta is nonnegative;
4. pseudoquery mMP@5 delta is positive in all three seeds.

Otherwise close this mechanism without an OPQ or CUDA follow-up. Even a pass
is claim-ineligible development evidence; it does not change a library default.
