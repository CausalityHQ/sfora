# MET-small equal-byte OPQ geometry diagnostic

## Question

The burned MET-small rank diagnostic shows that signed-int8 rounding is not the
learned64 limitation. This fixed screen asks whether the same 64-byte payload
improves when it uses more, narrower product subquantizers with fewer bits per
subquantizer.

Correction recorded after execution: Faiss specifications `OPQ64_768`,
`OPQ128_768`, and `OPQ256_768` all retain 768 dimensions. This experiment
changes partition granularity and per-subquantizer resolution, not retained
representation rank.

## Authority and arms

- normalized fit/gallery and validation-query embeddings come only from feature
  archive SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- Faiss is restricted to one CPU thread;
- every transform and codebook fits only the 38,307 gallery/train rows;
- every code is exactly 64 bytes;
- arms are fixed as `OPQ64_768,PQ64x8`, `OPQ128_768,PQ128x4`, and
  `OPQ256_768,PQ256x2`;
- each gallery code is decoded to the original 768-D space and scored against
  the unchanged float query with the existing deterministic MET top-five
  scorer.

Report mMP@5, R@1, code shape, fitting time, and normalized-gallery
reconstruction MSE for every arm.

## Frozen interpretation

`higher_rank_lower_bit_supported` is true if either lower-bit arm exceeds
OPQ64x8 by at least `0.002` mMP@5 and is no more than `0.005` worse in R@1.
Otherwise it is false.  Because validation has only 129 queries, this is a
mechanism screen on a burned split, not confirmation or a test-reveal gate.
No arm, threshold, thread count, or codec seed may change after execution.
