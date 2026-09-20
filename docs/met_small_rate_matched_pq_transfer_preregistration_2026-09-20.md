# MET-small rate-matched product-quantizer transfer

## Question

`RateMatchedProductQuantizer` already improves the 24-byte quality/storage
frontier on SOP and CUB.  This prospective diagnostic asks whether the same
fixed, public PCA80+OPQ24 construction transfers to a third domain with a
separate query/gallery protocol, rather than benefiting only self-retrieval
benchmarks.

This is an algorithm screen, not a kernel benchmark.  A CUDA kernel cannot
repair a quality failure, so custom CuTile or CUDA-Oxide work remains gated on
both a positive codec result and a measured runtime profile.

## Frozen authority

- input feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- fit/gallery population: all 38,307 normalized MET-small train embeddings;
- evaluation population: all 129 normalized official validation queries;
- the official test image is never read or scored;
- library source authorities:
  - `product_quantization.py` SHA-256
    `17a814b2ccb0fc997b352136119721638fe484220f37b3d36fa1216a3bc01d3d`;
  - `rate_matched_product_quantization.py` SHA-256
    `131e5feb57b4bc72674f671f898d49deafd2a7ce0dca9c3ef7d7f886e8865f01`;
  - `representation_ceiling.py` SHA-256
    `1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c`.

Every codec uses seed 50, 20 k-means iterations, four OPQ alternations, and
the library's deterministic one-thread fitting boundary.  All gallery codes
are hard `uint8` codes.  Float queries are scored by exact asymmetric lookup
distance without decoding gallery codes.

## Frozen arms

1. `pca80-opq24`: PCA from 768 to 80 normalized dimensions, then OPQ with 24
   one-byte subquantizers.  Payload: 24 bytes/vector.
2. `opq24`: OPQ in all 768 source dimensions with 24 one-byte
   subquantizers.  Payload: 24 bytes/vector.
3. `opq32`: OPQ in all 768 source dimensions with 32 one-byte
   subquantizers.  Payload: 32 bytes/vector.

Report mMP@5, Recall@1, per-query vectors, fit seconds, gallery-encode seconds,
query-score seconds, exact code shape, and shared codec parameter bytes.

## Frozen interpretation

`third_domain_supported` is true only when `pca80-opq24`:

- exceeds `opq24` by at least `0.005` mMP@5;
- is no more than `0.005` below `opq32` in mMP@5; and
- is no more than one validation query (`1/129`) below either baseline in
  Recall@1.

Also report query-class bootstrap intervals for every paired delta, but do not
replace the frozen point rule with post-hoc significance claims.  With only
129 validation queries and 111 query classes, this result is diagnostic and
claim-ineligible.  A positive result extends the cross-domain mechanism
evidence; a negative result rejects the current fixed PCA80 allocation on MET
without rejecting rate-matched coding generally.

No codec geometry, seed, threshold, metric, or query population may change
after execution.  The next step after either outcome is determined from the
result and the separately measured profile; no custom GPU implementation is
authorized by quality alone.
