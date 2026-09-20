# MET-small 64-byte product-residual codec preregistration

## Question

Does changing the representation from 64 independent OPQ/PQ bytes to 32
independent two-stage residual pairs improve powered MET-small retrieval at the
same 64-byte payload? This is a one-recipe, label-free, claim-ineligible
development falsifier. It is distinct from the previously failed 24-byte
full-dimensional greedy residual quantizer.

The incumbent is deterministic one-thread Faiss 1.12.0
`OPQ64_768,PQ64x8` at `0.582765` mMP@5 and `0.517705` R@1 on 3,050
deterministic pseudoqueries. Exact float is `0.590541 / 0.523934`.

## Frozen construction

1. Fit the incumbent OPQ64 rotation and PQ on the 35,257-row normalized
   gallery using one OpenMP thread.
2. Apply only the fitted orthogonal OPQ transform to gallery and queries.
3. Fit Faiss `ProductResidualQuantizer(768, 32, 2, 8)` in those coordinates:
   32 blocks of 24 dimensions, two 256-entry residual stages per block, and
   `max_beam_size=16` for every subquantizer. Do not sweep blocks, beam, seeds,
   stages, or bits.
4. Require exactly 64 encoded bytes per gallery row. For this quality
   reference, decode the codes and exhaustively rank by exact squared L2. Do
   not renormalize, prune candidates, rerank with originals, or use labels in
   fitting.

The runner must reproduce the incumbent via its registered ADC path and prove
that explicit squared L2 to its rotated decoded vectors produces identical
top-five rankings. It records reconstruction MSE, fit times, codebook bytes,
and the exact 8 MiB pair-norm table required by a future non-decoding scorer.
This is byte-matched payload, not equal total shared metadata; performance is
not claimed by this screen.

## Decision

Use paired query bootstrap intervals with 10,000 fixed-seed replicates on the
3,050-query primary split. Promote only if all hold:

1. product-residual minus OPQ mMP@5 is at least `+0.002`;
2. its 95% lower bound is above zero;
3. the R@1 delta 95% lower bound is above `-0.001`;
4. every authority, payload, and scorer-equivalence gate passes.

The 129-query shifted split is descriptive only. A failure closes this exact
two-stage product-residual representation under the inherited OPQ rotation and
bounded beam; it does not reject all additive quantization. A pass authorizes
unchanged fresh-domain replication, followed by total-memory and latency
qualification. CUDA/cuTile/CUDA-Oxide work remains blocked until those quality
gates pass.
