# MET-small second-moment OPQ64 precoder preregistration

## Question

Can a query-independent, full-rank second-moment coordinate pair improve
64-byte OPQ retrieval without changing any exact pre-quantization inner
product? This is a one-recipe, claim-ineligible development falsifier. It does
not tune on labels and a pass requires unchanged replication on fresh domains.

The incumbent is deterministic one-thread Faiss 1.12.0
`OPQ64_768,PQ64x8`, measured on 35,257 normalized gallery rows and 3,050
deterministic held-out pseudoqueries at `0.582765` mMP@5 and `0.517705` R@1.
Exact float scores `0.590541 / 0.523934`.

## Frozen construction

Let `S = X.T @ X / n` be the uncentred float64 second moment of the gallery.
Eigendecompose `S = V diag(lambda) V.T`, floor eigenvalues at
`lambda_max * 1e-6`, and define

- gallery coordinates `y = x V sqrt(lambda)`;
- query coordinates `z = q V invsqrt(lambda)`.

Then `z @ y.T = q @ x.T` before quantization. Fit one unchanged
`OPQ64_768,PQ64x8` to `y`, store exactly 64 bytes per gallery row, decode only
for this quality reference, and rank by exact inner product `z @ y_hat.T`.
There is no reconstruction renormalization, original-vector reranking,
candidate pruning, label use, seed sweep, eigenvalue-floor sweep, or alternate
split. The existing OPQ arm is refit and scored in the same process.

The script must require exact feature SHA-256
`0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`,
the fixed population, one OpenMP thread, exactly 64 code bytes, and equality of
the unquantized original and transformed top-five rankings. It records the
maximum sampled inner-product error, eigenvalue range/floor, stored transform
bytes, fit times, reconstruction errors, and the descriptive 129-query shifted
split.

## Decision

On the 3,050-query primary split, compute paired query bootstrap intervals with
10,000 fixed-seed replicates. Promote only if all hold:

1. candidate minus incumbent mMP@5 is at least `+0.002`;
2. its 95% interval lower bound is above zero;
3. the R@1 delta 95% lower bound is above `-0.001`;
4. every authority and exactness check passes.

The shifted 129-query split cannot reverse the primary decision. Failure closes
this exact full-rank second-moment precoder, not all linear transforms or all
quantizers. The separately prepared two-stage product-residual 64-byte codec is
the next bounded representation fallback. CUDA/cuTile/CUDA-Oxide work remains
blocked until a quality candidate passes and replicates.
