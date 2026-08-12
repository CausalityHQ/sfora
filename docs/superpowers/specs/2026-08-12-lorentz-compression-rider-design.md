# Lorentz Compression Rider Design

## Status

Prospective research design. This is a narrow compression/efficiency experiment,
not a claim that hyperbolic geometry is novel or globally superior. It may start
only from the same immutable modern frozen-embedding exports used by CTM.

## Question

Can a low-dimensional Lorentz head preserve more retrieval quality per stored
value than an equal-parameter cosine head, Matryoshka-style truncation, and
WCCN/Platt-cosine controls, while retaining ordinary GEMM/FAISS serving?

The live regime is `d in {8, 16, 32, 64}`. At 128 dimensions or more, modern
cosine transformer embeddings are expected to tie or outperform this method.

## Candidate

For frozen unit embedding `e in R^D`, learn one linear head `z = W e`, clip
`||z|| <= 2.5`, and lift it to the unit Lorentz hyperboloid:

```text
x0 = cosh(||z||)
xs = sinh(||z||) z / ||z||
x = (x0, xs),   <x,x>_L = -1
```

Use fixed curvature `-1`, ordinary AdamW, FP32, and distance-softmax proxies.
Compute distance with the stable identity

```text
u = (||xs - ys||^2 - (x0 - y0)^2) / 2
d_L(x,y) = log1p(u + sqrt(u(u+2))).
```

No Riemannian optimizer or FP64 fallback is allowed.

At inference, nearest Lorentz distance is maximum inner product after the
query sign transform `q_tilde = (q0, -qs)`. Therefore exact retrieval uses one
ordinary GEMM or FAISS `METRIC_INNER_PRODUCT`; IVF-PQ remains additive over
subquantizers. The geometry is not an excuse for a custom distance kernel.

## Controls

At identical dimension, parameter count, train identities, split, optimizer,
and seeds, compare:

1. Lorentz head;
2. normalized cosine-softmax linear head with matched negative weighting;
3. Matryoshka-style truncation head;
4. WCCN followed by calibrated cosine.

The matched hard-negative control is load-bearing because published analysis
shows that apparent hyperbolic gains can come from implicit negative weighting.

## Falsifier ladder

1. `L0`, no training: estimate relative delta-hyperbolicity on ten fixed
   2,000-row train-only subsamples. Close if mean `delta_rel > 0.30` on at least
   two of In-Shop, Cars196, and SOP.
2. `L1`, no training: PCA each train export to the registered dimensions, sweep
   a frozen Lorentz lift scale, and measure whether R@1 moves by at least three
   query-identity bootstrap standard errors. If L0 and L1 both fail, close.
3. `L2`, frozen-head training only: three seeds per method/dimension. At 16 and
   32 dimensions, Lorentz must beat the best matched control by at least 0.5
   R@1 point and three paired-bootstrap standard errors on label-disjoint
   validation, then replicate direction on Cars196. The advantage must shrink
   with dimension and be at most 0.2 point at 512 dimensions; a high-dimensional
   only gain is treated as a loss-weighting artifact.
4. `L3`, serving: FAISS IVF-PQ and augmented HNSW must lose no more than 0.3
   R@1 point at at least 10x the measured flat-scan throughput and match the
   Euclidean control at equal bytes.

Any nonfinite value, FP64 requirement, failed matched control, or failed
external replication closes the lane. Full backbone training is not authorized
by this design.

## Relationship to CTM and kernels

CTM is a zero-training `width+1` descriptor test; the Lorentz rider is a small
train-fitted compression head. They share frozen exports and retrieval metrics
but answer different questions. Neither should block the other.

The preferred implementation path is maintained GEMM plus a compiled
elementwise lift. A custom native kernel is considered only after profiling a
winning head and only if one unsupported operation consumes at least 10% of
step or query time after maintained fusion.

