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
query sign transform `q_tilde = (-q0, qs)`. With signature `(-,+,...,+)`, this
score is `<q,g>_L = -cosh(d_L(q,g))`, so its maximum is the nearest item. The
opposite transform `(q0,-qs)` yields `cosh(d_L)` and would make MIPS select the
farthest item; it is forbidden and must be a regression-test mutant. Therefore
exact retrieval uses one ordinary GEMM or FAISS `METRIC_INNER_PRODUCT`; IVF-PQ
remains additive over subquantizers. The geometry is not an excuse for a custom
distance kernel.

## Controls

At identical dimension, parameter count, train identities, split, optimizer,
and seeds, compare:

1. Lorentz head;
2. normalized cosine-softmax linear head with matched negative weighting;
3. Matryoshka-style truncation head;
4. WCCN followed by calibrated cosine.

The no-training lift also requires two functional controls. Spatial-only
Euclidean distance on `sinh(a) * z_hat` removes exactly the Lorentz time term.
Power-law norm scorers with powers `1` and `3` replace the `sinh/cosh`
functions while retaining the same angular and radial inputs. A result matched
by either family is norm-weighted rescoring, not evidence for hyperbolic
geometry.

The controls are exact, not fitted.  For clipped projected radii `a` and `b`
and angular cosine `c`, spatial-only ranks by

```text
sinh(a) sinh(b) c - sinh(b)^2 / 2.
```

For `p in {1,3}`, define `h_p(r)=r^p` and
`t_p(r)=sqrt(1+h_p(r)^2)`.  The power control ranks by

```text
h_p(a) / t_p(a) * h_p(b) * c - t_p(b).
```

This preserves the Lorentz scorer's normalized query factor, gallery radial
weight, and additive radial penalty while replacing its exponential radial
map by a fixed power map.  A power tie therefore falsifies attribution to the
specific `sinh/cosh` map rather than merely offering an easier endpoint.

The matched hard-negative control is load-bearing because published analysis
shows that apparent hyperbolic gains can come from implicit negative weighting.

## Falsifier ladder

1. `L0`, no training: estimate relative delta-hyperbolicity on ten fixed
   2,000-row train-only subsamples. For four points, sort the three paired
   distance sums as `L >= M >= S`, set `delta=(L-M)/2`, and report the exact
   convention `delta_rel=2*delta/diameter`. Close if mean `delta_rel > 0.30` on
   at least two of In-Shop, Cars196, and SOP. Paired column-permutation and
   spectrum-matched Gaussian nulls are mandatory because distance
   concentration can make an unstructured high-dimensional cloud appear to
   have low relative delta.
2. `L1`, no training: PCA each train export to the registered dimensions and
   set scales by fixed target median train radii `{0.125,0.25,0.5,1,2,4,8}`.
   The small-scale endpoint is PCA-Euclidean and the fully clipped endpoint is
   PCA-cosine. The primary statistic is the maximum, over interior scales, of
   `R1(scale) - max(R1(Euclidean), R1(cosine))`; its query-identity clustered
   bootstrap must recompute that maximum inside every replicate. L1 passes
   only if its lower bound exceeds zero and its point is at least three
   bootstrap standard errors on at least two datasets, and the winning
   sinh/cosh scorer beats both the spatial-only and power-law controls. If L0
   and L1 both fail, close.
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

## Interpretation boundary

For projected vector `x`, let `a=s*||x||` after clipping. Per query, Lorentz
retrieval ranks gallery rows by

```text
tanh(a_query) * sinh(a_gallery) * cosine(query,gallery) - cosh(a_gallery).
```

Thus the no-training L1 scorer is exactly a query-conditioned norm weighting
of cosine, not an independently identifiable geometric mechanism. L1 may
falsify the lane or nominate a useful low-dimensional norm-weighted rescoring
function; it cannot by itself establish a hyperbolic claim. Escalation to L2
requires the endpoint and function-family controls above to pass prospectively.

## Relationship to CTM and kernels

CTM is a zero-training `width+1` descriptor test; the Lorentz rider is a small
train-fitted compression head. They share frozen exports and retrieval metrics
but answer different questions. Neither should block the other.

The preferred implementation path is maintained GEMM plus a compiled
elementwise lift. A custom native kernel is considered only after profiling a
winning head and only if one unsupported operation consumes at least 10% of
step or query time after maintained fusion.
