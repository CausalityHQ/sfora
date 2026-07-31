# MOPC — mutual-ownership proxy calibration

**Gate 1 recorded 2026-07-31 before prior-art audit, implementation, or GPU
use.**

## Repository provenance

Using the exact epoch-10 In-Shop Proxy Anchor checkpoint and all 25,882 training
embeddings, the learned proxies and empirical normalized class centroids show a
large directional mismatch:

| measurement | result |
|---|---:|
| mean cosine(proxy, own centroid) | 0.09888 |
| 5th percentile | -0.00123 |
| proxy's nearest centroid is its own | 0.99975 |
| centroid's nearest proxy is its own | **0.70303** |
| image's highest-scoring proxy is its own | 0.65308 |
| correlation of proxy-centroid cosine with class LOO R@1 (n>=3) | 0.1391 |

Thus proxy ownership is nearly perfect in one direction but fails for about 30%
of empirical class centroids in the reverse direction. This is not simply the
known complaint that one centre cannot represent multiple modes: the aggregate
class representative itself lies in another proxy's Voronoi cell.

## Cross-disciplinary mechanism

MOPC borrows mutual acceptance from stable matching. Ordinary Proxy Anchor
optimizes sample-to-proxy terms but does not explicitly require the aggregate
class and its proxy to be reciprocal nearest neighbours. At a frozen warm-up
point, maintain training-only empirical class centroids (or an exact epochwise
refresh), and add a bounded centroid-side ownership constraint:

`sim(mu_c, p_c) >= max_{r != c} sim(mu_c, p_r) + margin`.

The standard Proxy Anchor objective remains intact. No member is pulled directly
to the centroid, no sub-centres are introduced, and the centroids are discarded
at test. The constraint stops once reciprocal Voronoi ownership is achieved, so
it need not erase within-class structure.

The claimed mechanism is *bidirectional ownership calibration between learned
proxies and empirical class support*, motivated by the measured 99.97% versus
70.30% asymmetry—not generic centre loss or additional sample attraction.

## Gate-2 attack required

Before implementation, search primary sources for:

- proxy-to-class-centroid alignment or reciprocal proxy assignment in DML;
- losses applied to batch/class means against learned proxies;
- prototype regularization, proxy synthesis, ProxyGML, SoftTriple, and proxy
  assignment methods;
- mutual-nearest prototype matching in supervised classification, face
  recognition, and re-identification;
- SwAV-style swapped assignments and class-prototype calibration.

MOPC is dead if existing work already enforces that empirical class centroids
select their labelled learned proxy over other proxies. It is also dead if the
centroid hinge is merely an algebraic minibatch aggregation of a standard
sample-to-proxy classification loss with no distinct effect.

