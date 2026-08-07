# Pass 98 — error-conditioned local/global fusion (2026-08-07)

## Dead at Gate 2

The repository's CUB error decomposition (48.1% local-evidence failures,
51.9% centroid-overlap failures) suggests routing a query between a local and a
global channel according to its failure type.  This would be a useful
measurement-driven idea only if its train-time object were new.

It is not. Global/local descriptor fusion and adaptive local-global gates are
well established in image retrieval, including GLAD, multi-head DML with global
and local representations, and unifying deep local/global image search.
Conditioning the gate on a proxy margin or centroid score changes the routing
criterion, not the mechanism. No GPU run occurred.
