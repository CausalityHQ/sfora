# Positive-cardinality diagnostic preregistration (2026-08-07)

The apparent unseen positive deficit uses a max over unequal same-class pools:
training rows have roughly 5.5 other images per identity, while query/gallery
has roughly 3.2. Before treating the −0.05305 positive change as a learned
mechanism, this CPU-only control samples exactly three same-class peers on
both sides (20 deterministic resamples per seed) from the cached matched
pre-head exports.

Decision thresholds, fixed before the recomputation:

- **Cardinality artifact supported** if the mean unseen-minus-seen positive
  gap shrinks below −0.020 (at least 60% attenuation from −0.05305).
- **A positive-side mechanism remains plausible** only if the gap stays below
  −0.040 with a four-seed bootstrap interval excluding −0.020.
- Intermediate values are inconclusive; no GPU method is authorized.
