# In-Shop proxy-geometry audit

**Measurement recorded 2026-07-31; no new GPU used.**

The audit combines `inshop_pa_epoch10_operating.train.npz` with
`arcg_inshop_pa_epoch10_seed0.pt`, normalizes the 512-dimensional embeddings,
learned proxies, and empirical per-identity centroids, and computes exact cosine
ownership.

The mean proxy-to-own-centroid cosine is 0.09888 (5th percentile -0.00123,
minimum -0.13277). Nevertheless, 99.975% of proxies have their labelled
centroid as the nearest empirical centroid. The reverse relation holds for only
70.303% of centroids. At the image level, only 65.308% score their labelled
proxy above every other proxy. Proxy-centroid cosine correlates only 0.1391 with
leave-one-out class R@1 among classes with at least three images.

The important observation is directional: proxies are class-specific relative
to other centroids, but many class centroids are more compatible with a foreign
proxy than their own. Any candidate derived from this result must distinguish
reciprocal ownership from centre attraction and must survive the extensive
proxy/prototype-assignment literature.

