# Published-checkpoint validation on corrected In-Shop pixels

Date: 2026-08-03. The interval and corpus profile were committed before inference
in audits 296 and 298.

The Proxy Anchor authors' published 512-dimensional BN-Inception checkpoint scores
**0.9176396118 Recall@1** on the replacement 256-by-256 corpus. This passes the
prospectively registered `[0.917, 0.921]` fidelity interval. The upstream
strict-negative-rank scorer, independent float64 Euclidean scorer, float64 cosine
scorer, and exact-tie expected scorer all return the identical value. There are no
nearest-neighbour ties, content duplicates, or cross-split content overlaps.

This is strong functional evidence that the mirror carries the benchmark pixels the
published checkpoint expects. It does not make the withdrawn `img_highres` experiments
valid. A new locally trained reference and every candidate comparison must use this
root and new artifacts.

The validated checkpoint SHA-256 is
`925cc1a1a5207f8f50ea6fa55189a2d8aed2523feca648132fe5cc74299f705a`.
The deciding DGX artifact is
`reports/generated/inshop_official_standard_published_proxy_anchor_diagnostic.json`.
