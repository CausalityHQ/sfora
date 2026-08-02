# In-Shop fragmentation cross-series retrieval preregistration

Recorded after the three-seed series-adjusted raw leave-one-out gap was known
(+5.476, +5.744, +5.587 points), but before computing any retrieval result with
same-series positives removed.

## Motivation

The existing class outcome counts a query correct when its nearest neighbour is
any image of the same item. Stable graph components align with filename series
at ARI 0.754--0.761, so a fragmented class may score well simply because each
query has a visually easy positive inside its own series. Exact matching on
series composition cannot remove that outcome-definition shortcut. The causal
question is whether fragmentation also helps retrieve a *different* series of
the same item.

## Locked analysis

Use the exact three epoch-10 In-Shop Proxy Anchor packs and the unchanged
symmetrized 1-NN fragmentation exposure. Parse the filename series token exactly
as in the prior audit. Retain identities containing at least two series.

For each query, rank the complete training split after excluding:

1. the query itself; and
2. every same-identity candidate with the same series token as the query.

Do not exclude other identities that happen to reuse the same numeric series
token. A query is correct only when the nearest remaining image has its identity,
which necessarily makes it a cross-series positive. Average correctness per
identity, then class-balance.

Report the unadjusted fragmented-minus-connected cross-series R@1 difference and
the primary exact-stratified estimate using the already locked sorted series-size
signature plus global quintiles of mean within-class cosine and nearest-foreign-
centroid cosine. Bins are recomputed independently per seed exactly as before.
Also report coverage and the two exposure-arm means. Do not select queries,
series pairs, or graph thresholds.

## Prediction and falsification

The prospective prediction is that the positive association is a same-series
retrieval shortcut: the primary adjusted fragmented-minus-connected
cross-series R@1 gap is **<= 0 in every seed**, at **>= 20%** retained-class
coverage.

- The shortcut prediction is falsified if the adjusted cross-series gap remains
  **> +2.0 points in all three seeds** at >=20% coverage.
- A mixture of signs, values in (0, +2], or insufficient coverage is
  inconclusive and cannot motivate a method.
- A replicated negative or zero result closes fragmentation-derived supervision:
  preserving the components helps only when evaluation supplies a same-series
  neighbour and does not transfer identity across its own modes.

A positive survival would still be observational, but it would rule out the
strongest outcome-definition artifact and motivate one further causal
diagnostic. It would not by itself make hierarchy, clustering, cross-series
mining, topology or diversity novel.

## Result (2026-08-02)

The registered test is **inconclusive because its overlap condition failed**.
Among 1,274 multi-series identities, 1,264 / 1,259 / 1,260 were fragmented in
seeds 0 / 1 / 2, leaving only 10 / 15 / 14 connected controls. Exact
series-signature and geometry matching retained only **7.46%**, **9.97%**, and
**6.83%** of eligible classes, below the locked 20% minimum.

The descriptive direction is nevertheless large and replicated. Unadjusted
fragmented-minus-connected cross-series R@1 was **-41.917**, **-22.047**, and
**-28.672 points**. The under-covered matched estimates were **-27.776**,
**-22.363**, and **-28.014 points**. These numbers cannot satisfy the registered
claim because the connected comparison arm is nearly absent; they do show why a
binary fragmentation exposure is poorly identified once the outcome requires a
different series.

The analyzer SHA-256 was
`6f379422371184654f0598765594eeba87a05efa5839ece85d2bdd64d2bc6700`.
Seed result SHA-256 values were
`7dac1406f44efdf4b660756d918cebe5394b1410c70126af0bb1b0cb8edbe6e0`,
`bc00a812573521ddb6515c51926f30b5fa65a8bac28492ea469710fcb05cef25`, and
`282740c1096d66c2eb492bbff56c519f16d86ffb0766d818229609037a707479`.

No method is authorized. The next defensible question is dataset-level rather
than another exposure model: determine prospectively whether official In-Shop
query/gallery evaluation normally offers each query a same-series positive.
