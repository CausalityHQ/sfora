# Pooler objective-conflict screen

One mechanism: ArcFace class-proxy pressure may oppose image-bank ranking
updates at the raw pooler. Test whether a **rank-priority pooler-gradient
projection** has a substantial operation to perform. This is gradient
projection prior art, not a novelty claim. It differs from earlier
Proxy-Anchor per-positive conflicts and the failed live-head bank freshness
arm. No coefficient, loss, optimizer, batch or source is changed by this
diagnostic.

Before changing training, run <=120s CPU on the SAME authenticated official
TRAIN fit cache/PCA seed179024 and first17 scheduled batches as the head-first
screen. Skip ranking in batches containing a singleton exactly as the current
trainer does. At initialization, separately differentiate native ArcFace
(margin0.3/scale64) and coefficient8 bank SmoothAP against each raw source
vector. Measure cosine and weighted-rank/ArcFace norm ratio only on nonzero,
finite rank-active query gradients. This is **pooler output geometry**, not
encoder parameter-gradient alignment; the latter uses a different metric.

Frozen advance rules: at least25% of valid source rows have negative cosine,
and the median weighted-rank/ArcFace gradient norm ratio is at least10%.
Require original source/cache/fit/PCA/schedule authority, >=10 active batches,
>=90% finite nonzero valid rows within active batches, and <=120s CPU wall.
Otherwise close this fixed pooler-projection configuration before GPU work.
Do not change its coordinate space or thresholds after this result.

Passing only authorizes a frozen bounded gradient/cost smoke with an exact
unchanged main-only control, plus a matched projection/sham or rescaling
control to distinguish direction from changed clipping magnitude. Projecting
an output gradient cannot guarantee parameter-level task alignment or unseen
quality. No held outcome, official query/gallery, R@1/mAP@R or serving
latency is read here; a new TRAIN holdout and paired seeds are later gates.

## Terminal cached result: KILL

The single CPU process exited0 in5.165150 main-function seconds. Protocol
and code were pushed at `e933e41f` before outcomes. Receipt
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-objective-conflict-cache-v1.json`,
SHA-256 `7ba7a0349518b093de0e1d0324371b6406ff6679f15abf6e2fc2857e01356eb1`.

| In-Shop official TRAIN fit-only initial source gradients | Measured | Decision |
|---|---:|---|
|15 active batches/960 valid source rows |12 opposed rows,1.25%; median cosine0.620746 | KILL: below25% conflict floor |
| Weighted-rank/ArcFace gradient norm ratio | Median0.509859 | Material rank signal, passes>=0.1 |
| Nonzero finite gradient coverage |960/960 active rows | Pass |
| CPU main wall |5.165150s | Pass<=120s |
| R@1/mAP@R, encoder training cost/VRAM, image-to-top-k p50/p95/p99/QPS | Not measured | No new claim |

Independent replay recomputed the exact conflict count/fraction, ratio
median, active/valid coverage, all criteria and negative decision. Ranking
is not a negligible initialization signal here, and its pooler gradient
mostly agrees with identity pressure. This rejects the specified initial
pooler-projection mechanism, without claiming parameter-gradient agreement
later in training. No GPU projection, held read, threshold search, production
edit or package rebuild followed. The earlier head warm-up remains closed.
The next comparison should change pretrained representation, keeping these
training/default objective gates intact; it must inspect already tested
substrates and pass cheap architecture/data/gradient/cost checks first.
