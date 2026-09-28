# Category pooling CPU falsifier: KILL

The original CPU job (native session `11287`) ended with exit 0 after
**18.776 seconds** of script wall. All three arms completed 100 finite updates
× 64 cached-feature rows. No DGX work, encoder updates or official evaluation
was performed. The frozen script and narrow regression check are committed
in `ac7daefb`; quality was first read after every arm finished training.

## Measured internal TRAIN result

In-Shop official TRAIN, **original fit identities only**: internal
product-disjoint A validation, **1,320 queries / 1,176 gallery images /
394 products**, fixed alternating query/gallery roles. Pretrained SigLIP2
Large/256 source features, common A-only image-PCA128 head initialization,
native ArcFace plus detached SmoothAP bank and exact int8/fp16-norm CPU scorer.

| Arm | Packed Recall@1 | mAP@R | Head/proxy training + initialization | Bank rows |
| --- | ---: | ---: | ---: | ---: |
| A-only control | 92.7273% | 0.726682 | 1.780 s | 2,264 |
| A+B true product identities | 92.8788% | 0.723057 | 4.921 s | 6,551 |
| Same A+B rows, B product labels shuffled | 91.0606% | 0.702958 | 10.503 s | 6,551 |

True pooling versus control: **+0.1515 percentage points Recall** (two net
queries), product-cluster 95% interval **[−0.2778, +0.5700] pp**; mAP change
**−0.3624 pp**. This misses the frozen +0.3 pp Recall floor, positive lower
bound and nonregressing mAP requirements. Versus shuffled labels, it gains
1.8182 Recall points, interval [+0.9817, +2.7508], and 2.0100 mAP points;
that control passes but cannot rescue the native-control failure.

Both pooling arms have identical feature samples and product counts; 99.51%
of B labels change under the sham. At equal total updates, pooling spends
half its slots on A; control spends all slots on A. Runtime is measured and
unequal despite equal update counts. Cache acquisition, encoder training,
serving p50/p95/p99, QPS and GPU VRAM were **not measured** by this experiment.
This is one optimizer seed; product bootstrap intervals condition on these
trained heads and do not describe variation across training seeds.

## Production decision

**Do not add a pooled-dataset loader or promote this recipe on this evidence.**
Close the fixed cached category-pooling proxy without additional updates,
seeds or GPU escalation. Its failure does not prove that actual joint
SOP/In-Shop encoder training is ineffective. The native production training
and serving recipe remains in place; the full joint quality-and-speed goal
is still unmet.

Independent verification reproduced metadata roles, every reported mean and
both 5,000-draw cluster intervals exactly; source matches the prior freeze.
Receipt SHA-256:
`ad179a96ac79b8b9d805db2d877bbea647970021bfe526794362b4974d1a2ecf`.
Raw receipt, freeze, original log and verification are in
[the evidence archive](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pooled-identity-f0-v1/).
