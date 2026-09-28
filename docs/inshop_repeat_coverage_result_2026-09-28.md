# Repeated coverage: exposure GO, cached quality KILL

The fixed sampling change is closed. It increased exposure of diagnosed tail
anchors but reduced internal unseen-product Recall. The opt-in was removed
from the production sampler, and the experimental entry point was removed
from HEAD. Original code remains reproducible at **`ead79135`**. Native
sampling and its package source are restored; **17 sampler tests pass**.

## F0: metadata-only mechanism gate

On the original In-Shop TRAIN-fit 13,283 rows / 2,004 products, native seed179026
and repeated coverage share their first **261 updates** exactly. Both cover
every fit row in 1,000 × 64 slots. Repeated coverage changes only the fallback
after the initial coverage queue empties.

| Schedule | Rank-active updates | Tail mean ranking appearances | Tail anchors with zero appearances |
| --- | ---: | ---: | ---: |
| Native coverage once, then uniform identities | 923 | 2.1800 | 23 |
| Repeated coverage passes | 945 | 3.9042 | 0 |

The **1.791×** exposure gain passes the frozen ≥1.5× gate, with valid batches,
complete row coverage and unchanged initial prefix. Script wall **0.555 s**;
no new features, images or quality scoring. This gate qualified the next CPU
test; it did not establish quality or novelty.

## F1: fixed cached-feature paired quality smoke

Original In-Shop official **TRAIN fit identities only**, internal fixed
product split: **6,757 training rows / 995 products**, and **3,440 queries /
3,074 gallery images / 997 unseen products**. Singleton products are excluded
identically. The original outer TRAIN holdout and official query/gallery
remain closed.

Both arms use frozen pretrained SigLIP2 Large/256 features, the same native
image-PCA128 initialization and proxies, Linear128, ArcFace margin0.3/scale64
+ coefficient8 detached SmoothAP, AdamW1e-4/decay0.05/clip1, seed179032,
**400 × 64 cached rows**, exact packed scorer and identical query/gallery roles.
Only repeated coverage differs. Losses match exactly through update **134**;
the first changed input is update **135**. Both finish 400 finite updates.

| CPU head/proxy arm | Packed Recall@1 | mAP@R | Training + initialization |
| --- | ---: | ---: | ---: |
| Native control | 91.7151% (3,155 hits) | 0.709160 | 14.656 s |
| Repeated coverage | 91.2791% (3,140 hits) | 0.707791 | 18.126 s |

Repeated minus native: **−0.4360 Recall percentage points**, product-cluster
95% interval **[−0.7424, −0.1442] pp**; mAP **−0.1369 pp**, interval
**[−0.4247, +0.1251] pp**. Both frozen quality gates fail. Total CPU script
wall **35.703 s**, original native session13257 exit0. These are cached-head
costs, not encoder images/second. Cache acquisition, full encoder training,
image-to-top-k p50/p95/p99, QPS and GPU VRAM were not measured.

Independent verification reconstructed metadata roles, finite histories,
exact prefix replay, means and both 5,000-draw product intervals. Receipt
SHA-256: `f69836f473f27f276598ec3cb64bcba9ae33e6596ca839e349523799e64baf76`.
Intervals condition on this optimizer seed and do not estimate seed variation.

## Product decision

**No encoder/GPU escalation, additional steps or seeds, or production
promotion for this fixed proxy.** More tail exposure alone did not improve
quality here. The result does not prove every class-weighting method or a
full encoder implementation fails. Keep the native recipe; do not rescue
this arm by adjusting its thresholds or repeating the same schedule search.
The full SOP + In-Shop quality-and-speed objective remains unmet.

Raw evidence: [F0](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-repeat-coverage-f0-v1/)
and [F1](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-repeat-coverage-cached-f1-v1/).
