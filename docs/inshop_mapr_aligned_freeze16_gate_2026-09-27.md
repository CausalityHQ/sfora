# In-Shop mAP@R-aligned rank loss gate, 27 September 2026

The first-16-block freeze saves 17.32% training wall but loses 0.007635
mAP@R against the same-seed first-12-block control on 12,599 unseen-product
official **TRAIN** held-only queries/gallery. This arm keeps the freeze-16
architecture, source cache, PCA, bank, batches, BF16, ArcFace, rank coefficient
8, optimizer and 1,000 × 64 image budget fixed. It changes only the existing
SmoothAP bank term: multiply each positive's soft precision by a differentiable
one-rank-width gate at that query's positive count R. That approximates the
reported mAP@R cutoff rather than optimizing full AP. It may improve the
multi-positive ranking loss; there is no claim of novelty or serving speed.

Before reading treatment quality, freeze this decision rule for seed 179024:

1. A 17-update smoke must show finite loss and gradients, identical source
   cache, split, schedule, PCA and first-ten image inputs to the archived
   freeze-16 smoke, exactly frozen embeddings plus blocks 0–15, and a changed
   upper encoder tensor. Otherwise stop.
2. Run one 1,000-update treatment. Compare its packed per-query scores against
   both archived same-seed controls, first-12-block freeze (R@1 **98.5475%**,
   mAP@R **0.839099**, wall **747.058 s**) and first-16-block freeze (R@1
   **98.4602%**, mAP@R **0.831464**, wall **617.658 s**). Require treatment
   mAP@R at least the freeze-12 point and a paired product-bootstrap 95%
   lower bound of its mAP@R difference at least **−0.003**; R@1 at least
   freeze-12 minus **0.2 percentage points**; and training wall including
   bank initialization at most **95%** of freeze-12. Source, checkpoint,
   per-query metric and finite-update receipts must validate. If any fails,
   stop before another seed or official TEST.
3. A passing first seed authorizes two independent paired TRAIN seeds and a
   distinct product-disjoint split. Only that replication could establish a
   learning-method gain. The existing official In-Shop TEST has prior reads;
   any later official result is exploratory. The full 24-block serving encoder
   is unchanged, so public image-to-top-k latency must be measured separately
   against UNICOM and cannot be inferred from this training gate.

The rank cutoff is fixed at R = the count of fit-bank positives excluding
self. The gate is `sigmoid(R + 0.5 - soft_rank)` with a rank scale of 1.0;
there is no tuning on held identities. The hypothesis is that full-bank
SmoothAP spends gradient on positives far below the evaluation cutoff,
especially products with many images. A null or negative gate falsifies this
specific hypothesis for the freeze-16 recipe.

## Terminal evidence and decision

The sole DGX Spark GB10 17-update smoke exited 0. It matched the archived
freeze-16 seed-179024 source cache, fit/held split, schedule, PCA and first ten
image inputs, had finite loss and gradients, and kept all **259** pretrained
embedding/block-0–15 tensors exact; **141** upper encoder tensors changed.
The [smoke receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-mapr-freeze16-179024-v1/smoke.json)
SHA-256 is `b9254d09c79af37075037cd9d618eb2e1ea7662a9d9cead5a5bad6c39ee7e7e3`;
its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-mapr-freeze16-179024-v1/smoke.journal.log)
SHA-256 is `0dcd229a5315571273aca4f1deb854b3a04d77739c5f83b2f96b503c2e57194d`.

The single 1,000-update treatment also exited 0, with **923** active rank
updates, finite gradients, held-image export and exact packed scoring. The
[source-bound decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-mapr-freeze16-179024-v1/decision.json)
checked the archived control receipt hashes, checkpoint and per-query metric
authorities, unchanged source modules, source cache, split, first-ten inputs,
schedule, PCA and hardware. On the same 12,599 official **TRAIN** held-only
symmetric queries/gallery:

| Seed 179024 arm | Packed R@1 | mAP@R | Training wall / 64,000 images | Images/s | Peak allocated CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Freeze embeddings + blocks 0–11, full SmoothAP | 98.5475% | 0.839099 | 747.058 s | 85.67 | 12.939 GB |
| Freeze embeddings + blocks 0–15, full SmoothAP | 98.4602% | 0.831464 | 617.658 s | 103.62 | 9.745 GB |
| Freeze embeddings + blocks 0–15, mAP@R cutoff | **98.4126%** | **0.833200** | **617.144 s** | **103.70** | **9.745 GB** |

The cutoff recovered **0.001737** mAP@R relative to freeze-16 but lost
**0.0476 percentage points** R@1. Against the required freeze-12 control,
mAP@R remains **−0.005898**, paired product-bootstrap 95% interval
**[−0.007951, −0.003816]**; R@1 is **−0.1349 pp**. Training wall including
bank initialization is **618.589 vs 748.621 s** (**17.37% faster**). The
source-bound mAP point and lower-bound gates fail, so
`advance_independent_seeds=false`: stop before other seeds, official TEST or
production promotion. The full 24-block serving path is unchanged and no
serving-latency improvement follows.

A post hoc grouping of the same per-query AP differences by held-product
size shows cutoff minus freeze-16 **−0.00077** for 2–3 images (650 queries),
**+0.00026** for 4–5 (4,846), **+0.00378** for 6–8 (2,421), and **+0.00255**
for 9 or more (4,682). It supports the intended multi-positive target but
does not reverse the frozen gate or establish causality across seeds.

The [full receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-mapr-freeze16-179024-v1/full.json)
SHA-256 is `ed8eaa9f1adbc09794b0ef41136bca36f3fe8f2daaf73aed947607fab9a68259`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-mapr-freeze16-179024-v1/full.journal.log)
SHA-256 is `32bb735045bc4d6fc12d7c8477164bd66969937d84ac0e15077efad079f1c600`;
and the decision SHA-256 is
`185c8e995f5eeeb7d2d9dd2c532d9239503e8ffe4cd5b3e615c39a3f5d5ab7e0`.
The analyzer source SHA-256 is
`295bce42c992a8d044f72242d6695d6276a0770af7f76bab9a6b7de69b2a1ff1`.
Two analyzer attempts stopped before issuing a decision because the archived
freeze-12 trainer has a historical filename suffix and lacks later optional
metadata fields; the corrected analyzer accepts only that pinned archived
receipt's schema. The full local repository suite passed **5,536 tests**, with
12 skips, after the training-code change.
