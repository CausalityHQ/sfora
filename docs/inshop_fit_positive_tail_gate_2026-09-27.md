# In-Shop TRAIN fit-positive-tail falsifier, 27 September 2026

The previously frozen 6,354-query/6,245-gallery TRAIN roles show that losing
one well-ranked positive during gallery thinning induces many misses. The
existing detached-bank SmoothAP uses a sigmoid comparison with slope 200;
large negative pair margins may leave weak gradients for a product's worst
positive. This is a mechanism hypothesis, not a measured training defect.

Before editing the loss, export the pinned seed-179026 `freeze_emb` 1,000-update
checkpoint (training receipt SHA-256
`76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc`)
on its **13,283 official In-Shop TRAIN fit images**. For each non-singleton
fit image, score the normalized 128-D float head against the full fit gallery,
exclude itself, and compute best different-product score minus worst same-product
score. Record all margins, source/checkpoint/model hashes, row inventory, wall
and peak PyTorch CUDA allocation. Partition labels reconstruct the fit rows;
no held TRAIN image is encoded or scored, and official query/gallery stay closed.

The frozen F0 rule advances the worst-positive bank-hinge lane only if at least
**10%** of eligible fit anchors have margin **>0.03**. Otherwise stop before
loss code or training. A pass establishes only that the proposed term has a
substantial target; it does not show that the term improves quality. Any later
paired arm must keep the same checkpoint source, architecture, data, seed,
schedule, budget and 130-byte exact packed scorer, and use an independently
frozen TRAIN-only quality/cost gate before official evaluation.

## Terminal F0 result

The first isolated DGX unit failed at import before loading the checkpoint;
its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-positive-tail-v1/failed-import.log)
has SHA-256 `ad6cefe6d42860e2bc34c45a82db2937ed8a32020a25cb44868dfae9db49cc45`.
The second failed at source authority before encoding because the check read
PyTorch's decorated wrapper path rather than the pinned helper module; its
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-positive-tail-v1/failed-helper-source.log)
has SHA-256 `257340b93b88a2a98743c8b70802cd1dd4b0ebc62ca4e77a76a9f61537e963e1`.
Both failures preceded a quality read. The corrected third unit exited 0
(invocation `7bf9a13f18dc4c088c7be779ec085b61`). Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-positive-tail-v1/receipt.json)
has SHA-256 `953be9e6eab846f22e3e187f043eebfaf3161af0bbe4a752f6c68987ffb7fb49`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-positive-tail-v1/success-journal.log)
has SHA-256 `ada6568b534af788f7f92c819b2a5352e6f1e3164918404bbb90fe7ada1bea31`.

Exactly **2,506/13,271 = 18.8833%** eligible TRAIN fit anchors exceed the
0.03 margin, so F0 **passes** its frozen 10% floor. The pinned feature export
and score took **70.675 s** on DGX Spark GB10; peak PyTorch allocated CUDA was
**1,886,321,152 bytes**. A local replay verified every finite margin, all
13,271 unique eligible ordinals, the count/fraction/decision and corrected
source SHA. This result permits a separately frozen paired TRAIN-only loss
screen; it does not support an official quality or method claim.

## Frozen F1 paired TRAIN screen

The treatment changes only the detached-bank rank loss: for each eligible
anchor, add `0.25 × mean[0.05 × softplus((best_negative − worst_positive +
0.02) / 0.05)]` to the existing SmoothAP term. Positives and negatives are
the same fit-bank rows already scored by SmoothAP; the anchor itself and padded
positive ordinals are excluded. The coefficient, margin and temperature match
the earlier rejected best-positive impostor screen, isolating which positive
is moved. This is prior-art-like batch-hard ranking, with no novelty claim.

Run paired seed-179026 `freeze_emb` control and opt-in worst-positive treatment
on DGX Spark, 17-update smoke then 100-update TRAIN screen. Hold fixed the
SigLIP2 Large/256 source, 13,283 fit products' images, 12,599 product-disjoint
held images, PCA head, 128-D output, BF16, batch64 schedule, augmentations,
ArcFace, bank refresh, optimizer and learning rates, native 130-byte packed
scorer and fixed 6,354-query/6,245-gallery TRAIN roles. Both arms must have
the same schedule/PCA/model, source-file and first-ten input batch hashes;
only the explicit loss arm differs.
No official query/gallery row is used.

Smoke must finish 17 finite optimizer steps with no skip, correct checkpoint
geometry and rank-call counts, treatment/control wall ≤1.10 and peak PyTorch
allocated CUDA ≤1.05. Failure stops before the 100-update pair. The 100-update
pair must repeat those authority/finiteness checks. Define `a` for each fixed
TRAIN-role query as the number of same-product images in the **full 12,599-row
held gallery**, excluding the query, whose packed score is strictly greater
than that query's best different-product score in the same full gallery.
The primary mechanism gate requires treatment minus control `Pr(a≥3)` among
the 6,354 fixed-role queries to be ≥**+1.0 percentage point**, with a paired
product-cluster bootstrap 95% lower bound above zero. Guardrails require
fixed-role packed R@1 delta ≥**−0.15 pp**, symmetric held packed mAP@R delta
≥**−0.002**, training wall ratio ≤**1.02**, and peak PyTorch allocated CUDA
ratio ≤**1.005**. A failure stops this treatment before more seeds, official
evaluation or production promotion. A pass permits a separate frozen
independent-seed full-budget TRAIN gate; it does not establish quality or SOTA.

## Terminal F1 result

The paired 17-update DGX Spark GB10 smokes exited 0. Their
[control](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/smoke/control.json)
and [treatment](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/smoke/treatment.json)
receipt SHA-256 values are `bdb385375a699718e069d4af5b6172adfbdd2cf62ab38b3d9101fb12bce47fbc`
and `908cfae29fa75eade88473f4afb0d8287dfaae2d87bbf3ed601f7a2ad69c32d4`;
their [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/smoke/journal.log)
has SHA-256 `9ac99a56c44248e21dee57a515afcb04a871b43a946b5498da07f6e1e8a32890`.
Source, schedule, PCA, rows, first-ten inputs and 14 rank-active steps matched.
Both losses and gradients were finite. Treatment/control training wall was
**0.9960×** and peak allocated CUDA **1.0003×**. Checkpoint audit found
400 vision and two head tensors in each arm, 195 frozen vision tensors equal
and 205 upper tensors changed. The frozen smoke gates passed.

Both 100-update runs then completed 100 finite updates and identical
89 rank-active steps. Their [control](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/full/control.json)
and [treatment](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/full/treatment.json)
receipt SHA-256 values are `1832d35513b2954936237c1e306b2ff3f0e2bb3e466a908823310642db916ef9`
and `33789dba69cf7a61afd2c0be67508c8d31e2a69cf977efaeaa00ef381bea777b`;
the [training journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/full/training-journal.log)
has SHA-256 `7852bb75fe690004fb9ff58f15b2893669aad1a2148a5fc773e696f1953b7a49`.
All paired input, source and schedule hashes matched. A scorer unit first
failed before writing a result because `tileiras` was absent from its isolated
environment; its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/full/failed-score-journal.log)
has SHA-256 `bb4641dc238c83c43db7e3c66a4cf2cb2f712df595332fe7dbf7839af5112ec8`.
The corrected scorer unit exited 0 (invocation
`dbd322b8f6f94c5f99a1464c5ca98911`) with the pinned native scorer and
`tileiras` binary. Its [raw paired result](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/full/pair-score.json)
has SHA-256 `726d9dfd72b9ca7e8f2594bb7d0a503da4109f8b8e34bef14142f077aab22784`;
the [terminal scorer journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-tail-f1-v1/full/success-score-journal.log)
has SHA-256 `c3f48f1ac74391799093bcbe405a9d5b6405bd32a17296a0118fe5c8bb2b1e0a`.

| Seed 179026, official TRAIN product-disjoint held panel | Control | Worst-positive treatment | Frozen gate |
| --- | ---: | ---: | --- |
| Fixed-role `a≥3`, 6,354 queries/full 12,599 held gallery | **5,131/6,354 = 80.7523%** | **5,123/6,354 = 80.6264%** | **−0.1259 pp**, product-bootstrap lower **−0.4284 pp**; fail ≥+1.0 pp and lower>0 |
| Fixed-role native packed Recall@1, 6,354 queries/6,245 gallery | **6,068/6,354 = 95.4989%** | **6,074/6,354 = 95.5933%** | +0.0944 pp; pass nonregression guard |
| Symmetric packed Recall@1 / mAP@R, 12,599 self-excluded held rows | **97.3411% / 0.775415** | **97.3014% / 0.775028** | mAP delta **−0.000387**; pass guard |
| Training wall including bank init, 6,400 sampled images | **77.405 s / 82.68 images/s** | **77.803 s / 82.26 images/s** | ratio **1.0051×**; pass ≤1.02× |
| Peak PyTorch allocated CUDA, DGX Spark GB10 | **12.939 GB** | **12.942 GB** | ratio **1.0003×**; pass ≤1.005× |

Local replay re-aggregated the 6,354 per-query hit and coverage entries from
the DGX scorer receipt and verified source/receipt hashes and the conjunction
of the frozen gates. The held embeddings remain on DGX, bound by the training
receipt hashes; they were not rescored locally. The primary
mechanism gate fails despite six net fixed-role R@1 rescues. Stop the
worst-positive treatment before other seeds, 1,000-update training, official
query/gallery or production promotion. The predeclared 100-update mechanism
readout is only an exploratory TRAIN decision; it does not prove the loss
cannot help at another budget or on another dataset. The opt-in production
loss path was removed in `97b4dceb`; commit `af2aaf71` preserves the exact
experiment source for replay.
