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
the same schedule/PCA/model hashes and first ten input batch hashes; source
files may differ only in the explicit loss option and its scoring receipt.
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
