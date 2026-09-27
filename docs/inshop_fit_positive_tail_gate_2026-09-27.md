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
