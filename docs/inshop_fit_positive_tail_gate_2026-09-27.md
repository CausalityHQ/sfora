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
