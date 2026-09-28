# Product-mean PCA confirmation: stop at Recall@1 guard

**Production decision:** retain native image-PCA initialization. Close promotion
of this fixed product-mean basis+centering initializer; do not run seed179030,
new official evaluation or new-checkpoint p99 certification for this arm.
The shared post-refresh bank-finiteness guard remains useful training code.

The [frozen gate](inshop_centroid_pca_seed_confirmation_gate_2026-09-28.md)
required nonnegative R@1 and mAP@R in every fresh paired seed. Both completed
pairs used 1,000 updates×64, identical sources/inputs, SigLIP2 Large/256,
native128 packed codes and fixed observed official TRAIN held roles:
6,354 queries, 6,245 gallery images, 1,993 held products. This is exploratory
on observed identities, not independent generalization or an official result.

| Seed | Packed R@1 native→products | R@1 effect, product95% interval (pp) | mAP@R native→products | mAP effect, product95% interval (pp) | Decision |
|---|---|---|---|---|---|
|179028|97.7022→97.8911%|+0.1889 [−0.0611,+0.4265]|0.850701→0.854244|+0.3543 [+0.1369,+0.5650]|Continue|
|179029|97.6865→97.6393%|−0.0472 [−0.2595,+0.1601]|0.848668→0.851962|+0.3294 [+0.1013,+0.5590]|Stop: 21 rescues, 24 regressions|
|179030|Unrun|Unrun|Unrun|Unrun|Frozen early stop|

All four children completed 1,000 stable updates and exact live/public packed
code/norm parity on64fit images/two32batches. All1,000 input hashes matched
within each pair; initialization, preflight, sources and checkpoints verified.
Resource rules passed in both pairs. Seed179029 stopped before the runner
stored resource criteria; the terminal verifier independently evaluated them.

| Seed | Training+bank native→products (s/64,000 images) | Training images/s native→products | Peak PyTorch CUDA (bytes, both) | Whole arm native→products (s) |
|---|---|---|---|---|
|179028|852.582→851.224|75.066→75.186|12,988,997,120|987.499→980.151|
|179029|855.577→849.087|74.803→75.375|12,988,997,120|992.935→978.914|

New-checkpoint image-to-top-k p50/p95/p99/QPS are **unmeasured**. Export time
and training throughput are not serving latency. No completed three-seed
aggregate or seed-population interval is claimed.

The sole original DGX unit `sfora-inshop-centroid-pca-confirmation-v1`, invocation
`3174cb583fed416480e0dd98022cde35`, closed with exit1 after **4,057.664128s**.
This was the expected controller stop `negative paired seed quality`, not a
numerical training failure. Receipt SHA:
`5a1b3b3af55c77c41b61092517f1008c8ef560dac135a5b8324d527a29865d43`.
[Terminal evidence and verification](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-centroid-pca-confirmation-v1/terminal/verification.json)
retain the full per-query vectors, receipts and original journal. Raw weights
and embeddings remain on DGX at the original result directory.

## Which layer next

A predeclared≤60s CPU diagnostic reused the four hash-pinned exports, read no
new images or official data, and matched every original GPU packed hit. It
finished in **2.115411s**. Seed179029 float R@1 was **97.6708→97.6550%**:
products lost one query before packing; packing amplified the net loss to
three. Thus packing is not the sole cause of the failed direction. This does
not prove all encoders, losses or product-mean initialization fail.
[Diagnostic and source](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-centroid-pca-confirmation-v1/terminal/float-packed-diagnostic.json)
bound the conclusion to these checkpoints. The next quality work should target
representation/learning with a distinct TRAIN-only causal falsifier, rather
than retrying this initializer or spending a p99 budget on a rejected arm.
