# In-Shop TRAIN color-miss falsifier

Before any descriptor or scorer change, test whether a fixed, cheap color
feature distinguishes the best same-product image from the top packed impostor
on the archived seed-179026 true-freeze misses. The 6,354-query/6,245-gallery
roles are product-disjoint from the 13,283 fit images and use only official
TRAIN standard retrieval pixels. The archived packed-score receipt has exactly
151 misses and supplies their query, best positive, and best impostor ordinals.

The frozen feature is an 8-bin-per-RGB-channel histogram from the central 60%
of each image, resized to 32×32. Distance is symmetric chi-square. Verify the
partition, role, receipt, and every sampled image SHA-256 before the read.
Advance only if the positive is closer than the impostor on at least **60%**
of the 151 misses **and** the median impostor-minus-positive distance exceeds
**0.02**. Otherwise stop the color lane before any model, gallery format,
scorer, or training change. A pass permits a separately frozen matched
TRAIN-only quality and image-to-top-k cost gate; it is not a quality gain,
algorithmic novelty, or official query/gallery claim.

## Terminal TRAIN result

The sole CPU-only DGX Spark unit `sfora-inshop-color-f0-v1` (invocation
`45f00dd6556044b68db91aa01a457160`) exited 0. Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-color-f0-v1/receipt.json)
has SHA-256 `6a13ee2015e45b28ded74482bddbaeac0d21b36634147da5f0eb0959d379942b`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-color-f0-v1/journal.log)
has SHA-256 `1479ca652ec0f4a8c622d785a09e43ad6fb62646f657b64581cf11cc30f1b765`.
The staged script SHA-256 was
`be02d6b7d04dc09693acf2f63355cfe311d9a99c19d9d693dbbe1438ff19c912`.
All sampled image hashes, partition, roles, and archived receipts passed their
source checks. Local replay of all 151 distance pairs reproduced the result.

The true image was color-closer on **84/151 = 55.6291%** of packed misses,
below the frozen **60%** floor. Median impostor-minus-positive distance was
**+0.040998**, above the frozen +0.02 floor, but the conjunction fails.
`advance=false`: stop this center-crop color lane before any training, packed
format, serving latency or official TEST change. This does not rule out a
different color representation, and no retrieval-quality improvement was
measured.
