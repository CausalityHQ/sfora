# In-Shop dual-depth compact-head preflight, 27 September 2026

The trained 22-block SigLIP2 screen lost mAP@R, but its **pretrained** block-22
source with an own fit-only PCA-128 head beat the 24-block source by 0.048174
mAP@R on the same held TRAIN products. Test whether both depths together
retain complementary information without another image export or extra encoder
blocks. Concatenate separately unit-normalized 22- and 24-block 1,024-D
pretrained source rows, fit one centered PCA-128 on only the 13,283 fit-product
official In-Shop TRAIN rows, and score the same 12,599 held-product rows with
the existing 130-byte packed scorer and stable ties.

Before reading the result, require the dual-depth packed mAP@R to exceed the
stronger 22-block own-PCA baseline **0.508923** by at least **0.005**, with a
paired product-bootstrap 95% lower bound above zero, and packed R@1 at least
the 22-block baseline **87.1736%**. Validate both cached feature/receipt and
model/row hashes, exact fit/held split, finite features and wire geometry.
Any failure stops this dual-depth treatment before training. A pass permits
one same-architecture 24-block, same-batch/budget/optimizer/scorer paired
100-update training feasibility screen with a 2,048-to-128 head; measured
training, memory and public image-to-top-10 latency must then pass separately.
The intended serving path would read an intermediate activation during the
existing 24-block forward, without a second encoder run, but this cost is
unmeasured. This already-observed TRAIN panel is exploratory and cannot
establish official quality, current SOTA or method novelty.

## Terminal TRAIN-only result

The sole DGX Spark GB10 unit `sfora-inshop-multidepth-head-v1.service`
(invocation `28b59d3f6d914bc3aa2860f4fc7c5b7e`) exited 0. The
[source-bound receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-multidepth-head-v1/receipt.json)
SHA-256 is `88a247c45c1c7567ae5aa92eb3dfd9a084281a9794887065a14815da8c1a0b42`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-multidepth-head-v1/service-journal.txt)
SHA-256 is `467329fcdf2471bd1a92425fef07344697a838e40e9d6a7d5a540d18aaeac8bd`.
The receipt binds both 25,882-row cache hashes, the baseline receipt,
product-disjoint fit/held row hashes, source, all 12,599 per-query outcomes
and the frozen gate.

| Pretrained fit-only 128-D packed source, 12,599 held TRAIN queries/gallery | Recall@1 | mAP@R |
| --- | ---: | ---: |
| 22 blocks, own PCA | **87.1736%** | **0.508923** |
| 24 blocks, own PCA | 81.6652% | 0.460749 |
| Concatenated 22+24 blocks, own PCA | **84.4670%** | **0.488134** |

Dual-depth minus the stronger 22-only control is **−2.7066 percentage points**
Recall@1 and **−0.020788 mAP@R**, paired product-bootstrap 95% interval
**[−0.025096, −0.016593]** for mAP@R. PCA fitting took **4.140 s**, total
cached-feature diagnostic **4.864 s**, peak allocated CUDA **85.704 MB**;
none of these are encoder training or public image-to-top-10 timings.

**Decision:** both frozen quality conditions fail. Stop this equal-weight
dual-depth/PCA head before training, official TEST or a production change.
The outcome does not rule out all learned nonlinear multi-layer heads; it
does remove the cheapest no-extra-encoder source case for this exact design.
