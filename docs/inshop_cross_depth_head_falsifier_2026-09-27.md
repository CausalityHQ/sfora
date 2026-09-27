# In-Shop cross-depth head falsifier, 27 September 2026

The 22-block SigLIP2 encoder trained 11.88% faster at 100 updates but lost
0.009100 packed mAP@R on 12,599 product-disjoint official **TRAIN** held
images, beyond its frozen floor. The 24-block and 22-block pretrained source
caches already contain the same 25,882 TRAIN rows. Test one fit-only linear
transfer before paying for another training run: align normalized 22-block
source vectors to normalized 24-block vectors by orthogonal Procrustes with
translation on the 13,283 fit-product rows, then compose the fitted map with
the 24-block fit-only PCA-128 head. The composed affine head fits the existing
single 1024-to-128 linear layer and adds **zero** serving operations. Compare
its packed held-only symmetric retrieval with the 22-block's own fit-only
PCA-128 and the 24-block fit-only PCA-128 on identical held rows. This is a
source-initialization probe, not trained-model or official TEST evidence.

Freeze this kill rule before reading held quality: advance to one matched
100-update 22-block training screen only if the transferred head exceeds the
22-block own-PCA packed mAP@R by at least **0.005**, its paired
product-bootstrap 95% lower bound for that mAP difference is **positive**,
and packed R@1 does not decline. Require exact source-cache/model/row hashes,
fit-only PCA and mapping, finite outputs and exact 130-byte packed scoring.
Otherwise stop this mapping arm with no retraining or official read. If it
passes, count the extra source-cache export when judging total training cost;
the existing separate exports took **216.163 s** for 24 blocks and
**180.906 s** for 22 blocks. A one-pass two-depth cache exporter would need
measurement before any production speed claim. Layer dropping and linear
feature distillation are prior art; no novelty or SOTA claim follows.

## Terminal TRAIN-only result

The single DGX Spark GB10 unit exited 0. The [source-bound probe](../scripts/probe_inshop_cross_depth_head.py)
verified both model and cache hashes, identical 25,882-row ordering, the
official TRAIN product-disjoint fit/held split, finite fit-only transforms,
and exact 130-byte packed scoring on all 12,599 held rows. Its Procrustes
self-check passed before the GPU run. The 24-block and 22-block source-cache
receipts were reused; no encoder was re-exported for this probe.

| Pretrained source with fit-only 128-D head | Packed held R@1 | Held mAP@R |
| --- | ---: | ---: |
| 24 blocks, own PCA | 81.6652% | 0.460749 |
| 22 blocks, own PCA | **87.1736%** | **0.508923** |
| 22 blocks, transferred 24-block PCA | 86.6497% | 0.503109 |

Transferred minus 22-own mAP@R is **−0.005814**, paired product-bootstrap
95% **[−0.006847, −0.004785]**; packed R@1 is **−0.5239 percentage
points**. Both frozen quality conditions fail, so `advance_training=false`:
stop this head transfer without another 22-block training run, batch-1
latency test, or official TEST read. The 22-block pretrained source's large
lead over 24-block pretrained source on this particular fit-only PCA task is
an exploratory clue about representation geometry. It does not overturn the
completed 100-update trained-model gate, where 22 blocks lost mAP@R, or
establish that a 22-block deployed model is more accurate.

The fit took **2.680 s**, and the whole cache-to-retrieval probe **4.550 s**
with **85.704 MB** peak allocated CUDA and **2.191 GB** peak parent host RSS.
The [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-cross-depth-head-v1/receipt.json)
SHA-256 is `943faa33e74a1bbf769e6b39d86482ac024a85f574ff6b829dc7328a034c84d7`;
the [exact transferred head](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-cross-depth-head-v1/transferred_head.npz)
SHA-256 is `c8b0f9a5e06e2937e35dc6256dddbd7656b9ef72e729def2d06687437dad8296`;
and the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-cross-depth-head-v1/journal.log)
SHA-256 is `f54985362f4f1e15a3d99a7ddb077d0b509f20a3324c0bb518c66b7e73bba93b`.
The probe source SHA-256 is
`23cd852d6c04e5140ff1f07049777b0ff77fea990ab133f41ea8707457514039`.
