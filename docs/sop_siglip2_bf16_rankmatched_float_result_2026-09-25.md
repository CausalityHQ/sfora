# SOP SigLIP2 BF16 rank-matched float result, 25 September 2026

The serial DGX Spark unit `sfora-siglip2-bf16-rankmatched-float-v1.service`
(invocation `a6c8b9aabdf74b8f8e44144690e8c990`) completed with systemd
`Result=success`, `ExecMainStatus=0`. The [unit journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-rankmatched-float-unit-v1.log)
has SHA-256 `f2817dddd82471d24f2f0db02fd4e2b0704f7780b988fb5d6f0117f4c1bd4aa2`.
The [source-bound comparison receipt](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-rankmatched-float-comparison-v1.json)
has SHA-256 `6646adc6a0c72838f398268876d41d317e94c1c0c1bde47b09eb424a251ac097`.
It validates the three existing bank and float receipts, the new three-arm
source/schedule/initialization/scorer authority, and exact native top-10.
The one-update [canary receipt](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-rankmatched-canary-v1.json)
has SHA-256 `3893e5aa4a8d9c4aac81296abd6470315f42c891e70006c9272db77de6613bb2`.

All quality rows use **SOP official TRAIN**, split by product into 53,700 fit
images and 5,851 heldout queries from 1,132 disjoint products. Queries search
the full 59,551-image TRAIN gallery with self exclusion. The selected holdout
and seeds `179023`, `179024`, `179025` make this an internal method screen,
not an official SOP TEST result. Every arm uses the same pretrained
SigLIP2-L/16@256 encoder, PCA-initialized 1024-to-128 head, ArcFace, class
schedule, 1,000 full-backbone updates, 130-byte packed wire and exact native
scorer. The bank arm uses coefficient 8 full-fit detached member-bank
SmoothAP; the new in-batch float arm uses coefficient 21.93, selected before
these quality reads to match the first-step feature-gradient norm.

| Seed | Bank R@1 | Matched float R@1 | Bank minus float | Bank mAP@R | Float mAP@R | Bank/float training wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 179023 | 91.5399% | 90.6683% | +0.8716 pp | 0.744382 | 0.722160 | 1.0039× |
| 179024 | 91.0443% | 90.3093% | +0.7349 pp | 0.741836 | 0.723463 | 1.0036× |
| 179025 | 91.0955% | 90.5486% | +0.5469 pp | 0.739623 | 0.723269 | 1.0080× |
| Mean | **91.2266%** | **90.5087%** | **+0.7178 pp** | **0.741947** | **0.722964** | **1.0052×** |

The heldout-product-clustered 5,000-draw bootstrap of the three-seed mean
gives bank-minus-matched-float Recall@1 **+0.7178 percentage points**, 95%
interval **[+0.4094,+1.0361] points**, and mAP@R **+0.018983**, 95%
interval **[+0.016111,+0.021941]**. The bank-minus-float R@1 point gain is
positive in all three seeds. The coefficient-21.93 float arm versus the
original coefficient-8 float arm changed mean R@1 by **-0.0342 points**,
95% interval **[-0.3057,+0.2389] points**, and mAP@R by **-0.001238**,
95% interval **[-0.003704,+0.001065]**. Raising the float coefficient did
not explain the bank gain on this panel. First-step gradient matching does
not equate subsequent gradients, optimizer steps or clipping, and this
comparison does not isolate candidate-set size from bank staleness and
detached-candidate effects. The interval is conditional on these three
trained seeds and this selected TRAIN holdout; it is not seed-population
uncertainty.

Accounted training wall averaged **1,147.383 s** for the bank arm and
**1,141.468 s** for matched float, or **55.779 versus 56.068 sampled
images/s** for 64,000 sampled image presentations per seed on NVIDIA GB10.
Mean float export was about 320 s and exact holdout scoring about 2.95 s;
these are separate from the reported training wall. Peak PyTorch CUDA
allocation was **21,409,459,200 B** for bank versus **21,089,141,248 B**
for matched float. Gallery storage is **130 B/image**. Raw matched-float
receipts: [179023](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-rankmatched-seed179023-float-v1.json)
(`4086dee4…1ada`), [179024](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-rankmatched-seed179024-float-v1.json)
(`fa935aa5…3fb7`), and [179025](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-rankmatched-seed179025-float-v1.json)
(`45170924…e4c9e`). The [original three-seed gate](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-member-bank-multiseed-v1.json)
binds the bank and original float receipts.

The frozen [rank-matched gate](sop_siglip2_bf16_rankmatched_float_gate_2026-09-25.md)
**passes**. The analyzer sets `claim_eligible=false`: the current evidence
does not establish official SOP TEST or In-Shop quality, trained-bank live
image-to-top-k latency, a published-SOTA comparison, or method novelty.
Next, measure bank versus ArcFace live encoding plus exact search on paired
GB10 requests, then run the already frozen cache-inclusive head-only speed
falsifier. Keep these GPU jobs serial. Freeze the resulting system before
official protocol reads and transfer checks.
