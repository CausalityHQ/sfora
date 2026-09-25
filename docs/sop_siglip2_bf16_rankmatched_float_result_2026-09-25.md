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

## Subsequent serving and training-speed diagnostics

Both planned serial DGX units ended with `Result=success` and exit status 0.
The paired live unit `sfora-siglip2-bf16-bank-live-seed179024-v1.service`
used invocation `1237a578154b47d2a0b64501be1132f2`. Its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-bank-live-seed179024-v1.json)
has SHA-256 `743aed5aa67b4aa39603e8d3eb698d5dbd28e400a48858e0f5d8438f65b5d52c`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-bank-live-seed179024-unit-v1.log)
has SHA-256 `99ace37d5fdedaababa21c02b9e58b20a3fcfbadcceb5460cfac9862b5171f1e`.
Both variants used the same trained seed, 32 selected SOP TRAIN holdout images,
preloaded PIL inputs, native-fp16 vision parameters, 130-byte resident gallery,
and exact top-10 API. AB/BA/BA/AB balanced 100 calls per variant at each batch
size on NVIDIA GB10.

| Batch | Trained bank p50/p95/p99 | ArcFace p50/p95/p99 | Bank throughput | ArcFace throughput |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15.879/18.548/18.985 ms | 15.712/18.108/19.479 ms | 62.310 queries/s | 62.684 queries/s |
| 32 | 266.277/293.314/303.670 ms | 268.287/290.913/303.871 ms | 119.594 queries/s | 119.767 queries/s |

This fixed-image diagnostic does not show a clear serving improvement over
the same architecture and cannot support a population p99 claim. The two
loaded models jointly peaked at 1,515,488,256 B of PyTorch CUDA allocation
after loading, and parent host RSS peaked at 7,476,781,056 B. The receipt
binds both training-receipt hashes and stable top-10 output digests. Full
live quality over all 5,851 holdout images was not measured in this run.

The cached-head unit `sfora-siglip2-cached-head-probe-v1.service` used
invocation `e95546a46af2425ea18ccf36fdfc2dad`. Its source-bound
[comparison receipt](evidence/compact_metric/sop-siglip2-substrate-v1/cached-head-probe-comparison-v1.json)
has SHA-256 `d47304a3bdc62ac710c486e7b18d49b7c6e622fa525e4cbaf9422c296adae459`;
raw [ArcFace](evidence/compact_metric/sop-siglip2-substrate-v1/cached-head-probe-arcface-v1.json),
[live-head bank](evidence/compact_metric/sop-siglip2-substrate-v1/cached-head-probe-live-bank-v1.json),
and [unit journal](evidence/compact_metric/sop-siglip2-substrate-v1/cached-head-probe-unit-v1.log)
are archived. On the same selected SOP TRAIN holdout, the head-only bank
reached **84.8573% Recall@1**, **0.617283 mAP@R** versus head-only ArcFace
**82.7551%**, **0.578572**. The paired difference is **+2.1022 percentage
points Recall@1**, single-seed product-bootstrap 95% interval
**[+1.6851,+2.5336] points**, and **+0.038711 mAP@R**, interval
**[+0.035386,+0.042034]**. All exact native top-10 checks passed.

The bank arm used **22.948 s** for 1,000 head/proxy updates and
**536.463 s** for the frozen speed numerator: 508.517 s initial source-cache
encoding plus head initialization and training. This was **0.4677×** the
pinned 1,146.908 s full-backbone bank training wall. Peak PyTorch allocation
in its head-training phase was **1,092,567,040 B**. The
[frozen probe gate](sop_siglip2_cached_head_speed_probe_2026-09-25.md)
**fails** because packed Recall@1 misses its 85% floor by **0.1427
percentage points**; it passes the paired gain, mAP nonregression, and time
criteria. The cache-plus-head numerator is a defined feasibility measure,
not a complete training or deployment wall. Frozen-encoder head fitting
cannot replace full-backbone training at the stated quality floor. A reduced
encoder-update hybrid remains unmeasured and requires a fresh train-only
selection split and frozen protocol before any tuning or claim.
