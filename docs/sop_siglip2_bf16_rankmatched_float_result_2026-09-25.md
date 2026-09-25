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
not a complete training or deployment wall. This particular frozen-encoder
probe did not reach its quality floor; other head schedules were not tested.
A reduced encoder-update hybrid remains unmeasured and would require a fresh
train-only selection split and frozen protocol before any tuning or claim.

## Three-seed unseen-product gallery check

The earlier [single-seed ArcFace sensitivity check](sop_siglip2_unseen_gallery_diagnostic_2026-09-25.md)
already showed a heldout-only gallery gain, but it did not test the later
three-seed matched float arms. A new source-frozen CPU diagnostic on the nine
existing exports ended successfully as DGX unit
`sfora-siglip2-unseen-gallery-v2.service`, invocation
`9493376b0396447ebc4c0b35d7438bf5`. Its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/unseen-gallery-three-seed-v2.json)
has SHA-256 `87adcb8d10058ca85ad5826954f9a8c1f228e9c5218ba0015fa83d2c77528862`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/unseen-gallery-three-seed-unit-v2.log)
has SHA-256 `2fdac5c5cb3e5063fd255c2cd9a60793a9efb73acd07fcc4fb8a58f9cf135499`.
The diagnostic and its fixed screen were pushed at `0ec1a8db` before these
quality outputs were read. The first successful
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/unseen-gallery-three-seed-v1.json)
is retained: its provenance field hashed a Torch decorator wrapper instead
of the wrapped scorer source. A provenance-only correction was pushed at
`96a2777f` and rerun as v2; all quality, gate, refresh, and input-authority
fields matched v1 exactly. All nine training receipts and exported embedding
files passed their SHA-256 checks. This uses the same selected 5,851 SOP TRAIN
holdout queries and 1,132 products, now with only those unseen-product rows
in the gallery and self exclusion. It is a retrospective sensitivity check,
not an independent confirmation or official SOP TEST result.

| Three-seed mean, packed 130 B/row | Bank | Matched float | Original float |
| --- | ---: | ---: | ---: |
| Recall@1 | 97.1686% | 96.7128% | 96.6501% |
| mAP@R | 0.880576 | 0.868607 | 0.867291 |

Bank minus matched float is **+0.4558 percentage points Recall@1**
(product bootstrap 95% **[+0.2376,+0.6790]** points) and **+0.011968
mAP@R** (95% **[+0.009545,+0.014506]**). Bank minus the original
coefficient-8 float control is **+0.5184 points Recall@1** (95%
**[+0.3168,+0.7232]**) and **+0.013285 mAP@R** (95%
**[+0.010915,+0.015708]**). The code-frozen port screen required at least
+0.36 points bank-minus-matched Recall@1, a positive product-bootstrap lower
bound, nonnegative mAP difference, and a positive lower bound versus original
float; it **survived**. These intervals condition on the selected holdout and
three seeds. The old full-TRAIN-gallery advantage partly involved trained-on
fit distractors, yet the bank gain remains positive when they are removed.

The replayed bank schedules had 64,000 sampled image presentations each.
Across seeds, **18,446 of 53,700 fit rows on average (34.35%)** were never
refreshed, 19,942 were refreshed once, and 15,312 more than once. Thus the
bank's initial pretrained geometry remains a substantial part of this method;
the result does not isolate larger candidate sets from stale features. The
new diagnostic uses the already validated packed scalar scorer on CPU but
does not repeat native top-10 parity on its smaller gallery. The earlier
single-seed sensitivity receipt did verify native top-10 on a smaller gallery,
and the official evaluator must verify it again for the frozen three-seed
system. The next decision is to port the frozen full-backbone bank and both
float controls to official SOP TEST, with every arm's cache, training, export,
packing, and search costs reported separately. No new training is selected
from this used holdout.
