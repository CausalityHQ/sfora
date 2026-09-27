# In-Shop TRAIN impostor census, 27 September 2026

The fixed-negative expected-gallery surrogate failed its frozen prediction
and near-boundary gates. Before any other rank-loss edit, re-export the same
seed-179026 true-freeze checkpoint on the same 12,599-image, product-disjoint
official **TRAIN** held panel and fixed 6,354-query/6,245-gallery roles.
Replay the previous per-query packed R@1 exactly. For each of its 151 misses,
record the best packed positive and impostor, their score margin, and the
query-to-impostor cosine in the pinned pretrained 1,024-D source cache.

Freeze this exploratory decision before the read: reopen a **specific
impostor** learning hypothesis only if at least 40% of the misses have packed
positive-minus-impostor margin in [−0.05, 0], and at most 25% have pretrained
query-to-impostor cosine ≥0.97. Otherwise stop this lane before trainer code
or training. These thresholds screen for a trainable close-margin mechanism;
they are not evidence of label errors, method novelty, or an official quality
gain. The official TEST split remains outside selection. Any surviving method
still needs a separately frozen paired TRAIN cost/quality gate and independent
seeds, with the same architecture, data, budget and packed scorer.

## Terminal TRAIN-only census

The sole DGX Spark GB10 service exited 0. Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-source-gap-179026/receipt.json)
has SHA-256 `8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c`;
the [service journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-source-gap-179026/service-journal.txt)
has SHA-256 `86d012a754ab07394201132c1383f63261abd8302e971813df6b61215f895d8b`.
The export took 67.105 s and replayed all 6,354 prior packed per-query
asymmetric hits exactly, including the same 151 misses. Local recomputation
checked the source hash, all 151 finite margins/cosines and the frozen rules.

| Same-checkpoint asymmetric misses | Measured | Frozen reopen rule |
| --- | ---: | ---: |
| Packed margin within 0.05 of best positive | 67/151 = **44.37%** | ≥40%; pass |
| Pretrained source cosine to best impostor ≥0.97 | 0/151 = **0%** | ≤25%; pass |

The source-cosine median is 0.8420 and maximum 0.9628; 24/151 are ≥0.90.
The 0.97 cutoff alone cannot establish that the other impostors are unrelated
or correctly labelled. Both exploratory thresholds pass, permitting a frozen
specific-impostor **TRAIN-only** learning screen. The result does not prove
that such a term improves R@1, mAP@R, training cost, or public serving speed;
no trainer or production serving code changed in this run.
