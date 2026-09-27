# In-Shop checkpoint-average source screen, 27 September 2026

Test one uniform, FP64-accumulated weight average of existing seed-179024,
179026 and 179027 `freeze_emb` 1,000-update checkpoints. Average the vision and
128-D head state tensors only; classifier proxies are unused at inference.
These checkpoints share the pretrained SigLIP2 Large/256 files, source feature
cache, fit/held product rows, PCA head initialization, frozen embeddings and
first twelve blocks, optimizer settings, training budget and packed scorer.
The 179024 trainer source differs from the later two because the latter added
an optional rank-loss arm; its baseline recorded zero recovered rank updates.
The probe must verify state geometry, identical frozen tensors, finite values,
and each receipt/checkpoint hash before evaluating. Any mismatch stops it.

Use only official In-Shop **TRAIN**: 13,283 fit-product images and 12,599
product-disjoint held images. Encode the held panel once at native 256 pixels,
pack to the existing 128 signed int8 coordinates plus fp16 inverse norm, and
score all 12,599 self-excluded symmetric queries with stable ordinal ties.
Also score the previously fixed 6,354-query/6,245-gallery roles. Do not fit a
transform on held rows or read official query/gallery labels. This is an
exploratory feasibility test because the TRAIN held panel has been repeatedly
inspected and the proposal was informed by previously observed official
query/gallery seed-miss overlap. The idea has no method-novelty claim.

Freeze this stop rule before export. Advance only if all hold:

1. Symmetric packed R@1 is at least **98.7269%** (best source seed 98.6269%
   plus 0.10 percentage points).
2. Product-cluster bootstrap 95% lower bound for the soup's per-query R@1
   minus the *per-query three-seed mean* is strictly positive. The source mean
   R@1 is **98.5660%**.
3. Symmetric mAP@R does not regress below the three-seed mean **0.839866**.
4. Fixed asymmetric packed R@1 does not regress below the archived seed-179026
   value **97.6235%**. This is a guard, not a claim of improvement over all
   three seeds on these roles.

A pass authorizes only an independent-seed and training-cost decision. Three
trained source models cost about three times one model's training; one-model
serving speed and 130-byte gallery storage remain hypotheses until checked.
The current official query/gallery result and its seed-specific misses cannot
select the method. Failure stops before new training, official TEST, or
production promotion.

## Terminal TRAIN-only result

The first DGX service completed export and scoring but failed to serialize its
receipt because one gate was a NumPy boolean. Its [failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-weight-soup-v1/failed-journal.log)
has SHA-256 `376a292d7b44f3c4f5ce4ed816951874ad3ac644ad148bba816650ee7a5a381c`.
The corrected script serializes before opening the output, and the sole rerun
`sfora-inshop-weight-soup-v2.service` exited 0 (invocation
`7d95f837e8274a6d97dfddc1e85dfafa`). Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-weight-soup-v1/receipt.json)
has SHA-256 `1a3a7007fd81bb61a8c94759eb373f34441a063b83f470988b1b22accd20ed59`;
the [successful journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-weight-soup-v1/success-journal.log)
has SHA-256 `d7d63bb0cd5ca9c3d322c42af56ea6c19b5644018778bc6fdf8644b696934d9f`.
The source script SHA-256 is
`7b5d5ad8e29facc84b7839d9e31fa572c5ee34d46a9d4e841c5785adbfa0f9c5`.
All 195 frozen embedding/lower-block tensors matched across sources.

| TRAIN held panel, packed scorer | Three source seeds, mean | Best source seed | Soup | Frozen decision |
| --- | ---: | ---: | ---: | --- |
| Symmetric R@1, 12,599 self-excluded queries/gallery | 98.5660% | 98.6269% | **98.6189%** | Fail: below best +0.10 pp |
| Symmetric mAP@R | 0.839866 | 0.841910 (seed 179026) | **0.847175** | Pass vs mean |
| Soup-minus-mean paired R@1, product-bootstrap 95% | reference 0 | — | **+0.0529 pp**, [−0.0376, +0.1451] pp | Fail: lower below zero |
| Fixed 6,354-query/6,245-gallery R@1 | — | seed-179026 97.6235% | **98.0327%** | Pass nonregression guard only |

The soup has **12,425/12,599** symmetric hits; the best source seed has one
more. The held-image encoder export took **66.685 s** and peak allocated CUDA
was **1.886 GB** on DGX Spark GB10. The three source training walls were
747.058, 744.921 and 746.701 s per 64,000 images, totaling **2,238.680 s**
to produce one soup; the source models' training peak was about 12.939 GB each.
The full soup probe used **11.729 GB** peak parent RSS. The export and storage
shape suggest one-model serving, but public latency was not measured.

Local replay verified the receipt hash, all 12,599 per-query means, and that
`advance` equals the conjunction of the four recorded predicates. Two frozen
predicates failed. Stop before fresh seeds, official query/gallery read, or
production promotion. The mAP and fixed-role gains are exploratory and do not
override the primary R@1 decision or the roughly threefold training cost.
