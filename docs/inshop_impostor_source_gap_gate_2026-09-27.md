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

## Same-role pretrained-source control

A separate [source-bound script](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-rank-179026/source_rank.py)
with SHA-256 `f03a65ac9ce9abc692b074bc205dbece6927f018bc81d6d8c6315f400e07309a`
used the pinned untrained 1,024-D SigLIP2 source cache and the same TRAIN
6,354-query/6,245-gallery roles. Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-rank-179026/receipt.json)
has SHA-256 `77ce284af0b15259127ef33fe8d046554040a977772e84c04e13aaeca6e29eb8`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-rank-179026/service-journal.txt)
has SHA-256 `636ca0741a128d22992487adc54f86e2d006725215fcbb4c31cbb3d9012111be`.
The service exited 0 and an independent replay checked all 6,354 hit flags
and paired counts. Pretrained source R@1 was **79.4775%**, compared with
**97.6235%** for the trained 128-D packed model: it rescued 15 of the 151
trained misses but lost 1,168 trained hits. This is a float 1,024-D
diagnostic, not an equal-byte or public serving baseline. It rules out using
the pretrained source ranking itself as the next quality treatment; the
specific trained impostor margin remains the candidate for a paired screen.

## Frozen paired 100-update method screen

Use seed 179026 and the existing `freeze_emb` true-freeze recipe as the
same-source control. The treatment changes only the bank rank term:

`rank = SmoothAP + 0.25 × mean[0.05 × softplus((best_negative − best_positive + 0.02) / 0.05)]`

The best positive and negative are taken from the same detached full bank
scores already used by SmoothAP, excluding the anchor itself. The 0.05
temperature covers the observed close packed margins; this is a frozen
training hypothesis, not a claim of novelty. Keep SigLIP2 Large/256, first
12 blocks and embeddings frozen, PCA head, ArcFace, rank coefficient 8,
augmentation, optimizer, 64-image schedule, fit/held products, BF16 runtime
and 128-D packed scorer identical. Add no fitted holdout parameter.

Run matched 17-update control/treatment smokes first. Require finite loss,
no skipped optimizer step, checkpoint geometry, identical schedule and first
ten input hashes, and treatment wall/peak CUDA ≤1.10× control. Then run
100 updates serially under the DGX lock. From the same held export, score
both the 12,599-image symmetric packed gallery and the fixed
6,354-query/6,245-gallery packed roles. Advance only if treatment-minus-
control asymmetric R@1 is at least **+0.30 percentage points** with a paired
product-bootstrap 95% lower bound strictly above zero, symmetric mAP@R does
not regress by more than **0.002**, and total training wall and peak CUDA are
each ≤**1.05×** control. A pass authorizes a separately frozen three-seed
1,000-update TRAIN gate. A failure stops the treatment without official TEST
or production promotion. All thresholds were set before either arm ran.

## Terminal paired TRAIN screen

The first DGX smoke unit failed before its first optimizer update because the
remote historical helper lacked `truncate_at_r`; its partial output is retained
as `freeze_emb-17-failed-import-v1`. The next source-pinned unit staged the
repository helper and completed both 17-update arms. Their schedule, source
manifest, geometry, and first ten input hashes matched. Both had finite losses
and gradients and no skipped steps. Treatment/control training wall was
**1.0049×** and peak CUDA **1.0003×**, passing the frozen 1.10× smoke bounds.

The single serial 100-update paired unit and subsequent scorer unit exited 0.
All comparisons below use the same official **TRAIN** product-disjoint held
split, seed 179026, SigLIP2 Large/256, 128-D packed Int8 scorer, and fixed
roles. They are exploratory; neither arm was selected on official TEST.

| Metric | True-freeze control | Specific-impostor treatment | Frozen decision |
| --- | ---: | ---: | --- |
| Fixed 6,354-query/6,245-gallery packed R@1 | **6,068/6,354 = 95.4989%** | **6,067/6,354 = 95.4832%** | Δ **−0.0157 pp**; paired product-bootstrap 95% **[−0.1908, +0.1688] pp**, fails ≥+0.30 pp and lower>0 |
| Full 12,599-image symmetric packed R@1 | **97.3411%** | **97.2061%** | diagnostic only |
| Full symmetric packed mAP@R | **0.775415** | **0.771910** | Δ **−0.003505**, fails ≥−0.002 |
| Training, 6,400 sampled images | **75.987 s**, **84.22 images/s** | **76.461 s**, **83.70 images/s** | **1.0062×** wall; passes ≤1.05× |
| Peak allocated CUDA on DGX Spark GB10 | **12.939 GB** | **12.942 GB** | **1.0003×**; passes ≤1.05× |

The treatment recovered 17 control misses and lost 18 control hits. The
asymmetric comparison [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-179026/raw/pair-score.json)
has SHA-256 `2c678504d82704a7929917c58a27d6e2f8a837aa15b6bed19fd16f114608c892`.
Control and treatment 100-update receipt hashes are respectively
`73aaa5b87350ea76b97bb3f3e28ee440d9adc8149be2879acd7c70c5666506aa`
and `815ee169f71a630299083740ffc969c74ee298289a48f09f9e9d7e98b53b9a6f`;
the [service journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-179026/raw/service-journal.txt)
hash is `4320e48435a1a8ff4ee3b60bbfa47705b760a506b2a3d0fb8748beaf3d1d539f`.
The [scorer source](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-179026/score_pair.py)
hash is `bc9850e2ee8e9c2122be25fc7533b285c183b359777fd224296fef7a3fafafc0`;
it verified both held-export hashes and the pinned role hashes before scoring.
Independent local replay confirmed 6,068 versus 6,067 hits and the 17/18
paired flips. The failed first smoke and the two source-pinned completed units
are identifiable in the journal. The held embeddings and checkpoints remain
on the DGX under `/home/riomus/runs/sfora-inshop-impostor-179026-v1/`, with
their SHA-256 values in the raw receipts.

**Decision:** reject this specific-impostor loss and stop before other seeds,
1,000-update training, official TEST, or serving promotion. It failed both
quality gates even though the incremental training cost was small. The
experimental implementation is retained only in source-bound evidence commit
`a2fb2bb9`, then removed from the production library path.
