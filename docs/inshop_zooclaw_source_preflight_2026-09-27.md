# Fashion-pretrained source screen on In-Shop TRAIN, 27 September 2026

The paired worst-positive loss and acquisition-group falsifier failed their
frozen TRAIN gates. A different representation source is the next distinct
quality/efficiency possibility. [ZooClaw-FashionSigLIP2](https://arxiv.org/abs/2606.27708)
releases a fashion-specialized checkpoint. Its paper evaluates image–text
retrieval, not this same-product image–image protocol; its proprietary training
catalog makes image overlap with In-Shop independently unverifiable. This
screen therefore remains exploratory and cannot establish a new learning
algorithm or SOTA.

## Frozen source and decision

Use only the released [checkpoint](https://huggingface.co/srpone/zooclaw-fashionsiglip2)
at revision `c6b34b2a7b069855c2684979a196a7af61751797`: `model.safetensors`
SHA-256 `aa9a0927a77b672697fd23bcca8e73ceff85dcde142c7c9fa7d1e46a2bbfeb20`,
`config.json` `fde47c741b67e7b9834e80d8377494096e244b11189e8a50f45a95a33a3579e9`,
and `preprocessor_config.json`
`fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90`.
The actual Transformers checkpoint loads as `SiglipModel`, with 12 vision
blocks, 768 hidden coordinates and 384-pixel input; a synthetic one-image
shape/finiteness smoke passed before any quality read. Use its pooled vision
output from `get_image_features`, then row-normalize it. This is a source
checkpoint comparison, not a faithful reproduction of the paper's benchmarks.

Encode all 25,882 official In-Shop **TRAIN** images once on DGX Spark GB10.
Fit one centered PCA-128 on the existing 13,283 fit-product rows only; pack
the 12,599 product-disjoint held rows as signed int8 plus f16 inverse norm and
score their symmetric self-excluded gallery with the existing exact scorer.
The stronger frozen pretrained reference is the 22-block SigLIP2 Large/256
own-PCA packed result in the [cross-depth receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-cross-depth-head-v1/receipt.json):
R@1 **87.1736%**, mAP@R **0.508923**. Its receipt SHA-256 is
`943faa33e74a1bbf769e6b39d86482ac024a85f574ff6b829dc7328a034c84d7`.

Advance to a separately frozen 100-update matched training feasibility and
public image-to-top-k latency gate only if the candidate's packed mAP@R is
at least **+0.005** above the reference, the paired product-bootstrap 95%
lower bound is **>0**, packed R@1 does not fall, the 25,882-image export wall
is **≤216.163 s**, and peak PyTorch allocated CUDA is **<10 GB**. The cost
limits match the earlier frozen TIPSv2 source screen; sequential export wall
is a feasibility bound, not a paired public latency or training-speed claim.
Any failure stops this source before fine-tuning, official query/gallery,
or production promotion. No threshold is changed after scoring. The source is
[`probe_inshop_zooclaw_source.py`](../scripts/probe_inshop_zooclaw_source.py).

## Terminal source-screen result

The sole locked DGX Spark GB10 unit `sfora-inshop-zooclaw-source-v1` exited 0
(invocation `7dcd416e64c3469392d32f19022abfd3`). Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-zooclaw-source-v1/receipt.json)
has SHA-256 `66611c638d8134a0fe0fedac52bce8f09e87234f692d67a641c4ebabc9f9e662`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-zooclaw-source-v1/journal.log)
has SHA-256 `3ba78de0b655c80d816e63049a059fa8f5efc7fc6ea4d323139b66100630164a`.
The remote source SHA matched the pushed script,
`a8425264abed52c59305b3017f8dfd296270d7cbc04b233cc888ea202ab00b9f`.
The pinned model, processor, baseline, partition and product-disjoint fit/held
row inventories passed source checks.

| Official TRAIN held-only symmetric gallery | Fashion checkpoint | 22-block SigLIP2 source | Frozen gate |
| --- | ---: | ---: | --- |
| Packed R@1, 12,599 self-excluded queries | **89.4198%** | **87.1736%** | +2.2462 pp; pass nonregression |
| Packed mAP@R | **0.545136** | **0.508923** | +0.036213; paired product-bootstrap 95% **[+0.029765, +0.042113]**, pass |
| Sequential 25,882-image export | **284.868 s** | **216.163 s** historical SigLIP2 Large/256 source export | **1.3178×**; fail ≤1.0× |
| Peak PyTorch allocated CUDA | **1.172 GB** | not paired | pass <10 GB |

The source's quality screen passes, but the **joint source gate fails export
cost** (`advance_training=false`). Stop this frozen source route before a
training arm, official query/gallery read, or production promotion. The
receipt is an exploratory source result, not a trained-model or public
image-to-top-k comparison. The cost gap motivates a separate exactness-pinned
stage profile; no later optimization may retroactively change this decision.

## Frozen exact-path cost diagnosis

Before changing preprocessing or encoder code, time decode, pinned processor,
host-to-device transfer and vision forward separately for both pinned sources
on the **first 320 official TRAIN images** in partition order, ten batches of
32 after one untimed warmup batch per arm. Use the exact image processors and
forward calls from the source-export paths, run the arms sequentially on the
same DGX Spark GB10, and record synchronized per-batch samples and allocated
CUDA peak. This short diagnostic has no quality read, p99 certification, or
advance gate. Its source is
[`diagnose_inshop_zooclaw_stage_cost.py`](../scripts/diagnose_inshop_zooclaw_stage_cost.py).

The first short profile exited 0 (invocation
`07483ab3c82448eaa246d663ec40dc7a`). Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-zooclaw-stage-v1/receipt.json)
SHA-256 is `b428baa8e49d998b358b87d46abeb91d47394da8929b2cd74bd7c44723263126`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-zooclaw-stage-v1/journal.log)
SHA-256 is `decae800453eec613a24b4a00a35e7812d8e504d1aacc6f39c1dada97d9a5820`.
On these ten batches, mean full stages were **0.303684 s/batch** for fashion
versus **0.339007 s/batch** for SigLIP2 Large. Fashion vision was faster
(**0.160325 vs 0.211309 s/batch**), while its processor was slower
(**0.084803 vs 0.071796 s/batch**). This subset conflicts with the full
export-cost contrast, so it is insufficient to infer production speed.

Before another cost conclusion, repeat the same pinned stage script with its
`--spread` selector on 320 evenly spaced official TRAIN images, still ten
batch-32 samples after one untimed warmup. This changes only the metadata-
selected diagnostic sample and reads no quality. If it agrees with the first
profile, the historical full-export comparison remains unmatched and requires
a new full paired cost gate; if it disagrees, diagnose the image-distribution
effect before changing production preprocessing.

The spread profile also exited 0 (invocation
`060159ec4be4453e9d307480b65e6d3f`). Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-zooclaw-stage-v2/receipt.json)
SHA-256 is `d6012af5e52720fbb9f76c75b9a865230be0c2dbbbf1ed524b248d19422ce2b4`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-zooclaw-stage-v2/journal.log)
SHA-256 is `715afb375af75365897c68235b63b91cf89eb4cf55d9ff786112b6e7d5f4c405`.
Mean total stages were **0.310970 vs 0.344294 s/batch** for fashion versus
SigLIP2 Large; vision **0.160444 vs 0.210867 s/batch**, processor
**0.090520 vs 0.074470 s/batch**. The second sample reproduces the first
profile's direction. Neither 320-image result revises the failed full-export
gate or establishes public image-to-top-k latency.

The next cost diagnosis is one unchanged full 25,882-TRAIN-image replay of
the pinned SigLIP2 Large/256 source exporter on this current DGX stack. Its
source and export helper hashes are
`30f46c7536f546a264295da9c22de9cde0738bd02f5ca1bd51dcd1a9b8a659b6`
and `e407b393ea94f1abd2cffffdc9cd61081da82de232e9b8037af7cd7cca32db2c`.
The replay tests whether the historical **216.163 s** is reproducible now;
it has no quality scoring or advancement rule. A slower current baseline
would explain why the short paired profiles and historical unpaired exports
disagree, but would not retroactively pass the frozen source screen.
