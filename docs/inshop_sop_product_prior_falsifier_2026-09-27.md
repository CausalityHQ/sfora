# SOP product-prior transfer to In-Shop: TRAIN-only source screen

The current In-Shop true-freeze model still has no official result, and the
older selected bank model is 1.2177 points below the dated published UNICOM
In-Shop R@1 reference. Repeated In-Shop-only loss and crop variants did not
close the gap. A causally distinct intervention is to initialize In-Shop
fine-tuning from the **already trained SOP product encoder**. SOP shares the
same SigLIP2 Large/256 architecture and product-identity task, and its
training cost has already been paid for the SOP product. This tests whether
the upper encoder learned transferable product discrimination; it adds no
serving layers, dimensions or gallery bytes. It is not a method-novelty claim.

First, run one no-fit source screen on the fixed official **In-Shop TRAIN**
13,283-fit/12,599-held product-disjoint split and 6,354-query/6,245-gallery
held roles. Compare the pinned pretrained SigLIP2 source cache (SHA-256
`f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885`)
with the pooled 1,024-D source from the selected SOP seed-179024 true-freeze
1,000-update checkpoint (SHA-256
`2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172`).
Use the same pinned processor, model architecture, normalized float cosine,
fixed roles and ordinal ties for both. Verify the pretrained control exactly
replays its archived 79.4775% R@1; report mAP@R, per-query paired hits,
product-cluster uncertainty, export wall and peak CUDA. No In-Shop-held label
or image is used for fitting. This is a **source-only early stop**, not a
packed serving or official quality result.

Advance to any trainer edit only if the SOP source improves R@1 by at least
**+1.00 percentage point** over the pretrained source, the paired
product-bootstrap 95% lower endpoint is strictly positive, and mAP@R is
nondecreasing. A smaller effect is too weak to motivate a new warm-start
training campaign against the 1.22-point older official gap. A failure stops
this transfer lane before fine-tuning or an official query/gallery read.
Passing permits a separate frozen 17-update paired smoke, then matched
100-update control/treatment on a TRAIN-only panel, followed only if promising
by independent paired 1,000-update seeds. The sole treatment difference will
be the encoder initialization; PCA head, fit products, augmentation, bank,
optimizer, input schedule, scorer and update budget stay matched. Charge the
SOP checkpoint's acquisition cost explicitly, including any amortization
across deployed datasets. Requalify quality and full image-to-top-k latency
for a changed checkpoint; architecture identity alone does not certify speed.

## Terminal TRAIN-only source result

The sole DGX Spark GB10 unit `sfora-inshop-sop-product-prior-v1`
(invocation `ccf45f7a87fe48468221f6d2719835c6`) exited 0. Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-sop-product-prior-v1/receipt.json)
has SHA-256 `8c4dca4f93b1a9aaec590dc0f63efdcc4be324a47736232fa6dc3d5ef0832ca8`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-sop-product-prior-v1/journal.log)
has SHA-256 `794600cde86014e90ab9f7de53d230564c009405d4cb93ccb14aa34c76fda663`.
The source SHA-256 is
`cc07f4f0ec695e66f41919f01eab0c98199ddc24b13e41b4e5645b4dada8970c`.
The pretrained control's **6,354 per-query R@1 flags** replayed its earlier
source-rank receipt exactly. Local replay checked the per-query means,
source digest and frozen decision.

| Official In-Shop TRAIN held roles, float 1,024-D source | Pretrained SigLIP2 | SOP product-prior vision |
| --- | ---: | ---: |
| R@1, 6,354 queries / 6,245 gallery | **79.4775%** | **88.5899%** |
| mAP@R | **0.508068** | **0.633295** |
| Paired query flips | — | **656** wins, **77** losses |

The paired R@1 gain is **+9.1124 percentage points**, product-cluster
bootstrap 95% **[+8.2336, +10.0081] points**. The fixed +1.00-point,
positive-lower-bound and nondecreasing-mAP rules pass. SOP-source export took
**67.721 s**; total source screening took **70.186 s**, with peak allocated
CUDA **1,885,763,584 bytes**. These are source descriptors before any
In-Shop fitting and before the 128-D packed head. They justify a matched
warm-start training smoke, but do not estimate its final quality, training
cost or public image-to-top-k latency. The existing In-Shop trainer initializes
its PCA head and detached bank from the pretrained source cache; the treatment
must first acquire its **own SOP-source fit cache** so head/bank and encoder
come from the same initialization. Its cache cost must be charged.

## Frozen paired 17-update training integrity gate

The source screen used different image processor and precision paths across
arms, so its +9.1124-point difference is not a weights-only causal estimate.
Before training, export the SOP-vision source cache through the **same**
`export_inshop_siglip2_train_features.py` path that made the pretrained control
cache: pinned model snapshot, default processor, FP16 weights, batch 32, all
25,882 official TRAIN rows. Require the SOP vision checkpoint SHA-256 above
in the cache receipt and trainer, strict state loading, exact row/model/cache
hashes, and no overlap with another DGX GPU unit. The treatment PCA head,
proxies, and detached bank must all derive from its own SOP cache.

Run paired `freeze_emb` 17-update jobs, seed 179024, with the same source code,
dataset, preflight, augmentation schedule, processor, LR, optimizer, worker
count, and BF16 train precision. The control uses the archived pretrained
cache; the treatment uses the new SOP cache. The only experimental change is
the coherent encoder/source initialization. Freeze these **KILL-only** rules:

- Both finish 17/17 updates with finite losses and preclip gradients and no
  optimizer skip; their first ten input-batch hashes and executed schedule
  hash match exactly.
- Each checkpoint reloads strictly. Embeddings and lower twelve blocks equal
  that arm's actual loaded initialization, and at least one trainable upper
  tensor changes. The treatment cache SHA and vision checkpoint SHA agree
  with its receipt. Any mismatch kills the lane before 100 updates.
- Treatment 17-update training wall and peak allocated CUDA must each be at
  most 1.20 times the paired control. This screens gross runtime regressions;
  it does not establish an optimized training cost.

The 17-update gate has **no quality promotion rule**. If it passes, freeze a
separate paired 100-update TRAIN-held packed R@1/mAP@R gate, with both fixed
asymmetric query/gallery roles and symmetric held-gallery guard, before
launching it. The 100-update read can kill this lane, never prove final
quality. Only paired full-budget seeds and official independent evaluation
can support a deployed quality claim. Charge SOP acquisition and new cache
export separately from In-Shop fine-tuning; unchanged architecture only
motivates, but does not replace, a new serving latency measurement.

## Terminal 17-update integrity result and next cheap falsifier

The first systemd invocation `fe541272d0ef45e6b27f3091e08325ee`
completed its control training, then stopped on a replay-script
`KeyError`: the archived control receipt predates the
`executed_schedule_sha256` field. Its first ten input hashes, full schedule
hash and cache hash matched. The resumed unit
`72de9cb945484444ba91e4fafba9e50f` reused that terminal control,
exported the new SOP source cache, and completed treatment. The
[analysis receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-sop-warmstart-smoke-v1/analysis.json)
SHA-256 is `6f584ab981857215e8ad44092e9e4a70d961c01faa40230e8ecdd8c4fdd53f0b`.
All frozen checks passed: exact paired input/schedule hashes, 17 finite
gradients and losses per arm, no skipped step, strict checkpoint hashes,
frozen embeddings/lower twelve blocks equal their own loaded initialization,
and changed upper weights. Control/treatment training took **15.006/14.835 s**
with **12,196,663,296 bytes** peak allocated CUDA each; wall ratio 0.98865,
CUDA ratio 1.0. The SOP cache export took **213.095 s**, separate from
training. This gate is GO for a quality falsifier, not evidence of retrieval
improvement.

Before a 100-update training run, score the two already acquired source
caches through one identical float-1024D normalized-cosine scorer on the same
fixed official In-Shop TRAIN 6,354-query/6,245-gallery held roles. This
removes the previous export-path confound without another encoder pass.
Advance only if SOP-cache R@1 beats pretrained-cache R@1 by at least **+1.00
point**, product-cluster paired bootstrap 95% lower endpoint is positive,
and mAP@R does not decrease. Otherwise KILL transfer before more training.
