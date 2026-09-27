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
