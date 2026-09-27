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
