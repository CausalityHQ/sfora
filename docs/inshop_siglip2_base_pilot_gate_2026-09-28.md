# SigLIP2 Base representation pilot

The head-first and pooler-conflict configurations are closed. Change encoder
capacity/pretrained representation rather than another loss/update knob.
The [official SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-256)
checkpoint is public, Apache2.0, revision
`3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`. Its resolved vision config is
12 blocks/768 coordinates/patch16/256px. The baseline is authenticated
SigLIP2 Large/256,24 blocks/1024 coordinates, revision787800c8990e6f058423089178e718139608408c.
This is a smaller-substrate test, not an invented pretraining method or SOTA
claim. Fine-tuning, not frozen-source recall, remains the later target.

One earlier candidate was official DINOv3 ViT-B16,
revision5931719e67bbdb9737e363e781fb0c67687896bc. It is manually gated and
DGX's existing access returned `GatedRepoError`401 even for config. No GPU
job or weight download was launched. The operator was notified of the access
requirement; this candidate stays closed pending a normal access grant.

Acquire only immutable Base config, weights and processor. Pin weights to
official LFS SHA256 `6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863`
and config SHA256 `7b5aedcb8893e31376e129c1ffd7a5392f1a806dbc793ce53eda220c2ec59edf`.
The first acquisition unit `sfora-siglip2-base-acquire-v1`, invocation
`51cc824cd9fe4bca8f2cac2cc049a4fa`, exited1 before downloading because the
installed `hf` CLI lacked `click`. The one retry uses the installed
`huggingface_hub.snapshot_download` API; no dependency was added.

Before a full cache or training run, one <=120-second paired GPU pilot:

- Select512 non-singleton products from original official TRAIN **fit** rows
  using SHA256(`inshop-siglip2-base-pilot-v1\0`+label), one query/gallery
  image each by ascending relative-path SHA. No held or official evaluation.
- Native float16-autocast pooled features from both encoders, batch32, same
 1024 selected images. Require exact native processor pixel parity, finite
 outputs and verified source/model/partition/fit/image inventories.
- Fit each centered PCA128 on only its512 gallery features; pack query and
 gallery with existing signed-int8/f16-norm code and ordinal cosine scoring.
 Report paired512-query prototype R@1 (this fit-only pilot is not held image
 retrieval), raw-float R@1 and compact R@1. Do not call it generalization.
- Time10 interleaved encoder-forward calls per arm at batch32 after3 warmups,
 on the same fixed processed images. Median measured Base encoder time must
 be<=0.8x Large; these calls do NOT measure decode/preprocess/image-to-top-k
 p99 or product QPS.
- Advance only if packed Base-minus-Large R@1>=-3pp with paired product-
 bootstrap95 lower>=-5pp, time ratio<=0.8, whole pilot<=120s and CUDA<10GB.
 A gross quality, authority, numerical or resource failure kills this fixed
 Base checkpoint route before more encoder export/training.

Passing only authorizes a separately frozen full fit-cache/100-update
training feasibility and public-path design, with matched data/compute/scorer
controls and independent TRAIN evaluation. No automatic long training,
10k-serving gate, official read, default or package change follows a pilot.
The prior DINOv2/224, MODA and FashionSigLIP source rejections remain intact.

## Terminal source-pilot decision: KILL

The API acquisition retry `sfora-siglip2-base-acquire-v2`, invocation
`a26d72406f6c451a9e0c744e84d07bf0`, exited0; the full1,500,985,224-byte
weights matched their official SHA. Both acquisition journals are preserved
in `siglip2-base-acquisition-v1/` under the evidence directory.

The ORIGINAL locked DGX pilot `sfora-inshop-siglip2-base-pilot-v1`, invocation
`ff89908a2d834b5c8e344c6402e220c7`, exited0. Every one of1024 native processed
images matched pixel-for-pixel between encoders. Source/partition/fit/model
authority passed; no original run was duplicated.

| Official TRAIN fit-only512 query/gallery pairs | Large baseline | Base candidate | Decision |
|---|---:|---:|---|
| Own-gallery-PCA128 packed prototype accuracy |69.53125%|66.796875%|Delta-2.734375pp passes-3pp point floor|
| Raw float prototype accuracy |71.484375%|69.140625%|Diagnostic only|
| Paired product-bootstrap95 lower on packed delta |—|-6.0546875pp|KILL: below-5pp floor|
| Batch32 encoder-only p50,10 interleaved calls/arm |260.833207ms|85.766922ms|Ratio0.328819; speed gate passes|
| Vision parameter count |315,956,224|92,930,304|Counted actual vision tensors, excludes text tower|
| Main-function pilot wall/combined allocated CUDA peak |22.102866s /1,988,969,472bytes|Shared measurement|Both resource gates pass|

Two models'1024-image export wall was13.140899s. Neither that wall nor
batch32 encoder p50 is full image-to-top-k latency. No p95/p99/QPS
certification, R@1/mAP@R on a held protocol, training cost or training gain
was measured. Prototype pairs are a fit-only gross-deficit screen.

Receipt `inshop-siglip2-base-pilot-v1/receipt.json` SHA-256
`a1eac9d9dfde11304b9a150ced10506db71309b40028dd8afa2ec2bac1693a5d`.
Independent replay verified512 class hits per arm, paired delta/lower,
all10 per-arm forward times/median ratio and all frozen criteria. Replay
extracts the `id_` path component rather than assuming a path offset: the
standard tree's relative paths include `Img/img/`. The packed scorer's
lower-ordinal tie test and Ruff pass. GPU jobs are terminal/idle.

**Stop this fixed Base source route before full cache export, training,
another official read or production promotion.** Do not remove its failed
uncertainty guard, add queries after this result, or call its encoder gain a
matched-quality product speed win. Production source/package remain unchanged.
The external DINOv3 access request is pending; no hidden gated-model download
or alternative mirror was attempted. The joint quality/speed target is unmet.
