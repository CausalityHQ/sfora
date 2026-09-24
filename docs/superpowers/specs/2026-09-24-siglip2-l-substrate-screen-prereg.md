# SigLIP2 L/16 substrate feasibility screen (exploratory preregistration)

## Question and choice

Can a pretrained L-class encoder with fewer image tokens produce a better
130-byte image descriptor and faster full image-to-top-10 retrieval than the
authenticated pretrained UNICOM L/14@336 substrate? This tests a starting
representation, not a Sfora training-method advance. The closed B/16 slot
readout, MaxSim rerank, and 336 area-pool variants are not reused. The
alternative of another B/16 loss screen cannot close the several-point
official SOP gap by itself; the alternative of full L/14 retraining is much
more expensive before a faster substrate is known. Select the public
Apache-2.0 `google/siglip2-large-patch16-256` model at immutable Hugging Face
revision `787800c8990e6f058423089178e718139608408c` for this first
substrate screen. Its native 256-pixel processor is part of the treatment.
No new model weights are trained in this screen.

## Authority and split

Read the 59,551 official Stanford Online Products **TRAIN** image rows in
their authenticated archive order; do not read TEST arrays, images, or
metadata. The existing seed-179019 product partition contains 53,700 fit
rows and 5,851 holdout queries from 1,132 unseen-to-fit products. The
authenticated UNICOM L/14@336 TRAIN archive SHA-256 is
`1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a`.
Reuse its row identities and native-preprocessing descriptors as the
reference. Authenticate the SigLIP2 revision, model weight hash, processor
files, ordered image IDs/labels/paths, source code, and hardware in the raw
receipt. Keep the full-resolution SigLIP2 image descriptor and its image
preprocessing fixed across all rows.

For each encoder independently, normalize its raw image descriptor, fit a
centered PCA-128 map on the 53,700 **fit** rows only, project all 59,551
TRAIN rows, normalize and pack with Sfora's signed-int8 plus f16-inverse-norm
format (130 bytes). Record float full-width, float PCA-128, and packed
PCA-128 holdout-only Recall@1/mAP@R. Primary quality metric is packed
PCA-128 Recall@1 for the 5,851 holdout queries against the 59,551 TRAIN
gallery with self exclusion and exact deployed score arithmetic. Also report
packed full-gallery mAP@R and per-query outcomes. Use one fixed
1,132-product paired bootstrap (5,000 draws, seed 179019) for candidate
minus UNICOM. This reused development split only screens feasibility; no
official-quality or SOTA claim follows.

## Performance and gates

Measure native image decode/preprocess, encoder, packing, exact packed
top-10 search, and full image-to-top-10 timing on the same DGX GB10, same
sample images and resident 59,551-item gallery, at batch 1 and batch 32.
Run at least 10 alternating AB/BA blocks after warmup. Report p50/p95,
throughput, CUDA allocation, host RSS, model and gallery bytes, and timing
sample count. A p99 claim remains unavailable until the product's 10,000-call
protocol is run. The encoder cost must include SigLIP2's own processor and
vision tower; do not present a resident-tensor number as end-to-end latency.

Advance this substrate to a matched learning-method experiment only if
packed full-TRAIN-gallery Recall@1 beats the pretrained UNICOM L/14@336
reference by at least **+1.0 percentage point**, the paired product 95%
lower endpoint is above zero, packed holdout mAP@R does not regress by more
than **0.003** (paired lower endpoint above −0.003), and full image-to-top-10
p50 is at most **0.85×** reference at both batch sizes. These are screen
thresholds, not a product win. If any condition fails, close this exact
SigLIP2 substrate and choose a materially different substrate or training
signal. Never substitute an estimate for a measured cell. Any passing
substrate must then undergo source-recipe matched supervised training with
a compact-head control, independent seeds, official SOP and In-Shop
evaluation, CUB/Cars transfer, and certified p99 before a joint claim.

## Resource and failure rules

The public model card reports Apache-2.0 and a 3,526,204,360-byte
`model.safetensors` file; this is a model-file size, not training memory.
Pin the revision before download and do not accept gated licenses. Run one
durable job, preserve its original exit status, per-stage checkpointed
receipts, and the exact downloaded file hashes. Stop on row mismatch,
nonfinite descriptor, source-hash drift, failed score parity, or device OOM.
There is no simultaneous duplicate GPU or download job. The code and
preregistration are committed before the main comparison starts.
