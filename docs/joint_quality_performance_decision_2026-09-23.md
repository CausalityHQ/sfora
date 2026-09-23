# Joint quality and performance decision, first training-code gate

This is a dated research checkpoint, not a state-of-the-art claim. The
[target contract](similarity_quality_performance_sota_target_2026-09-23.md)
requires quality above authenticated published references on SOP and In-Shop
and lower measured image-to-top-k latency on the same hardware. Neither joint
gate is met yet. The official test splits below were observed previously, so
the local scores are exploratory product evidence.

## Comparison and training design

The published quality reference is the supervised UNICOM ViT-L/14 at 336 px:
24 transformer layers, width 1024, 16 attention heads and a 768-dimensional
output. Its reported 91.2% SOP and 96.7% In-Shop Recall@1 are dated reference
gates, not a verified 2026 global frontier. We also compare with the official
OML ViT-S/16 at 224 px, whose 384-dimensional float descriptor is a strong
efficient local control, and with a native packed exact-search control for
the serving component. A published number alone does not establish a paired
speed comparison: we need the checkpoint and a protocol-matched run on the
same machine.

The first learning-method screen uses the authenticated UNICOM ViT-B/16 at
224 px (12 transformer layers, width 768, 12 heads) because training and
encoding it are cheaper. It starts at the same pretrained checkpoint in every
arm. Each arm learns a 768-to-128 projection, normalizes the output, uses the
same ArcFace classifier, and deploys the same 128 signed bytes plus f16
inverse norm. The causal arms are ArcFace alone, ArcFace plus SmoothAP on
float cosine, and ArcFace plus SmoothAP on the exact packed forward score
with a straight-through gradient. Match train identities, augmentation,
class-balanced batch sequence, optimizer, update count, seeds, and checkpoint
selection. Use only train identities for recipe selection. Train the entire
backbone under the frozen recipe in the first causal screen: the earlier
head-only rank experiment failed its gate. Compare paired independent seeds
before attributing a gain to the new loss. The
published full-width B/16 SOP result is 88.8%, so this cheaper architecture
is only a feasibility step toward the L/14 quality gate.

The primary quality panel is official SOP (59,551 training images, 60,502
test images, 11,318 train and 11,316 test product identities; leave-one-out
test retrieval) and official In-Shop query/gallery retrieval. CUB-200-2011
and Cars196 check transfer. Every arm reports Recall@1, mAP@R and per-query
results for both float and actually deployed packed descriptors. Measure
training updates/s, VRAM, image encoding and packing, gallery bytes, and
native and full image-to-top-k p50/p95/p99 at batch 1 and 32 on the same GPU.

## Quality and serving status

| Dataset and split | Model or system | Recall@1 | mAP@R | Gallery bytes/item | Image-to-top-k p99 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SOP official test, 60,502 self-retrieval queries | UNICOM ViT-L/14@336 | 91.2% | — | 768 f32 output before indexing | — | Published [UNICOM Table 4](https://arxiv.org/pdf/2304.05884); protocol audit pending |
| SOP official test, same queries | UNICOM ViT-B/16@224, pretrained float | 69.9812% | 0.420759 | 768 f32 before indexing | — | Exploratory reproduced pretrained checkpoint; no SOP fine-tuning |
| SOP official test, same queries | Same B/16 with train-only PCA-128, float | 67.2903% | 0.393903 | 128 f32 before indexing | — | Exploratory projection control; [raw per-query result](evidence/compact_metric/unicom-b16-sop-pretrained-screen-v2.json) |
| SOP official test, same queries | Same B/16 with train-only PCA-128 and int8 wire | 67.2325% | 0.393442 | 130 | — | Exploratory, no SOP fine-tuning; same raw result |
| SOP official test, same queries | Same B/16 full-backbone trained, selected packed-rank 128-D | 81.2221% | 0.562186 | 130 | — | Exploratory one-seed result selected on train-identity holdout; [per-query receipt](evidence/compact_metric/sop-full-backbone-packed-rank-official-test-seed179019-1000-v1.json) |
| SOP official test, same queries | OML ViT-S/16@224 + Sfora compact profile | 85.9757% | 0.641825 | 130 | — | Exploratory, packed representation; [raw profile](evidence/compact_metric/oml-vits16-sop-packed-profile-verification-v1.json) |
| In-Shop official query/gallery | UNICOM ViT-L/14@336 | 96.7% | — | 768 f32 output before indexing | — | Published [UNICOM Table 4](https://arxiv.org/pdf/2304.05884); evaluator uses normalized prefix-512 Euclidean |
| In-Shop official query/gallery | UNICOM ViT-L/14@336 + Sfora compact profile | 95.4283% | 0.800020 | 130 | — | Exploratory, [local result](compact_metric_selector_result_2026-09-19.md) |
| In-Shop official 14,218-query/12,612-gallery split | OML ViT-S/16@224, raw 384-D Euclidean | 92.0945% | 0.685148 | 1536 as f32 | — | Exploratory local reproduction of the published 92.1% control; [raw receipt](evidence/compact_metric/oml-vits16-inshop-baseline-v1.json) |

The independent [STIR paper](https://arxiv.org/pdf/2304.13393) reports
88.3% SOP and 95.0% In-Shop Recall@1 for a ViT-S/16 system with a symmetric
pixel-level top-five reranker. It is a useful method comparison, but its
additional inference stage and backbone make it a different latency and
storage point from a single compact embedding. These published comparisons
are a dated, checked reference panel, not an exhaustive current-frontier
audit or a claim that any local system surpasses state of the art.

The later [LoCoRe CVPR 2025 paper](https://arxiv.org/html/2503.21772v1)
reports 83.8% SOP and 87.9% In-Shop Recall@1 for its base model when it
re-ranks 100 images from its own global-descriptor shortlist. Its shared
global-only controls are 80.8% and 88.5%, respectively. Its method compares
re-rankers on the same shortlist and uses additional local descriptors;
these numbers do not replace the stronger UNICOM single-descriptor quality
gates or provide a same-hardware image-to-top-k speed comparison.

The OML float source rescored with the same stable-ordinal evaluator remains
86.5575% Recall@1 and 0.654393 mAP@R, with
[raw matched-scorer evidence](evidence/compact_metric/oml-vits16-sop-matched-scorer-v1.json).
The published full-width models and the 130-byte compact profiles have
different storage budgets. The current [paired search replay](evidence/compact_metric/oml-vits16-sop-packed-1m-replay-v2.json)
uses a tiled SOP gallery, not one million distinct images. It reports search
medians near 1.05 ms at batch 1 and 2.8 ms at batch 32 on a DGX GB10, with
50 calls per arm. Those timings exclude decode, encoding, and packing; their
p99 values are diagnostic only. They cannot fill the end-to-end column above.
A first end-to-end paired replay was terminated after 41 minutes with **zero
timed calls**: its first `tileiras --gpu-name sm_121 --opt-level 3` compilation
was still using one CPU core and about 5.7 GiB RSS. The original B/16 query
path also lacked the source-vector normalization required by its train-fitted
PCA head, so that replay could not produce a valid quality-matched result.
The process group exited after termination; no overlapping copy was started.
Cold compilation remains a measured startup concern, and a corrected paired
timing run is pending. This observation does not establish hot-path p99.

The authenticated B/16 feature export contains the official 59,551 train and
60,502 test images, checkpoint SHA-256
`c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef`,
and archive SHA-256
`16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f`.
On the DGX GB10, train-only PCA-128 fitting took 2.900 s, transforming all
test features 0.091 s, and packing them 0.028 s. The stable-ordinal scorer
took 5.158 s for full-width float, 3.076 s for PCA float, and 3.252 s for
PCA packed over all 60,502 queries. These aggregate scoring times have an
order effect and different descriptor widths; they are not paired serving
latencies or evidence that int8 search is faster. The three-arm raw screen is
[`unicom-b16-sop-pretrained-screen-v2.json`](evidence/compact_metric/unicom-b16-sop-pretrained-screen-v2.json),
SHA-256 `a7c65b7b5dda1a8884f08ea384f150ef98ac20c77607c0b8b2ed2fbbe6c1053d`.

For arm selection, a separate deterministic 90/10 **class-disjoint partition
of SOP training identities** (seed 179019) has 53,700 fit images and 5,851
validation images. With the same pretrained B/16 checkpoint, fit-only PCA-128,
and matched scorer, validation Recall@1 was 84.0882% full float, 82.0714%
PCA float, and 82.0543% packed; mAP@R was 0.591994, 0.564800, and 0.565342.
The [raw holdout screen](evidence/compact_metric/unicom-b16-sop-train-holdout-screen-v1.json)
has per-query outputs, SHA-256
`0480525d64fdf8a40bd8f1c3cde8dcaa1bb644236af48b0e22dbc975766b9d1d`.
This split is substantially easier than the official SOP test for the
pretrained checkpoint; use it for **paired arm deltas and checkpoint choice**,
not as a prediction of absolute official-test quality.

## Causal training-cost diagnostic

The production [packed rank scorer](../src/sfora/deployed_code_rank.py) now
uses the actual signed-int8 code and f16 inverse norm in its forward pass,
with straight-through gradients for rounding. A matched float rank control
shares the SmoothAP computation. In the GB10 synthetic fixed-feature screen,
each arm used 128-dimensional inputs, four images per identity, seed 179019,
five warmups and 50 timed forward/backward calls. No optimizer, image encoder,
or dataset quality was measured. The two arm orders were float→packed and
packed→float.

| Batch | Float p50, two orders (ms) | Packed p50, two orders (ms) | Packed/float p50 range | Quality result |
| ---: | ---: | ---: | ---: | --- |
| 32 | 1.850 / 1.600 | 2.176 / 2.154 | 1.18–1.35× | Not measured |
| 128 | 2.156 / 2.170 | 2.403 / 2.382 | 1.10–1.11× | Not measured |

The candidate costs about 0.2–0.5 ms more per synthetic rank step here.
Peak CUDA allocated values were affected by previous arms in the same process,
so this receipt does not establish an arm-level VRAM difference. Source and
raw samples are bound in
[`deployed-code-rank-step-dgx-v1.json`](evidence/compact_metric/deployed-code-rank-step-dgx-v1.json):
NVIDIA GB10, PyTorch 2.12.1+cu130, script SHA-256
`a618831606d8171f96303833e5c39d712eefe0ee628f15ff5ad1661887182a26`,
scorer SHA-256
`3adbdbedb9867e67d7038d3a6697e738c73eeab4bae4c3a1a302538a97c25c6d`,
raw receipt SHA-256
`a82b9fc65fdc0e59aef6bf656c7361ffb3867029b720891f944549f0f7a501e8`.
This timing receipt predates the final CUDA-autocast correctness fix in the
production scorer. Its arms ran without an outer autocast context, so it is a
diagnostic of that precise version, not a performance claim for the final
implementation. The final scorer has separate CUDA fp16 and bf16 autocast
forward/loss parity tests.
The initial confounded comparison is retained as
[`deployed-code-rank-step-confounded-v0.json`](evidence/compact_metric/deployed-code-rank-step-confounded-v0.json)
and must not be used to assert a speed win.

## Next measured decision

Direct train-only PCA compaction of the pretrained B/16 checkpoint is closed
as an immediate quality path: its full-width float result is 16.58 percentage
points below the OML float SOP source. Of the 2.75-point drop from full-width
B/16 to packed PCA-128, 2.69 points arise by projecting to float PCA-128 and
only 0.06 from int8 packing. In 5,000 bootstrap resamples of SOP test product
identities, the paired Recall@1 change for PCA float versus full float was
−2.691 percentage points (95% interval −2.899 to −2.485), while packing
versus the same PCA float was −0.058 points (−0.123 to +0.003). The paired
mAP@R losses were −2.686 points (−2.816 to −2.555) for projection and
−0.046 points (−0.073 to −0.020) for packing. These intervals describe an
exploratory result on a previously observed test split; they do not turn it
into a confirmatory method claim. See the
[per-query class-clustered decomposition](evidence/compact_metric/unicom-b16-sop-loss-decomposition-v1.json),
SHA-256 `33c33a8da9a2ca06143a2a6ef1c6b761770e6aea60f36751f8959894e168379a`.
This does not falsify supervised B/16 or all learned compact heads: UNICOM
reports 88.8% SOP Recall@1 after supervised fine-tuning. The first learning
experiment trained a 128-dimensional head and the backbone together on
SOP train identities, beginning with a matched ArcFace control and adding
float-rank and packed-rank arms. It matched starting checkpoint, classifier,
batch identities and order, total updates, scorer, and seed. Measure training
time/throughput/VRAM and encoder-plus-packed-search latency before promoting
an arm. One seed is a feasibility screen; at least three paired seeds and a
second dataset are needed for an algorithmic claim. The OML compact profile
remains the efficient product baseline for a future paired end-to-end timing;
no image-to-result p99 result exists yet.
The full-backbone feasibility runner is
[`train_sop_compact_backbone.py`](../scripts/train_sop_compact_backbone.py):
all three arms share the fixed 90/10 train-identity split, 32 identities ×
4 images per batch, PCA-initialized 768-to-128 head, imprinted ArcFace
classifier, the authenticated backbone's internal fp16 attention with gradient
scaling, AdamW optimizer topology, 1,000 updates,
and exact packed validation scorer. The rank arms only add the specified
2.0-weight float or packed SmoothAP term. The coefficient was frozen from
the first train-fit batch before full training: at weight 0.1, the float and
packed rank gradients were only 1.031% and 1.007% of the ArcFace gradient
at the head output. At weight 2.0, the same measured gradients imply 20.62%
and 20.15%. This is an initialization diagnostic on cached features, not a
guarantee about later training; see the
[raw gradient receipt](evidence/compact_metric/sop-compact-rank-gradient-diagnostic-v1.json),
SHA-256 `54f4504626b35598e99c550da78dc6b68745677070e052e5232d4b93de86dfaa`.
A one-seed result can screen this
mechanism but cannot establish an algorithmic improvement.
The first 1,000-update full-backbone ArcFace control completed on the DGX
GB10 with seed 179019. On the **SOP train-identity holdout**, packed Recall@1
increased from 82.0543% to 91.8988%, and packed mAP@R from 0.565342 to
0.726940. The final float metrics were 91.8305% and 0.726542. These are
validation scores on classes disjoint from the fitting identities, not the
official SOP test and not a SOTA comparison. Training took 688.306 s for
1,000 updates (1.453 updates/s), with 0.685 s median and 0.691 s 95th
percentile step duration and 12.411 GB peak CUDA allocation. The
[complete per-query receipt](evidence/compact_metric/sop-full-backbone-arcface-seed179019-1000-v1.json)
has SHA-256
`8a533ab4668576dae14f15c183553af16310f88b7e8df70272da5ab3fe841662`;
the remote checkpoint SHA-256 recorded in that receipt is
`a2568c671336b2c5a97587634ed8312c026625e172eadcba0a0df57abf799672`.
All ten source hashes in the receipt match this checkout. The matched
1,000-update float-rank arm has now finished:
packed Recall@1 92.2236% and mAP@R 0.732604 on the same holdout, versus
91.8988% and 0.726940 for ArcFace. Training took 688.380 s and used the same
12.411 GB peak CUDA allocation. The seed, split, batch schedule, pretrained
checkpoint, source hashes, initial head, and initial classifier match the
control receipt. A paired 10,000-resample product-identity bootstrap gives
packed Recall@1 delta +0.325 percentage points (95% interval +0.122 to
+0.539) and mAP@R delta +0.005664 (+0.004178 to +0.007177). This is an
intra-seed query uncertainty summary; independent training seeds remain
required before attributing a method improvement. The
[float-rank per-query receipt](evidence/compact_metric/sop-full-backbone-float-rank-seed179019-1000-v1.json)
has SHA-256
`6ff9e916da1cf709d93a7e3f09470b38817ec93410a8860aff09ff9f2cfd798e`.
The packed-rank run has since completed and passes the same schedule, split,
source, initial-head, and initial-classifier checks. The one-seed holdout
decision is:

| 1,000-update arm, seed 179019 | Packed Recall@1 | Packed mAP@R | Train seconds | Peak allocated CUDA bytes |
| --- | ---: | ---: | ---: | ---: |
| ArcFace control | 91.8988% | 0.726940 | 688.306 | 12,410,501,120 |
| ArcFace + float SmoothAP | 92.2236% | 0.732604 | 688.380 | 12,410,501,120 |
| ArcFace + deployed-packed SmoothAP | 92.1894% | **0.734052** | 691.429 | 12,410,501,120 |

The predeclared train-holdout packed mAP@R rule selects **packed-rank** for
one official SOP test evaluation. Its mAP@R advantage over ArcFace is
+0.007112 (paired product-identity bootstrap 95% interval +0.005561 to
+0.008743); over float-rank it is +0.001448 (+0.000242 to +0.002651).
Its Recall@1 is +0.291 percentage points against ArcFace (+0.071 to +0.513)
and −0.034 points against float-rank (−0.221 to +0.156). These within-seed
intervals do not account for training-seed variance. The
[packed-rank per-query receipt](evidence/compact_metric/sop-full-backbone-packed-rank-seed179019-1000-v1.json)
has SHA-256
`3540d86f9f684031a4be3e8e0169bb355463b9805482c5e8f3d6d31674b34172`.
The one frozen **official SOP test** evaluation has now measured the selected
packed-rank checkpoint: 81.2221% Recall@1 and 0.562186 mAP@R for the deployed
130-byte descriptor; float 128-D scoring gives 81.2486% and 0.562395.
Encoding the 60,502 images took 133.390 s, packing 0.021 s, and aggregate
packed leave-one-out scoring 3.281 s on the DGX GB10. These aggregate stage
times are **not** batch-1 or batch-32 image-to-top-k p99. The complete
[official per-query receipt](evidence/compact_metric/sop-full-backbone-packed-rank-official-test-seed179019-1000-v1.json)
has SHA-256
`03ab8c7a0c92afbcee4cbc62d76e9020530de06a9c0e191d0faf202f46c3492d`,
binds the three holdout-selection receipts, exact test inventory, source, and
trained checkpoint, and passed per-query aggregate checks. The result is
**4.754 percentage points below** local OML packed SOP Recall@1 and
**9.978 points below** the published UNICOM L/14@336 reference. The trained
float-to-packed loss is only 0.026 percentage points, so the first priority
is the learned representation or training recipe, not finer quantization.
This B/16 result improves 13.990 points over its own pretrained packed PCA
control, yet fails the joint quality gate. The next experiment must change
representation capacity or training method while preserving a matched packed
scorer; repeated tuning of this already-failed 128-D B/16 point is unwarranted.
Second-dataset transfer and image-to-top-k p99 remain unmeasured for this
checkpoint.

The distinct **4,000-update ArcFace budget diagnostic** completed on the
same B/16 checkpoint, seed, 53,700/5,851 training-identity partition,
PCA-initialized 128-D head, deterministic batch prefix, and constant-rate
screen recipe. Its train-only holdout packed Recall@1 was **92.8388%** and
mAP@R **0.755029**, versus 91.8988% and 0.726940 at 1,000 ArcFace updates.
The within-seed class-bootstrap differences are +0.940 percentage points in
Recall@1 (95% interval +0.481 to +1.429 points) and +0.028089 mAP@R
(+0.023543 to +0.032718). Training took 2,745.797 s on the GB10 with
12,410,501,120 peak allocated CUDA bytes, about four times the 1,000-update
training cost. The raw [per-query receipt](evidence/compact_metric/sop-full-backbone-arcface-seed179019-4000-v1.json)
has SHA-256
`4b6a74d0e258d5b6a3d4b770d0fc9945fb02e0dc02a0f2d515c5e118d1c4e9de`;
the DGX checkpoint is
`/home/riomus/runs/sfora-sop-compact-backbone-179019/full-arcface-seed179019-4000.pt`,
SHA-256 `28df81a7fb7fcdd4f6f5dbbfdebd19702d9b65db519cbb890d76c299e9e5441c`.
Source, split, initialization, and per-query aggregate hashes were verified
against the 1,000-update ArcFace receipt. This is evidence that the first
screen was under-budget on its own holdout. It is **not** an official-test
result or a matched-budget algorithmic comparison with the rank arms.

An authenticated, paired **pretrained architecture screen** reused the same
5,851-image SOP training-class holdout and exact image rows for B/16@224 and
L/14@336. B/16 full-width/PCA-128 packed Recall@1 was 84.0882%/82.0543%;
L/14 was 87.5064%/86.1220%. The L/14 packed mAP@R was 0.623416 versus
B/16's 0.565342. The larger backbone improves the pretrained packed score
by 4.068 percentage points and loses 1.384 points when compressed from its
full-width float output, versus 2.034 points for B/16. This is a train-only
architecture diagnostic, not a trained L/14 result or official-test claim.
The [raw paired screen](evidence/compact_metric/sop-pretrained-architecture-holdout-b16-l14-seed179019-v1.json)
has per-query values and SHA-256
`f13ad36978fe7afa4c8754d153da42468c7cc518e957824aa3334c1179bf5476`.

The upstream UNICOM
[`retrieval.py`](https://github.com/deepglint/unicom/blob/d71992ed969e6c271436ac0a0ee1f3ca61474ac0/unicom/retrieval.py)
generic defaults are **not** the SOP training recipe. The authenticated
[`sop_vit_b_16.sh`](https://github.com/deepglint/unicom/blob/d71992ed969e6c271436ac0a0ee1f3ca61474ac0/unicom/scripts/sop_vit_b_16.sh)
launches one GPU with batch 64, 64 epochs, OneCycle peak backbone learning
rate 1e-5, classifier multiplier 10, ArcFace margin 0.25 and scale 32, and
zero weight decay. The corresponding L/14@336 script launches eight GPUs at
batch 16 **per GPU**, also for 64 epochs with margin 0.25 and scale 32.
Our 1,000-update screen used about 2.4 image-count passes over its fit images,
constant backbone learning rate 1e-5, scale 64, margin 0.3, and weight decay
0.05. Its failure therefore does not isolate backbone capacity from an
under-budget, off-reference recipe. The
next causal gate is a longer B/16 ArcFace training run with a reference-like
schedule and periodic **train-holdout** checks, paired with a full-width
768-D control under the same inputs and budget. Only if that recipe approaches
the published B/16 result should full L/14 fine-tuning or teacher-to-student
distillation consume the substantially larger training budget. The existing
official test result is a feasibility gate and must not select checkpoints.

The prepared reference-like compact control keeps the same 53,700-image fit
partition, balanced 16-product × 4-image sampler, PCA-initialized 128-D head,
and packed holdout evaluator. It changes the optimizer to upstream-like
OneCycle peak rates (backbone 1e-5, head/classifier 1e-4), zero weight decay,
ArcFace margin 0.25 and scale 32, and the upstream timm training augmentation.
It trains for 53,760 updates, the equivalent of 64 passes by image count, with
durable **diagnostic-only** checkpoints and train-only holdout receipts at
4,000, 8,000, 16,000, 32,000, and 48,000 updates, plus the final checkpoint.
The 4,000-update diagnostic is not a stop gate; none of the intermediate
checkpoints is a resumable training state. The best completed checkpoint is
selected by holdout packed mAP@R, then packed Recall@1, then earlier step.
Because the balanced sampler differs from upstream's shuffled image sampler,
the compact head is new, only 90% of SOP train identities are fitted, and
gradient clipping at norm 1.0 is additional, this is **reference-like**, not
a faithful published UNICOM reproduction. The upstream B/16 script's
single-GPU batch 64 avoids a global-batch mismatch, but its classifier is
full-width and uses `num_feat=512` in PartialFC.
The completed 4,000-update constant-rate screen is a separate budget
diagnostic using the original recipe. Its measured gain must be interpreted
separately from the longer reference-like control now running on the DGX.

The reference-like run's first **train-identity holdout** diagnostic at 4,000
of 53,760 updates recorded packed Recall@1 **90.2239%** and mAP@R
**0.696166** (float 90.2410% and 0.695663). Its immutable [raw receipt](evidence/compact_metric/sop-reference-arcface-seed179019-step4000-v1.json)
has per-query outputs and SHA-256
`35341869a4336f756331779ccd37bf3a8f837fc0e463498af9b46750adc6eaa2`;
the remote checkpoint SHA-256 recorded by the trainer is
`9ad0375f73bd0eb7af9ccb48450cf210b9e7f7ab32a43dd700e4092a4a3803da`.
The source-manifest digest matches the frozen `b8f85611` snapshot
(`79dc20905e9bac9ed865fbbbece6d69623cad753812bd1d75d2599889466e427`).
The older constant-rate 4,000-update screen reached 92.8388% packed Recall@1
and 0.755029 mAP@R on the same holdout, but the recipes differ in schedule,
augmentation, margin, scale, weight decay, and batch size. This early
OneCycle checkpoint is still in warmup, so the comparison is diagnostic and
does not select the final model or support a method claim.

The official SOP evaluator for this long recipe is prepared but must wait for
all six checkpoints and the final training receipt. It checks the frozen
training-source snapshot and each checkpoint digest, selects the highest
train-holdout packed mAP@R (then Recall@1, then earlier step), binds the
selected model tensors to that receipt, and records the evaluator's loaded
source hashes. An atomic claim in the shared SOP dataset directory is keyed
by the final checkpoint digest, preventing an automatic second official-test
exposure after a copied receipt, changed output path, or crashed evaluation.
The first diagnostic receipt's eleven source-file hashes
were independently compared with git commit `b8f85611` and all matched.
The evaluator also requires a pinned [binary manifest](evidence/compact_metric/sop-test-image-sha256-v1.bin)
of one SHA-256 digest per official test image in metadata order. Its SHA-256 is
`28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1`;
the [builder receipt](evidence/compact_metric/sop-test-image-sha256-v1.json)
has SHA-256
`33555493dc1b7eb074f8b53ea135ba90953d3eb0a98820a5bce9bbb0e1a9168d`.
The 60,502 images occupy 1,442,004,273 source bytes. The evaluator hashes the
same bytes it decodes for inference and rejects a mismatch. This pins the
local DGX corpus used before model selection; it does not independently
certify the images against a publisher-supplied checksum.
The evaluator's aggregate encoding time includes JPEG decoding and SHA-256
verification; public image-to-top-k p50/p95/p99 require a separate matched
serving benchmark.

A matched full-width B/16 control is prepared with the same 53,700 fit images,
class-disjoint holdout, pretrained checkpoint, seed, batch schedule, reference
augmentation, ArcFace margin/scale, OneCycle schedule, and 53,760 updates. Its
trainable 768-to-768 head starts as the identity, so its initial float
descriptor matches the pretrained 768-dimensional source; class proxies are
imprinted only from fit identities. The runner records both 768-D float and
770-byte packed holdout results. Paired comparison with the 128-D run can
attribute a quality difference to embedding width and its associated head
initialization, but the two systems occupy different gallery-storage and
search-cost points. It has not been launched while the compact DGX job runs.
The shared global gradient clip may reduce backbone updates by different
amounts as the head and classifier widths change. A full-width float result
therefore diagnoses the trained-system capacity, not width in isolation. The
770-byte fixed-scale int8 wire also has a different quantization error than
the 130-byte wire. After training, fit a PCA-128 projection on full-width
**fit identities only**, then score the held-out identities in float and
packed form. This will test whether full-width learning followed by compact
deployment improves the same 130-byte product profile without another GPU
training run.

The one-update ArcFace canary passed an image-level step-zero parity check:
the packed validation Recall@1 and mAP@R were exactly 0.8205435 and 0.5653424,
matching the cached-feature screen. Its full-backbone update took 2.182 s
including batch loading, with 10.766 GB peak CUDA allocation; initial and
final validation took 14.55 and 15.12 s. Its after-one-update packed metrics
(0.8207144 Recall@1, 0.5662724 mAP@R) are a plumbing check, not a quality
finding. The [raw canary receipt](evidence/compact_metric/sop-full-backbone-arcface-canary-v1.json)
has per-query evidence, SHA-256
`0e1c49673c23c3dc084833c7447ed1e9025126bb915fdaee8c4f5a9f0b8d4011`;
its DGX checkpoint is
`/home/riomus/runs/sfora-sop-compact-backbone-179019/canary-arcface-1.pt`,
SHA-256 `eed4eafeb81e4a078acd5828727bb37c1a982ed7dec8c39b6e269fd2ab25b9ad`.
An additional one-update packed-rank canary used the earlier 0.1 coefficient;
its initial ArcFace and raw SmoothAP losses were 7.893748 and 0.060270.
That coefficient was superseded before the full runs. The
[raw packed canary](evidence/compact_metric/sop-full-backbone-packed-canary-coefficient0p1-v1.json)
is retained under SHA-256
`d9febf3b4caa780c1cb328137929dbf31d2741fb88b4161eb7e5a6bd7311b85b`.
