# Joint quality and performance decision, first training-code gate

## 24 September exact affine L/14 head F0: smaller safe baseline

The UNICOM L/14 evaluation head is
`Linear(589824→1024) → BatchNorm → Linear(1024→768) → BatchNorm`, with no
activation. In evaluation mode it can be composed into one 589,824→768
affine layer. This is exact in real arithmetic; finite-precision GEMM
reordering must be checked. The new optional
[inference transform](../src/sfora/inference_head.py) implements the
composition and rejects training-mode or incompatible heads. Two numerical
unit tests pass, including nontrivial BatchNorm running statistics, affine
parameters and biases. The [raw F0 receipt](evidence/compact_metric/l14-exact-head-fold-f0-v1.json)
has SHA-256 `82911b63163185ba9ed1db099afecea1c88365b78cee10aa5ec7e1b81d880ec7`.

On 32 sampled SOP **training** images from the pinned pretrained L/14@336
checkpoint, the largest absolute 768-D descriptor difference is
`7.153e-7`, largest relative L2 difference `1.355e-6`, and mean normalized
cosine 1.0. These 32 images are a numerical smoke check, not retrieval
quality evidence. The fold removes about 0.604 GB of fp32 head weights
before package overhead, compared with the original head. It leaves the
same 768-D output and 770-byte packed gallery wire.

| DGX GB10, resident preprocessed SOP train tensors | Original GPU p50 | Exact-fold GPU p50 | Fold/original ratio |
| --- | ---: | ---: | ---: |
| Full encoder, batch 1 | 28.686 ms | 26.094 ms | 0.9096 |
| Full encoder, batch 32 | 399.444 ms | 393.422 ms | 0.9849 |
| Head only, batch 1 | 9.708 ms | 7.211 ms | 0.7428 |
| Head only, batch 32 | 26.049 ms | 20.140 ms | 0.7732 |

Each cell contains 100 calls in ten alternating blocks. For full-encoder
batch 1, the median paired block-median ratio is 0.9077, with a descriptive
10,000-resample block-bootstrap 95% interval 0.9000–0.9117; for batch 32
it is 0.9857 (0.9840–0.9873). The DGX job exited zero in 110.19 s,
with peak CUDA allocation 7.90 GB. The head remains fp32 and runs outside
UNICOM's internal fp16 transformer-block autocast. This is a short
encoder-only screen: no retrieval quality, full image-to-top-k or contract
p99 was measured. It is an execution-preserving serving control, not a new
similarity-learning method.

The compact Sfora path calls `F.normalize(source)` **between** the UNICOM
head and its 768→128 projection; its fitted centered PCA also has a mean.
Thus composing those stages directly into a 128-output affine layer would
change the current product function. A 128-output fold is exact only for a
separately specified, purely affine downstream head without that intermediate
normalization, or under additional conditions such as a bias-free projection
followed by final normalization. This control must not silently replace the
130-byte product path. Both exact-fold and rank-512 heads should be compared
on the same eventual compact wire and matched full pipeline; the two F0
receipts are separate runs and do not establish their direct paired ordering.

## 24 September L/14 low-rank head screen: a viable latency lever, not a quality win

The [raw train-only head receipt](evidence/compact_metric/l14-lowrank-head-train-holdout-v1.json)
(SHA-256 `fafe5cf48fc92f1fb9c1ab6019199a6fafca6487e7ee35a3f541d7d4cc95e71e`)
comes from the pinned **pretrained** UNICOM ViT-L/14@336 checkpoint and
5,851 images from SOP **training identities**, held out by the established
seed-179019 class partition. The [versioned probe](../scripts/probe_l14_head_lowrank.py)
replaces only its 1,024 × 589,824 token-flattening linear layer with the
optimal rank-512 Frobenius approximation `U(UᵀW)` obtained from the top
eigenvectors of `WWᵀ`; it uses no SOP labels to build that approximation.
The rank-512 matrix captures 99.1766% of the source weight matrix's squared
Frobenius norm. That spectral number alone predicts neither retrieval quality
nor latency. The remaining encoder, BatchNorm layers and 768-D output are
unchanged. This is a known low-rank factorization, not a new Sfora learning
method or evidence about a supervised L/14 checkpoint.

| SOP train-identity holdout, 5,851 self queries | Original L/14 | Rank-512 head |
| --- | ---: | ---: |
| 768-D full-float Recall@1 / mAP@R | 87.5064% / 0.645904 | 87.5064% / 0.645906 |
| 770-byte packed Recall@1 / mAP@R (off the 130-byte target) | 87.5064% / 0.645945 | 87.4893% / 0.646260 |
| 768-D upstream-prefix-512 Euclidean Recall@1 / mAP@R | 87.3184% / 0.643466 | 87.2500% / 0.643385 |

The original full-float 87.5064% / 0.645904 exactly reproduces the pinned
pretrained architecture holdout receipt at its reported precision. In the
packed comparison, rank-512 gains four queries and loses five. The paired
1,132-product class-bootstrap 95% descriptive interval for the packed
Recall@1 difference is **−0.1197 to +0.0849 percentage points**
(10,000 resamples, NumPy PCG64 seed 179019). The mean cosine between source
and factorized normalized descriptors is 0.999982, but this is only a
coordinate diagnostic. The published UNICOM normalized-prefix Euclidean
scorer loses four correct queries and gains none on this split, a
**−0.0684-point** change. Its post-hoc product-bootstrap descriptive 95%
interval is −0.1393 to −0.0168 points. No In-Shop or official-test quality
retention has been measured for this factorization. These intervals are
retrospective diagnostics, not a predeclared retention test.

| Preprocessed SOP training images resident on GB10 | Original encoder GPU p50 | Rank-512 encoder GPU p50 | Ratio |
| --- | ---: | ---: | ---: |
| Batch 1 | 28.960 ms | 23.964 ms | 0.8275 |
| Batch 32 | 401.654 ms | 389.419 ms | 0.9695 |

Each timing cell has 100 calls in ten alternated blocks on the same model
with the head module swapped outside the timed call; both arms use fp32
stored head weights, run the head outside autocast, and use UNICOM's
internal fp16 autocast for transformer blocks. The median
of ten paired block-median ratios is 0.8267 at batch 1 (10,000-resample
block-bootstrap 95% descriptive interval 0.8207–0.8284) and 0.9692 at
batch 32 (0.9677–0.9701). The measured head saves about 1.205 GB of fp32
weight storage before package overhead. The DGX job exited zero in 255.19 s,
with peak CUDA allocation 5.45 GB and peak host RSS 10.46 GB. The local
repository suite passed 5,446 tests with 12 skips; the new focused probe
tests passed 3/3 and Ruff passed. These are encoder-only timings on resident
tensors, **not** decode/pack/search, serving p99, training throughput, or
a joint quality/performance advance over the published supervised UNICOM
91.2% SOP and 96.7% In-Shop Recall@1 references.

The fold F0 remains rejected. The rank-512 head is retained as a low-cost
candidate for the next **matched full image-to-top-k** batch-1/batch-32
screen and a train-only In-Shop retrieval check. It should advance to a
supervised L/14 training arm only if the same-stack full-pipeline latency
improves at both batch sizes and In-Shop train-only quality stays within a
predeclared retention tolerance. A 10,000-call interleaved paired p99 panel
and official split evaluation remain required for a product claim. Parallel
Claude Opus 5.5 and GPT-6 Astra terminal-result critiques were started after
the receipt; their findings will be reconciled before that next gate.

## 24 September full-width transfer development result

### Frozen source-to-trained weight blend diagnostic

The [five-arm blend receipt](evidence/compact_metric/sop-fullwidth-step48000-weight-blend-development-v1.json)
(SHA-256 `66caaefd0800940ec4dc6efdda753243d29449f6b0ae5e80baaeb8f6157043dd`)
evaluates α = 0, 0.25, 0.5, 0.75, 1 in
`(1−α)·pretrained + α·SOP-trained` for both the B/16 backbone and its
identity-to-trained 768-D head. All arms use the same signed-int8 packed
cosine scorer, 770 bytes per item, 5,851 **SOP train-identity holdout**
images and 5,864 **CUB development classes 1–100** images. The α=0 and α=1
per-query float/packed scores reproduce the pinned training and CUB receipts;
CUB endpoint feature-array hashes match. The DGX GB10 job exited 0 in 155.69 s,
with 6.02 GB peak host RSS and 1.15 GB peak CUDA allocation. This is a
development diagnostic, not an official-test or SOTA measurement; the SOP
holdout had already selected checkpoint step 48,000.

| Trained weight fraction α | SOP train holdout packed Recall@1 / mAP@R | CUB development packed Recall@1 / mAP@R |
| ---: | ---: | ---: |
| 0 | 84.0540% / 0.591906 | 87.5000% / 0.618651 |
| 0.25 | 89.5744% / 0.679477 | 87.2613% / 0.610725 |
| 0.5 | 93.1977% / 0.760299 | 86.4939% / 0.587066 |
| 0.75 | 95.3512% / 0.816001 | 84.7033% / 0.536931 |
| 1 | 96.0349% / 0.836934 | 80.1330% / 0.430912 |

At α=0.75, CUB development Recall@1 gains **4.5703 points** versus fully
trained (paired 100-class bootstrap descriptive 95% interval **+3.6134 to
+5.6121**) while SOP train-holdout loses **0.6836 points** (paired
1,132-product interval **−1.0012 to −0.3745**). At α=0.5, CUB gains 6.3608
points and SOP loses 2.8371 points versus fully trained. The intervals use
10,000 class/product resamples with NumPy PCG64 seed 179019 and the receipt's
paired per-query outcomes. No tested blend
jointly improves SOP holdout and CUB development over their respective
endpoint best scores. The free blend curve is therefore a control for a
future anchored-training arm, not itself a solved quality method. The CUB
development labels were not used to fit or select α; any future selection
needs a preregistered rule and an independent confirmation panel. Because
the B/16 official SOP test remains below the 91.2% published L/14 reference,
an L/14-capacity encoder with a **measured** latency improvement is the next
architecture gate before expensive retraining.

Independent Claude Opus 5.5 and GPT-6 Astra result reviews agreed that the
blend does not justify immediate B/16 anchor retraining. They differed on
whether to prioritize a broader serving-stack panel or a bounded fold probe;
the chosen next gate is a **no-training L/14 fold falsifier with a faithful
same-stack baseline**. The authenticated pretrained UNICOM L/14@336 checkpoint
is available locally, but a reproduced SOP-supervised 91.2% checkpoint is
not. First measure sequential L/14 and a precisely specified 24-to-12
parallel-residual fold on the same GB10, same 336-pixel inputs, precision,
packing, gallery and top-k scorer. Time encoder-only and full image-to-top-k
at batch 1 and 32 on SOP **training** requests; include the existing B/16
and OML S/16 as separately labelled controls. A short paired screen can
reject a slow fold, but any p99 advancement must satisfy the target contract's
10,000 calls per cell, 10 interleaved blocks and paired interval, with
nonregressing p50 and throughput. Only if folding passes the latency gate
should it receive quality-retention and distillation experiments. A folded
pair approximates `x + f₁(x) + f₂(x + f₁(x))` by
`x + f₁(x) + f₂(x)`; retaining weights does not preserve the function.
The proposal is an unmeasured hypothesis with related prior art, not a novel
or superior method claim. Separately, benchmark the faithful local L/14
reference's full pipeline latency before claiming any candidate is faster
than the quality reference; B/16-versus-OML timing cannot supply that number.

### L/14 optimistic fold F0: rejected for this gate

The [raw F0 timing receipt](evidence/compact_metric/l14-parallel-fold-encoder-f0-v1.json)
(SHA-256 `5868a869a074855ff1ca82a4c394a2db04dbe04cad1eba33ea66fae3f2cc09d5`)
used the pinned pretrained UNICOM L/14@336 checkpoint and 32 sampled SOP
**training** images on DGX GB10. The concrete probe is *more aggressive* than
the whole-block parallel expression above: it evaluates both attention and
MLP branches of each pair at the same input, absorbs LayerNorm affines into
fused layers, and uses 12 fused blocks. The weight mapping passes a CPU
parallel-oracle test; on one real pretrained pair, the maximum absolute
output difference from the explicit four-branch CUDA oracle is 0.007324
under fp16 autocast. This validates the intended approximation only within
the recorded numerical tolerance, not equivalence to the sequential model.

| Region, resident preprocessed train tensors | Original L/14 GPU p50 | Optimistic fold GPU p50 | Fold/original p50 ratio |
| --- | ---: | ---: | ---: |
| Full encoder, batch 1 | 28.934 ms | 31.316 ms | 1.0823 |
| Full encoder, batch 32 | 395.306 ms | 282.180 ms | 0.7138 |
| Transformer blocks only, batch 1 | 18.292 ms | 20.633 ms | 1.1280 |
| Transformer blocks only, batch 32 | 360.841 ms | 244.041 ms | 0.6763 |

Each cell has 100 calls in 10 alternated blocks. The median of the 10 paired
block-median **full-encoder batch-1 ratios** is 1.0851, with a descriptive
10,000-resample block-bootstrap 95% interval of 1.0807–1.0884 (NumPy PCG64
seed 179019). The batch-1 direction is therefore a measured regression in
this short screen, despite a batch-32 improvement. The fold fails the
preregistered need to improve both batch sizes; **do not fund fold healing or
training on this implementation**. The original-versus-folded output
embedding cosine averages 0.052 over the 32 images, indicating a large
representation change but measuring neither Recall@1 nor mAP@R. The F0 uses
resident GPU tensors and measures no decode, packing, search or 10,000-call
p99 interval. It cannot establish full image-to-top-k performance or a SOTA
claim. The full local L/14 serving baseline and a route to higher quality
than its published supervised result remain open work.

The [authenticated CUB development receipt](evidence/compact_metric/sop-fullwidth-step48000-cub-development-v1.json)
(SHA-256 `93c2e04a94f618c0e48884ae59182e327d2d7205b6c059160cf5207ac72590d9`)
scores all 5,864 images in CUB-200-2011 **classes 1–100** by self retrieval,
excluding each query itself. It never fits on CUB and never reads test classes
101–200. The selected SOP ArcFace B/16 step-48,000 checkpoint and its SOP
fit-identity PCA projection are pinned by SHA-256, and the evaluator was
committed as `63304672` before the DGX run. All three rows below use the same
images, canonical UNICOM preprocessing, exact deployed signed-int8 scorer,
and the NVIDIA GB10. This is exploratory development evidence, not a new
official-test or state-of-the-art result.

| CUB development representation | Recall@1 | mAP@R | Gallery bytes/item | Image encoding, full split |
| --- | ---: | ---: | ---: | ---: |
| Pretrained UNICOM B/16, 768-D packed | 87.5000% | 0.618651 | 770 | 13.54 s |
| SOP-trained B/16 step 48,000, 768-D packed | 80.1330% | 0.430912 | 770 | 12.78 s |
| Same trained B/16, SOP fit-only PCA-128 packed | 75.3752% | 0.338990 | 130 | 12.78 s plus PCA |

The trained full-width arm loses **7.3670 percentage points Recall@1** and
**0.187739 mAP@R** against the same-width pretrained control. A paired
10,000-resample bootstrap over CUB development classes gives descriptive 95%
intervals of −8.8640 to −5.9569 points for Recall@1 and −0.206211 to
−0.169734 for mAP@R. Thus the transfer loss is already present before
projection; its precise training mechanism is not yet identified. The SOP
fit-only PCA additionally loses **4.7578 points Recall@1** and **0.091922
mAP@R** against trained full width (class-bootstrap intervals −5.8128 to
−3.7313 points and −0.102492 to −0.081336). This projection improved SOP
train-identity holdout quality at equal 130-byte storage, but it does not
generalize to this CUB development split. Packing itself changes the
full-width CUB Recall@1 by zero points versus float, so quantization is not
the source of the large observed drop.

The next diagnostic will compare source-to-trained checkpoint interpolation
on the **SOP train-identity holdout** and this already-used CUB development
half, with identical image bytes and packed scorer. It will test whether an
anchored weight path recovers transfer while retaining SOP fit/holdout gains.
The blend coefficient and any later training recipe must be selected on SOP
train identities; CUB development only tests robustness and cannot become an
untouched confirmation. If interpolation fails, test a transfer-aware
projection or an L/14-capacity encoder under a matched performance protocol.
The completed read-only Fable architecture proposal suggested folding pairs
of L/14 blocks to reduce serial depth, but its quality and latency effects
are unmeasured. A no-training encoder timing gate must precede any costly
distillation or retraining.

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
gates, not a verified 2026 global frontier. The authenticated upstream SOP
scorer normalizes the full 768-dimensional output, keeps the first 512
coordinates without a second normalization, and ranks by Euclidean distance
with self excluded. The local
[`score_symmetric`](../src/sfora/sop_evaluation.py) exposes this distinct
reference path; its compact candidate path still scores the deployed code.
The [pinned upstream README](https://github.com/deepglint/unicom/blob/d71992ed969e6c271436ac0a0ee1f3ca61474ac0/unicom/README.md)
separately reports **74.5% zero-shot** and **91.2% supervised** SOP Recall@1
for L/14@336. The archived pretrained L/14 source scored **74.5099%** on the
60,502-image official test through Sfora's full-width cosine evaluator
([receipt](evidence/compact_metric/sop-compact-selector-official-v1.json));
that numerical match is not a reproduction of either upstream protocol because
the upstream evaluator uses normalized-prefix-512 Euclidean distance. The
released pretrained checkpoint and its archived features cannot serve as the
91.2% supervised reference checkpoint. No SOP-supervised L/14 checkpoint was
identified in the pinned upstream model loader; a faithful local runtime
comparison still requires an authenticated supervised checkpoint or a
documented reproduction of its fine-tuning recipe.
The pinned upstream SOP launch scripts pass `--transform origin_clip`, which
reuses the authenticated model's deterministic resize/center-crop transform
for training. The current Sfora B/16 ArcFace runs instead use timm random
crop, RandAugment and random erasing. That is a real training-recipe difference,
not an explanation proven to cause the observed quality gap. The trainer now
offers an explicit `--reference-transform origin_clip` control; its default
remains `timm`, and the chosen mode and source transform digest are recorded in
new receipts. A future run must compare the two modes on the same SOP train
identities, architecture, projection width, seed, schedule and packed scorer
before attributing any difference to augmentation. It would remain a Sfora
control because the sampler, head and model-selection rule still differ from
upstream UNICOM.
The control will be chosen, if run, by the already-frozen train-identity
holdout packed mAP@R rule, with packed Recall@1 and earlier step as ties. A
cross-arm audit must verify identical seed, fit/holdout rows, schedule, width,
source checkpoint, objective and deployed scorer; only the transform and its
source-code hashes may differ. The existing SOP official test has already
been observed, and evaluating a second transform there would remain
exploratory, not a clean confirmation or a basis for selecting the arm.
The `analyze_sop_reference_progress.py --timm ... --origin-clip ...` control
audit rejects mismatched diagnostic receipts, including changes in the frozen
train split, schedule, seed, width, or scorer source; it does not turn a
single-seed contrast into an algorithmic gain claim.
For the 768-dimensional local control, that path scores its additional trained
linear head; the checkpoint is selected on packed train-holdout mAP@R, not the
upstream scorer. It is not an exact reproduction of UNICOM training. No new
reference-quality measurement follows from adding the scorer. We also
compare with the official OML ViT-S/16 at 224 px, whose 384-dimensional float
descriptor is a strong efficient local control, and with a native packed
exact-search control for the serving component. A published number alone does
not establish a paired speed comparison: we need the checkpoint and a
protocol-matched run on the same machine.

The exact-real prefix-score ranking identity is now proved in
[`Euclidean.lean`](../formal/SforaProofs/Euclidean.lean), but this does not
prove float32 rankings. A synthetic near-duplicate CPU probe changed top-1
between the expanded float32 score and float64 direct distance. To test
whether that precision risk is already a bottleneck in the available real
descriptors, the [source-frozen train-only audit](../scripts/audit_sop_train_prefix_precision.py)
used the authenticated pretrained B/16 archive (SHA-256 `16b4554d…`),
1,024 evenly spaced SOP **training** queries, and all 59,551 training rows.
The float32 expanded scorer, evaluated as individual queries and in a batch,
and the float64 scorer selected the same top-1 ordinal on every query; all
three sampled Recall@1 values were 69.7266%. This sampled negative result
does not bound the error rate on the trained model, the official SOP test, or
near-duplicate cohorts. The [raw receipt](evidence/compact_metric/sop-train-prefix-precision-probe-1024-v1.json)
has SHA-256 `729f8b281b9c174be5815716baaeae7ea5f6dbee0753c0cab27a19983001c954`;
the committed audit script matches its recorded SHA-256
`5f24962fe457593bf22186ce135341cf5e1c90056b943f70ba0e965e8fd35f7a`.
For the next product decision, train-only quality and full image-to-top-k
latency take priority over changing this source-matched scorer.

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

| Quality dataset and split | Model or system | Recall@1 | mAP@R | Gallery bytes/item | Image-to-top-k p99 on separate SOP train-image replay | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SOP official test, 60,502 self-retrieval queries | UNICOM ViT-L/14@336 | 91.2% | — | 768 f32 output before indexing | — | Published [UNICOM Table 4](https://arxiv.org/pdf/2304.05884); scorer audited, checkpoint reproduction pending |
| SOP official test, same protocol | UNICOM ViT-B/16@224, full width | 88.8% | — | 768 f32 output before indexing | — | Published [UNICOM Table 4](https://arxiv.org/pdf/2304.05884); no local fine-tuned checkpoint reproduction |
| SOP official test, same protocol | UNICOM-pretrained ViT-B/16 + ArcFace + TCM, full width | 89.1% | — | 768 f32 output before indexing | — | Published [ICLR 2024 Table 10](https://proceedings.iclr.cc/paper_files/paper/2024/file/16336d94a5ffca8de019087ab7fe403f-Paper-Conference.pdf); same backbone family, different loss and storage budget; no local reproduction |
| SOP official test, same protocol | SEE ViT-S 128-D float | 85.9% | — | 512 as 128 f32 | — | Published [IJCAI 2025 Table 1](https://www.ijcai.org/proceedings/2025/1214.pdf); width-matched recent method, no local checkpoint or paired runtime |
| SOP official 60,502-image test split, author protocol | LaFG ViT, language-guided training | 87.1% | — | Not reported | — | Published [CVPR 2026 Table 3](https://arxiv.org/pdf/2512.06255); ImageNet initialization plus LLM/VLM-derived training targets and A100 training, no matched scorer or runtime; not a 130-byte comparison |
| SOP official test, same queries | UNICOM ViT-B/16@224, pretrained float | 69.9812% | 0.420759 | 768 f32 before indexing | — | Exploratory reproduced pretrained checkpoint; no SOP fine-tuning |
| SOP official test, same queries | Same B/16 with train-only PCA-128, float | 67.2903% | 0.393903 | 128 f32 before indexing | — | Exploratory projection control; [raw per-query result](evidence/compact_metric/unicom-b16-sop-pretrained-screen-v2.json) |
| SOP official test, same queries | Same B/16 with train-only PCA-128 and int8 wire | 67.2325% | 0.393442 | 130 | — | Exploratory, no SOP fine-tuning; same raw result |
| SOP official test, same queries | Same B/16 full-backbone trained, selected packed-rank 128-D | 81.2221% | 0.562186 | 130 | — | Exploratory one-seed result selected on train-identity holdout; [per-query receipt](evidence/compact_metric/sop-full-backbone-packed-rank-official-test-seed179019-1000-v1.json) |
| SOP official test, same queries | B/16@224 full-backbone ArcFace, 48k train-selected, packed 128-D | 86.4715% | 0.651811 | 130 | batch 1: 16.679–17.464 ms; batch 32: 215.513–219.152 ms | Exploratory one-seed quality; diagnostic p99 on separate SOP **training** image/gallery split, 50 calls per arm/order; [quality receipt](evidence/compact_metric/sop-reference-arcface-seed179019-official-test-v1.json), [timing receipt](evidence/compact_metric/sop-trained-b16-vs-oml-paired-image-to-topk-bm1-v2.json) |
| SOP official test, same queries | B/16@224 full-backbone ArcFace, 48k train-selected, packed 768-D | 87.9078% | 0.679219 | 770 | — | Exploratory one-seed quality; no full-pipeline p99 yet; [per-query receipt](evidence/compact_metric/sop-fullwidth768-seed179019-official-test-v1.json) |
| SOP official test, same queries | OML ViT-S/16@224 + Sfora compact profile | 85.9757% | 0.641825 | 130 | batch 1: 7.298–7.462 ms; batch 32: 208.876–210.693 ms | Exploratory packed quality; same paired training-split diagnostic timing; [quality profile](evidence/compact_metric/oml-vits16-sop-packed-profile-verification-v1.json) |
| In-Shop official query/gallery | UNICOM ViT-L/14@336 | 96.7% | — | 768 f32 output before indexing | — | Published [UNICOM Table 4](https://arxiv.org/pdf/2304.05884); evaluator uses normalized prefix-512 Euclidean |
| In-Shop official query/gallery | UNICOM ViT-L/14@336 + Sfora compact profile | 95.4283% | 0.800020 | 130 | — | Exploratory, [local result](compact_metric_selector_result_2026-09-19.md) |
| In-Shop official query/gallery | SEE ViT-S 128-D float | 92.8% | — | 512 as 128 f32 | — | Published [IJCAI 2025 Table 1](https://www.ijcai.org/proceedings/2025/1214.pdf); matched descriptor width, different training and wire |
| In-Shop official 14,218-query/12,612-gallery split | OML ViT-S/16@224, raw 384-D Euclidean | 92.0945% | 0.685148 | 1536 as f32 | — | Exploratory local reproduction of the published 92.1% control; [raw receipt](evidence/compact_metric/oml-vits16-inshop-baseline-v1.json) |

The 24 September 2026 primary-paper spot check adds LaFG as a recent SOP
method comparison: its paper specifies the official 59,551-image train and
60,502-image test split, cosine-distance Recall@K, and 87.1% SOP Recall@1.
It does not report a matched 130-byte wire, end-to-end runtime, or In-Shop
result. This bounded check does not certify a global frontier; the published
UNICOM 91.2% SOP and 96.7% In-Shop values remain the dated quality gates.

The one-time selected B/16 SOP official receipt has SHA-256
`d2e25fc66ce22df71d8dbeeff04330a26b2f55cdc1cf3ebcbaddba6bcae55629`.
Its float descriptor scored 86.4633% Recall@1 and 0.651951 mAP@R, so
signed-byte packing did not cause the large quality shortfall. It binds the
48,000-update checkpoint SHA-256
`a337acf9fff0b789ad305a2241da469202f9448580bb90ef373570c74c615cbd`
and completed one-time official test claim. The OML feature archive's
60,502 test IDs and labels exactly match this receipt's order. Relative to
the 130-byte OML packed control, paired product-clustered 10,000-resample
bootstraps (PCG64 seed 179019; 11,316 test products) give +0.496 percentage
points Recall@1 (95% interval +0.217 to +0.781) and +0.009986 mAP@R
(+0.006721 to +0.013331). This is a cross-encoder system comparison,
not a matched learning-method ablation. The candidate remains below the
published full-width B/16 88.8% reference and L/14 91.2% gate. Its 95.20%
train-identity holdout therefore did not predict official-test quality;
the next run must diagnose representation width and generalization before a
new learning-method claim.

A paired **train-only gallery-size audit** now isolates one source of that
holdout mismatch for the *pretrained*, full-width B/16 encoder. All 5,851
class-disjoint holdout images are the same queries in both arms, with the same
normalized 768-D float32 cosine scorer and self exclusion. The small gallery
contains those 5,851 images; the large gallery adds the 53,700 fit-identity
images as distractors, for 59,551 rows total. Recall@1 falls from **84.0882%
to 69.6633%** (−14.4249 percentage points). Exactly 844 formerly correct
queries become incorrect and none change in the other direction, as required
when only distractors are added. A paired bootstrap over holdout product
identities gives a descriptive 95% interval of **−15.6398 to −13.2496
points**. The [raw per-query receipt](evidence/compact_metric/sop-b16-train-gallery-size-audit-v1.json)
binds the [audit script](../scripts/audit_sop_gallery_size_train_holdout.py) and
the authenticated source feature archive; its SHA-256 is
`5b0bdfdd945c224abd669961f5ff6f0f6352bf841bfb6dc431a48e099e6d63fb`.
The loss is largest among products with fewer positive images:

| Holdout product images | Query rows | Small-gallery Recall@1 | 59,551-row Recall@1 | Hits lost |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 450 | 75.3333% | 54.8889% | 92 |
| 3 | 555 | 80.5405% | 63.7838% | 93 |
| 4–5 | 1,356 | 83.4808% | 69.9115% | 184 |
| 6–12 | 3,490 | 86.0172% | 72.4069% | 475 |

The ratio of large-gallery to small-gallery **miss rates** is 1.83, 1.86,
1.82 and 1.97 respectively across these bands. The larger percentage-point
loss for two-image products is therefore not evidence by itself that training
underweights them: their baseline miss rate is already higher, and each query
has fewer distinct positive gallery images. A class-size weighting arm needs a
matched control and a separate positive-coverage diagnosis.

The 1,775 failed queries in the large-gallery arm chose 1,690 distinct wrong
gallery items; no single item won more than three failed queries. Among the
844 newly lost queries, the median winning-wrong minus best-positive cosine
margin was **0.03993** (25th–75th percentile 0.01762–0.08191). This
pretrained-encoder diagnostic shows widely distributed, often nontrivial
wrong-negative margins. It weakens a simple small-set-of-hubs explanation,
without ruling out a learned per-item scoring method on a different encoder.

This identifies a large
gallery-size effect for one pretrained encoder; it does **not** measure the
effect for the 48k trained encoder or prove the remaining official-test
distribution shift is zero. A size-matched trained-encoder diagnosis remains
necessary before changing its learning objective. The fit-identity distractors
were seen during trained-model learning, so a 59,551-row replay with them is a
labelled diagnostic bracket, not an unbiased unseen-identity checkpoint
selector. An unseen-only holdout curve or a separate class-disjoint pilot is
needed for that selection decision.

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
An additional primary-source check of the
[ICCV 2025 AdvRF paper](https://arxiv.org/pdf/2507.21742) found **84.2%** SOP
Recall@1 in its Table 4 on the official 59,551/60,502-image split. Its
deployed retrieval backbone is ResNet-50, and it does not provide a paired
In-Shop result in that paper. It is a later, useful method comparison but does
not raise the UNICOM SOP quality gate. This is a scoped source audit, not a
certificate that no stronger 2026 result exists.

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
Cold compilation remains a measured startup concern. In a later 59,551-row
batch-32 diagnostic the original 32-row merge generated 909,031 PTX lines;
`ptxas` was still at 100% of one CPU core and about 11 GB RSS after 47 minutes.
That single job was stopped before a result. The merge now uses one query row
per CUDA block while the score stage retains its 32-row MMA tile. Its
4,097-row batch-32 GPU boundary test passed, including exact top-10 ties;
a second test with distinct query rows passed against the scalar reference.
The subsequent paired replay completed every warmup and timed call. Its
full-gallery cold compilation duration was not measured separately.
An additional, uncommitted runtime-bound kernel candidate replaces the
gallery-row and merge-entry const generics with scalar arguments. CuTile 0.1.1
still keys integer scalars by divisibility class (powers of two capped at
16), so this could reduce a per-gallery-size JIT to at most five row-bound
variants per kernel and batch shape, not to one universal binary. It has only
passed a Rust release compile without GPU execution and is excluded from the
paired timing receipt. Exactness, compile reuse, and hot latency must be
checked after the active DGX training run releases the GPU.
The paired replay script had also pinned the rejected RC5 output-initialization
pilot library (`b7440d39…`) rather than the accepted RC4 library. Its input
authority now requires the released `39602d0e…` binary, its `b7c57022…`
Python API, and the immutable
pretrained B/16 quality-screen receipt (`a7c65b7b…`). A focused check binds
those hashes to the release decision and committed evidence. This corrects
the intended benchmark input; it is historical evidence for that earlier
pretrained replay, not the library binary used by the trained paired result.
The benchmark also requires the pinned SOP test-image content manifest
(`28a3ec05…`) and verifies the exact bytes of its first 32 query images before
timing. A CPU-only DGX preflight matched all 32 images against that manifest;
the trainer remained active, so the GPU timing replay was not run concurrently.
The replay now checks that all 32 freshly encoded query features match the
cached gallery source for each encoder (minimum normalized cosine 0.999)
before opening the native gallery or recording timings. This guards against
an encoder checkpoint or preprocessing mismatch; no new GPU timing is claimed.
A second, train-image [paired replay](../scripts/benchmark_sop_trained_image_to_topk_pair.py)
completed on the DGX GB10 with the selected 48k B/16 checkpoint and OML
ViT-S/16. It encoded the same 59,551 SOP training images live, used the first
32 as timed queries and the remaining 59,519 as equal-size 130-byte galleries,
and ran 50 full image-to-top-10 calls per arm, batch, and AB/BA order. It
authenticated both quality receipts, the selected checkpoint and train
holdout, full-gallery/live-query feature parity, and GPU process occupancy.
No SOP test pixels were used for timing. The [raw receipt](evidence/compact_metric/sop-trained-b16-vs-oml-paired-image-to-topk-bm1-v2.json)
has SHA-256 `31c49af66af61563b5beda1a93b89fa48badc635e1cf61a13569ef3724815a27`.

| SOP training-image split, 59,519-row gallery | B/16 p50 in AB/BA order | OML p50 in AB/BA order | B/16 p99 in AB/BA order | OML p99 in AB/BA order |
| --- | ---: | ---: | ---: | ---: |
| Batch 1, full image-to-top-10, ms/call | 12.940 / 12.364 | 6.249 / 5.468 | 17.464 / 16.679 | 7.462 / 7.298 |
| Batch 32, full image-to-top-10, ms/call | 203.569 / 203.990 | 194.708 / 195.605 | 219.152 / 215.513 | 210.693 / 208.876 |

B/16 gallery encoding took 130.520 s versus 96.834 s for OML; packing took
0.020 s versus 0.049 s. At batch 1, B/16's p50 preprocessing was
5.636–6.041 ms versus OML's 1.565–1.976 ms, and encoder/transfer was
6.259–6.394 ms versus 3.165–3.191 ms. Native search was only 0.307–0.348
versus 0.257–0.268 ms at batch 1, and 0.509 versus 0.506–0.508 ms at batch
32. The quality gain of 0.496 percentage points over OML therefore does not
pay for slower full-pipeline serving or encoding in this comparison. This is
a 50-call diagnostic, not a certified p99, a matched-architecture method
ablation, or a SOTA claim. The next performance change must target image
preprocessing/encoding or select a smaller backbone with comparable quality;
further search-kernel tuning cannot close the measured batch-1 gap alone.

These stage measurements do not isolate why the two nominally similar
resize/crop pipelines have different recorded preprocessing times; an
identical-pixel, interleaved component probe is still required before changing
the transform or attributing the difference to the encoder architecture.
Inspection of the authenticated transform sources showed that both use
bicubic resize to 224 pixels, center crop to 224 pixels, and tensor conversion;
UNICOM additionally converts an already-RGB image to RGB inside its transform.
A CPU-only interleaved probe on the same SOP training image, while the DGX
trainer was active, measured 6.532 ms median for that UNICOM sequence and
6.585 ms for the OML sequence over 35 timed calls each. Their tensors were
identical when the same normalization constants were used. This one-image
contended CPU check is diagnostic, not a serving latency measurement. The
paired benchmark's old `decode_preprocess_ns` stage also includes host-to-GPU
transfer. Its source now records `host_decode_preprocess_ns` and
`host_to_device_ns` separately while preserving the aggregate stage. The latter
is wall time around `.cuda(non_blocking=False)`, which includes allocation and
synchronization overhead; it is not a pure device-copy kernel time. The
GPU split measurement is pending an idle-GPU replay.

The CPU packed-search fallback now selects the top-k cutoff before sorting
the retained candidates, while resolving cutoff ties by the lower gallery
ordinal. It also scans at most 64 query rows per gallery block. With the
default 65,536-row gallery block, one f32 score plane is at most 16 MiB
regardless of total query count; top-k output storage still grows with the
number of queries. A paired **synthetic-code CPU diagnostic** on this aarch64 devbox
(59,519 rows × 128 dimensions, top-10, one Torch thread, 10 interleaved
blocks) measured median complete packed-search calls of 7.179 → 2.990 ms at
batch 1, 173.800 → 34.671 ms at batch 32, and 1,328.501 → 250.541 ms at
batch 256 for full-sort control → cutoff
selection. Ordinals and scores matched exactly in both arms. The
[raw receipt](evidence/packed_topk/cpu-selection-synthetic-v1.json) has SHA-256
`b7845aba2d70e8873baf76d732035be82674787e1e39b00cc7aab327a2237b43`.
This changes the CPU fallback only; the vectors are synthetic and the numbers
do not measure an untiled memory baseline, DGX GPU image-to-top-k, or a quality
improvement.

The same selected SOP-trained B/16 checkpoint was also evaluated without any
target-dataset fitting on class-disjoint CUB and Cars identities. These are
transfer diagnostics, not matched published-model reproductions. Both arms
use a 128-D head and the same 130-byte packed scorer; the control uses the
pretrained B/16 and fit-only initial head, while the candidate has the SOP
updated backbone and head. Thus the result detects a joint training effect
but does not yet isolate the head from backbone drift.

| Transfer test self retrieval | Pretrained packed Recall@1 / mAP@R | SOP-trained packed Recall@1 / mAP@R | Difference in Recall@1 | Raw receipt |
| --- | ---: | ---: | ---: | --- |
| CUB-200-2011 classes 101–200, 5,924 queries | 83.8960% / 0.507338 | 73.3288% / 0.333417 | −10.5672 percentage points | [CUB result](evidence/compact_metric/sop-b16-cub-transfer-v1.json), SHA-256 `c96a2a3892506fc0564139f8f94c20ff08f01021296fed2a1ae3673004755812` |
| Cars196 classes 98–195, 8,131 queries | 96.2120% / 0.655341 | 92.6577% / 0.435613 | −3.5543 percentage points | [Cars result](evidence/compact_metric/sop-b16-cars-transfer-v1.json), SHA-256 `f9484ccf7bd77d06a4c1877b1131a4181469fa86f553a35d6d97424eba382c08` |

The next controlled training run keeps the same B/16 initialization, SOP
fit identities, ArcFace recipe, batch schedule and update budget, and changes
the head width from 128 to 768 along with its initialization and proxy
geometry. It will compare train-holdout packed and
float ranking, then run the one-time official SOP test only after a train-only
checkpoint choice. If full width recovers quality or transfer, compression is
a likely bottleneck; if it does not, constrain the backbone update and test
it under the same split and budget. Neither branch is evidence of a new
algorithmic or SOTA gain without the paired independent-seed panel.
The first two full-width checkpoint receipts are now published. These are
interim measurements on the **5,851-image SOP training-identity holdout**,
not on the official test identities. Both rows use the same seed, image
inventory and update count; the full-width arm also changes initialization
and proxy geometry. Packed storage is 770 bytes per gallery item at 768-D
versus 130 bytes at 128-D.

| Updates | 128-D packed Recall@1 / mAP@R | 768-D packed Recall@1 / mAP@R | Full-width receipt |
| ---: | ---: | ---: | --- |
| 4,000 | 90.2239% / 0.696166 | 92.4799% / 0.737175 | [step 4,000](evidence/compact_metric/sop-fullwidth768-seed179019-step4000-holdout-v1.json), SHA-256 `48d86bc5f8633591c9e661399160a598d4531227e894b1df91ceb3443e32e1d8` |
| 8,000 | 92.5996% / 0.746960 | 94.0181% / 0.789077 | [step 8,000](evidence/compact_metric/sop-fullwidth768-seed179019-step8000-holdout-v1.json), SHA-256 `792e3de74acda3be7095b6d37bd3e1ed582e91c4c9f22ffc00bd026851bd628c` |
| 16,000 | 93.8643% / 0.779284 | 95.0094% / 0.813594 | [step 16,000](evidence/compact_metric/sop-fullwidth768-seed179019-step16000-holdout-v1.json), SHA-256 `ae4eedee1360551bdc86be78194d86780b11df20e77d5687b441bf87c21f7e3d` |

The packed Recall@1 gap narrows from 2.2560 to 1.4185 to 1.1451 percentage
points as training proceeds. At step 16,000, the full-width arm wins 95 of the
5,851 identical holdout queries that the compact arm misses and loses 28 that
the compact arm gets right. Resampling training identities 2,000 times with
seed 179019 gives a descriptive 95% percentile interval of +0.7920 to
+1.5256 points for the packed difference. This does not isolate embedding
width because the head initialization, proxy geometry, and backbone updates
also differ. In the same step-16,000 receipts, float-to-packed scoring changes
top-1 label correctness for only 9 full-width queries (6 float wins, 3 packed
wins) and 10 compact queries (3 float wins, 7 packed wins). The float Recall@1
values are 95.0607% and 93.7959%, respectively. Thus the observed packed
quality gap is principally present before byte packing; a different code
format alone has little measured room to close it at this checkpoint. These
counts are paired train-holdout diagnostics, not a general quantization bound.
None of these checkpoints has been selected for the official
test, and these observations do not justify substituting a 770-byte product
for the 130-byte target. A queued train-only probe freezes both step-8,000
backbones, replays their trained heads, and fits three 128-D PCA maps using
fit identities only: on full-width headed features, full-width source features,
and compact-run source features. The same deployed packed scorer then measures
linear compressibility on the shared holdout. This is a matched-step diagnostic,
not a clean backbone-versus-head training intervention: the two backbones
followed different head, proxy, and gradient trajectories. The full-run
train-selected checkpoint and official SOP test remain pending.
The PCA probe is a diagnostic baseline, not a new learning method. The
[ICLR 2025 AE-SVC/(SS)²D paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a2370db7c99791ad5d9f3ef48ad6d464-Paper-Conference.pdf)
already studies PCA against learned compact retrieval embeddings, and
[CVPR 2025 PFML](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
studies a potential-field metric-learning loss on SOP and transfer sets.
Neither supplies protocol-matched 130-byte image-to-top-k latency evidence
for this Sfora candidate. A proposed new loss or projection must identify
its difference from these methods and pass a matched training ablation.
The CUB and Cars transfer evaluators now have a four-arm diagnostic option:
pretrained/trained backbone crossed with initial/trained 128-D projection head,
with both float and 130-byte packed retrieval. This will show whether either
component alone preserves target retrieval and whether their combination has
an interaction. These mismatched pairs are diagnostic functions, not trained
deployable models. Their GPU evaluation is pending the full-width run.
The transfer loss is broad across target identities: the trained packed arm
has lower class-level Recall@1 for 91 of 100 CUB classes and 82 of 98 Cars
classes, computed from the per-query records and class labels in the two raw
receipts. The trained float arm reaches 73.3288% on CUB and 92.7807% on
Cars, so byte packing is not the cause of the large transfer losses. The
full-width trainer is active on the DGX; a watcher tied to its
original PID and start time will run the six-checkpoint train-only selector
and official SOP evaluator once after training and GPU release. Its result is
still pending.

A read-only invention review proposed a smaller OML S/16 **query** encoder
against a frozen B/16 packed **gallery**. Independent Opus and Astra critiques
found this is an asymmetric retrieval system with substantial prior art, not
evidence of a new symmetric SOP similarity method. The existing teacher's
official 0.496-point edge over OML is narrow, its transfer has regressed, and
its gallery construction is slower. A ridge compatibility map would test only
linear fit, so either outcome is insufficient to fund full-backbone training.
The OML checkpoint is externally trained on SOP, so this run's train-identity
holdout is not a clean unseen-identity validation set for any OML-initialized
student unless checkpoint provenance proves otherwise. No holdout codes may
enter a fit-gallery training loss, even as negatives. The idea is shelved as a
SOTA route with the current teacher. A future bounded product pilot would
need a stronger 128-D teacher, fit-only gallery, self masking, a matched
compatibility-loss control, actual packed-forward parity, and separate
asymmetric protocol reporting. A better query function could in principle
beat its teacher's symmetric ranking, so teacher quality is not a formal
ceiling; that outcome is presently unmeasured.

An optional `--certify-p99` mode now records 20 interleaved AB/BA blocks with
500 timed full image-to-top-10 calls per arm at each of batch 1 and 32,
retains every stage sample, checks result stability across blocks, and records
GPU process occupancy, clocks, power, and utilization at block boundaries.
It bootstraps ten consecutive AB/BA superblocks to preserve the order balance
when estimating the trained B/16 to OML p99 ratio. The fixed diagnostic gate
requires its upper 95% bound below 1, a one-sided paired-superblock sign-test
`p < 0.05`, and nonregressing pooled p50 and mean latency at both batch sizes.
Its interval is conditional
on exchangeable superblocks; boundary telemetry cannot prove continuous GPU
exclusivity. The receipt remains diagnostic and claim-ineligible. This mode
has only CPU-level scheduler and statistical tests; the completed 50-call GPU
diagnostic above did not run this certification mode. No p99 interval or
full-pipeline speed win has been established.

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
At 16,000 and 32,000 updates, packed holdout Recall@1 was 93.8643% and
94.9410%, while mAP@R was 0.779284 and 0.805460. The same 5,851 holdout
images in 1,132 products appear in both receipts. A paired, product-clustered
10,000-resample bootstrap (PCG64 seed 179019, resampling the 1,132 products)
of the 32,000-minus-16,000 per-query scores gives
+1.077 percentage points Recall@1 (95% percentile interval +0.718 to +1.447)
and +0.026177 mAP@R (+0.022297 to +0.029857). This is a train-identity
diagnostic, not an official-test or independent-seed result. Its continued
improvement supports completing the frozen schedule rather than selecting a
checkpoint from the previously observed official test. The DGX
receipts are
`/home/riomus/runs/sfora-sop-reference-b8f85611-179019/arcface-seed179019-53760.step16000.json`
(SHA-256 `bad3fab826a5b0d317054b001f623bca26b602eba5871c140000577698f01af5`)
and `arcface-seed179019-53760.step32000.json`
(SHA-256 `13eb5df7c49686c6ff28cdbedf3323b0973c729be19f89ba22376b3999a85cd4`).
The 48,000-update receipt records 95.1974% packed holdout Recall@1 and
0.812453 packed mAP@R. Relative to 32,000 updates on exactly the same holdout
rows, paired product-clustered 10,000-resample intervals (PCG64 seed 179019)
give +0.256 percentage
points Recall@1 (95% interval +0.034 to +0.474) and +0.006993 mAP@R
(+0.004920 to +0.009048). The source, split, schedule, and checkpoint
provenance fields match the 32,000-update receipt. The 48,000-update receipt is
`/home/riomus/runs/sfora-sop-reference-b8f85611-179019/arcface-seed179019-53760.step48000.json`
(SHA-256 `42350440edf3dcb972c6d2a9372a1adc75e6363319efc31a047815e81cea29c0`).
The completed 53,760-update checkpoint reached 95.3341% packed holdout
Recall@1 and 0.811912 packed mAP@R, versus 95.1974% and 0.812453 at
48,000 updates. Thus the frozen train-only mAP@R rule selects the
48,000-update checkpoint; the later Recall@1 increase does not override the
primary selection metric. The final [raw training receipt](evidence/compact_metric/sop-reference-arcface-seed179019-final-v1.json)
has SHA-256 `a020cfc246df0423b617cdba45acbf5001174c44d1cda888d063fe2822dce3fb`
and binds the final checkpoint SHA-256
`5c8a68975f959df89da45324bda6877627499807dd4f64d6fea98ba14ac9f85d`.
It records 23,543.867 seconds of training, 2.2834 updates per second,
7,804,527,104 peak CUDA allocated bytes, and 6,613,139,456 peak host RSS
bytes on one NVIDIA GB10. The receipt also records 91.745 seconds of
checkpoint diagnostic overhead separately. This is one seed and a
train-identity holdout; the one-time official evaluation is running and these
holdout scores cannot establish the published quality gate.
On identical holdout queries, the final-minus-48,000 paired product-clustered
10,000-resample bootstrap (PCG64 seed 179019) gives +0.137 percentage points
packed Recall@1 (95% interval +0.034 to +0.260) and −0.000541 packed mAP@R
(−0.001625 to +0.000575). The mAP@R interval crosses zero; the frozen
selection rule still chooses the 48,000-update checkpoint by its observed
holdout mAP@R.

A passive 60-sample, 1 Hz DGX trace spanning the diagnostic had median SM
utilization 96% and median GPU power 51 W; five samples had SM utilization
below 90%, including four at zero. Its raw path is
`/home/riomus/runs/sfora-sop-reference-b8f85611-179019/arcface-seed179019-53760.gpu-dmon-60s.txt`
(SHA-256 `8193209b4911d691e9b6d173e097e121187eacfdbca722b5172a7b587dbaff0f`).
This short, diagnostic-spanning trace is not a training-only throughput or
latency profile.

Because the balanced sampler differs from upstream's shuffled image sampler,
the compact head is new, only 90% of SOP train identities are fitted, and
gradient clipping at norm 1.0 is additional, this is **reference-like**, not
a faithful published UNICOM reproduction. The upstream B/16 script's
single-GPU batch 64 avoids a global-batch mismatch, but its classifier is
full-width and uses `num_feat=512` in PartialFC.

For a later 768-D width control, the holdout receipt now records a separate
`upstream_prefix512_euclidean` metric using UNICOM's full-vector normalization,
first-512 truncation, and Euclidean ranking. Its checkpoint rule remains the
same packed mAP@R rule as the 128-D control. Applying the upstream scorer to
our trained affine head does not make this width control a faithful UNICOM
training reproduction.

The completed 4,000-update constant-rate screen is a separate budget
diagnostic using the original recipe. Its measured gain must be interpreted
separately from the completed longer reference-like control.

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

At 8,000 updates, the same run's train-identity holdout improved to packed
Recall@1 **92.5996%** and mAP@R **0.746960** (float 92.5825% and 0.747406)
across all 5,851 holdout queries. The [raw per-query receipt](evidence/compact_metric/sop-reference-arcface-seed179019-step8000-v1.json)
has SHA-256
`845a568319701281737cac94dd367fef50d2996d7cc05f7537a3dca5f88baaad`,
independently matched against the remote file. The trainer records checkpoint
SHA-256 `35d9b2416cb78680cc4df967af9c7465cc55ad091e744867873aa6a11b1dfe00`.
The per-query AP averages independently reproduce both reported mAP@R values.
This is still a diagnostic from training classes; no official-test result or
method improvement follows from it.
The [paired progress receipt](evidence/compact_metric/sop-reference-holdout-progress-step4000-to8000-v1.json)
compares identical queries and resamples all queries within each of the 1,132
held-out product identities together (10,000 draws, seed 179019). From 4,000
to 8,000 updates, packed mAP@R rose **+0.050794** (product-bootstrap 95%
interval **+0.045802 to +0.056082**) and Recall@1 rose **+2.376 percentage
points** (**+1.900 to +2.864**). Of 5,851 queries, 167 changed from miss to
hit and 28 from hit to miss. This within-run interval does not include
training-seed variation or predict the official-test result. The raw progress
receipt has SHA-256
`5bfc147f8cec80d2836165bbd1ac9dc5da5888a2f22f990afc7bee544fee6905`.

At 16,000 updates, the same fixed train-identity holdout reaches packed
Recall@1 **93.8643%** and mAP@R **0.779284** (float 93.7959% and 0.779206).
The [raw per-query receipt](evidence/compact_metric/sop-reference-arcface-seed179019-step16000-v1.json)
has SHA-256 `bad3fab826a5b0d317054b001f623bca26b602eba5871c140000577698f01af5`,
matching the original DGX file; its checkpoint digest is
`c30e4df3fbb97a2ae30e419ee938babdb085dee0a7cbcf177801655e4da847df`.
The [paired 8,000-to-16,000 progress receipt](evidence/compact_metric/sop-reference-holdout-progress-step8000-to16000-seed179019-v1.json)
uses 10,000 product-identity bootstrap draws. Packed mAP@R rose **+0.032323**
(95% interval **+0.028240 to +0.036599**) and Recall@1 rose **+1.265
percentage points** (**+0.864 to +1.662**); 108 queries changed from miss
to hit and 34 from hit to miss. The progress receipt SHA-256 is
`320f53e6c5a9d35b9f53b0a7c2ce924961123e5dd3207d73ccd1d2ebdd084c26`.
These are within-run train-holdout diagnostics on one seed. The original
53,760-update trainer later completed; this intermediate checkpoint alone
did not establish an official-test or latency result.

At 32,000 updates, packed Recall@1 reaches **94.9410%** and mAP@R
**0.805460** on the same 5,851-query train-identity holdout; float reaches
94.9923% and 0.805731. The [raw checkpoint receipt](evidence/compact_metric/sop-reference-arcface-seed179019-step32000-v1.json)
has SHA-256 `13eb5df7c49686c6ff28cdbedf3323b0973c729be19f89ba22376b3999a85cd4`;
the original DGX checkpoint was independently hashed to
`0ca583b11f11b45c5f716f4b870bc0273d39ee4419021a88de407517ad48ca18`,
as recorded by the trainer. The [paired 16,000-to-32,000 receipt](evidence/compact_metric/sop-reference-holdout-progress-step16000-to32000-seed179019-v1.json)
has SHA-256 `9ad3ebcaea377aab4a07bbdc525bdebe4cd62e5227a59184fb120b6cae0ebbd2`.
Across the same queries, packed Recall@1 rose **+1.077 percentage points**
(product-bootstrap 95% interval **+0.718 to +1.447**) and mAP@R rose
**+0.026177** (**+0.022409 to +0.029965**). There were 87 new Recall@1
hits and 24 lost hits. At this checkpoint, packed minus float Recall@1 is
−0.051 percentage points (product-bootstrap interval −0.134 to +0.017)
and packed minus float mAP@R is −0.000271 (−0.001005 to +0.000409).
Thus the measured train-holdout improvement from further backbone training
is much larger than the current packing difference. These are one-seed,
train-only diagnostics; they do not predict the official-test score or prove
that a rank loss cannot help.

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

The CUB transfer evaluator completed an exploratory class-disjoint check
after SOP selection. It compares the authenticated
pretrained UNICOM ViT-B/16@224 with the SOP-fit-only PCA 128 head against
the train-selected SOP-finetuned ViT-B/16@224 and trained 128 head. Both use
the same CUB-200-2011 classes 101–200 test self-retrieval protocol (5,924
images), 128-D float and 130-byte packed scoring, with no CUB fitting. The
script binds the selected checkpoint to the official SOP receipt, reconstructs
and hashes the fit-only initial head, checks every CUB image against the pinned
source tar, and writes per-query Recall@1 and mAP@R. The measured packed
Recall@1 was 83.8960% pretrained versus 73.3288% SOP-trained; the raw
receipt and mAP@R are reported above. It cannot establish an SOP or In-Shop
SOTA claim.

The Cars196 transfer evaluator is also prepared for the SOP-selected model.
It uses the pinned `tanganke/stanford_cars` revision
`9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40` already present in the DGX
cache. Four Arrow shard SHA-256 values and the earlier Cars feature archive
bind the image bytes and evaluation order. Classes 98–195 across both original
image partitions give 8,131 evaluation images; their label order exactly
matches the earlier authenticated archive. It compares the same pretrained
PCA-128 and SOP-finetuned 128-D arms in float and 130-byte packed form, with
no Cars fitting. The measured packed Recall@1 was 96.2120% pretrained versus
92.6577% SOP-trained; the raw receipt and mAP@R are reported above. Arrow
here is the external Cars dataset cache format, not a Sfora library or index
dependency.

A matched full-width B/16 control is prepared with the same 53,700 fit images,
class-disjoint holdout, pretrained checkpoint, seed, batch schedule, reference
augmentation, ArcFace margin/scale, OneCycle schedule, and 53,760 updates. Its
trainable 768-to-768 head starts as the identity, so its initial float
descriptor matches the pretrained 768-dimensional source; class proxies are
imprinted only from fit identities. The runner records both 768-D float and
770-byte packed holdout results. Paired comparison with the 128-D run can
attribute a quality difference to embedding width and its associated head
initialization, but the two systems occupy different gallery-storage and
search-cost points. The full-width run started after the paired serving and
transfer checks finished; its result is pending. No second training copy is
running.
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

## Frozen-encoder negative coverage diagnostic, 24 September

On 512 deterministic anchors from the SOP **train fit identities**, the
authenticated pretrained UNICOM B/16@224 768-D cosine features expose a
specific learning opportunity. A 53,700-image fit bank has a wrong-product
nearest neighbor above the anchor's best same-product image for **28.125%** of
anchors. A random 16-product, four-image-per-product proxy batch (15 other
products) exposes such a negative for **2.5391%**. The full-bank nearest
wrong-product cosine exceeds the sampled-batch nearest wrong-product cosine
by a median **0.36554**. These are frozen-feature, train-fit-only diagnostic
measurements; they are neither trained-model quality gains nor official-test
results. The proxy random class draw is not a replay of the trainer's actual
batch schedule or augmentation. The [reproducible receipt](evidence/compact_metric/sop-b16-pretrained-negative-coverage-probe-v1.json)
binds the archive and script hashes.

A matched CPU head screen then compared the strongest wrong-product negative
from each actual scheduled 64-image batch against the strongest from the
53,700-image fit bank. Both arms used the same frozen pretrained B/16 encoder,
identity-initialized trainable 768-D linear head, source-feature hardest
positive, softplus cosine-triplet loss, Adam at 1e-4, 256 batches, seed
179019, 770-byte packed scorer, and 5,851-image disjoint-product SOP train
holdout. This was an exploratory method screen, not independent-seed or
full-backbone evidence.

| SOP train holdout, 5,851 queries | Packed Recall@1 | Packed mAP@R | Head-training time, local CPU |
| --- | ---: | ---: | ---: |
| Frozen identity head | 84.0540% | 0.591906 | 0 s |
| In-batch hardest negative | 85.3358% | 0.609979 | 2.27 s |
| Full-fit-bank hardest negative | 83.2849% | 0.577061 | 2.29 s |

The bank arm lost **2.0509 percentage points** Recall@1 relative to the
in-batch arm, with product-bootstrap 95% interval **[-2.6069, -1.4975] pp**;
its mAP@R fell **0.032918**, interval **[-0.037340, -0.028432]**. There were
185 in-batch-only hits and 65 bank-only hits. Mining took 14.85 s before
either arm trained. The [raw per-query receipt](evidence/compact_metric/sop-b16-bank-negative-head-screen-v1.json)
and [compact summary](evidence/compact_metric/sop-b16-bank-negative-head-screen-summary-v1.json)
retain the source, schedule, code, timing, and paired evidence. The naive
static, single-strongest-bank-negative formulation is rejected for the next
full-backbone run; the coverage gap alone did not predict a quality gain. This
screen kept pretrained mining indexes fixed and propagated gradients through
both sides of each selected pair. It does not test a refreshed or detached
memory bank, and one training schedule does not measure variation across
training seeds. Opus 5.5 and GPT-6 Astra independently found no
comparison-breaking code flaw but agreed that the result cannot reject bank
methods as a class. The decision now prioritizes the already-queued
full-width-to-PCA-128 matched 130-byte control, which requires no new backbone
run, followed by the separately staged trained-checkpoint gallery diagnosis.
The latter's seen-fit distractors remain diagnostic only. The original
full-width DGX trainer and downstream watchers remain active, so this screen
used local CPU and did not contend for their GPU.

## Matched 32,000-update width checkpoint, 24 September

The original full-width B/16 trainer published its step-32,000 diagnostic
checkpoint while continuing the same 53,760-update run. Its checkpoint file
SHA-256 `a6ee8dff640ac79c45b009414c1b3e289574c4d2d237cc9e90c64fa56f2c99c0`
matches the receipt. The compact and full-width receipts bind identical SOP
official-**train** holdout image IDs and labels, 5,851 queries/gallery items,
split hashes, source checkpoint, schedule, seed 179019, ArcFace reference
recipe, and update count. Their heads, class proxies, gradient response, and
packed storage widths differ, so this is a trained-system width comparison,
not an isolated dimension-only effect.

| Step 32,000 SOP train holdout | Packed bytes/item | Packed Recall@1 | Packed mAP@R |
| --- | ---: | ---: | ---: |
| Direct compact 128-D | 130 | 94.9410% | 0.805460 |
| Full-width 768-D | 770 | 95.8469% | 0.831014 |

Full width gained **0.9058 pp** packed Recall@1, product-bootstrap 95%
interval **[0.5318, 1.2841] pp**, and **0.025554** mAP@R, interval
**[0.021424, 0.029679]**. It won 85 queries the compact arm missed and lost
32 the compact arm hit. The [full-width raw receipt](evidence/compact_metric/sop-fullwidth768-seed179019-step32000-holdout-v1.json)
and [reproducible paired summary](evidence/compact_metric/sop-compact-vs-fullwidth-seed179019-step32000-paired-v1.json)
record the hashes and per-query evidence. The original trainer and downstream
jobs remain live. The already-queued fit-only PCA-128 probe is pinned to the
matched **step-8,000** checkpoints and will answer an early-stage
compressibility question. It cannot establish whether the step-32,000 gain
survives 130-byte deployment. That requires fitting PCA on the later
full-width checkpoint's fit-identity features and comparing the resulting
holdout descriptor with a matched compact checkpoint. The staged trained
feature exporter can supply those features after the existing GPU queue
finishes. Official SOP test results are still pending.

The separate [matched step-48,000 PCA audit](../scripts/audit_sop_matched_fullwidth_pca.py)
is prepared for that later check. It pins the existing [compact diagnostic
receipt](evidence/compact_metric/sop-reference-arcface-seed179019-step48000-v1.json),
requires the full-width export to reproduce its original holdout scores,
fits PCA on fit identities only, and records a usable projection with its
normalization rules and input hashes. It has passed focused lint, type check,
CLI import, and paired-score checks against the actual step-32,000 receipts;
it has **not** run on step-48,000 trained features. The compact 48,000-update
checkpoint was selected using this same holdout, so the eventual matched-step
comparison remains exploratory and will disclose that selection.

## Matched 48,000-update width checkpoint, 24 September

The original full-width trainer published its step-48,000 diagnostic while
continuing the same 53,760-update run. The remote checkpoint hashes to
`232f7cee93e39fa242d8461f8dc8cee684d7228eef01f8fe9399b9782e80a1b2`,
matching the receipt; the copied receipt matches the remote SHA-256
`b0b857e02a26fe27560aa9721912db99f6e54f60ff6c43ffbead16403164010b`.
The [raw full-width receipt](evidence/compact_metric/sop-fullwidth768-seed179019-step48000-holdout-v1.json)
and [paired width audit](evidence/compact_metric/sop-compact-vs-fullwidth-seed179019-step48000-paired-v1.json)
bind the same official SOP **train** class-disjoint holdout: 5,851 query/gallery
images from 1,132 products, seed 179019, matched image rows, source checkpoint,
schedule, ArcFace recipe and update count.

| Step 48,000 SOP train holdout | Packed bytes/item | Packed Recall@1 | Packed mAP@R |
| --- | ---: | ---: | ---: |
| Direct compact 128-D | 130 | 95.1974% | 0.812453 |
| Full-width 768-D | 770 | 96.0349% | 0.836934 |

The full-width packed descriptor gains **0.8375 pp** Recall@1, paired
product-bootstrap 95% interval **[0.4900, 1.1909] pp**, and **0.024481**
mAP@R, interval **[0.020585, 0.028321]**. It recovers 77 queries the
compact arm misses and loses 28 compact hits. Compared with its own step-32,000
checkpoint, full width gains **0.1880 pp** packed Recall@1 and **0.005920**
mAP@R; the [within-run progress audit](evidence/compact_metric/sop-fullwidth-reference-holdout-progress-step32000-to48000-seed179019-v1.json)
records paired intervals. This remains a one-seed, train-holdout, unequal
storage comparison. The selected compact checkpoint was chosen on this
holdout. The fit-only PCA-128 audit on the staged train-feature export must
show whether the full-width advantage survives at 130 bytes/item. The final
trainer, official SOP evaluation, transfer checks and export queue are still
active; no official-test result is inferred from this holdout.

## Full-width training closeout, 24 September

The original 53,760-update full-width run completed on the NVIDIA GB10. Its
[final receipt](evidence/compact_metric/sop-fullwidth768-seed179019-final-v1.json)
matches the remote SHA-256
`73408b5ceddace5f23cca4bbf071ec1a8df3e72db0cc9b20a7e7ae9ae2e1404f`;
the final checkpoint independently hashes to the receipt's
`dc5b94a02176b6a8437466f7e1d7fc8dfa5136318fc1f0e95d3f15242a795c66`.
It trained seed 179019 on 53,700 SOP train-fit images from 10,186 product
identities, batch 64 with four images per identity. Training time excluding
diagnostics was **24,095.94 s (6.693 h)**, or **2.231 updates/s**, with
step-time p50 **0.4438 s** and p95 **0.4475 s**. Peak allocated CUDA memory
was **7.916 GB** and peak host RSS **6.453 GB**. Diagnostic overhead added
100.90 s and the final validation took 20.28 s. The earlier compact run's
corresponding training receipt records 23,543.87 s, 2.283 updates/s, and
7.805 GB peak allocated CUDA memory; these are separate runs with differences
in their recorded trainer, evaluation, and loss-module source hashes, so the
2.3% throughput difference is descriptive rather than an isolated width cost.

The final full-width packed train-holdout score is **96.0520% Recall@1** and
**0.836346 mAP@R**. The frozen selection rule maximizes packed mAP@R, with
Recall@1 and earlier step only as ties. It therefore selects the already
verified step-48,000 checkpoint at **0.836934 mAP@R**; the final step's
0.0171-point Recall@1 increase does not override that rule. The official SOP
evaluator subsequently completed from its original immutable source snapshot;
its result is recorded below.

The independent local CPU serving check used the authenticated 59,551-image
SOP **train** feature archive as one gallery, with normalized 128-coordinate
prefix and 768-coordinate pretrained descriptors. The 128-D prefix is a
width-only timing control, not a trained compact quality candidate. Both arms
used the same exact packed scorer, gallery order, k=10, four CPU threads on a
four-vCPU Neoverse-V2, 25 warmups and 200 timed calls per cell. A stable full
sort agreed with the production selector on four sampled queries at each
width.

| CPU packed search | Gallery bytes | Batch-1 p50 / p95 | Batch-32 p50 / p95 | Pack time |
| --- | ---: | ---: | ---: | ---: |
| 128-D timing control | 7,741,630 | 1.214 / 2.383 ms | 22.727 / 32.107 ms | 0.066 s |
| 768-D pretrained | 45,854,270 | 4.656 / 33.245 ms | 48.632 / 59.558 ms | 0.978 s |

The [raw 200-call receipt](evidence/compact_metric/sop-pretrained-cpu-packed-width-cost-v1.json)
records every duration, source and script hashes, RSS, and hardware. The wider
descriptor costs 5.92 times the persistent gallery bytes and, in this CPU
check, 3.84 times the batch-1 median search latency. The p95 variation and
200-call p99 are diagnostic; neither arm includes image encoding, and this
does not replace the pending DGX full-pipeline and CuTile measurements.

## SOP fit-label conflict census, 24 September

The [train-only census](evidence/compact_metric/sop-pretrained-b16-fit-cross-label-census-v1.json)
uses the authenticated pretrained B/16 feature archive (SHA-256
`16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f`),
L2-normalizes its 768-D embeddings, and scores each of the 53,700 fit images
against all fit images from a *different* labeled product. It excludes the
5,851 train-identity holdout and all official-test features. The exact CPU
scan took 41.46 s with four threads and 1.39 GB peak RSS. Thresholds were
chosen after a 2,048-query screen, so these are descriptive counts, not a
confirmatory test.

| Nearest cross-label cosine | Fit queries | Fraction of 53,700 |
| --- | ---: | ---: |
| At least 0.90 | 2,218 | 4.13% |
| At least 0.95 | 280 | 0.52% |
| At least 0.98 | 33 | 0.061% |

The closest examples include a chair/sofa pair with identical image bytes
(SHA-256 `7e6e498cad0fc1f65b1156d524e21ee20fc07b15106efd038c48886b7c060c36`)
and a bicycle/lamp pair with identical image bytes
(`a567462f4edd496bdf5cd00da5bbde64131c283e3cf396bfd58c0fac26b13d9a`).
Different SOP product IDs can therefore impose contradictory class negatives
on visually identical images. Other high-cosine pairs can be legitimate
lookalikes. The measured high-similarity incidence alone does not establish
that label conflicts caused the CUB/Cars transfer decline or that merging
labels improves SOP retrieval. A conflict-aware objective needs a matched
train-only ablation and transfer check before promotion; the current evidence
does not justify launching that GPU experiment ahead of the queued width/PCA
and official-quality results.

## Full-width official SOP and compact-backbone transfer factorial, 24 September

The step-48,000 full-width checkpoint was selected on the class-disjoint SOP
training holdout under the frozen packed mAP@R rule before its one-time
official-test evaluation. The [official receipt](evidence/compact_metric/sop-fullwidth768-seed179019-official-test-v1.json)
has SHA-256
`ff25925f85232087fa753bb75130f71f7f376aafe602e37fb771d0b08f7643f3`,
matching the DGX original. Its remote claim file matches the receipt's SHA-256
`a0662b9b629ea26bae21388b59c6d6ebcbe582174c1e5da4fbf32ad4a54f232e`.
The selected checkpoint SHA-256 is
`232f7cee93e39fa242d8461f8dc8cee684d7228eef01f8fe9399b9782e80a1b2`.
All 60,502 official SOP test images from 11,316 unseen products were used as
queries against that test gallery with self-exclusion.

| SOP official test system | Packed Recall@1 | Packed mAP@R | Bytes/gallery item |
| --- | ---: | ---: | ---: |
| Trained B/16, selected step 48,000, 768-D | 87.9078% | 0.679219 | 770 |
| Trained B/16, selected step 48,000, 128-D | 86.4715% | 0.651811 | 130 |

The two receipts have identical ordered test images, labels, manifest, and
evaluator source. The full-width trained system gains **1.4363 percentage
points Recall@1** and **0.027407 mAP@R** at **5.923 times the gallery
storage**. A paired bootstrap resampling all 11,316 product clusters 5,000
times with seed 179019 gives a descriptive 95% interval of **+1.2794 to
+1.5976 percentage points** for Recall@1 and **+0.025996 to +0.028824** for
mAP@R. This compares two trained heads and checkpoints, not a dimension-only
intervention. It is one seed with already observed official test results, so
the interval does not establish variation across training seeds.

The new result remains **0.8922 percentage points below** the published
UNICOM B/16 SOP 88.8% reference and **3.2922 points below** the L/14 91.2%
reference. Those publications did not use this 770-byte packed scorer; a
source-compatible prefix-512 float comparator is still needed. The new
receipt's 132.220 s encode and 5.074 s packed-score totals cover one offline
test pass. They are not batch-1 serving latency or certified p99. The
full-pipeline and equal-byte PCA checks remain queued.

The completed four-arm [CUB transfer factorial](evidence/compact_metric/sop-compact128-seed179019-cub-transfer-factorial-v2.json)
and [Cars transfer factorial](evidence/compact_metric/sop-compact128-seed179019-cars-transfer-factorial-v2.json)
retain per-query metrics, source hashes, and class-disjoint test protocols.
They use the selected **compact 128-D** checkpoint, not the full-width
checkpoint. They cross pretrained versus SOP-trained compact backbones with a fixed fit-only
initial 128-D head versus the trained 128-D head, without fitting on CUB or
Cars. The receipts match their DGX originals at SHA-256
`f020215194d87ba3afc5ae3169649fc7739f8f3df2a2f1df5023cbda4a06336c`
and `a897dc61acffc5720a7549b2b1806373bfcfe1df2557b33667df929b90cfc033`.

| Transfer test and split | Pretrained backbone + initial head | Pretrained backbone + trained head | SOP-trained backbone + initial head | SOP-trained backbone + trained head |
| --- | ---: | ---: | ---: | ---: |
| CUB-200-2011 classes 101–200, 5,924 self queries: packed Recall@1 / mAP@R | 83.8960% / 0.507338 | 84.2167% / 0.508154 | 73.7002% / 0.341901 | 73.3288% / 0.333417 |
| Cars196 classes 98–195, 8,131 self queries: packed Recall@1 / mAP@R | 96.2120% / 0.655341 | 95.9907% / 0.645332 | 93.4448% / 0.483354 | 92.6577% / 0.435613 |

Holding the initial head fixed, SOP backbone training reduces CUB Recall@1
by **10.1958 points** and Cars by **2.7672 points**. Holding the pretrained
backbone fixed, swapping in the trained head changes CUB by **+0.3207** and
Cars by **−0.2214 points**. Backbone updates are thus the dominant measured
source of transfer loss **for the compact run**. Full-width-backbone transfer
has not yet been measured. This factorial does not isolate the mechanism within
backbone training. The next controlled experiment should retain pretrained
backbone geometry more strongly, use source-matched `origin_clip`
augmentation as a separate arm, select on SOP train identities plus transfer
checks, and assess equal-byte PCA before another full run.

## Equal-byte step-8,000 PCA diagnostic, 24 September

The already queued [PCA diagnostic](evidence/compact_metric/sop-matched-step8000-pca128-v2.json)
completed on the original 5,851-image, 1,132-product SOP **training**
identity holdout. The local receipt's SHA-256
`d4b332df82025eaba2ab5ffbdfb3590ca5a241455c4f529385583fcf948e537b`
matches the DGX original. Its three 768-to-128 PCA projections were fit only
on the 53,700 training-fit images. Source checkpoints, feature arrays,
projection arrays, holdout order, and exact replay of both source packed
scores are hashed in the receipt. The comparison is diagnostic at the common
step 8,000; it is not the train-selected product checkpoint or an official-test
result.

| SOP train-identity holdout, step 8,000 | Packed Recall@1 | Packed mAP@R | Bytes/item |
| --- | ---: | ---: | ---: |
| Trained 128-D compact head | 92.5996% | 0.746960 | 130 |
| Trained 768-D head, no projection | 94.0181% | 0.789077 | 770 |
| Trained 768-D head, fit-only PCA-128 | 93.8301% | 0.785465 | 130 |
| Trained 768-D backbone source feature, fit-only PCA-128 | 93.2319% | 0.767266 | 130 |

The equal-byte full-width-head PCA arm exceeds the compact trained head by
**1.2306 percentage points Recall@1** and **0.038505 mAP@R**. Resampling
the 1,132 products 5,000 times (seed 179019) gives descriptive paired 95%
intervals of **+0.7766 to +1.6670 percentage points** and **+0.033163 to
+0.043750 mAP@R**. The full-width-head PCA arm retains most of its
unprojected holdout quality at this early step. Different head/proxy geometry
also changed backbone training, so this comparison does not isolate a pure
width effect or identify a superior final model. The selected step-48,000
equal-byte check remains necessary; its feature export started after the
replacement image-to-top-k stage split completed.

## Paired image-to-top-k stage attribution, 24 September

The original queued stage-split watcher failed before timing: its launcher
omitted `CUTILE_TILEIRAS_PATH` and the CuTile JIT tried to execute a
`tileiras` command absent from `PATH`. The compiler binary existed and the
separate exactness watcher had passed with its path pinned. The original
watcher and dependent exporter both reached terminal `exit=1` with no
benchmark output. A distinct v3 watcher pinned the compiler path and completed
once; the dependent export was safely replaced by a distinct v2 watcher. No
overlapping benchmark or feature export was started.

The [v3 raw stage receipt](evidence/compact_metric/sop-compact128-b16-vs-oml-stage-split-v3.json)
matches its DGX original at SHA-256
`557728bbc20610a7d8ef155fc25268f8c9708f893c258c2d207488b704d6df16`.
It compares the selected SOP-trained compact B/16 with the OML ViT-S/16 on
the same SOP **training** query images and 59,519-row gallery, in AB/BA
order, 50 calls per arm and batch shape. It is a diagnostic timing result, not
a certified p99 or a matched-architecture quality comparison.

| Batch | System | Image-to-top-k p50 | Host decode/preprocess p50 | Encoder/transfer p50 | Native search p50 | Image-to-top-k p99 diagnostic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | SOP-trained B/16, 130 bytes/item | 12.124–12.263 ms | 5.335–5.514 ms | 6.265–6.268 ms | 0.294–0.295 ms | 16.819–17.306 ms |
| 1 | OML S/16, 130 bytes/item | 5.918–5.926 ms | 1.704–1.810 ms | 3.207–3.222 ms | 0.262–0.265 ms | 7.625–12.969 ms |
| 32 | SOP-trained B/16, 130 bytes/item | 199.440–204.133 ms | 147.871–151.381 ms | 48.484–48.516 ms | 0.498–0.504 ms | 218.395–218.619 ms |
| 32 | OML S/16, 130 bytes/item | 192.751–196.264 ms | 138.765–142.900 ms | 47.164–47.226 ms | 0.507–0.512 ms | 211.640–215.023 ms |

At batch one, both host image preprocessing and encoder execution account for
the B/16 gap; packed search does not. At batch 32, preprocessing accounts for
most of the remaining median gap. A kernel-only speedup cannot produce a joint
image-to-result win here. The next serving change must target these measured
stages while preserving descriptor quality and the exact preprocessing
contract. The 10,000-call paired p99 gate remains unmet.

## Selected step-48,000 equal-byte PCA result, 24 September

The dependent compact and full-width SOP **train** feature exports completed
from their selected step-48,000 checkpoints. The [fit-only PCA audit](evidence/compact_metric/sop-fullwidth-step48000-fit-pca128-v1.json)
then fit one centered 768-to-128 projection on the 53,700 full-width
training-fit head features and scored the unchanged 5,851-image,
1,132-product class-disjoint train holdout. The receipt and the
[projection artifact](evidence/compact_metric/sop-fullwidth-step48000-fit-pca128-v1.npz)
match their DGX originals at SHA-256
`458b45f31e61a1ad04c5213ca25cd53857a55dcc9b8b56e0359faa98d9294299`
and `b1ec4a628788b55242a2d8b233392f264be4a8c3a6eabe37a88b297373a09e91`.
The audit replays both source packed holdout scores before comparing them.
No official SOP test features enter the PCA fit or audit.

| SOP train-identity holdout, selected step 48,000 | Packed Recall@1 | Packed mAP@R | Bytes/item |
| --- | ---: | ---: | ---: |
| Trained full-width B/16 head | 96.0349% | 0.836934 | 770 |
| Same full-width head, fit-only PCA-128 | 95.6076% | 0.828906 | 130 |
| Separately trained compact B/16 head | 95.1974% | 0.812453 | 130 |

At equal 130-byte storage, PCA of the full-width trained head exceeds the
compact trained head by **0.4102 percentage points Recall@1** with product
bootstrap 95% interval **0.0000 to +0.8168 points**, and **0.016453 mAP@R**
with interval **+0.011246 to +0.021308**. It loses **0.4273 points
Recall@1** and **0.008028 mAP@R** relative to the 770-byte full-width
version. The Recall@1 equal-byte interval reaches zero, and the official-test
quality of the PCA version is unmeasured. The early step-8,000 equal-byte gain
did not fully persist to the selected checkpoint. This is one seed, and the
separately trained heads changed backbone optimization as well as the
descriptor width. The result supports a 130-byte candidate for the next
controlled experiment but no SOP or joint SOTA claim.

The independent Claude Opus 5.5 and GPT-6 Astra read-only method critiques
both identified the missing **full-width transfer** measurement and favored
a pretrained-feature anchor only as a controlled hypothesis. The CUB
classes 1–100 result is now recorded above: trained full width loses 7.3670
points against pretrained full width, and SOP-fit PCA loses another 4.7578
points. The next no-training gate is a frozen pretrained-to-trained weight
blend curve on the SOP train holdout and these transfer development classes,
followed by Cars classes 0–97 (8,054 images) if the CUB and SOP curve is
promising. No CUB/Cars features may fit the projection or update the model.
The already observed CUB/Cars test halves are descriptive evidence and will
not choose the blend or a new training arm.

If no free blend supplies the desired SOP/transfer tradeoff, compare two
same-source, same-seed 8,000-update full-width ArcFace arms: a control and
one with a frozen-pretrained cosine feature anchor on the same augmented
image tensor. Keep the 53,760-step OneCycle schedule, timm augmentation,
batch sampler, classifier, and fit-only PCA-128 deployment identical. Only
promote the anchor if it improves packed SOP train-holdout quality and
CUB/Cars development transfer beyond the control's weight-blend Pareto
curve; report training time and VRAM as part of that gate. A known feature
anchor or weight blend alone is not a novel Sfora method or a SOTA claim.
Source-matched `origin_clip` augmentation is a separate controlled factor.
