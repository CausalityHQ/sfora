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
selection. Use only train identities for recipe selection. First test a head
feasibility pilot, then train the backbone under the frozen recipe; compare
paired independent seeds before attributing a gain to the new loss. The
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
| SOP official test, same queries | OML ViT-S/16@224 + Sfora compact profile | 85.9757% | 0.641825 | 130 | — | Exploratory, packed representation; [raw profile](evidence/compact_metric/oml-vits16-sop-packed-profile-verification-v1.json) |
| In-Shop official query/gallery | UNICOM ViT-L/14@336 | 96.7% | — | 768 f32 output before indexing | — | Published [UNICOM Table 4](https://arxiv.org/pdf/2304.05884); evaluator uses normalized prefix-512 Euclidean |
| In-Shop official query/gallery | UNICOM ViT-L/14@336 + Sfora compact profile | 95.4283% | 0.800020 | 130 | — | Exploratory, [local result](compact_metric_selector_result_2026-09-19.md) |

The OML float source rescored with the same stable-ordinal evaluator remains
86.5575% Recall@1 and 0.654393 mAP@R, with
[raw matched-scorer evidence](evidence/compact_metric/oml-vits16-sop-matched-scorer-v1.json).
The published full-width models and the 130-byte compact profiles have
different storage budgets. The current [paired search replay](evidence/compact_metric/oml-vits16-sop-packed-1m-replay-v2.json)
uses a tiled SOP gallery, not one million distinct images. It reports search
medians near 1.05 ms at batch 1 and 2.8 ms at batch 32 on a DGX GB10, with
50 calls per arm. Those timings exclude decode, encoding, and packing; their
p99 values are diagnostic only. They cannot fill the end-to-end column above.

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
reports 88.8% SOP Recall@1 after supervised fine-tuning. The next learning
experiment should train a 128-dimensional head and then the backbone on SOP
train identities, beginning with a matched ArcFace control. Add float-rank
if the control has room; prioritize packed-rank if the new trained head
shows a material int8-specific loss. Match starting checkpoint, classifier,
batch identities and order, total updates, scorer, and seed. Measure training
time/throughput/VRAM and encoder-plus-packed-search latency before promoting
an arm. One seed is a feasibility screen; at least three paired seeds and a
second dataset are needed for an algorithmic claim. The OML compact profile
remains the efficient product baseline for a future paired end-to-end timing;
no image-to-result p99 result exists yet.
