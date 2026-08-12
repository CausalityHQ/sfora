# Foundation-to-Edge Similarity Pareto Design

**Status:** proposed engineering-SOTA program pending adversarial review.

## 1. Decision

The primary bet is not a new metric-learning loss. It is a reproducible system
that transfers current foundation-model retrieval quality into one compact,
fast descriptor and then serves it with maintained ANN software or, only where
measurement justifies it, a fused Triton/CUDA search kernel.

The ordered lanes are:

1. establish current frozen and linearly adapted DINOv3, SigLIP2, and UNICOM
   anchors under one evaluator;
2. train a cached-feature Matryoshka adapter on the strongest anchor;
3. only if needed, distill that adapted teacher into one DINOv3 ConvNeXt-Tiny
   or ViT-S query encoder with nested 128/256/512-D outputs;
4. retain the smallest prefix that passes a prospectively frozen quality gate;
5. use FAISS/cuVS/ScaNN first, and write a kernel only on a real 1M--100M
   workload whose trace proves search is material.

This is an engineering Pareto program. Foundation pretraining, linear metric
adaptation, relational distillation, Matryoshka representation learning,
quantization, and dense/ANN search are occupied prior art. A successful result
may claim a reproduced quality/latency/storage operating point, not a new loss.

## 2. Why this route

The local BN-Inception lane is useful mechanism evidence but is too far behind
modern supervised and foundation-model anchors to make small objective changes
the shortest path to SOTA. DINOv3 releases ViT and distilled ConvNeXt models;
its official model card already reports strong global retrieval probing.
SigLIP2 is a complementary vision-language encoder. ILIAS (CVPR 2025) shows
that foundation models are strong at open-world instance retrieval, that a
linear multi-domain adaptation can improve them, and that 100M-distractor
retrieval remains unsaturated.

The key system hypothesis is falsifiable: a compact student can inherit enough
of a complementary teacher's neighbourhood structure to beat older specialist
models while being materially faster and smaller than the teacher. If it
cannot, the project keeps the strongest reproducible foundation anchor and
closes the distillation claim.

Hyperbolic/Poincare geometry is not a primary arm. Instance identity has no
registered hierarchy, and existing hyperbolic DML behavior collides with
norm-aware and hard-negative Euclidean controls. Curvature is reconsidered
only if a frozen geometry audit finds reproducible radial signal that a matched
Euclidean norm-aware transform cannot reproduce.

## 3. Reproducible foundation ladder

### 3.1 Models

Pin exact upstream revisions, weight digests, licenses, processors, input
resolutions, pooling rules, and output dimensions for:

- `facebook/dinov3-vits16-pretrain-lvd1689m`;
- `facebook/dinov3-vitb16-pretrain-lvd1689m` when access and memory permit;
- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`;
- `google/siglip2-base-patch16-256`;
- the code-backed MLCD B/L exports from the UNICOM successor repository;
- Marqo FashionSigLIP as a domain-specific collision control, never as clean
  evidence against possible In-Shop pretraining contamination;
- the released UNICOM ViT-B/16 export under its official normalize-full,
  truncate-512-without-renormalization Euclidean geometry; and
- the strongest faithful local DADA/VPTSP-G reproduction available at the
  time of execution.

An unavailable gated weight is recorded as unavailable; it is never silently
replaced. Paper-only values remain context, not acceptance thresholds.

### 3.2 Source-fidelity gate F0

For every executable anchor:

- authenticate model and processor bytes/revisions;
- compare framework output against the upstream example or a frozen fixture;
- record exact preprocessing, pooling, dtype, and normalization;
- verify cache reload is byte- and row-order stable;
- measure batch-1/8/32 encoder p50/p95 after warm-up, peak memory, parameters,
  MACs, descriptor width, and descriptor bytes; and
- reject any arm whose native and repository evaluators disagree outside a
  prospectively frozen tolerance.

No training begins until F0 is green.

### 3.3 Frozen screen F1

Export train/query/gallery descriptors once for In-Shop and SOP. Evaluate the
native geometry plus a label-blind grid of normalized cosine, normalized
Euclidean, and native unnormalized geometry. Geometry and any linear adapter
hyperparameters are selected only from identity-disjoint training identities.
The official test split is read once after freezing.

Report R@1/10/20/30 for In-Shop, R@1/10/100 for SOP, mAP@R where defined, and
the full quality/cost table. Continue only if at least one foundation anchor
either beats the strongest faithful local anchor or offers a strict encoder or
descriptor Pareto point within a frozen 0.40 R@1 equivalence margin.

Because DeepFashion images may overlap web-scale pretraining, an In-Shop-only
zero-shot win is contamination-suspect. Cars196 replication is mandatory for
the adapter and SOP or ILIAS replication is mandatory before a general
instance-retrieval claim.

## 4. Cached-feature Matryoshka adapter

### 4.1 Primary adapter

On the strongest F1 anchor, fit at most a 5M-parameter `D -> 1024 -> 512` MLP
with nested normalized prefixes `{64,128,256,512}` and a fixed normalized
margin-softmax objective over official training identities. Compare it with a
bias-free linear 512-D adapter, PCA/whitening, fixed-width heads, and the frozen
backbone. The backbone stays frozen and features from a frozen set of canonical
and flip/multi-crop views are cached once. Every adapter receives the same
sampler, steps, optimizer search budget, cached views, and identity-disjoint
selection split.

The adapter is retained only if its paired training-identity bootstrap lower
bound on validation R@1 exceeds zero and its mean validation gain is at least
0.50 R@1 point. The held-out test result must not lose to its frozen source.
The 128-D prefix is additionally compared with the published 92.6 Hyp-ViT
128-D capability row, clearly labelled unmatched until reproduced. Test
outcomes cannot select the adapter.

### 4.2 Optional complementary teacher

Only after the single-anchor adapter passes, concatenate the two strongest
validation-selected normalized anchors from different pretraining families,
fit one 512-D adapter on training identities, and normalize. At inference
evaluation this teacher is reported honestly as a two-encoder upper bound; it
is never a deployed candidate.

Kill fusion before student training if either is true:

- its validation gain over the better single adapted teacher is below 0.30 R@1
  point or its one-sided 95% identity-bootstrap lower bound is not positive;
- a same-parameter single-anchor adapter, PCA/whitening, or score averaging
  recovers at least 90% of its gain.

This prevents an expensive teacher from entering the program without measured
complementarity.

## 5. Compact student

### 5.1 Architecture and asymmetric serving

The null deployment control is the released DINOv3-S or ConvNeXt-Tiny frozen
encoder plus its own Section 4 adapter; Meta already distilled these backbones,
so another distillation must beat this control rather than a weak unadapted
student. Only if the null loses the registered quality target, fine-tune one
small student against the adapted P2 teacher. Add one 512-D head whose prefixes
`{128,256,512}` are independently L2-normalized.

The primary deployment is explicitly asymmetric: gallery rows are encoded
offline by the teacher adapter and queries by the faster student into the same
selected prefix. A symmetric student/student index is a required control. The
system reports gallery build cost and cannot claim the teacher's speed as the
query encoder's speed. There is no reranker, ensemble at query time, query
expansion, or gallery-dependent transform.

### 5.2 Cached targets and loss

Teacher descriptors are computed once from frozen canonical/multi-view inputs and stored
with image-byte hash, transform hash, model digest, row ID, and FP32 descriptor
hash. Training uses an augmented student view and the matching canonical
teacher target. The target cache is immutable and no teacher forward occurs in
the timed student epochs.

For each prefix `d`, use:

```text
L_d = L_supervised(z_d, y)
    + lambda_feature * (1 - cosine(z_d, stopgrad(t_d)))
    + lambda_relation * KL(softmax(S_t/tau) || softmax(S_d/tau))
L = sum_{d in {128,256,512}} alpha_d L_d
```

`t_d` is a train-only learned linear projection of the 512-D teacher into the
student prefix, fitted before student optimization and then frozen. `S_t` and
`S_d` exclude self entries and use the same live identity-balanced batch.
Teacher logits, student logits, and reductions are FP32. The backbone and head
are trained together unless the F2 cost gate chooses the explicitly labelled
head-only diagnostic.

The loss is occupied relational/feature distillation plus MRL. It has no
novelty claim.

### 5.3 Matched controls

Run, in order:

1. frozen student plus trained head;
2. supervised student without distillation or nested-prefix losses;
3. supervised MRL student;
4. single-teacher feature distillation;
5. single-teacher feature plus relational distillation;
6. dual-teacher feature plus relational distillation;
7. post-hoc PCA and INT8 quantization of the best full-width descriptor.

Every trainable control uses identical initialization, sampler, transforms,
optimizer, steps, EMA policy, and GPU-hour cap. The dual teacher cannot receive
extra tuning trials.

### 5.4 Student falsifier F2

Run one structural smoke, then three paired short-run seeds. Continue to full
training only if one prefix satisfies all:

- mean validation R@1 is at least the supervised MRL student's value plus
  0.30 point, with all three seed differences positive; and
- measured encoder p95 plus exact-search p95 is at least 20% lower than the
  best teacher at matched hardware, or descriptor storage is at least 2x
  smaller with quality inside the 0.40-point equivalence margin; and
- the trained student beats the frozen small-backbone-plus-adapter null by at
  least 0.50 R@1 point at matched query latency. Otherwise use the null and
  close additional distillation.

Kill dual-teacher distillation if the single-teacher arm recovers at least 90%
of its gain. Kill relational distillation if feature-only recovers at least
90%. Kill MRL if independently trained fixed-width heads dominate every
prefix at matched width and training cost.

### 5.5 Full claim F3

Use six paired seeds for the surviving student and strongest matched control.
Quality improvement requires mean In-Shop R@1 gain at least 0.50 point and a
one-sided 95% paired lower bound above zero. A cost claim requires a two-sided
90% paired interval entirely inside `[-0.40,+0.40]` R@1 point and at least 20%
improvement in encoder-plus-search p95 or descriptor storage. SOP is a hard
secondary non-inferiority gate once its exact recipe is available. Report all
seeds, final and selected checkpoints, FLOPs, wall time, GPU-hours, memory,
latency, width, bytes, and build time.

## 6. Search and native-kernel lane

### 6.1 Workloads

In-Shop and SOP are quality/correctness workloads and cannot authorize custom
search. The systems lane requires:

- a named consumer and recorded request trace;
- at least 1M real gallery vectors, with ILIAS 5M/100M as the preferred public
  characterization benchmark when its assets are available;
- pre-embedded query vectors;
- batches 1 and 8 as decisive;
- `k in {1,10,100}`;
- complete timing of query transfer, search, result transfer, and top-k; and
- search at least 30% of this end-to-end p95.

Without those facts the kernel lane remains closed.

### 6.2 Maintained baselines

Benchmark exact PyTorch/cuBLAS, FAISS-GPU or cuVS exact, FAISS Flat CPU, and
when approximation is allowed FAISS IVF-PQ/OPQ, HNSW, ScaNN, and RaBitQ under a
matched recall grid. Record index build time, storage, recall@k, p50/p95/p99,
queries/s, and bytes transferred. CPU/GPU latency is never divided into an
algorithmic speedup.

### 6.3 Kernel trigger K0

A custom operator is authorized only if the strongest maintained same-device
baseline leaves at least a 20% measured p95 gap to a roofline estimate and the
profile identifies a fusible scan/dequantize/top-k bottleneck. Triton is tried
first. Native CUDA follows only for a measured missing primitive, register or
shared-memory limit, synchronization pattern, or code-generation failure.

### 6.4 Candidate kernel

The first candidate is a fused rowwise-INT8 inner-product scan with FP32 query
scales, blockwise top-k, and a final stable merge. It emits exact results for
the registered quantized score, not approximate claims about the FP32 model.
An optional exact-FP32 refinement reranks a prospectively bounded survivor set
and is compared with maintained IVF-PQ/RaBitQ refinement.

Correctness requires:

- an INT32 overflow proof for every supported width;
- stable `(score descending,row_id ascending)` ties;
- reference agreement on random, adversarial, saturation, zero, and duplicate
  vectors;
- no out-of-bounds access under sanitizers;
- forward agreement across legal tiles within frozen tolerance; and
- identical results for repeated execution of the fixed reduction tree.

Keep the kernel only if it delivers at least 1.5x search p95 and 15% end-to-end
p95 improvement over the strongest maintained same-device control at matched
recall, with no regression hidden at another claimed batch or k. Otherwise
delete the custom path and ship the maintained index.

## 7. Budget and order

1. Finish already running jobs without duplication.
2. Implement only the F0 model/processor/export audit and existing evaluator
   adapters.
3. Run F1 frozen screens; close unavailable or weak anchors.
4. Fit single and dual linear adapters; close fusion if complementarity fails.
5. Run the three-seed F2 student falsifier.
6. Spend the six-seed F3 budget only on the surviving prefix/control pair.
7. Acquire or name a real 1M+ workload and benchmark maintained search.
8. Write Triton/CUDA only after K0.

The first local implementation budget is CPU tests plus at most 6 GB10 hours
for F0/F1 exports. Cached adapter work is capped at 4 GB10 hours, and F2 is
capped at 25 GB10 hours total. F3 receives a separate
prospective cap after measured short-run throughput. Search characterization is
read-only over frozen embeddings. Kernel work receives no GPU budget before
K0.

## 8. Claims and stop rules

Allowed claims are a reproduced foundation retrieval anchor, an adapted-anchor
quality result, a compact-student Pareto point, and a measured search-kernel
speedup. They remain separate until each gate passes.

Prohibited:

- calling distillation, MRL, linear adaptation, or INT8 search novel;
- comparing a two-encoder teacher's quality with a one-encoder student's speed
  without showing both full system rows;
- tuning geometry, teacher choice, prefix, or temperature on test identities;
- calling non-significance equivalence;
- using a synthetic million-row expansion as a named production workload;
- writing a kernel because it is interesting rather than measured necessary;
- using hyperbolic geometry to rescue a failed Euclidean student; or
- calling a local gain SOTA without reproducing the contemporary anchor and
  matching its data, evaluator, and inference contract.

## 9. Primary sources

- DINOv3 official repository and model card:
  <https://github.com/facebookresearch/dinov3>
- SigLIP2 official model family and documentation:
  <https://huggingface.co/docs/transformers/model_doc/siglip2>
- ILIAS, CVPR 2025:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Kordopatis-Zilos_ILIAS_Instance-Level_Image_retrieval_At_Scale_CVPR_2025_paper.html>
- Matryoshka Representation Learning, NeurIPS 2022:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html>
- FAISS, cuVS, ScaNN, and official model repositories are pinned to exact
  revisions during F0/K0 rather than cited only by mutable main branches.
