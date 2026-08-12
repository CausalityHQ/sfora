# Foundation-to-Edge Similarity Pareto Design

**Status:** proposed reproducible-Pareto program under adversarial repair.

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
- MobileCLIP-S2 as the ungated compact fallback;
- `google/siglip2-base-patch16-256`;
- the code-backed MLCD B/L exports from the UNICOM successor repository;
- Marqo FashionSigLIP as a domain-specific collision control, never as clean
  evidence against possible In-Shop pretraining contamination;
- the released UNICOM ViT-B/16 export under its official normalize-full,
  truncate-512-without-renormalization Euclidean geometry; and
- the strongest faithful local DADA/VPTSP-G reproduction available at the
  time of execution, with the faithful BN-Inception Proxy Anchor reproduction
  as the registered fallback comparator when neither reproduction is ready.

An unavailable gated weight is recorded as unavailable; it is never silently
replaced. If neither DINOv3 compact model is accessible, MobileCLIP-S2 becomes
the registered student null; if no compact source-fidelity model is available,
the student lane closes and the adapted anchor is the terminal result.
Paper-only values remain context, not acceptance thresholds.

### 3.2 Source-fidelity gate F0

For every executable anchor:

- authenticate model and processor bytes/revisions;
- compare framework output against the upstream example or a frozen fixture;
- record exact preprocessing, pooling, dtype, and normalization;
- key every cache by model revision and weight digest, processor-config digest,
  transform/view ID, resolution, dtype, normalization, dataset-row digest, and
  split; invalidate every cache written under the older mutable-tag schema;
- verify cache reload is byte- and row-order stable;
- measure batch-1/8/32 encoder p50/p95 after warm-up, peak memory, parameters,
  MACs, descriptor width, and descriptor bytes; and
- reject any arm whose cross-checkable native and repository metrics disagree
  outside a metric-specific prospectively frozen tolerance. UNICOM and DADA
  are cross-checked only at R@1; repository R@10+ values are labelled as such.

No training begins until F0 is green.

### 3.3 Frozen screen F1

Export train/query/gallery descriptors once for In-Shop and SOP. Evaluate the
native geometry plus a label-blind grid of normalized cosine, normalized
Euclidean, and native unnormalized geometry. Geometry and any linear adapter
hyperparameters are selected only from identity-disjoint training identities.
Before the first official-test access, commit a test-read register listing each
arm, checkpoint, metric, purpose, and single permitted evaluation. Any
unregistered read invalidates the confirmatory claim.

Report R@1/10/20/30 for In-Shop, R@1/10/100 for SOP, mAP@R where defined, and
the full quality/cost table. Raw-frozen rows are descriptive. Fit the same
bias-free 512-D linear probe to every F0-green anchor using cached
training-identity features and select only on the identity-disjoint validation
split. The comparator is the strongest faithful local anchor that is F0-green
and has been re-evaluated with this identical probe protocol on the same
identity-disjoint validation split; published or official-test values,
including BN-Inception PA 91.5201 and DADA/VPTSP-G capability rows, are
descriptive only. Continue only if at least one probe comes within 1.0
validation R@1 point of that same-split comparator, or is strictly
Pareto-dominant on encoder p95 or descriptor bytes while inside the 0.40-point
margin.

If no probe clears this margin, the foundation-transfer lane closes. The only
remaining authorized quality work is the conditional DADA/VPTSP-G fidelity
line in Section 7, whose purpose is to establish a trustworthy contemporary
comparator for the negative result required by Section 8, not to rescue the
failed probes.

Because closed LVD/WebLI corpora make contamination unfalsifiable, In-Shop,
Cars196, and SOP are reproduction/generality rows, not contamination evidence.
ILIAS's post-cutoff construction is mandatory and non-substitutable before a
zero-shot or general foundation-transfer claim. SOP remains a separate
generality check. If ILIAS does not run, every claim is restricted to In-Shop
and SOP reproduction and the general foundation-transfer claim closes.

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

Run three paired adapter seeds. Retention is decided only on the
identity-disjoint validation split: resample validation query identities while
holding its gallery fixed, require the one-sided 95% paired-bootstrap lower
bound above zero, mean gain at least 0.50 R@1 point, and all three seed gains
positive. The registered official-test result is descriptive and cannot
trigger fallback. The 128-D prefix is additionally compared with the published
92.7 Hyp-ViT 128-D capability row, clearly labelled unmatched until
reproduced.

Descriptor storage is a Section 4 claim: compare teacher/teacher retrieval at
the validation-selected prefix with the same teacher at 512 dimensions and
with PCA/PQ/OPQ/INT8 controls. The 64-D prefix is diagnostic only; the
deployable prefix is selected from `{128,256,512}` so the later student can
serve it. Require the 90% query-identity paired-bootstrap interval to lie
entirely inside `[-0.40,+0.40]` R@1 point and require at least 2x smaller
gallery rows. It is not credited to the later student.

### 4.2 Optional complementary teacher

Only after the single-anchor adapter passes, concatenate the two strongest
validation-selected normalized anchors from different pretraining families,
fit one 512-D adapter on training identities, and normalize. At inference
evaluation this teacher is reported honestly as a two-encoder upper bound. It
is not used as the deployed gallery, query encoder, or student teacher; the
deployable lane uses the strongest single adapted teacher. This diagnostic
therefore cannot authorize extra student tuning or inherit any speed claim.

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
student. F2 therefore fine-tunes exactly one small student against the single
adapted teacher of Section 4.1 and retains it only if it clears both the F2
bullet-3 asymmetric threshold and bullet-4 symmetric non-inferiority threshold
against their serving-mode-matched nulls. Add one 512-D head whose prefixes
`{128,256,512}` are independently L2-normalized.

The primary deployment is explicitly asymmetric: gallery rows are encoded
offline by the teacher adapter and queries by the faster student into the same
selected prefix. Coordinate semantics are frozen by the teacher: gallery
vectors are the normalized first `d` coordinates of the adapted teacher and
student outputs are trained directly against those exact coordinates. No
independent rotation or projection is fitted on either side after distillation.
A mixed teacher-gallery/student-query evaluator and a symmetric student/student
index are required controls. The system reports gallery build cost and cannot
claim the teacher's speed as the query encoder's speed. There is no reranker,
ensemble at query time, query expansion, or gallery-dependent transform.
A gallery contains rows from exactly one encoder; mixed teacher/student rows
are prohibited because their within-space scores are not comparable.

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

`t_d` is exactly the independently normalized first `d` coordinates of the
frozen adapted teacher; it is also the gallery representation in asymmetric
serving. `S_t` and `S_d` exclude self entries and use the same live
identity-balanced batch.
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
6. the same frozen compact backbone used by the student plus a cached-feature
   alignment into the teacher's
   serving coordinates, using the same at-most-5M-parameter `D -> 1024 -> 512`
   adapter class, optimizer-search budget, and training identities as Section
   4.1; a bias-free linear alignment is a required nested ablation;
7. MobileCLIP-S2 as an external compact retrieval null; and
8. post-hoc PCA, INT8, PQ, and OPQ compression of the best full-width
   descriptor.

Every trainable control uses identical initialization, sampler, transforms,
optimizer, steps, EMA policy, and GPU-hour cap. Cached discrete views are used
by every adapter control; live student augmentations are used by every student
control. The distillation grid is frozen to
`lambda_feature in {0.25,1}`, `lambda_relation in {0,0.25}`, `tau in {0.05,0.1}`
with uniform `alpha_d`; the same eight-trial cap applies to matched controls.
The dual-teacher diagnostic cannot receive student trials. PCA/whitening and
every alignment map are fitted on training-identity features only.

### 5.4 Student falsifier F2

Run one structural smoke, then three paired short-run screening seeds. F2 is
non-confirmatory; F3 uses disjoint seeds. Select one primary prefix and the
single primary In-Shop R@1 endpoint from validation; all other prefixes and
datasets are descriptive. Continue to full training only if that prefix
satisfies all:

- mean validation R@1 is at least the supervised MRL student's value plus
  0.30 point in the primary asymmetric serving configuration, with all three
  seed differences positive; and
- measured query-side encoder p95 plus search p95 is at least 20% lower than
  the strongest single-encoder teacher at matched hardware; and
- the trained student beats the Section 5.3 control-6 nonlinear aligned null by
  at least 0.50 R@1 point at matched query latency and serving mode; and
- in the symmetric student/student versus own-adapter/own-adapter comparison,
  compute each validation query identity's R@1 difference after averaging that
  identity over the same three screening seeds, require the 90%
  query-identity paired-bootstrap interval of those seed-averaged differences
  to lie wholly above `-0.40` point, and require every per-seed mean difference
  to exceed `-0.40` point. Otherwise use control 6 for asymmetric serving and
  the own-adapter compact null for symmetric serving, and close additional
  distillation.

For the asymmetric comparison, the matched null is Section 5.3 control 6 in
the teacher's serving coordinates. The Section 5.1 own-adapter null is compared
only in the symmetric student/student versus null/null configuration. Bullet 3
is gating in the primary asymmetric configuration, and bullet 4 is a separate
gating non-inferiority requirement for the symmetric configuration.

Kill relational distillation if feature-only recovers at least 90% of its gain.
Kill MRL if independently trained fixed-width heads dominate every prefix at
matched width and training cost.

### 5.5 Full claim F3

The quality branch starts with six paired seeds and imports the parent
program's variance-only extension to at most 12: mean In-Shop R@1 gain at least
0.50 point, one-sided 95% paired lower bound above zero, and disclosed paired
SD/MDE. The cost branch starts with eight paired seeds and permits one
variance-only extension to 12; its 90% paired interval must lie entirely in
`[-0.40,+0.40]` R@1 point while query-side encoder-plus-search p95 improves at
least 20%. Failure to power either branch closes it; non-significance is not
equivalence. Storage is credited separately to the Section 4 teacher-prefix
result, or to a symmetric student/student comparison—never to asymmetric
query distillation. SOP is a hard secondary non-inferiority gate once its exact
recipe is available. Report all seeds, final and selected checkpoints, FLOPs,
wall time, GPU-hours, memory, latency, width, bytes, and build time.

## 6. Search and native-kernel lane

### 6.0 Training acceleration and kernel trigger T0

FAISS/cuVS are inference-search baselines; they do not accelerate learning.
Training speed is measured as **wall-clock time to a frozen validation-quality
target**, plus final quality at a frozen GPU-hour budget. The first training
accelerator is algorithmic: cached foundation features remove backbone
forward/backward entirely from Section 4 adapter optimization. The later
student uses maintained BF16, channels-last, compiled graph, fused optimizer,
and supported attention/MLP kernels, each profiled separately before
composition.

A custom Triton/CUDA training operator is authorized when a profiler on the
actual surviving student shows one uncovered similarity, relational-logit,
normalization, or reduction region consumes at least 10% of steady-state step
time, or its materialized intermediates prevent the registered useful batch.
The candidate may tile and fuse normalization, similarity GEMM epilogues,
self-mask, teacher/student log-softmax, KL reduction, and analytical descriptor
gradients without storing the full pair matrix. Dense GEMMs remain cuBLASLt
unless fusion around them creates the measured gain.

The training kernel must match an FP64/ordinary-PyTorch forward/backward oracle,
preserve both query and database descriptor-gradient roles, support arbitrary
labels and legal batch tails, reject nonfinite inputs, and pass fixed-tree
repeatability plus cross-tile tolerance tests. Keep it only if it provides at
least 1.5x operator speed, at least 10% end-to-end step speed, and at least 15%
lower median time-to-the-frozen-quality-target over paired runs. A larger batch
counts only if it improves quality at matched GPU-hours. Otherwise retain the
maintained implementation. The time-to-quality test uses at least four paired
seeds and is paid from the surviving F2/F3 allocation; it is not an additional
GPU budget.

### 6.1 Workloads

In-Shop and SOP are quality/correctness workloads and cannot authorize custom
search. The systems lane requires:

- a named consumer and recorded request trace;
- at least 1M real vectors from that named consumer's own gallery;
- pre-embedded query vectors;
- batches 1 and 8 as decisive;
- `k in {1,10,100}`;
- complete timing of query transfer, search, result transfer, and top-k; and
- search at least 30% of this end-to-end p95.

Without those facts the kernel lane remains closed.
ILIAS, LAION, SIFT1M, and synthetic vectors are characterization only and can
never authorize K0. On one 128-GB GB10, in-memory characterization is capped at
ILIAS-5M at the deployed prefix; 100M x 512-D FP32 is 204.8 GB and out of
scope. Any public-set embedding cost is separately budgeted before execution.

### 6.2 Maintained baselines

Benchmark exact PyTorch/cuBLAS, FAISS-GPU or cuVS exact, FAISS Flat CPU, and
when approximation is allowed FAISS IVF-PQ/OPQ, HNSW, ScaNN, and RaBitQ under a
matched recall grid. Record index build time, storage, recall@k, p50/p95/p99,
queries/s, and bytes transferred. CPU/GPU latency is never divided into an
algorithmic speedup.

FAISS and cuVS must be pinned and built/imported for CUDA 13 and `sm_121` before
the gate. If either is unavailable, disclose it and use the registered tiled
FP32 same-device primitive; never fabricate the missing row. ScaNN is included
only if an authenticated aarch64 build exists, otherwise it is recorded as
unavailable.

### 6.3 Kernel trigger K0

A custom operator is authorized only if the profiler assigns at least 30% of
search p95 to a named fusible scan/dequantize/top-k stage, the gap is measured
against achieved effective bandwidth of a same-access-pattern streaming
reference on this device, and the projected post-fusion p95 still clears the
1.5x search and 15% end-to-end keep gates below. Triton is tried first. Native
CUDA follows only for a measured missing primitive, register/shared-memory
limit, synchronization pattern, or code-generation failure.

### 6.4 Candidate kernel

The first candidate is a fused rowwise-INT8 inner-product scan. Gallery and
query coordinates use symmetric round-to-nearest INT8 codes in `[-127,127]`
with `-128` prohibited, one registered FP32 scale per vector, exact INT32
products/accumulation, a canonical fixed-order scale application, blockwise
top-k, and a final stable merge. It emits bit-exact results for that registered
quantized score, not exact claims about the FP32 model. Optional FP32 reranking
of a bounded approximate survivor set reports recall against exhaustive FP32;
it is called exact only if it reuses the parent's certified `L/U/theta`
containment construction.

Correctness requires:

- an INT32 overflow proof for every supported width;
- bit-exact INT32 accumulator agreement across legal tiles and stable
  `(canonical quantized score descending,row_id ascending)` ties;
- reference agreement on random, adversarial, saturation, zero, and duplicate
  vectors;
- no out-of-bounds access under sanitizers;
- FP32-scale application agreement with the fixed rounding schedule; and
- identical results for repeated execution of the fixed reduction tree.

Recall is measured against registered exhaustive FP32 top-k, and the INT8 path
is always labelled approximate relative to that target. Keep the kernel only
if it delivers at least 1.5x search p95 and 15% end-to-end p95 improvement over
the strongest maintained same-device control at matched FP32 recall, with no
regression hidden at another claimed batch or k. Otherwise
delete the custom path and ship the maintained index.

## 7. Budget and order

1. Finish already running jobs without duplication.
2. Implement the missing F0 surface explicitly: revision-addressed model and
   processor loading; content-addressed multi-view cache v2; asynchronous
   export; batch latency/memory/MAC instrumentation; SOP R@100; and
   metric-specific native-evaluator fixtures. Also implement train-only
   PCA/INT8/PQ/OPQ comparators and pin an importable FAISS/cuVS build before
   any Section 4 compression claim. Budget 10 engineer-days of CPU/dev work;
   this estimate consumes no GB10 hours.
3. Implement and test a PyTorch linear/MLP Matryoshka adapter; run F1 frozen
   screens plus cheap linear probes and close unavailable or weak anchors.
   Budget 5 engineer-days of CPU/dev work; this estimate consumes no GB10
   hours, which are charged to the ledger below.
4. Fit the single adapter and optional two-encoder diagnostic; close fusion if
   complementarity fails.
5. Before F2, implement cached external-teacher targets, the small-backbone
   training path, asymmetric aligned-space evaluator, and matched nulls.
   Budget 8 engineer-days of CPU/dev work; this estimate consumes no GB10
   hours, which are charged to the ledger below.
6. Run the three-seed F2 student falsifier. Spend F3 only on the
   validation-selected prefix/control pair.
7. Estimate and, if affordable, run the ILIAS quality evaluation. If it does
   not run, close the general foundation-transfer claim.
8. If the T0 profiler trigger fires, run at least four paired maintained-versus-
   candidate time-to-quality seeds inside the surviving F2/F3 allocation.
9. Acquire or name a real 1M+ workload and benchmark maintained search.
10. Write Triton/CUDA only after K0.

This design supersedes the parent's unlaunched post-queue allocation while
retaining its gross 120-GB10-hour ceiling. Before a new job, the execution
ledger enumerates every already-spent protocol-identical run as `H_spent` and
sets `H_remaining = 120 - H_spent`; no rounded or estimated value can authorize
work. Each line records both its cap and hours already charged to it; its
remaining allocation is `H_x = max(0, cap_x - charged_x)`. Frozen exports/F0
have cap 6, cached adapters 4, F2 student screening 25, conditional
DADA/VPTSP-G fidelity 25, ILIAS/public characterization reserve 10, and F3 50.
The fidelity line exists only if no current same-split probe reaches the
fallback anchor. F3 receives at most
`max(0, min(50, H_remaining - H_F0 - H_adapter - H_F2 - H_fidelity - H_reserve))`,
where every subtracted `H_x` is the remaining unspent allocation rather than a
cap already included in `H_spent`. Before each launch, the exact cumulative
ledger plus that job's registered maximum duration must be at most 120; this
admission rule is authoritative over nominal line allocations. F3 is cut
first when capacity contracts and conditional fidelity next. If both reach
zero and the admission rule still fails, the confirmatory lane closes and no
new job is authorized. T0's at-least-four paired runs are charged to the
surviving F2/F3 line rather than added on top. Measured short-run throughput
may shrink but never enlarge these caps. If the reserve cannot cover ILIAS
encoding, ILIAS does not run and the general foundation-transfer claim closes.
Kernel work receives no GPU budget before T0/K0.

## 8. Claims and stop rules

Allowed claims are a reproduced foundation retrieval anchor, an adapted-anchor
quality result, a compact-student Pareto point, and a measured search-kernel
speedup. They remain separate until each gate passes.

Prohibited:

- calling distillation, MRL, linear adaptation, or INT8 search novel;
- calling asymmetric metric transfer or backward-compatible representation
  learning novel;
- comparing a two-encoder teacher's quality with a one-encoder student's speed
  without showing both full system rows;
- tuning geometry, teacher choice, prefix, or temperature on test identities;
- calling non-significance equivalence;
- using a synthetic million-row expansion as a named production workload;
- writing a kernel because it is interesting rather than measured necessary;
- using hyperbolic geometry to rescue a failed Euclidean student; or
- calling a local gain SOTA without reproducing the contemporary anchor and
  matching its data, evaluator, and inference contract.

Every quality row reports its absolute distance to published 95.5 supervised
UNICOM B/16 and 96.5 VPTSP-G ViT-L/14 capability rows. The document and result
use “reproducible Pareto” rather than “SOTA” until a contemporary operating
point is reproduced under matched data, evaluator, and inference contracts.

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
- Asymmetric Metric Learning for Knowledge Transfer (CVPR 2021) and
  Backward-Compatible Training (CVPR 2020) bound the asymmetric serving claim.
