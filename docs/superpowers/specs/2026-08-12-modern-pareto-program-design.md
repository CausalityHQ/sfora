# Modern Similarity-Learning Pareto Program

**Status:** adversarially reviewed design; implementation has not started

**Goal:** first establish a locally reproducible modern quality baseline, then
improve either quality, training cost, inference cost, or more than one of
these without mixing their attribution. The output is a Pareto frontier with
ordinary Git commits and reproducible benchmark records, not a blended score.

## 1. Decision

The local BN-Inception Proxy Anchor lane is useful mechanism evidence but is
roughly 1.5 points behind the audited DADA pipeline and roughly 4 points behind
the published supervised UNICOM B/16 result. Another small correction on that
old pipeline is not the shortest path to a competitive system.

The previous two proposals are closed:

- PARC duplicated the occupied Matryoshka/adaptive-retrieval family and offered
  no full-width quality mechanism.
- ORBIT duplicated Panorama's learned orthogonal transform and progressive
  exact bounds, now present in FAISS.

The replacement has three ordered lanes:

1. **quality foundation:** reproduce code-backed one-GPU modern anchors and
   audit their retrieval geometry before inventing a loss;
2. **training throughput:** apply supported compiler, layout, precision, and
   fused-optimizer improvements to the winning anchor;
3. **retrieval throughput:** use existing exact/approximate libraries first and
   consider a custom certified mixed-precision screen only where retrieval is
   a material part of online latency.

The lanes may compose only after each independently passes. A kernel never
receives credit for R@1, and a different backbone or pretraining corpus never
receives credit as a loss improvement.

## 2. Reproducible quality foundation

### 2.1 Anchor ladder

The primary trainable anchor is the strongest code-backed recipe that can be
run faithfully on one GB10, not the largest unaudited leaderboard number.

1. **UNICOM ViT-B/16 released-weight probe.** Reproduce the official zero-shot
   In-Shop path first: normalize the full embedding, truncate to 512 dimensions
   without renormalizing, and rank by Euclidean distance. The documented 74.6
   R@1 is a pipeline check, not a supervised baseline.
2. **PA+DADA ResNet-50.** Port the committed one-GPU `configs/inshop.yaml`
   recipe and target the published 93.0 operating point. A same-runtime
   `oproxy` arm bounds the aggregate pipeline gap but is not called a clean
   DADA ablation because the official arms also differ in loss weight, proxy
   optimizer, learning rate, positive mixing, and compute.
3. **VPTSP-G ViT-B/16.** Reproduce the released one-GPU
   `config/inshop_basic.yaml` recipe and its code-backed 92.5 target. This is
   the transformer/PEFT cross-check.
4. **Published-only rows.** CRT 94.48, VPTSP-G ViT-L/14 96.5, supervised
   UNICOM B/16 95.5 remain capability references until their exact local
   recipe, weights, and hardware adaptation are audited. They are not starting
   baselines and are never acceptance thresholds.

The active PA/MCPS, compactness, and UNICOM jobs finish unchanged. DADA and
VPTSP-G do not overlap them.

### 2.2 Reproduction contract

Each trainable anchor gets one structural smoke, then three preregistered seeds
for descriptive reproduction. The faithful reproduction uses every official
training identity. A separate selection arm withholds 10% of training
identities, chosen prospectively by ordering
`SHA256(UTF8("200\0" + identity_string))`; its selected epoch index is transferred to the corresponding
full-data run. The withheld arm's own score is reported separately and is not
subject to the publication-fidelity bound.

Report every seed and three checkpoint views: frozen-final; best-epoch-on-test,
labelled as the authors' published selection rule and prohibited from
supporting a new claim; and the full-data checkpoint at the transferred epoch,
the sole paired baseline for a new method. Do not tune a new method against
test R@1.

Record:

- In-Shop R@1/10/20/30 and SOP R@1/10/100 only when a committed source recipe
  supports them;
- train images/s, optimizer steps, wall time, peak allocated and reserved GPU
  memory, encoder latency at batches 1/8/32, parameter count, and MACs;
- descriptor width, input resolution, pretraining, sampler, transforms,
  optimizer, schedule, precision flags, and evaluation geometry;
- deviations from the authors' environment, including compatibility patches;
- mean, standard deviation, and individual seeds, without presenting three
  seeds as a powered superiority test.

A reproduction may serve as a modern paired anchor only when the authors'
test-selected view is within 1.0 In-Shop R@1 point of the published target and
the run is structurally faithful. A larger gap is a diagnostic only. "Cannot
run faithfully" means an objective condition: the audited six-epoch preflight
cannot build the model, produces a non-finite objective, performs no expected
optimizer update, or fails reload/evaluation. It is not a judgment based on the
eventual score. If DADA fails that checklist, VPTSP-G becomes the primary
trainable anchor and the DADA gap remains disclosed. If no faithful anchor is
within 1.0 point, the smallest-gap faithful reproduction becomes the paired
baseline, its degradation is disclosed, and every claim is scoped to an
improvement over a degraded local reproduction rather than the frontier.

One repository evaluator is verified against each source's native In-Shop R@1
within a prospectively registered tolerance and is used for every geometry and
cross-anchor row. Native evaluators remain separate reproduction-fidelity
outputs; their inconsistent R@10+ and pseudo-mAP fields are not mixed. SOP is a
hard secondary gate only after the primary anchor's committed SOP recipe is
audited. Otherwise any initial claim is explicitly scoped to In-Shop and a
cross-dataset claim remains deferred.

### 2.3 Geometry audit before invention

Modern methods confound representation and retrieval geometry. UNICOM's
published evaluator uses truncated, unrenormalized Euclidean distance; HIER
uses hyperbolic distance; most proxy methods use normalized cosine. Before a
new loss is proposed, frozen descriptors from every reproduced anchor are
evaluated under a label-blind grid:

- native published geometry;
- normalized cosine;
- Euclidean before and after normalization;
- registered prefix widths with and without renormalization.

The grid is fixed before test labels are read. Any selection uses an
identity-disjoint validation split constructed only from training identities.
The test split is evaluated once. This audit answers whether the apparent
frontier gap is representation, radial information, truncation, or ordinary
Euclidean/cosine metric choice. Hyperbolic DML remains a prior-art collision
control, not an experimental arm in this one-GPU budget. The audit does not
retroactively change a published reproduction.

Only a diagnosed failure mechanism authorizes a new quality method. Candidate
families must then survive a prior-art audit and matched controls. A likely
first hypothesis is to preserve useful radial confidence while regularizing
angular class separation, but AdaFace, MagFace, norm-aware losses, hyperbolic
DML, and ranking distillation are mandatory collision controls; no name or
formula is frozen before the geometry audit.

### 2.4 Quality and cost claims

A new learning method uses a single primary endpoint, In-Shop R@1. Secondary
endpoints are disclosed but do not rescue the primary. The directional claim
starts with six paired seeds and a one-sided 95% lower confidence bound. After
seed six, one preregistered sample-size re-estimation may use only the paired
variance (not the observed mean) to reach 80% power at `+0.50` point, up to 12
seeds. The re-estimation powers the one-sided test under alternative
`Delta=+0.50`; the observed-mean floor below is a separate relevance filter.
If more than 12 are required, the experiment is inconclusive and closes.

Quality-improvement acceptance requires:

- mean paired gain at least `+0.50` percentage point;
- one-sided 95% paired confidence interval with lower bound above zero;
- when the audited SOP recipe and budget exist, its non-inferiority margin and
  seed count are frozen from the actual comparison's paired SD measured over at
  least six seeds; if the funded seed count cannot attain that gate, SOP is
  descriptive and the claim remains explicitly In-Shop-only; and
- no material training or inference regression hidden from the Pareto table.

The report states the arm-specific paired standard deviation and minimum
detectable effect. An estimate from three or fewer seeds is never used to
declare adequate power.

The cost-improvement branch uses equivalence rather than failure to reject a
null. It begins with eight paired seeds and requires the 90% paired confidence
interval for In-Shop R@1 to lie entirely inside `[-0.40, +0.40]` point (TOST),
while measured training wall time or encoder latency improves by at least 20%.
One variance-only extension to at most 12 seeds is allowed; inability to attain
equivalence closes the branch. A non-significant difference alone is not
equivalence. Any quality-method regression above 5% in training or inference
cost is material and disclosed; above 20% blocks a quality-only claim.

### 2.5 Zero-training Pareto hypothesis: calibrated tail moment

The prior-art search found no credible unoccupied loss expected to supply the
1.5--4 point gap to the modern frontier. One narrow deployment hypothesis is
cheap enough to falsify before training and is retained without a novelty
claim.

For frozen unit embedding `u in R^D`, fix a train-only basis and deployment
width `d`. Split `u = [h; t]`, where `h = u[:d]`, and retain one additional
scalar `r = ||t||_2`. Fit a single label-blind coefficient on training items:

```text
lambda_raw = sum_(q,g in P) r_q r_g <t_q,t_g>
             / sum_(q,g in P) (r_q r_g)^2
lambda = clip(lambda_raw, 0, 1)
S_lambda(q,g) = <h_q,h_g> + lambda r_q r_g
              = <[h_q; sqrt(lambda) r_q],
                  [h_g; sqrt(lambda) r_g]>.
```

`P` is the fixed top-50 **head-only-inner-product** neighbor set among training
rows; it cannot select on the tail-product response being fitted. Labels are
never used for pair selection or the point coefficient; train identities are
used only for the two-way clustered uncertainty calculation described below.
The result is an ordinary `(d+1)`-dimensional inner
product, compatible with maintained GEMM/FAISS/cuVS kernels. At `d=128`, FP32
gallery storage is 129 values instead of a 512-D anchor, a 3.97x reduction.
The encoder still emits the full descriptor: this is a gallery/search saving,
not an encoder-speed claim. Tail-norm computation, any PCA projection, and the
global PCA matrix are included in build cost, query latency, and storage.

The hypothesis is that deterministic truncation discards a class-correlated
tail product that can be estimated from the two observed tail norms. It also
explains a specific UNICOM behavior: after full normalization and truncation,
unnormalized prefix Euclidean ranking is equivalent for a fixed query to
`<h_q,h_g> + r_g^2/2`, a query-independent tail-energy reward rather than
cosine on a renormalized prefix.

The zero-training test freezes the subset of `{64, 128, 256, 512}` strictly
below the export's full width, the basis (native and train-only PCA), and
`lambda` before test evaluation. Matched row-width controls are
renormalized `(d+1)` truncation, plain `(d+1)` truncation, `lambda=0`, UNICOM's
`r_g^2/2` form, PCA `(d+1)` renormalization, and sign/permuted-pair controls.
The PCA mean and matrix are charged to its total deployment storage, so PCA is
not called same-total-byte even though its row width matches. UNICOM has two
distinct ceilings: the published `official_512` geometry (full normalize,
first 512 coordinates, unrenormalized Euclidean) and a separately labelled
`full_width_768` diagnostic. The published Hyp-ViT 128-D 92.6 row is a labelled
same-storage frontier reference, not a matched control. The test runs first on
the UNICOM export. Cars196 and local PA require explicit adapters for their
existing split-specific archive schemas; they are not passed to the UNICOM
three-split loader. Cars196 remains required before a general claim, and
DADA/VPTSP-G follow only after reproduction.

Native CTM at `d=128` is the primary 129-value candidate. PCA-CTM is reported
as a diagnostic from the frozen basis grid; if it alone passes the quality
gates, it requires its own total-storage Pareto calculation including the PCA
mean and matrix and cannot inherit the native 3.97x claim.

At `d=128`, continuation requires all of:

- at least `+0.30` In-Shop R@1 over renormalized 129-D truncation with a paired
  10,000-resample bootstrap that resamples query identities only while holding
  the gallery fixed, with 95% lower bound above zero;
- no mAP@R loss larger than 0.10 point versus renormalized 129-D truncation;
- recovery of at least 50% of the
  `official_512`-minus-renormalized-129 R@1 gap;
- superiority to every matched row-width quality control, including PCA, while
  separately passing the native CTM total-storage comparison; and
- replication on Cars196 before a general claim.

If renormalized 129-D already matches or exceeds `official_512` R@1, use that
simpler compressed descriptor and close this hypothesis; the 50% recovery
quantity is evaluated only for a strictly positive `official_512` gap. If the
fitted coefficient clips to zero, or its encoded FP32 scalar is identically
zero, close before any query/gallery grid is evaluated. Otherwise, kill it if
a two-way train-identity cluster bootstrap confidence interval for
`lambda_raw` includes or lies below zero. The bootstrap freezes the observed
head-only pairs and weights each pair by the product of the resampled counts of
its query and gallery identities, accounting for both appearances of a row.

The tail null freezes the same head pairs and every row's tail radius, then
permutes only nonzero tail directions with PCG64 seeds 206 through 237. Its
one-sided exact permutation p-value is
`(1 + count(null_lambda_raw >= observed_lambda_raw)) / 33`; continuation
requires `p <= 0.05`. This is a falsification diagnostic rather than a source
of coefficient tuning. Also kill CTM if every
width gains less than 0.10 point, a simpler equal-total-storage control matches it, the
external replication fails, or an exact predecessor ranking by partial inner
product plus a fitted `lambda r_q r_g` term is found. Do not tune width or
`lambda` on test. Composition with INT8 screening is later engineering and does
not rescue a failed descriptor.

## 3. Training-throughput lane

Profile the reproduced anchor before custom code. Benchmark one change at a
time, then their compatible composition:

- BF16/FP16 autocast with registered FP32 reductions;
- fused AdamW or the source-equivalent fused optimizer;
- channels-last where numerically valid;
- `torch.compile` with compile time reported separately;
- the fastest supported attention/MLP primitive for the backbone;
- pinned-memory/asynchronous input transfer and data-loader tuning.

Every comparison fixes images, sampler order, optimizer steps, effective batch,
loss, and seed. Warm-up and compilation are excluded from steady-state
throughput but included in total-run break-even reporting. GPU nondeterminism is
handled with paired repeated runs and disclosed distributions, not promises of
bitwise reproducibility.

The compatible composition is the gated unit. Individual changes are measured
for attribution and retained in the candidate composition when they do not
reduce images/s or force a batch/accumulation change; they need not each clear
20%. The final composition is kept only if:

- steady-state images/s improves by at least 20%;
- end-to-end wall time improves at the registered training budget;
- peak memory does not cause a hidden batch or accumulation change; and
- the Section 2.4 TOST quality gate passes once for the full composition.

A custom Triton/CUDA training kernel is authorized only when profiling shows
an uncovered operator consumes at least 10% of step time and supported PyTorch,
cuDNN, cuBLASLt, or compiler fusion cannot remove it. Otherwise use the
maintained primitive.

## 4. Retrieval workloads and controls

The small In-Shop and SOP galleries are correctness and quality datasets, not a
justification for custom search. At 512 dimensions their raw FP32 galleries are
about 25.8 MB and 124 MB respectively. The preflight computes the lower-bound
scan time as `rows * width * 4 / measured_device_bandwidth` and compares the
measured exact-search p95 with batch-1 encoder p95. The expected outcome is
search well below 10% and closure of custom retrieval work for ordinary image
queries.

The only route that can reopen native search is an actually served workload
with at least 1M rows, a named consumer, a recorded request trace, and a
pre-embedded query stream, such as stored-embedding deduplication or
recommendation. In that profile the denominator is explicitly search plus
transfer p95 for the served request; an image encoder is not fabricated into
the denominator. No such consumer/trace is present in this repository as of
2026-08-12, so Sections 5 and 6 are dormant and custom retrieval work is closed
until external workload evidence exists. A self-created LAION benchmark cannot
reopen it by being renamed a serving profile.

Performance characterization uses:

- the largest real metric-learning gallery available;
- a fixed public real-image subset at 1M and, if storage permits, 10M rows,
  embedded by the same frozen model (for example a registered LAION subset),
  used only for characterization;
- SIFT1M/DEEP1M only as separately labelled standard vector-search controls,
  never as evidence about the model's descriptor distribution;
- synthetic expansions only as separately labelled scaling diagnostics;
- online query batches `1` and `8` as decisive workloads, with `64` and `512`
  reported only as offline throughput characterization;
- `k in {1, 10, 100}` and widths 128, 512, and the model's native width.

For each batch and gallery separately report p50/p95/p99 latency, queries/s,
bytes read, index build time, storage, median/p95/p99 survivor fraction, and
end-to-end encoder-plus-search latency.

Primary controls are:

1. one registered tiled FP32 scoring/top-k primitive on GPU;
2. cuVS or FAISS-GPU exact search on the same device;
3. FAISS CPU Flat;
4. FAISS `IndexFlatPanorama` on CPU, including PCA composition;
5. FAISS `IndexRaBitQ` and `IndexRaBitQFastScan` as approximate compressed
   controls;
6. HNSW/IVF-PQ at a fixed recall/latency grid when approximation is allowed.

CPU and GPU latency are never divided to claim an algorithmic speedup.
Panorama is a composable transform/pruning stage, not merely a rival: benchmark
Panorama alone, quantization alone, and PCA/Panorama followed by quantization.

## 5. Certified mixed-precision exact screen

This is an engineering candidate, not a novel similarity-learning method. It
is attempted only after existing libraries and only for the named 1M+ serving
profile that passes its search-plus-transfer materiality gate. SIFT1M,
DEEP1M, and synthetic/LAION characterization alone cannot authorize it.

### 5.1 Exact reference

Register one scoring primitive that is used both to exhaustively score the
reference gallery and to rerank survivors. Freeze its tile shape, split-K
factor, accumulator dtype, and per-row reduction tree so these are independent
of gallery or survivor count. Set
`torch.backends.cuda.matmul.allow_tf32 = False` and
`torch.set_float32_matmul_precision("highest")`, record those flags plus
`torch.backends.cudnn.allow_tf32`, and use
stable `(score descending, row id ascending)` ordering. Index differences are
permitted only among rows with exactly equal registered FP32 scores and only
according to that stable rule. An ordered FP64 oracle diagnoses containment;
it does not redefine the FP32 target.

A full GEMM reference and a gather-dot reranker are not silently treated as
numerically interchangeable.

### 5.2 First candidate: per-row symmetric INT8

The gallery stores one signed INT8 code in `[-127, 127]` per coordinate, one FP32 scale, one
upward-rounded FP32 residual norm, and one upward-rounded FP32 norm bound per
row. At width 128 this is 140 bytes versus 512 bytes for FP32, so total storage
with the original gallery is 127.3% of FP32. At the anchors' 512-dimensional
width it is 524 auxiliary bytes beside 2,048 FP32 bytes, or 125.6% total.

The query is quantized per query. INT8 products accumulate exactly in INT32:
`d * 127^2 <= 2^31 - 1` for `d <= 133,152`; at width 128 the worst magnitude is
`2,064,512`. Dequantization and all interval arithmetic use outward-rounded
FP32 operations. A higher-precision query is a secondary variant only if an
actual GB10 cuBLASLt/Triton path supports it faster than the INT8 control; the
spec does not assume a nonexistent mixed datatype tensor-core instruction.

For FP32 `q` and gallery row `g_j`, decoded approximations `q_hat`, `g_hat_j`,
and conservative residual bounds satisfy

```text
epsilon_q   >= ||q - q_hat||_2
epsilon_g_j >= ||g_j - g_hat_j||_2
G_j         >= ||g_j||_2

s_hat_j = <q_hat, g_hat_j>
E_j = epsilon_q * G_j + ||q_hat||_2 * epsilon_g_j + delta_numeric_j
L_j = s_hat_j - E_j
U_j = s_hat_j + E_j.
```

`G_j` and `||q_hat||_2` are measured and upward-rounded; unit norm is not
assumed. The registered `delta_numeric_j` covers scale representation, exact
INT32-to-FP32 conversion, dequantization, final FP32 operations, and a
prospective forward-error bound for the registered FP32 reference reduction
schedule. Its derivation is prospective and stress-tested against the FP64
oracle; it is never fitted to top-k success.

Let `theta` be the kth-largest `L_j`. Rows with `U_j >= theta` survive. The
registered FP32 primitive reranks every survivor, so a valid interval returns
the exact registered top-k and keeps all boundary ties.

Per-block scaling, FP8, and FP4 are later candidates only if they beat per-row
INT8 on measured GB10 latency after accounting for split reductions and their
wider interval bounds.

### 5.3 Zero-training falsifier

The query partition is disjoint from the indexed gallery and registered before
any interval is measured; identity labels are not required. Before kernel work,
a reference implementation measures each batch and gallery size separately. It
derives a predicted survivor distribution from
the registered interval widths and the held-out empirical score distribution,
before reading correctness outcomes.

The candidate continues only where all are true:

- every interval contains the FP64 diagnostic score and every certified top-k
  equals the registered FP32 top-k;
- for both batches 1 and 8, median/p95/p99 survivor fractions are at most three
  times their prospective predictions and also below absolute ceilings
  5%/10%/20%;
- predicted total bytes, including code scan and FP32 survivor reread, are at
  most 50% of exhaustive FP32 for each batch; batch-8 FP32 reread uses the
  deduplicated union of its eight per-query survivor sets;
- combined FP32 plus auxiliary storage is at most 130% of one FP32 gallery;
- the named pre-embedded workload passed the 10% search-plus-transfer
  materiality gate; and
- no choice depends on test labels, test queries, or post-outcome thresholds.

Compare the answer-to-byte frontier against Panorama, RaBitQ, and composed
Panorama-plus-quantization. Bytes may be compared across backends; latency may
not. SIFT1M is an optimistic diagnostic because its non-negative, effectively
8-bit coordinates flatter symmetric INT8 and is never decisive. Do not require
one method's first bound to be tighter than another's differently staged bound.

## 6. Native retrieval kernel gate

Only a passing falsifier and profiler authorize a custom kernel. The candidate
may fuse INT8 scoring, outward intervals, kth-lower selection, survivor
compaction, and exact reranking. Transfers, query transforms, FP32 gallery
access, and top-k are inside the timed boundary.

For each decisive batch independently, success requires:

- exact registered top-k on every correctness query;
- at least 1.5x p95 latency speedup over the strongest exact same-device GPU
  control on the named real 1M+ pre-embedded serving profile;
- a total-byte point on or above the best matched-exactness frontier from
  Panorama, RaBitQ with exact refinement, and their composition; same-device
  restrictions apply to latency, not byte accounting;
- the Section 5 survivor and 130% storage gates;
- positive search-plus-transfer p95 improvement for the pre-embedded profile;
- no regression at another claimed batch hidden by an aggregate throughput
  number.

Panorama comparisons remain CPU-vs-CPU unless a supported same-device backend
exists. RaBitQ and ANN comparisons report recall and are not called exact.
Failure deletes the custom kernel branch; it does not trigger threshold tuning.

## 7. Staged budget and sequencing

1. Finish the existing PA/MCPS, compactness, and UNICOM queue unchanged.
2. Reproduce UNICOM zero-shot and run the DADA six-epoch compatibility smoke.
   Extrapolate its measured steps/s to the exact 28,800-step full run and freeze
   the GPU-hour budget before launching it.
3. Reproduce one primary trainable anchor; run VPTSP-G only as the planned
   transformer cross-check, not as an unbounded sweep.
4. Run the frozen-descriptor geometry audit, including the calibrated-tail
   zero-training falsifier, and profile training/search cost.
5. Apply maintained training optimizations and evaluate their cost branch.
6. Close retrieval kernel work immediately if real online search is below 10%
   of end-to-end p95.
7. Otherwise benchmark existing FAISS/cuVS/Panorama/RaBitQ controls and run the
   zero-training interval falsifier.
8. Write custom training or retrieval kernels only after their respective
   profiler and falsifier gates pass.
9. Propose one quality method only from a diagnosed modern-anchor failure, then
   preregister the Section 2.4 powered paired experiment and matched prior-art
   controls.

No hyperparameter sweep is authorized by this design. Each failed stage closes
its branch or returns to diagnosis; it does not widen a gate.

The post-queue ceiling for this program is 120 GB10 GPU-hours: at most 40 for
the primary DADA reproduction, 25 for the VPTSP-G cross-check, 5 for exports and
geometry/search profiling, at most 40 for training-speed evidence, and 10
reserve. For a TOST claim the projected cost is registered exactly as
`2 * n_seeds * measured_full_run_hours` from the six-epoch smoke, less only
already reusable, protocol-identical baseline seeds. If the remaining ceiling
cannot fund the Section 2.4 required `n`, the equivalence-claim branch closes;
the margin is never widened and a throughput-only microbenchmark remains
descriptive. If the total ceiling binds, drop VPTSP-G first, then an unaudited
SOP extension, then non-composed performance ablations. A later quality
invention receives a separate preregistered budget.

Before scheduling retrieval controls, pin and build/import FAISS and cuVS for
the GB10 (`sm_121`, CUDA 13). If either is unavailable, disclose it and use the
registered tiled FP32 primitive as the strongest exact same-device fallback;
do not invent a missing baseline result. If the pinned source lacks a named
class, drop that comparison; any later cross-backend byte gate names only the
remaining authenticated controls.

## 8. Claims and stop rules

Allowed after evidence:

- a locally reproduced modern operating point;
- a paired quality improvement over that exact anchor;
- a quality-equivalent training speedup under TOST;
- an exact same-device retrieval speedup with identical registered top-k;
- a deployment-storage Pareto point, stated as `X` R@1 at `d+1` FP32 values
  versus `Y` R@1 at the registered deployment anchor, with recovered
  truncation-gap fraction, matched row-width controls, and total storage
  (including any PCA basis) disclosed;
- a combined Pareto point whose quality and systems components each passed.

Stop or prohibit:

- do not call CRT, supervised UNICOM, or VPTSP-G 96.5 locally reproducible
  before their recipes are actually reproduced;
- do not call three seeds a powered superiority test;
- do not interpret a non-significant quality difference as equivalence;
- do not build a custom search kernel for In-Shop/SOP-scale serving when search
  is below 10% of online p95;
- do not compare CPU Panorama latency with GPU latency as an algorithmic ratio;
- do not claim novelty for Panorama, RaBitQ, ordinary quantization, or dense
  low-precision GEMM;
- do not keep a custom kernel that misses its preregistered gate;
- do not hide failed seeds, final checkpoints, memory, build time, or storage.

## 9. Primary evidence

- UNICOM official code and ICLR 2023 paper.
- DADA official `configs/inshop.yaml` and AAAI 2024 paper.
- VPTSP-G official `config/inshop_basic.yaml` and ICLR 2024 paper.
- Panorama paper and [FAISS](https://github.com/facebookresearch/faiss) release
  `v1.13.2` (`070fbcdd93ee086ddacb1bbd0aa078e31864cdd9`) plus inspected upstream
  commit `a424dcb809fd725c44dd976d9063febd4837d16a`, specifically
  `faiss/IndexFlat.{h,cpp}` and
  `faiss/impl/Panorama.{h,cpp}`.
- RaBitQ SIGMOD 2024 paper and the same FAISS revision, specifically
  `faiss/IndexRaBitQ.{h,cpp}`, `faiss/IndexRaBitQFastScan.{h,cpp}`, and their IVF
  variants.
- PDX, FEXIPRO, L2AP, and ADSampling as exact/partial-distance prior art. BOND
  is omitted until an unambiguous primary citation is audited.

This program starts from maintained implementations and reproducible evidence.
It does not rename occupied work, and it does not let systems wins disguise a
quality gap.
