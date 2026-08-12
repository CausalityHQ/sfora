# PE-HTGC PartialFC: Positive-Exact Gauge-Calibrated Feature Sampling

**Status:** approved for deterministic falsification; training is conditional on
the gates below

**Target:** improve UNICOM-style supervised image retrieval without changing
the backbone, descriptor width, class sampling, or retrieval protocol, while
repairing a source-confirmed inconsistency in class-sharded feature sampling.

## 1. Decision

The selected candidate is **PE-HTGC PartialFC**. It combines three changes that must
remain separately attributable:

1. **coherent masks:** every class shard evaluates the same sampled feature
   coordinates on a distributed step; and
2. **positive-exact margins:** each sample's target-class cosine is computed in
   all `D` coordinates before ArcFace; and
3. **gauge-calibrated negatives:** the large negative-class matrix uses
   inverse-probability-scaled sampled inner products divided by full-vector
   norms rather than independently re-normalizing each sampled subspace.

The first change is a distributed-correctness repair and is a no-op at world
size one. The positive-exact/HT-negative estimator is the research mechanism
and remains testable on one GPU. Neither change is described as a SOTA
improvement until it beats the controls under the evidence ladder in Section 9.

The operator has instructed the research loop to choose the recommended path
without waiting for repeated decisions. This design therefore proceeds to its
falsifier after review; it does not authorize a training claim in advance.

## 2. Source-confirmed problem

The inspected official UNICOM revision is
`d71992ed969e6c271436ac0a0ee1f3ca61474ac0`.

For an output dimension `D=768` and `sample_num_feat=K=512`,
`PartialFC_V2.forward` does the following on every rank:

1. all-gathers the full embeddings and labels;
2. owns a contiguous shard of class prototypes;
3. independently draws `K` coordinates with the rank-local default CUDA RNG;
4. selects those coordinates from embeddings and local prototypes;
5. separately normalizes both selected vectors;
6. computes local class-shard logits; and
7. combines the shard logits through one distributed softmax.

The feature axis is replicated while the class axis is sharded. Consequently,
one softmax compares class logits computed in different random coordinate
systems. Reassigning class indices to ranks can change a realized loss even
though that reassignment has no semantic meaning.

The implementation is stochastic rather than algebraically undefined, but its
objective inherits an unintended dependency on world size and class-shard
placement. Independent masks also have an accidental advantage: with four
ranks they touch a coordinate with probability

```text
1 - (1 - K/D)^4 = 80/81 = 98.765% per step,
```

whereas one globally coherent mask touches only `K/D = 66.667%`. Any experiment
that compares those paths without a coverage control is confounded.

The official evaluator creates a second mismatch: it normalizes all 768
coordinates, keeps the first 512, and then uses Euclidean distance without
renormalizing the truncated vectors. Training instead selects first and then
normalizes. The evaluator reports R@1 only; an R@10 claim requires a corrected,
explicit evaluator extension.

## 3. Alternatives considered

### 3.1 Globally synchronized UNICOM masks

Broadcast one uniform `K`-of-`D` mask and keep UNICOM's selected-subspace
cosine. This restores exact sharding invariance for a realized step. It is the
necessary distributed control, but it is established replicated-tensor RNG
hygiene, is a no-op on one GPU, and reduces per-step coordinate coverage.

### 3.2 Multi-mask selected-subspace averaging

Evaluate several coherent masks on the same batch and average their losses.
This reduces within-step mask variance and restores coordinate coverage, but it
multiplies classifier work and collides closely with multi-sample dropout and
multi-mask gradient averaging. It remains a variance/coverage control.

### 3.3 PE-HTGC PartialFC (selected)

Use a coherent mask, compute the target-class full-space cosine exactly, and
estimate only negative full-space cosine numerators with Horvitz–Thompson
inverse-probability scaling while retaining exact full-vector norms. This
removes the random selected-norm temperature, gives ArcFace an exact bounded
positive cosine, keeps almost all of the `K/D` classifier-matmul saving, and is
measurable at world size one. Its known failure mode is negative-logit
variance: an estimate can leave `[-1, 1]` and require clipping. That failure is
measured prospectively rather than hidden.

### 3.4 Full-768 PartialFC

Disable feature sampling. This is the simplest scientific control and may be
the best method despite 50% more feature-side classifier work than `K=512`.
PE-HTGC has no value if it cannot outperform or meaningfully match this control at
lower measured cost.

## 4. Method

Let `z_i in R^D` be an embedding, `w_c in R^D` a class prototype, and `R` a
uniform subset of `K` coordinates sampled without replacement. Let
`pi = K/D`. Define

```text
n_ic(R) = sum_{j in R} z_ij w_cj
d_ic    = clamp(||z_i||_2, 1e-12) * clamp(||w_c||_2, 1e-12)
h_ic(R) = (1/pi) * n_ic(R) / d_ic
q_iy    = <z_i, w_y> / d_iy
```

The candidate pre-margin logit is

```text
s_ic(R) = q_iy                                      if c = y_i
          clamp(h_ic(R), -1 + eps_arc, 1 - eps_arc) otherwise
```

with `eps_arc` equal to the smallest frozen FP32 safety constant that keeps the
existing ArcFace `arccos` finite for estimated negatives. The initial
implementation uses `eps_arc=1e-7`; alternatives are not tuned after outcome
inspection. Exact positive cosines use the existing numerical clamp only for
roundoff at the unit boundary; a positive value outside `[-1-1e-6, 1+1e-6]`
is a structural failure rather than an estimator clipping event.

Because every coordinate has inclusion probability `pi`, before clipping,

```text
E_R[h_ic(R) | z_i, w_c] = <z_i, w_c> / (||z_i||_2 ||w_c||_2).
```

For negative classes, the expected sampled **logit** and, under ordinary
interchange conditions, its gradient equal the full-space cosine logit and
gradient. The positive logit and its gradient are exact for every mask. The
nonlinear cross-entropy of sampled negative logits is not claimed to be an
unbiased estimator of the full-softmax loss. Rank-shared masks are still
required: unbiased marginal logits do not remove the class-partition
dependence induced by rank-correlated logit noise inside `logsumexp`.

Computing exact positives adds `O(BD)` work for a global batch of `B`; the
negative classifier remains `O(B C K)` for `C` classes. The cost is negligible
when `C` is large but is measured rather than assumed.

The ArcFace margin, scale, class sampling, distributed cross-entropy, optimizer,
backbone, image transforms, and training schedule remain unchanged.

## 5. Distributed mask contract

At every step, rank zero draws exactly one sorted `int64` mask of `K` unique
coordinates from `[0, D)`. The mask is broadcast to all ranks before prototype
or embedding selection. All ranks verify an identical mask hash.

The implementation must not infer coherence from equal seeds. It uses an
explicit collective so that RNG consumption elsewhere cannot desynchronize
the classifier.

The official independent-rank path remains available as a named control. It is
never silently replaced in a baseline run.

## 6. Evaluation contract

One frozen 768-dimensional embedding export supports three no-training views:

1. `official_512`: normalize 768, slice `[0:512]`, Euclidean distance;
2. `prefix_unit_512`: slice `[0:512]`, normalize 512, Euclidean distance; and
3. `full_unit_768`: normalize 768, Euclidean distance.

For every view, compute Recall@1, Recall@10, Recall@20, Recall@30, and mAP@R.
Recall@K means that at least one of the first `K` gallery identities equals the
query identity. Stable gallery order breaks exact distance ties. No test-time
augmentation, gallery fitting, reranking, or learned post-processing is used.

The corrected candidate deployment view is selected before training:

- use `prefix_unit_512` when the intended product descriptor is 512-D;
- use `full_unit_768` only in a separately labelled 768-D lane.

The higher-scoring test view is not chosen after seeing candidate training
outcomes.

## 7. Deterministic falsifier

The first implementation is a CPU-only functional and four-process Gloo test.
It uses fixed FP64 tensors for the mathematical oracle and fixed FP32 tensors
for production-parity checks.

### 7.1 Probe A: estimator identity

For energy-concentrated embeddings and prototypes, enumerate all `K`-of-`D`
masks at `D=8, K=4`.

- the mean unclipped HT negative logit must equal the full cosine to `1e-12` in FP64;
- its mean gradient must equal the full-cosine gradient to `1e-11`;
- the positive logit and gradient must equal full cosine for every mask;
- the official selected-subspace cosine must show a nonzero bias on at least
  one frozen pair; and
- the test must fail if the `D/K` factor or full norms are removed.

### 7.2 Probe B: class-sharding invariance

Use four Gloo ranks, eight classes, a fixed global batch, and a fixed shared
mask. Compute the distributed loss under two class-to-rank permutations while
permuting prototypes and labels consistently.

- synchronized official logits and synchronized PE-HTGC logits must be invariant
  to `1e-6` in FP32;
- independent-rank masks must violate invariance by at least `1e-4` for the
  frozen adversarial fixture; and
- the same global tensors at world sizes one and four must agree for the
  coherent paths to `1e-6`.

The fixture is frozen before running the test against production code. It may
be constructed analytically, but it cannot be selected from candidate training
outcomes.

### 7.3 Probe C: variance and clipping

Across 4,096 frozen masks on deterministic synthetic tensors, record:

- logit bias and variance;
- gradient mean-squared error versus full-768;
- fraction of negative logits that would clip;
- exact-positive numerical-bound violations; and
- coverage for one, two, three, and four coherent masks.

PE-HTGC proceeds only if all are true:

1. absolute mean logit bias is at most `1e-4` before clipping;
2. clipped negative fraction is at most `0.02` and positive bound violations are zero;
3. gradient MSE is lower than official selected-subspace cosine; and
4. no nonfinite loss or gradient occurs.

If negative clipping exceeds the threshold, single-mask PE-HTGC closes. A multi-mask PE-HTGC
variant is a new conditional arm, not an automatic threshold relaxation.

## 8. No-training checkpoint gate

Before any UNICOM fine-tuning, reproduce the released ViT-B/16 zero-shot
In-Shop result and evaluate the three frozen views from identical cached
embeddings.

Define paired query-bootstrap differences using 10,000 `PCG64(206)` resamples.
The evaluation correction is adopted only if `prefix_unit_512` exceeds
`official_512` by at least `0.001` R@1 and its 95% paired lower bound is
positive. Otherwise it remains a reported diagnostic rather than a claimed
improvement.

This gate cannot establish a supervised SOTA result and cannot authorize
selecting a different prefix.

## 9. Training evidence ladder

Training begins only after all deterministic probes pass.

### 9.1 Eight-epoch seed-zero smoke

Run from the same official initialization:

- `U-OFF`: official independent-rank selected-subspace cosine;
- `U-SYNC`: coherent selected-subspace cosine;
- `PE-HTGC-1`: coherent one-mask positive-exact HTGC;
- `FULL-768`: no feature sampling.

On one GPU, `U-OFF` and `U-SYNC` are byte-semantic aliases and only one is
executed. On at least four GPUs, both are required. Record clipping, coordinate
coverage, throughput, peak memory, and final registered metrics.

PE-HTGC passes the smoke only if:

1. final R@1 improves over the matched official control by at least `0.002`;
2. final R@10 is not lower than the control;
3. negative clipping remains below `0.02` and positive bound violations remain zero;
4. epoch time is no more than `1.10` times the official sampled path; and
5. all losses and gradients remain finite.

### 9.2 Conditional variance control

If PE-HTGC-1 is finite and directionally positive but its paired improvement is
below `0.002`, run `PE-HTGC-M3`: compute the exact positive once, evaluate
negative logits under three coherent masks on the same batch, average the three
ArcFace losses, and perform one optimizer step. This arm is explicitly a
multi-sample variance control and carries no standalone novelty claim.

PE-HTGC-M3 survives only if it beats PE-HTGC-1 and its measured gain justifies its
classifier-time increase. Otherwise the multi-mask branch closes.

### 9.3 Full paired experiment

Only a passing smoke proceeds to the official 128-epoch ViT-B/16 In-Shop
recipe for seeds 0, 1, and 2. GPU training is statistically reproducible rather
than bitwise deterministic. All arms use paired initializations, data order,
checkpoint epochs, and evaluation.

The candidate counts as a matched mechanism improvement only if:

- its final-checkpoint R@1 paired delta versus every cheaper surviving control
  is positive in at least two of three seeds;
- its mean paired R@1 delta is at least `0.002` and exceeds its sample standard
  error;
- R@10 and mAP@R do not regress by more than `0.001` in mean;
- the clipping, cost, and finite-value gates remain satisfied; and
- no best-epoch or test-selected checkpoint replaces the registered final
  checkpoint.

`FULL-768` is a mandatory control. If it matches or beats PE-HTGC at acceptable
cost, the sampling mechanism is not the preferred method.

## 10. SOTA boundary

Published supervised UNICOM In-Shop R@1 is 95.5 for ViT-B/16, 96.0 for
ViT-L/14, and 96.7 for ViT-L/14@336. The released B/16 checkpoint is a
pretrained zero-shot model, not the fine-tuned 95.5 checkpoint.

A SOTA or top-10 claim requires all of the following:

1. reproduce the exact comparable official UNICOM training point;
2. exceed it with the same backbone, input size, descriptor width, data,
   pretraining, and inference lane;
3. report paired multi-seed uncertainty and all registered retrieval metrics;
4. replicate the direction on at least two of CUB, Cars196, and SOP; and
5. compare against current descriptor-only systems without mixing rerankers,
   ensembles, transductive scoring, or different pretraining budgets.

Until those conditions hold, the result is a source repair, a deterministic
mechanism result, or a matched In-Shop improvement—not SOTA.

## 11. Prior-art boundary

- UNICOM already samples feature coordinates for embedding/prototype margin
  softmax; that component is not new.
- Megatron-LM already requires synchronized RNG for stochastic operations over
  replicated tensor regions; mask broadcast is a correctness repair.
- Horvitz–Thompson inverse-probability estimation is classical and is not new.
- Multi-sample dropout and multi-mask gradient averaging occupy loss averaging
  over several masks; PE-HTGC-M3 is only a control.
- Nested Dropout, Slimmable Networks, and Matryoshka Representation Learning
  occupy ordered/nested deployable subspaces; PE-HTGC uses uniform coordinates and
  does not claim that territory.
- Partial FC samples classes rather than feature coordinates.

The narrow research claim, if experiments support it, is that combining a
globally coherent feature mask, an exact positive cosine, and full-norm
inverse-probability negative cosines corrects the gauge and selected-norm
defects of coordinate-sampled, class-sharded margin softmax while retaining
nearly all of its classifier-compute saving.

## 12. Components

Implementation has four isolated units:

1. a pure sampled-logit function with official and PE-HTGC modes;
2. a distributed mask provider with an explicit broadcast contract;
3. a CPU/Gloo falsifier with fixed fixtures and JSON output; and
4. an evaluator that computes the frozen official, corrected-prefix, and
   full-vector metrics from one embedding export.

The initial falsifier does not train a model, modify the official checkpoint,
or launch a GPU process. Ordinary Git commits, tests, and experiment reports
are sufficient; no separate handoff-authentication framework is introduced.

## 13. Failure handling

- Duplicate/out-of-range masks, zero or nonfinite full norms, nonfinite logits,
  inconsistent rank mask hashes, incomplete retrieval splits, or inconsistent
  source/model identifiers invalidate an attempt.
- Existing reports are never overwritten.
- Failed gates close the exact arm; they do not authorize threshold tuning,
  seed replacement, or post-hoc switching between 512-D and 768-D evaluation.
- A new mechanism after closure receives a new design rather than inheriting a
  favorable subset of this candidate's evidence.

## 14. Primary references

- UNICOM, ICLR 2023: <https://openreview.net/forum?id=3YFDsSRSxB->
- Official UNICOM source: <https://github.com/deepglint/unicom>
- Partial FC, CVPR 2022: <https://arxiv.org/abs/2203.15565>
- Megatron-LM: <https://arxiv.org/abs/1909.08053>
- Horvitz and Thompson, JASA 1952:
  <https://www.jstor.org/stable/2280784>
- Multi-Sample Dropout: <https://arxiv.org/abs/1905.09788>
- Nested Dropout, ICML 2014:
  <https://proceedings.mlr.press/v32/rippel14.html>
- Slimmable Neural Networks, ICLR 2019:
  <https://openreview.net/forum?id=H1gMCsAqY7>
- Matryoshka Representation Learning, NeurIPS 2022:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html>
