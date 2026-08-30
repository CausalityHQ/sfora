# Token-Set Proxy Anchor: preregistered SFORA method search

Status: **F0 implementation in progress; Cars test split remains fenced.**

## Target and evidence boundary

The current published Cars196 zero-shot retrieval horizon is SAGA at **97.0 ± 0.3
R@1** over three seeds (arXiv:2606.15134, Table 1). It uses the Qwen3-VL-8B
vision tower, a 4096-dimensional pooled descriptor, and a frozen MLLM training
supervisor. The older 94.9 AdvRF result is no longer the target.

SFORA's archived `google/siglip-base-patch16-224` report is useful but burned
diagnostic evidence, not a new claim. Under the official 98/98 class split it
records 96.4580 R@1 for the frozen 768-dimensional pooler and 96.7409 for a
train-only projection. The report predates model-revision and source-recipe
authority, and its Cars test outcomes have already influenced this search.

The new method must satisfy both conditions:

1. final three-seed Cars mean R@1 is at least **97.4**, placing the point estimate
   above SAGA's reported mean plus its 0.3-point seed deviation; and
2. its paired mean gain over the same-backbone pooled control is at least **0.5
   point**. If the backbone crosses the horizon but the method does not meet this
   paired gate, the result is recorded as a substrate crossing, not a method win.

No Cars test image, embedding, nearest-neighbour identity, or label-dependent test
statistic may enter F0 or F1. The final method/configuration is sealed before F2.

## Method: TSPA

Token-Set Proxy Anchor (TSPA) preserves localized fine-grained evidence through
training and retrieval rather than pooling it away.

For one image, a pinned SigLIP vision encoder emits patch tokens
`h_i in R^768` and its global pooled descriptor. A saliency head initialized from
the pretrained attention pooler selects `K=32` tokens with deterministic
lowest-index ties. The head maps selected tokens to normalized
`z_i in R^128`, maps the global descriptor to normalized `g in R^512`, and assigns
normalized nonnegative saliency weights `a_i`.

Each training class has a global proxy `gamma_c` and a set of `M=16` normalized
token proxies `p_cj`. The image-to-class score is

```text
s(x,c) = (1-lambda) <g_x, gamma_c>
       + lambda sum_i a_xi max_j <z_xi, p_cj>,       lambda = 0.25.
```

Proxy Anchor acts on this score with its fixed official `alpha=32` and
`delta=0.1`. A diversity hinge penalizes token-proxy pairs whose cosine exceeds
0.5. The diversity hinge enters the total objective with fixed coefficient 0.1.
Mean within-class token-proxy cosine above 0.95 is a preregistered collapse:
even a high headline then cannot support a TSPA mechanism claim.

At evaluation, each image keeps one global vector and its 32-token set. The
single-stage, gallery-independent pair score is

```text
S(x,y) = (1-lambda) <g_x,g_y>
       + lambda/2 [sum_i a_xi max_j <z_xi,z_yj>
                  + sum_j a_yj max_i <z_yj,z_xi>].
```

There is one model, one image view, no test-time fitting, no gallery adaptation,
no candidate-generation stage, and no reranking. The larger multi-vector storage
lane is disclosed explicitly.

## What is and is not novel

- DIML (ICCV 2021) performs optimal-transport structural matching only on a
  globally retrieved top-K candidate set. TSPA's learned token-set score is the
  sole retrieval metric and uses no first-stage shortlist.
- SFORA `region_proxy_anchor` trains fixed ResNet grid regions against one global
  proxy and previously lost 3.6 points. TSPA instead uses transformer patch
  tokens, learned saliency, a set-valued class proxy, and a global/set residual
  composition. The fixed-grid arm remains a negative control.
- SFORA `proxy_anchor_subcenter` represents each class with multiple global
  centers but each image with one global vector. It cannot express token-part
  conjunctions.
- SAGA uses attribute-aware MLLM supervision but deploys one pooled descriptor.
  TSPA has no MLLM and keeps the learned token decomposition at inference.
- ColBERT supplies the cross-field MaxSim precedent, but not class proxies,
  class-disjoint visual metric learning, or the global/set residual operator.

The novelty claim is the conjunction of learned token-set class proxies,
single-stage symmetric image-set retrieval, and paired evidence that this
mechanism adds at least 0.5 point over the identical pooled foundation encoder.

## Custom kernel

`fused_set_maxsim` streams one query/gallery image pair per Triton program. It
computes the `K x M` token dot products, both directed maxima, and both weighted
reductions without materializing a `(B,N,K,M)` tensor. The CPU eager operator is
the scientific oracle. CUDA parity is mutation-locked against that oracle before
any GPU screen.

F0 requires only the forward kernel. F1 adds an autograd kernel whose backward
stores deterministic lowest-index winners and scatter-adds gradients only through
those winners. Until backward parity is proven, training must use the eager
reference rather than silently claiming a fused training path.

## Falsifier chain

### F0: frozen token-set viability

- Authority: `google/siglip-base-patch16-224` at exact model commit
  `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.
- Data: Cars **train** classes 82..97 only; classes 98..195 are inaccessible.
- Selection: top 32 final-layer tokens using the pinned pretrained attention
  pooler's weights; no learned parameters.
- Scores: pooled cosine, pure symmetric token MaxSim, and fixed `lambda=0.25`
  hybrid.
- Pass gates: pooled R@1 >= 82%; pure set R@1 >= pooled - 1 point; hybrid R@1
  >= pooled. Failure rejects TSPA before training.

### F1: mechanism screen

Precompute pinned tokens for Cars train classes only. F0 has burned classes
82..97, so F1 may not use them for selection. Use classes 0..48 to train and
49..81 as the untouched validation band. Compare three paired seeds of:

1. the same SigLIP substrate with a pooled Proxy Anchor head;
2. TSPA;
3. token-shuffled TSPA, preserving parameters and kernel work while destroying
   part identity.

For the shuffled arm, each seed freezes one individually mixed bijection over the
F1 training examples such that every image receives the complete token set and
pretrained attention weights of an image from a different class. A fixed 64N
sequence of seeded, validity-preserving swaps mixes the initial derangement, and
each Cars training class must receive tokens from at least eight distinct source
classes. Global features and labels remain attached to the original image.
Validation is never shuffled. This keeps the token branch shape and work
identical while removing class-consistent local evidence; permutation within one
unordered token set or a coherent class-to-class relabeling would be vacuous.

The paired seeds are 17, 29, and 43. Each arm trains its frozen-feature head for
40 epochs with batch size 128 and AdamW (`lr=3e-4`, weight decay `1e-4`), using
the exact official Proxy Anchor `alpha=32`, `delta=0.1`; TSPA additionally uses
the fixed diversity coefficient 0.1, margin 0.5, and collapse threshold 0.95.
Global/token projection dimensions remain 512/128 with 16 token proxies per
class. Frozen token features are retained in float32. There is no checkpoint
selection: only the final epoch is evaluated. CUDA execution requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32-disabled PyTorch matmuls, and IEEE-input
Triton dot products.

Proxy Anchor is the query-to-class proxy surrogate; it does not directly
optimize the symmetric image-to-image MaxSim expression. F1 therefore supports
the mechanism only through final retrieval plus the paired token-shuffle
contrast, not by claiming that the deployment score itself was a pair loss.

Proceed only if TSPA exceeds pooled control by at least 0.5 point on average,
exceeds token-shuffled by at least 0.5 point, and token proxies do not collapse.
Both paired gains must also be nonnegative at every seed; the receipt records
the three paired deltas and their population standard deviations.
All hyperparameters and the final source digest are then sealed.

### F2: one final Cars qualification

Train the sealed pooled control and TSPA on all Cars train classes 0..97 for three
paired seeds. Evaluate immutable final checkpoints once on test classes 98..195.
No checkpoint selection, adaptive rerun, or test-conditioned configuration is
permitted. A method claim requires mean TSPA R@1 >=97.4 and paired gain >=0.5
point. Otherwise continue the research search; do not relabel a substrate win.
