# Amortized Local-Scale Potential (ALSP) Design

## Objective

Test whether the gallery-local density defect observed on frozen In-Shop and
Cars196 embeddings can be learned from training data and transferred to unseen
classes as a unary retrieval potential. The target is a composable similarity
method, not another transductive reranker:

\[
  S(q,g)=2\langle z_q,z_g\rangle-\widehat\mu(g),
\]

where `z_q` and `z_g` are unit embeddings and `mu_hat(g)` is predicted without
reading the test gallery graph. The query-side potential is omitted because it
is constant for all candidates of one query and cannot change ranking.

The working name is **Amortized Local-Scale Potential (ALSP)**. This is a
mechanism probe and item-bias baseline, not the final novelty or a SOTA claim.
An independent cross-field review identified the stronger learning candidate as
an equipotential constraint that reduces cross-sample variance of local
potential; ALSP first tests whether such a potential is representable from the
embedding at all.

## Evidence that motivates the candidate

The fixed transductive score

\[
  S_{oracle}(q,g)=2\langle z_q,z_g\rangle-\mu_{50}(g)
\]

improved Recall@1 on three frozen embedding pairs:

- published Proxy Anchor In-Shop: `+0.0020396680`, exact paired `p=0.02148`;
- reproduced Proxy Anchor seed 0 In-Shop: `+0.0024616683`, exact paired
  `p=0.01222`;
- prospectively fixed Cars196 PFML: `+0.0129135408`, exact paired
  `p=2.97e-05`.

Here `mu_50(g)` is the mean cosine similarity from `g` to its 50 closest
nonself gallery neighbours. These results establish a cross-dataset geometry
defect, but the score itself belongs to established local-scaling/CSLS prior
art and is not novel.

## Mechanism

The chemistry analogy is a learned chemical potential. `mu_50(g)` measures the
local crowding energy of a gallery item. A crowded item receives an unfairly
high chance of becoming a nearest neighbour; subtracting its potential corrects
that response bias. ALSP asks whether the embedding already contains enough
information to predict that potential from the item alone.

For the first falsifier, the predictor is deliberately simple and convex:

\[
  \widehat\mu(z)=w^Tz+b.
\]

`w` and `b` are fitted by ridge regression on frozen training embeddings and
their train-only `mu_50` targets. A nonlinear head is not authorized in this
first experiment because it would add tuning freedom before predictability is
established.

If the frozen falsifier passes, the learned method is a scalar head attached to
the embedding network. During training it predicts a stop-gradient local-scale
teacher computed from a training memory bank. At inference the head emits the
unary potential directly; no test-gallery statistics, labels, adaptation, or
reranking are used.

## Alternatives considered

1. **ALSP (selected):** a scalar unary potential distilled from train-only
   neighbourhood density. Cheapest decisive test and directly grounded in the
   replicated defect.
2. **Local anisotropic precision:** predict a low-rank local precision matrix
   and use a symmetrized anisotropic similarity. It may model more than density,
   but is substantially costlier and overlaps probabilistic/uncertainty-aware
   embeddings. It is deferred unless a scalar potential is insufficient.
3. **Homeostatic density equalization:** penalize memory-bank density or
   k-occurrence variance during training. This overlaps established
   uniformity/hubness regularization, and repository evidence already shows
   that reducing hubness can lower CUB Recall@1. It is rejected as the first
   continuation.

## Frozen-embedding falsifier

### Inputs

Use only the corrected Proxy Anchor seed-0 train/query/gallery archives. The
official published checkpoint lacks a coordinate-compatible training archive
and therefore cannot be used to fit this predictor. Input paths and SHA-256
digests must be persisted in the result.

### Split and targets

1. Sort distinct train labels by `SHA256(int64_label_bytes), label`.
2. Assign the first 80% of labels to fit and the remainder to validation. A
   label may not occur in both partitions.
3. Compute `mu_50` once for every train row against the complete train pool,
   excluding self with an explicit row identity. The label split controls model
   fitting and validation; it must not redefine two different density pools.
4. Standardize target values using fit-partition statistics only for numerical
   fitting, then invert that transform so every predicted potential is in the
   original cosine-similarity units.

### Predictor selection

Fit ridge models with an unregularized intercept for the fixed grid
`lambda in {1e-6, 1e-4, 1e-2, 1, 100}`. Normalize both `Xc.T @ Xc` and
`Xc.T @ yc` by the fit-row count before applying `lambda`, so the grid has a
sample-count-independent meaning. Select the model with the lowest validation
mean squared error; break ties by the earlier grid entry. No test
embedding, test density, label, or retrieval result may participate in this
selection.

The selected predictor is then refit on all training rows using the selected
lambda and train-only targets. The inference potential is the predicted raw
`mu_50`; there is no fitted test-time weight, clipping, or monotone transform.

### Test evaluation

Evaluate exactly once on the compatible seed-0 query/gallery pair:

- raw cosine;
- ALSP prediction;
- constant-potential identity control;
- a predictor fit to one fixed PCG64 permutation of train targets;
- 20 fixed-PCG64 random-direction unary potentials, each centered and scaled to
  the ALSP prediction's gallery mean and standard deviation;
- 20 fixed-PCG64 row permutations of the frozen ALSP gallery prediction, which
  preserve its exact marginal distribution while breaking item assignment;
- diagnostic-only ALSP multipliers `{0.5, 1.0, 2.0}` and the least-squares
  slope/intercept mapping predicted potential to true gallery density. These
  diagnose scale mismatch and never alter the frozen `1.0` decision arm;
- the true test-gallery `mu_50` transductive oracle, reported only as an upper
  bound after the ALSP prediction and configuration are frozen.

Use the repository's stable full-order top-1 convention for every arm. Report
Recall@1, wrong-to-right and right-to-wrong transitions versus raw, and the
exact two-sided McNemar p-value. Also report Pearson correlation, Spearman
correlation, and MSE between predicted and true test-gallery density; these are
mechanism diagnostics, not selection criteria.

### Prospective decision

Let `G_alsp` be ALSP Recall@1 minus raw Recall@1 and `G_oracle` the fixed
transductive oracle gain.

ALSP passes only if all conditions hold:

- Pearson correlation with test-gallery `mu_50` is at least `0.20`;
- `G_alsp >= 0.001`;
- `G_oracle > 0` and `G_alsp >= 0.30 * G_oracle`;
- exact paired McNemar `p < 0.05`;
- the permuted-target gain is strictly less than `G_alsp - 0.00025`.
- `G_alsp` is strictly greater than the linear empirical 95th percentile of the
  20 random-direction gains.
- `G_alsp` is strictly greater than the linear empirical 95th percentile of the
  20 assignment-permutation gains.

Otherwise the unary predictability hypothesis is killed. There is no nonlinear
rescue on the same test pair. A failure may motivate the separately specified
context-feature ANC approach, but that would require a new prospective design.

## Leakage and reproducibility constraints

- CPU NumPy only for the frozen falsifier; `CUDA_VISIBLE_DEVICES` is empty.
- Launch with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
  `OPENBLAS_NUM_THREADS=1`; record these values and the NumPy version.
- No test labels enter fitting or model selection.
- Test density is computed only for the frozen oracle and post-freeze mechanism
  diagnostics.
- Every array is finite and uses explicit `float32` embedding inputs with
  `float64` fitting/reductions.
- The result is written atomically without overwriting an existing destination.
- The evaluator validates input schema, unit norms, unique IDs, disjoint label
  split, and exact output schema.
- Tests independently recompute small-fixture density targets, ridge solutions,
  tie breaking, ranking, paired counts, and decision predicates.

## Follow-up only if the falsifier passes

The preferred learning continuation is an equipotential constraint, not a claim
that an item-bias head is itself novel. Add

\[
  L_{eq}=\operatorname{Var}_{i\in B}\left[
    \tau\log\frac{1}{|B|-1}\sum_{j\ne i}
    \exp(\langle z_i,z_j\rangle/\tau)
  \right],\qquad \tau=0.1,
\]

to the unchanged Proxy Anchor objective. Compare matched `mu in {0, 1e-2,
1e-1, 1}` arms across at least three seeds. ALSP remains a diagnostic baseline
and optional auxiliary head, not the principal contribution.

1. Proxy Anchor baseline;
2. baseline plus equipotential variance at each fixed dose;
3. an equal-compute generic regularization control.

The first GPU gate is not a SOTA claim. It passes only if mean raw Recall@1
improves, the paired/seed uncertainty excludes zero, and the post-hoc
transductive oracle gain shrinks monotonically with dose. That dose-response is
the causal signature that the model internalized the defect rather than merely
benefiting from another regularizer. Backbone-matched Proxy Anchor/HIST are the
fair first comparison; larger-pretraining ViT systems are reported separately.

## Prior-art boundary

ALSP is adjacent to, and must be compared against:

- local scaling, Mutual Proximity, and CSLS, which compute transductive
  neighbourhood corrections rather than amortizing them into a unary head;
- density-aware metric learning, which changes class-cluster training geometry
  rather than distilling a per-item retrieval potential;
- neighbourhood-aware confidence calibration and quality-aware face scoring,
  which estimate confidence/quality rather than the gallery-side bias term of
  nearest-neighbour ranking;
- probabilistic embeddings, which predict uncertainty distributions rather
  than this specific local-scale potential.
- cohort score normalization, including Z/T/S/AS-Norm in speaker and face
  verification, which estimates normalization statistics from an external
  cohort. The narrow delta is amortizing a gallery-side potential into a
  feed-forward predictor, not inventing cohort normalization.

The defensible ALSP claim, if all gates pass, is narrow: **a train-only scalar
head can amortize a gallery-local scaling correction to unseen-class retrieval
and improve raw nearest-neighbour ranking without observing the test gallery
graph**. This algebraically resembles an item-bias term in matrix factorization,
so it is evidence for representability and a baseline—not the novelty claim.
Broader claims require the separately gated equipotential training result.
