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
candidate, not a novelty or SOTA claim.

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
3. Compute `mu_50` for every train row against all other rows in its own
   partition, excluding self with an explicit row identity rather than relying
   on a diagonal position after blocking.
4. Standardize target values using fit-partition statistics only for numerical
   fitting, then invert that transform so every predicted potential is in the
   original cosine-similarity units.

### Predictor selection

Fit ridge models with an unregularized intercept for the fixed grid
`lambda in {1e-6, 1e-4, 1e-2, 1, 100}`. Select the model with the lowest
validation mean squared error; break ties by the earlier grid entry. No test
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
- `G_alsp >= 0.30 * G_oracle`;
- exact paired McNemar `p < 0.05`;
- the permuted-target gain is strictly less than `G_alsp - 0.00025`.

Otherwise the unary predictability hypothesis is killed. There is no nonlinear
rescue on the same test pair. A failure may motivate the separately specified
context-feature ANC approach, but that would require a new prospective design.

## Leakage and reproducibility constraints

- CPU NumPy only for the frozen falsifier; `CUDA_VISIBLE_DEVICES` is empty.
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

Add one scalar potential head to the Proxy Anchor embedding model. Train it
against a stop-gradient memory-bank `mu_50` teacher alongside the unchanged
Proxy Anchor objective. Compare three matched arms across at least three seeds:

1. Proxy Anchor baseline;
2. baseline plus an equal-parameter arbitrary scalar auxiliary head;
3. baseline plus ALSP distillation and raw ALSP inference.

The first GPU gate is not a SOTA claim. It passes only if mean raw Recall@1
improves, the paired/seed uncertainty excludes zero, and the post-hoc
transductive oracle gain shrinks, which is the causal signature that the model
internalized the correction. A stronger reproducible checkpoint such as
Hyp-ViT is a later transfer benchmark, not the first expensive run.

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

The defensible claim, if all gates pass, is narrow: **a train-only scalar head
can amortize a gallery-local scaling correction to unseen-class retrieval and
improve raw nearest-neighbour ranking without observing the test gallery
graph**. Broader claims of novel calibration, density awareness, or SOTA are
not authorized by this design.
