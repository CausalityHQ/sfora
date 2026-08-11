# Asymmetric Hard-Negative Cohort Residual Similarity Design

## Status and motivation

This is a prospective mechanism test, not a SOTA claim. The frozen ALSP result
showed that a train-only scalar predicts gallery density well (`Pearson=0.741`)
yet worsens Recall@1, while the true gallery graph improves it. Post-outcome
diagnosis found only `67.4%` agreement between predicted and true density order
inside raw top-10 candidate sets. The failure is therefore local and
query-conditional, not simply an inability to predict marginal density.

One explicitly exploratory calculation on the already-observed official pair
motivated this candidate: subtracting the centroid of the query's 50 nearest
training embeddings improved Recall@1 by `0.005415670277113538` with exact
paired `p=1.095840141656609e-06`. That number is contaminated hypothesis
generation and cannot validate the method. The test below uses only a new,
prospectively frozen held-out-label retrieval construction inside the training
archive.

## Candidate

Call the method **Asymmetric Hard-Negative Cohort Residual similarity
(AHNCR)**. Let `q` and every cohort/gallery embedding be unit-normalized. Let
`C` contain only embeddings from training identities disjoint from every
evaluation identity. For fixed `k=50`, define

\[
  H_k(q)=\operatorname{topk}_{c\in C}\langle q,c\rangle,
  \qquad m_k(q)=\frac1k\sum_{c\in H_k(q)}c,
\]

and score gallery item `g` by

\[
  s_{AHNCR}(q,g)=\langle 2q-m_k(q),g\rangle.
\]

The cohort is class-disjoint, so its nearest members are known semantic
negatives rather than pseudo-positive query-expansion items. The subtraction
removes the hard-negative direction most confusable with the query. It is
deliberately asymmetric: the gallery is not centered, the residual is not
renormalized, and no test-gallery neighborhood is read.

## Alternatives rejected for this gate

1. **Local-order potential distillation** would train ALSP on pairwise local
   density order instead of global MSE. It is more trainable but is too close to
   an outcome-aware rescue of the killed unary hypothesis.
2. **Cohort-profile transport or kernels** compare query/gallery distributions
   over training prototypes. They are expressive but introduce temperature,
   prototype count, transport regularization, and fusion choices before the
   basic cohort-residual mechanism is established.
3. **Symmetric adaptive data normalization** is retained as a control, not the
   candidate. It is known in speaker verification and did not reproduce the
   exploratory asymmetric gain.

## Frozen held-out-label construction

Use only
`inshop_corrected_pa_seed0_train_final.npz` with SHA-256
`67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
The official query/gallery arrays and the ALSP result must not be opened by the
falsifier.

This construction also separates the two explanations exposed by the ALSP
failure. A unary cohort-density control asks whether a smooth population field
is sufficient; AHNCR asks whether the *query-conditioned hard-negative
direction* is the missing quantity. The shuffled-centroid null then tests
whether any improvement truly depends on matching that direction to its query,
rather than on the marginal distribution of centroid vectors.

1. Labels with at least two rows are evaluation-eligible. Sort those labels by
   `(SHA256(little-endian int64 bytes), integer label)`.
2. The first `80%` of eligible labels plus every singleton label form the
   cohort; the remaining `20%` of eligible labels form the evaluation
   identities. All rows of a label stay in one side.
3. For each evaluation label, choose as query the row with the smallest
  `(SHA256(example_id UTF-8 bytes), example_id)` pair. Its remaining rows form
  the gallery.
4. Split evaluation labels into four reporting shards by the first two bits of
   `SHA256(UTF-8 "AHNCR-shard-v1:" || little-endian int64 label bytes)`. The
   domain separator makes this hash independent of the hash used to choose the
   held-out tail. Shards affect only reporting, not scoring or selection.
5. Use stable first-index `argmax`, float32 matrix products, float64 means and
   statistics, block size `256`, and one BLAS thread.

There is no hyperparameter selection, fitted model, test-time graph, or second
attempt.

The fixed coefficient `2`, `k=50`, and lack of residual renormalization were
chosen before this held-out-label result is observed. They may not be changed
afterward, even if one of the controls suggests a nearby alternative.

## Controls

Evaluate all arms from each shared query-gallery product:

- `raw`: `q·g`;
- `ahncr`: `(2q-m_k(q))·g`;
- `global_mean`: `(2q-mean(C))·g`;
- `positive_expansion`: `(2q+m_k(q))·g`;
- `unary_cohort_density`: `2q·g-mu_C(g)`, where `mu_C(g)` is the mean of
  `g`'s top-50 similarities to `C`;
- `symmetric_ad_norm`: cosine between independently normalized
  `q-m_k(q)` and `g-m_k(g)`;
- 20 fixed-PCG64 (`seed=20260814`) permutations assigning the observed AHNCR
  query centroids to different queries. These preserve centroid values while
  breaking query-centroid alignment.

Persist Recall@1, gain versus raw, wrong-to-right, right-to-wrong, and exact
two-sided binomial McNemar p-value for every named arm. Persist the 20 shuffled
gains and their linear empirical 95th percentile.

## Prospective decision

AHNCR passes the held-out mechanism gate only if all conditions hold:

- pooled Recall@1 gain is at least `0.003`;
- exact paired McNemar `p < 0.01`;
- at least three of four label-shard gains are strictly positive;
- gain exceeds `global_mean`, `positive_expansion`,
  `unary_cohort_density`, and `symmetric_ad_norm` by at least `0.001` each;
- gain is strictly above the linear 95th percentile of the 20 shuffled-centroid
  gains;
- wrong-to-right transitions exceed right-to-wrong transitions.

Any failed predicate kills AHNCR without coefficient, `k`, normalization, or
nonlinear tuning. A pass authorizes an external-dataset frozen-embedding test,
not GPU training and not a SOTA claim.

## Reproducibility and output

- CPU only with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.
- Exact schema records input SHA, NumPy version, thread environment, label and
  row split hashes, cohort/query/gallery counts, all arm statistics, null draws,
  predicates, and final decision.
- Publish canonical JSON by same-directory exclusive temp plus hard-link
  no-replace; fsync, strict reload, validate, and remove only the owned temp.
- Existing destination, symlink input/output, malformed IDs, non-unit/nonfinite
  arrays, empty shards, or inconsistent recursive relations are structural
  failures.

## Prior-art boundary

AHNCR combines two old ideas: negative-centroid relevance feedback (Rocchio)
and adaptive cohort centering/normalization from speaker verification. Cumani
and Sarni's 2023 Adaptive Data Normalization independently centers both trial
embeddings using adaptive impostor cohorts. Standard image query expansion
moves toward presumed positives in the evaluation index. AHNCR instead uses a
guaranteed class-disjoint training cohort, moves only the query away from its
nearest hard-negative centroid, retains the gallery unchanged, and uses no
evaluation graph.

The defensible contribution, if it survives independent datasets, is this
specific asymmetric open-set retrieval rule and its use as a mechanism for
hard-negative direction removal. Broad claims about cohort normalization,
negative feedback, or query expansion are not novel. Relevant primary sources
include [Adaptive Data Normalization](https://www.isca-archive.org/interspeech_2023/cumani23_interspeech.pdf),
[Rocchio relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/the-rocchio71-algorithm-1.html),
and [learned image query expansion](https://arxiv.org/abs/2007.08019).
