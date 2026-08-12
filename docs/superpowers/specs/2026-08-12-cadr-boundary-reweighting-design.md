# CADR: Cosine-Anchored Diagonal Boundary Reweighting

## Status

This is a prospective sequential-research test, not a SOTA claim. ALSP killed a
train-only unary gallery correction. AHNCR killed a cohort residual under its
registered screen, which saturated at raw `797/797`. Neither result tests a
strictly pair-conditioned score learned from source labels and evaluated on
identities unseen by the encoder.

The official In-Shop query/gallery pair has already informed earlier research.
It is therefore sealed until the train-only gate below passes, and its later
result is evidence for this single frozen candidate only.

## Candidate

For unit embeddings `q,g in R^512`, CADR scores

```
s_u(q,g) = sum_j u_j q_j g_j.
```

This is a diagonal bilinear similarity. It has 512 parameters, requires no
test-gallery graph or cohort, and reduces exactly to raw cosine at `u=1`.
Unlike diagonal Mahalanobis distance on `(q-g)^2`, it adds no separate unary
query or gallery term.

Fit `u` and an intercept `b` with balanced, weighted logistic regression on
pair features `x=q*g` and labels `y in {-1,+1}`:

```
mean_i omega_i log(1 + exp(-y_i (u.x_i + b)))
    + lambda * mean_j (u_j - 1)^2.
```

Positive and negative examples each have total weight `1/2`. The intercept is
unregularized. Direct anchoring at `u=1`, rather than regularizing only the
mean-free component, makes the finite safe limit raw cosine.

## Train-only construction

The only Stage-A input is
`inshop_corrected_pa_seed0_train_final.npz`, SHA-256
`67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.

1. Eligible labels have at least two rows. Sort them by
   `SHA256("CADR-split-v1:" || little-endian-int64(label))`, then by label.
   The first floor(`0.8*n`) are fit labels; the rest are validation labels.
   Singletons are excluded. Labels never cross the split.
2. Within each pool, positives are unordered distinct same-label pairs. If a
   label has more than 30, retain the first 30 under SHA-256 of the two ordered
   UTF-8 example IDs with domain `CADR-positive-v1:`.
3. Within each pool, for every row, find its ten highest-cosine rows having a
   different label, with stable row-index tie breaking. Canonicalize each as an
   unordered row-index pair and deduplicate. These are boundary negatives.
4. Products are float32; objectives, gradients, and statistics are float64.
   Fit with SciPy L-BFGS-B, `maxiter=500`, `ftol=1e-12`, `gtol=1e-8`, starting
   exactly at `u=ones(512), b=0`. Nonconvergence is structural failure.
5. Select `lambda` from `[1e-4,1e-3,1e-2,1e-1,1]` by minimum validation
   balanced log loss; exact ties choose the larger lambda. Then refit on all
   eligible train labels with the selected lambda.

The continuous held-out loss avoids the saturated closed-set Recall@1 screen.
It does not itself establish open-set transfer.

## Stage-A gate

Before the official query/gallery arrays may be opened, all must hold:

- selected lambda is less than `1`;
- `std(u)/abs(mean(u)) >= 0.01` and `mean(u) > 0`;
- validation balanced log loss improves by at least 1% relative to a
  two-parameter Platt calibration of raw cosine fit on the fit pairs;
- validation balanced log loss improves by at least 0.5% relative to a
  diagonal WCCN control selected on the same validation pairs.

WCCN uses `u_j = 1/(within_variance_j + tau)`, normalized to mean one, with
`tau/median(within_variance)` in `[0.1,0.3,1,3,10]`; ties choose larger tau.
For each fixed WCCN vector, fit only a Platt scale and intercept on fit pairs
before computing validation log loss; this prevents arbitrary score scale from
handicapping the control.
Failure closes CADR without touching the official pair.

## One official unseen-identity evaluation

If Stage A passes, refit CADR and controls on all train labels, then evaluate
exactly once on:

- query SHA-256 `ef5278fd9aae7a6398a6c74133e6acc0ded05e39647087bdf78459223b9eb761`;
- gallery SHA-256 `6eb89ff57e7a6002f2ba71f9659e04dabd0cafdb1996be3d85f5211731ba861a`.

Use full-gallery stable top-1 ranking. Arms are raw cosine, CADR, WCCN, a full
pipeline fixed label-permutation control, and 20 PCG64 seed `20260813`
mean-matched random reweightings whose mean-free L2 norm equals CADR's. Split
queries into four reporting shards using the first two bits of
`SHA256("CADR-shard-v1:" || example_id UTF-8)`.

PASS requires every predicate:

- Recall@1 gain over raw is at least `0.002`;
- exact two-sided paired McNemar `p < 0.01` and wrong-to-right exceeds
  right-to-wrong;
- at least three of four shard gains are positive;
- gain is strictly above the linear 95th percentile of random-control gains;
- gain minus label-permutation gain is at least `0.001`;
- gain minus WCCN gain is at least `0.001`, and the direct CADR-vs-WCCN paired
  McNemar test favors CADR at `p < 0.025`.

If all predicates except the WCCN attribution predicates pass, the result is
`REWEIGHTING_ONLY`: CADR is not supported, but WCCN may be independently
replicated. Any other miss is `KILL`. No lambda, pair-mining, basis, nonlinear,
or normalization rescue is allowed on this pair.

## Controls, outputs, and reproducibility

Stage A and B are CPU-only with CUDA hidden and OMP/MKL/OpenBLAS fixed to one
thread. Persist exact input hashes; split and pair hashes; optimizer status;
all lambda losses; fitted-vector hashes and summaries; Platt/WCCN controls;
per-arm correct counts, transitions, McNemar p-values, shard recalls; random
control draws; ordered predicates; and the final decision.

Publish canonical JSON with exclusive same-directory temp creation, hard-link
no-replace, fsync, strict reload/revalidation, and owned-temp-only cleanup.
Stage A and B are separate commands/artifacts so a failed Stage A cannot read
query/gallery bytes.

## Prior-art and novelty boundary

Diagonal metric learning and bilinear similarity learning are established:
LMNN, OASIS, and direct Mahalanobis ranking all precede this work. CADR is not a
new metric-learning family. Its narrow research question is whether a
cosine-anchored, hard-boundary, label-disjointly selected diagonal residual can
transfer a frozen deep metric to unseen identities after unary and cohort
corrections fail. Relevant primary sources include:

- Weinberger and Saul, LMNN: https://jmlr.org/beta/papers/v10/weinberger09a.html
- Chechik et al., OASIS: https://research.google/pubs/large-scale-online-learning-of-image-similarity-through-ranking/
- Lim and Lanckriet, ranking metrics: https://proceedings.mlr.press/v32/lim14.html
- Wen et al., SimPLE: https://arxiv.org/abs/2310.09449
- Vasudeva et al., hard-negative embeddings: https://openaccess.thecvf.com/content/ICCV2021/html/Vasudeva_LoOp_Looking_for_Optimal_Hard_Negative_Embeddings_for_Deep_Metric_ICCV_2021_paper.html

A PASS authorizes a preregistered independent-dataset frozen-embedding
replication, not GPU training and not a SOTA claim.
