# Pass 31 local evidence-aware audit: XTail

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_xtail_proposal_pass31_2026-08-06.md`  
Independent-review prompt: `docs/opus_xtail_review_prompt_2026-08-06.txt`  
Review-prompt SHA-256: `5dcb313f42fc454bace49c762682497bbde0c812d4f0e09ee1560024f64fd1b4`  
Durable independent review: `27d72c0378ce4a13` (running when this audit was frozen)

This audit was written without reading the independent review result.

## Verdict

**DEAD at Gates 1 and 2, independently with a locally adverse gradient and no
direct frontier-crossing frozen method.** No diagnostic, preregistration,
implementation, or GPU is warranted.

## Gate 1: a sixth return to an explicitly unsupported premise

XTail assumes that a POT/GPD shape fitted to roughly 36 within-batch negative
image similarities identifies the deepest negative order statistic in a much
larger gallery of unseen identities. The repository contains no prospective
measurement of GPD fit, threshold stability, seen-to-held-out tail-index
transport, gallery-multiplicity error decomposition, or causal response to
tail pressure. The verified evidence boundary has repeatedly recorded this
premise as absent.

The proposal's dataset-size ordering and illustrative GPD arithmetic are not
measurements. They choose the desired effect ordering from nominal population
sizes and then forecast `+0.009/+0.011/+0.016/+0.020` on CUB, Cars, In-Shop,
and SOP. No saved corrected observation connects those ratios to retrieval
errors. The load-bearing assertion that tail shape is class-generic and
transfers across disjoint identities is exactly what must be measured, not a
consequence of location/scale invariance.

## Gate 2: exact internal collision and occupied public mechanism

XTail is the sixth blind version of the same return-level mechanism:

- Pass 9 EVPC fitted a negative-score GPD and proposed a deployment-size
  return-level repair.
- Pass 14 RLM fitted a differentiable POT/GPD image/proxy tail and trained a
  margin against its extrapolated maximum.
- Pass 15 EGR-PFML used an XBM-style image queue, a per-anchor GPD hard-negative
  tail, and a full-gallery return-level margin.
- Pass 23 PORTAL used PWM, a nominal training-set-sized population, and
  differentiable gallery-tail risk.
- Pass 26 PORT used per-anchor class-block negative tails, differentiable PWM,
  a population return level, and a retrieval loss.

XTail removes the queue, uses the batch top quartile, shrinks per-anchor PWM
shape estimates toward a pooled EMA, and places `u+A*a` against a soft maximum
of positives. Those are estimator and wrapper changes inside the identical
supervision object: **train on observed seen-identity similarities as though a
fitted GPD extrapolation revealed the unseen gallery maximum**. Internal Gate 2
rejects rediscovery regardless of publication status.

Primary literature also occupies the mechanism. WEINCE (arXiv:2606.00262)
uses anchor-wise online EVT correction of top normalized contrastive scores
during training. TriSim (CVPR 2026) fits generalized-Pareto similarity tails
and uses their probabilities in a retrieval-training loss. Recall@k surrogates,
top-k/CVaR and ranked-list objectives occupy the observed-tail reduction;
AnchorFace/OneFace/UniTSFace and XBM address the same operating-depth problem
by increasing the empirical negative set. The proposal omitted TriSim and all
five exact repository predecessors. Differentiating the PWM estimate and using
`N_train` rather than a memory bank do not create a new supervision referent.

Primary sources:

- WEINCE: <https://arxiv.org/abs/2606.00262>
- TriSim: <https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf>
- Recall@k surrogate: <https://arxiv.org/abs/2108.11179>
- AnchorFace: <https://ojs.aaai.org/index.php/AAAI/article/view/20063>

## Frozen mathematical failures

### The threshold gradient explicitly rewards a negative similarity

With fixed membership and shape coefficient, the frozen target is

```
q = u + A(mean_E(s) - u) = A mean_E(s) - (A-1)u,
```

and the proposal correctly derives `dq/du = 1-A < 0`. Gradient descent
therefore **raises the threshold negative similarity** to reduce the loss. The
threshold is the `(k+1)`-st order statistic and is not in the exceedance mean,
so this is a valid local coordinate direction until an ordering tie. This is
the same monotonicity failure found in Pass 14 RLM. The inequality `q >= u`
does not imply that minimizing `q` minimizes `u`; an upper bound can fall while
its lower bound rises.

Nor does XTail dominate the hardest observed negative. For a cliff tail with
one exceedance `s_max-u=d` and the other `k-1` exceedances near zero,
`q=u+(A/k)d`. Here `k≈36` while the claimed `A` range is `2.9--18.3`, so
`q<s_max`. The supposed deployment-depth target can ignore a negative already
observed in its own batch. A fixed top-k/rank-weight loss can reproduce the
live gradient without any unseen-population interpretation.

### The collapse and saturation arguments fail through normalized cosine

At identical normalized descriptors all pairwise scores are one. Although the
scalar loss has a nonzero derivative with respect to the written positive
score, the tangent derivative of cosine between identical normalized vectors
is zero. Exact collapse is stationary through the descriptor, as in the prior
return-level proposals; a high scalar loss is not a first-order escape proof.

When `u+A*a >= 1`, `q=min(1,...)` removes every negative-tail gradient. The
remaining target requires `s+ > 1+delta_X` to make the margin negative, which
is impossible for cosine. This creates an irreducible active loss with a dead
tail branch, not a controlled deep quantile. The hard clips on shape similarly
produce flat estimator regions.

### The fitted sample is not the claimed population

The nominal 145 negatives are only 29 other identities with five correlated
images each. The top quartile is a mixture over those identities, shares one
anchor, and lies in the distribution bulk rather than an asymptotic tail.
Repeated samples from a hard class are neither iid tail draws nor 36
independent identity extremes. The law also changes every optimizer step. A
GPD fit to this sampler-conditioned, moving mixture cannot acquire
unseen-identity or full-gallery semantics by substituting `N_train` into its
quantile formula.

The EMA is underdefined as an autograd object. A persistent buffer must be
detached across steps; “gradient flows only through the current 0.1 term” is
executable only if the forward uses a separate expression such as
`0.9*buffer_detached + 0.1*current` and then updates the stored detached value.
The proposal does not specify that state transition.

## Controls, forecasts, and objective mismatch

A4 (constant `A`) and A7 (no nominal extrapolation) are necessary but not
sufficient. Missing decisive controls include a rank-affine/top-k mean loss
matched to the actual per-rank gradient, an adaptive margin using the same
`u,a` statistics without a GPD interpretation, and a hardest-observed-negative
floor. Beating a differently tuned constant does not show that 36 dependent
exceedances estimated an unseen-gallery tail.

The frozen standalone method crosses no supplied frontier on CUB or Cars and
lands below PA+DADA on In-Shop. Its SOP `0.830` forecast is a `+0.001` tie
against reference SD `0.003`, and it exists only after adding XTail to an
invented multi-proxy stand-in. The In-Shop `0.934` row likewise assumes an
unmeasured additive multi-proxy contribution. The proposal openly calls the
former a tie and the latter unadjudicable. Even its own forecasts therefore do
not supply a decisive frozen method that fulfills the standing objective.

The cost estimate is directionally small but incomplete: the pairwise Gram may
already exist for another loss, while differentiable per-row sorting, fp32 PWM
state, and retained autograd graphs cost more than the stated 90 KB matrix.
This engineering correction is immaterial to the rejection.

## Mechanism lesson

Six independent rediscoveries of “training sees fewer negatives than the
gallery, so extrapolate a GPD tail” reveal an attractive search prior, not an
evidence-backed direction. Future variants must be killed before GPU unless a
new prospective repository measurement first establishes tail transport and
causal error repair, and the supervision object is materially different from
EVPC/RLM/EGR-PFML/PORTAL/PORT and WEINCE/TriSim. More careful shrinkage or a new
margin wrapper is not enough.
