# Pass 202 — measurement-conditioned hill-climb

Date: 2026-08-09  
Status: **NONE — no GPU candidate**

This pass searched locally from the project's newest measured defects rather than
from an unconstrained idea list. Fable and Claude Opus both hit provider limits;
their silence was not treated as evidence. A separate Codex/Sol audit and an
independent primary-source check reached the same boundary: the measurements needed
to justify a further move are precisely the pending Pass 201 CIS isolation panel and
Pass 200 RSTA Stage A.

## Evidence boundary

- Pass 201 is still `PENDING_SOURCE`: the actual batch class count, aligned
  shared-confuser excess, equal-update-norm coalition advantage, and held-out
  virtual-update effects are unmeasured.
- RSTA has not yet measured the contextual B=180 fields. The existing `0.592177`
  receiver-own versus `0.559383` transported-donor alignment is a singleton,
  clean-eval diagnostic and cannot be silently promoted to contextual provenance.
- The old best-over-training “selection correction” interpretation was retracted;
  it is not evidence for a new candidate.

## 1. Atomic full-union field — control only

For selected representatives `u_i`, normalized proxies `p_c`, bundle labels `U`,
and `m` representatives:

`L_AFU = (1/(mK)) sum_{i,c} BCE(u_i^T p_c, 1[label(p_c) in U])`.

This is the missing per-image, full-union, no-coalition control. Conditional on `U`,
it has no cross-image embedding Hessian blocks. The source audit proves only that its
per-member scalar coefficient `1/(mK)` differs from summed CIS's `1/(K sqrt(m))` by
the exact factor `sqrt(m)`. Neither the single arm (`0.9167253` raw best,
`0.9132789` final) nor complementary arm (`0.9184133`, `0.9150373`) implements it,
so neither supplies positive provenance.

Gate verdict: **CONTROL, NOT A METHOD**. Assigning the randomly co-batched label
union to each image is artificial multi-label supervision, adjacent to established
[compositional multi-label embeddings](https://openaccess.thecvf.com/content/WACV2021/html/Li_Compositional_Embeddings_for_Multi-Label_One-Shot_Learning_WACV_2021_paper.html).
It remains mandatory for CIS attribution. Its cheapest falsifier is already in Pass
201: at equal PA-update norm, fail if held-out foreign suppression has UCB `<= 0` or
owner-margin change has UCB `< 0`. Theoretical cost is `O(mKd)` arithmetic and
`O(mK)` auxiliary memory, with no extra encoder forward or inference work.

## 2. Shared-confuser intersection loss — dead at Gate 2

For foreign-proxy probabilities `q_ic`, define
`G_c = exp((1/m) sum_i log q_ic)` and penalize `mean_c G_c`.
The between-class failure fraction (`51.9%` on corrected CUB) and localized confusion
agreement (`2.3886%` versus `0.1466%`) motivate the observable, but the aligned
shared-confuser excess in the exact CIS context remains unmeasured.

Its derivative is
`dG_c/ds_ic = G_c (1-q_ic)/m`: the executed operator is context-dependent negative
pair weighting. That training mechanism is already covered by
[General Pair Weighting / Multi-Similarity](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html).
Gate verdict: **DEAD, Gate 2**. No code or GPU.

## 3. Backward-only class-excluded normalization — no-go before Gate 2

Let `c_i` be a class-excluded normalized raw head output and use the straight-through
descriptor `z_tilde = stopgrad(z) + c - stopgrad(c)`. Forward PA logits remain
ordinary while the encoder receives the class-excluded normalization Jacobian.

The relevant measurements are adverse: post-hoc CE-BN was `+1.357` pt, but hard
training lost `22.936` final points, soft CE-BN lost `2.469`, and CEGT produced
exactly `0.000` final and only `+0.007` raw-best. No measurement isolates the
normalization Jacobian as the useful component. It is also adjacent to
[Backward Gradient Normalization](https://arxiv.org/abs/2106.09475) and
[Batch Normalization Preconditioning](https://www.jmlr.org/papers/v23/20-1135.html),
which modify backward flow or directly condition gradients.

Gate verdict: **NO-GO — Gate 1 unresolved and prior-art risk high**. No code or GPU.

## Decision

Do not manufacture a fourth idea from unmeasured premises. Resolve Pass 201 and RSTA;
their signed, held-out, equal-norm measurements determine the next legitimate local
step. A positive operator-specific residual will provide provenance for a refinement;
a negative result closes that branch and redirects the search.
