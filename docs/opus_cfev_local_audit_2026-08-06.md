# Pass 35 local evidence-aware audit: CFEV

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_cfev_proposal_pass35_2026-08-06.md`  
Frozen proposal SHA-256: `137d5e8c0702418f3b995b5bcd5d55518914fda06971276b3d6d60c2dce6d21e`  
Independent-review prompt: `docs/opus_cfev_review_prompt_2026-08-06.txt`  
Durable independent review: `7a60e7322b874e88` (running when this audit was frozen)

This audit was written without reading the independent review result.

## Verdict

**DEAD at Gates 1 and 2, independently, with the written loss also
non-executable as its claimed GPD risk. No preregistration, implementation, or
GPU is warranted.**

## Gate 1: directly adverse repository evidence

CFEV assumes that a POT/GPD fit to the pooled upper 10 percent of roughly four
thousand dependent negative image-pair scores identifies the much deeper
unseen-gallery maximum, and that nominal gallery size supplies a causal
training lever. No corrected measurement in this repository supports that
premise. The repository has already measured the opposite in the nearest
frozen diagnostic.

Pass 31 XTail's cold simulations found that a batch tail-shape fit added only
`0.0087` incremental R-squared beyond threshold and mean exceedance, while a
constant coefficient predicted the deep quantile better (`0.0918` versus
`0.1575` RMSE). Shape-estimator noise was 67--86 percent of the complete
batch-to-SOP depth signal. A hard-negative mixture returned coefficient
`13.38` where the required value was `5.09`. CFEV pools across anchors and
therefore changes the estimator variance, but it supplies no prospective
measurement that reverses the failed tail-transport premise.

The nominal `log(M/m)` ratios and the proposed `0.5` R@1-points-per-nat
coefficient are hypotheses chosen from dataset sizes, not measurements in this
repository. They cannot satisfy provenance Gate 1.

## Gate 2: eighth internal gallery-tail recurrence

CFEV is the eighth return to the same supervision object:

1. Pass 9 EVPC fit a negative-score GPD and proposed a deployment-size return
   level.
2. Pass 14 RLM fit a differentiable POT/GPD tail and trained against its
   extrapolated maximum.
3. Pass 15 EGR-PFML added an image queue and full-gallery return level.
4. Pass 23 PORTAL used PWM and differentiable train-set-sized gallery risk.
5. Pass 26 PORT used class-block tails and a population return level.
6. Pass 31 XTail used in-batch PWM and a training-set-sized deep quantile.
7. Pass 33 POTER used two GPD extreme layers and estimated gallery error.
8. Pass 35 CFEV pools the same negative-similarity tail, extrapolates its
   survival to `M=|D_train|`, and wraps it as estimated R@1 error.

Identity-half cross-fitting and a tail-agreement penalty are estimator
scaffolds. They do not change what supervision exists: observed
seen-identity similarities are treated as a fitted parametric tail that reveals
an unobserved gallery maximum. Internally, that object is closed regardless of
whether the precise scaffold has a publication.

Publicly, the proposal omits the same near work already resolved in this
repository: WEINCE applies online EVT correction to top normalized contrastive
scores; TriSim (CVPR 2026) fits generalized-Pareto similarity tails inside
retrieval training; LDReg (ICLR 2024) differentiates an in-batch EVT/LID tail
estimator through the representation; Recall@k, ranked-list, top-k/CVaR, and
Histogram Loss occupy empirical tail/risk reductions. Post-hoc EVT retrieval
calibration is additionally adjacent. CFEV may retain a narrow compositional
distinction in its identity cross-fit, but not a new supervision object.

Primary sources already frozen in earlier audits:

- WEINCE: <https://arxiv.org/abs/2606.00262>
- TriSim: <https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf>
- LDReg: <https://arxiv.org/abs/2401.10474>
- Recall@k surrogate: <https://arxiv.org/abs/2108.11179>
- Histogram Loss: <https://proceedings.neurips.cc/paper_files/paper/2016/file/325995af77a0e8b06d1204a171010b3a-Paper.pdf>

## Frozen executable failures

### The scale transform destroys the fitted GPD scale

The proposal derives a positive PWM scale `sigma_raw` and then executes

    sigma_hat = softplus(sigma_raw) + 1e-4.

Ordinary softplus is not a positivity clamp that preserves an already-positive
scale. It maps realistic cosine-tail scales as follows:

| `sigma_raw` | written `sigma_hat` | inflation |
|---:|---:|---:|
| 0.01 | 0.698260 | 69.83x |
| 0.03 | 0.708360 | 23.61x |
| 0.05 | 0.718560 | 14.37x |
| 0.10 | 0.744497 | 7.44x |
| 0.20 | 0.798239 | 3.99x |

Thus the executable loss does not use the fitted tail scale. For a representative
`u=0.30`, `s+=0.70`, `xi=-0.2`, and `zeta=0.1`, the written transform
gives `sigma_hat=0.71856`, `q=0.055423`, and already
`M*q=9.98` at the no-extrapolation control `M=180`.

In `L=log(1+M*q)`, sensitivity to `log q` is `M*q/(1+M*q)`. It is
`0.9089` at `M=180` and `0.9997` at SOP's `M=59551`: the 331x change in
nominal gallery size changes this gradient multiplier by only 10.0 percent.
Once saturated, changing `M` primarily adds `log M` to the loss, an
optimizer-irrelevant constant. CFEV's claimed `log(M/m)` causal lever
therefore does not follow from its executable objective.

### Repairing the scale exposes an unhandled support violation

For negative GPD shape, survival exists only where
`1 + xi*(s+-u)/sigma > 0`. The proposal soft-clamps `xi` but never enforces
this support. With the same values and the intended fitted
`sigma=0.05`, the support argument is `-0.60`; the written logarithm is not
real. This is not an obscure boundary because cosine similarity is bounded and
the proposal itself expects a negative Weibull-domain shape. The claim that the
survival is “smooth, everywhere-finite” is false.

For `s+<u`, POT supplies no tail formula for the required bulk probability.
The positive softplus maps the negative excess back to a small nonnegative
excess and returns `q<=zeta`, whereas the true exceedance probability below
the threshold must be greater than `zeta`. The quantity is therefore not an
R@1-risk estimator in precisely the weak-positive regime that should receive
the largest penalty.

### The cross-fit does not score tail transfer

`L_{B|A}` fits the negative tail on A and combines it only with positive
scores from B. It never evaluates A's predicted survival against negative
scores from B. The mirror term does the converse with the same shared encoder,
and `L_agree` merely equalizes two global fitted scalars. Resampling halves
therefore does not make the objective “fall only if the similarity law
transfers”; a common encoder can reduce both in-sample tail summaries without
any out-of-half predictive check. The U-statistic argument concerns a fixed
estimand, as the proposal concedes, not the learned optimum or the validity of
GPD extrapolation.

The global pooled tail also estimates a marginal inter-class-pair law, while
R@1 is decided by a query-conditional tail and correlated gallery blocks.
Roughly four thousand pair scores contain only 22--23 identities per half and
share endpoints heavily; pair count is not effective tail sample size.

### The scale control is not norm matched

Let `g` be the PA embedding gradient and choose the auxiliary gradient `h`
so `||h||=r||g||`, with `r=0.5`. The CFEV total norm is

    ||g+h||/||g|| = sqrt(1+r^2+2*r*cos(theta)),

which ranges from `0.5` to `1.5` depending on direction and is `1.118`
when orthogonal. C1 uses `(1+r)g`, whose ratio is always `1.5`. It matches
the CFEV arm only if the two gradients are perfectly aligned, so
`CFEV-C1` does not isolate the claimed mechanism from optimizer scale.

## Forecast and protocol mismatch

The proposal forecasts misses on CUB and Cars and only point-estimate crossings
on SOP (`+0.003` against PFML) and In-Shop (`+0.009` against an
uncertainty-free external row). Its invented PA reproductions are not measured
baselines. The repository has also established that In-Shop's actual gallery
contains 12,612 images, whereas CFEV uses 25,882 as its load-bearing
multiplicity, a 2.05x mismatch already recorded for POTER.

Even if the method were repaired, the search protocol would require an
In-Shop-first preregistered screen, raw and final-state/selection-audit
reporting, and out-of-sample replication. It cannot reach those gates because
Gate 1 is adverse and Gate 2 is an exact internal recurrence.

## Correct pieces worth preserving

- R@1 is indeed a maximum-negative versus maximum-positive event, and
  batch-to-gallery operating depth is a real measurement target.
- The untransformed Hosking--Wallis PWM formulas and `xi->0` GPD limit are
  algebraically standard.
- Using only training-set metadata for a frozen nominal population avoids
  direct test leakage.
- The proposal states its failures, external-reference uncertainty, and
  non-crossing forecasts more honestly than most candidates.
- Its M-sweep, empirical-survival control, negative-gradient masks, and
  raw/final-state reporting requirements are useful diagnostics outside any
  novelty claim.

## Mechanism lesson

Eight blind rediscoveries now show that gallery-tail extrapolation is a strong
language-model invention attractor, not a repository-supported lead. A more
careful estimator, identity split, second extreme layer, or new wrapper does
not create a new supervision referent. Future blind prompts should continue to
withhold the catalogue, but local Gate 2 must reject this family immediately.
The next candidate must leave EVT/GPD/return-level/gallery-multiplicity
supervision unless a prospectively frozen repository measurement first
reopens it.

