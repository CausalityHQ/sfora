# Return-Level Margin local precheck

Date: 2026-08-05. Status: proposal frozen; mandatory independent review
pending provider-capacity reset. This is not the final protocol verdict.

Frozen proposal: `docs/fable_rlm_proposal_pass14_2026-08-05.md`.

## Exact internal collision

Blind pass 9's EVPC near-miss fitted a per-anchor GPD to negative proxy scores
and tried to extrapolate from fitted training classes to deployment gallery
size. Its partition rescaling was constant in the parameters, and the pass
explicitly identified the nonconstant repair:

```
R_M = u + (sigma / xi) * ((M * tail_fraction)^xi - 1)
```

RLM is that return-level repair, with PWM estimates, detached threshold and
shape, a realized-max guard, and an image-level copy. Therefore this is not a
fresh mechanism relative to the repository; it is a more complete
instantiation of the exact repair already rejected in
`docs/fable_blind_pass9_outcome_2026-08-05.md`.

## Executable-gradient reduction

On a fixed top-k membership region, RLM detaches `u`, `xi`, and

```
A = ((N * pi)^xi - 1) / xi.
```

Its fitted scale is

```
sigma_i = (1 - xi) * mean_{j in top-k}(s_ij - u_i).
```

Consequently the only EVT-branch derivative with respect to each selected
negative score is the same constant

```
A * (1 - xi) / k.
```

The guard selects the larger of this branch and the ordinary realized maximum.
Thus the descent field is uniform top-k tail/CVaR-style negative pressure with
a detached adaptive scalar, or ordinary hardest-negative pressure. `c` changes
that scalar and the hinge activation threshold; it does not create a new
observation or gradient direction. The proposed hot-LSE control is useful but
does not separate RLM from observed-tail risk, pAUC, DRO, or top-k mining.

The proposition's parameter-space corollary is also false as stated. Nonnegative
partial derivatives with respect to individual similarities do not imply that
no network-parameter descent direction can increase any impostor similarity:
all similarities share an embedding and backbone, so lowering the aggregate
loss can increase some coordinates. The upper clamp additionally permits
`t_i < s_i^max` whenever `s_i^max > 1 - 1e-4`, contradicting the claimed exact
upper-bound corollary at the specified boundary.

## Gate-1 identification failure

No corrected repository measurement identifies a batch-to-gallery extreme-tail
gap, a seen-to-fresh tail-index shift, or a gain from EVT calibration. The
training proxy tail is actively optimized and is not an unbiased sample from
the unseen-class image tail. Using multiple proxies also changes the score law
through a maximum over `M`, while deployment ranks images rather than class
proxies. The image-level twin fits only a small class-balanced batch and then
extrapolates under iid/exchangeability assumptions to a training-set-sized
gallery; it does not validate those assumptions.

The corrected In-Shop evidence establishes stable query errors, not
max-margin calibration. Historical CUB/SOP evidence is quarantined where not
independently recomputable. There is therefore no eligible positive
provenance for the causal premise.

## Primary prior-art boundary

WEINCE, *When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE*
(ICML 2026, arXiv:2606.00262), directly treats contrastive top-1 learning as an
extreme-value problem for bounded cosine similarities. It fits anchor-wise
online tail statistics, detaches the fitted tail quantities, substitutes a
Weibull endpoint-shortfall correction into the training logits, adds no
parameters, and reports kNN R@1. GPD return levels rather than Weibull
shortfall logits change the tail estimator/scalarization, not the train-time
EVT correction mechanism.

TriSim (CVPR 2026) additionally fits a generalized Pareto tail inside a
retrieval training pipeline and uses the resulting probabilities to weight a
triplet loss for false-negative mitigation. Its cross-modal labels and causal
purpose differ, so it is adjacent rather than exact, but it disproves the
proposal's broad claim that vision EVT is only post-hoc calibration.

## Forecast warning

The proposal supplies no measured mapping from tail exceedance probability to
R@1 gain. Its own crossing probabilities are only about 0.60--0.62 for mean
crossing, while stronger significance crossings are about 0.20--0.35. The
base scores are forecasts rather than measured current-digest references.
Those numbers do not justify GPU work if Gates 1--2 fail.
