# Candidate 271: proxy-equilibrium stopping

**Verdict: DEAD at Gate 0 in its current form. No implementation and no GPU.**

## Proposed speed mechanism

Inspired by chemical equilibrium, monitor only training-set proxy/embedding geometry:
proxy displacement, proxy-to-class-centroid alignment, within-class dispersion, and
the net attractive/repulsive proxy force. Stop when their robust moving slopes and
force residuals remain below fixed tolerances. The intended result is the same final
retrieval quality with substantially fewer updates and no validation/test split.

## Why the apparent provenance is invalid

The corrected SOP run reached raw best-test R@1 **0.7812 at epoch 21** and **0.7837 at
epoch 23** of a 60-epoch recipe. Those are plausible early quality values, but they are
test-split evaluations. They cannot motivate or tune a supposedly training-only stopping
criterion without leaking the target. The run persists only final proxies; it does not
record the proposed geometric trajectory at each epoch. Therefore there is no artifact
showing that a training-only equilibrium event occurred near epochs 21--23, preceded
the final-quality plateau, or generalized across a seed/dataset.

The preregistered final proxy-clock diagnostic is cross-sectional, not temporal. A
single final point cannot validate an early-stopping rule.

## Prior-art horizon (not the deciding gate)

Even with provenance, the claim would need a narrow defense. Mahsereci et al.,
[Early Stopping without a Validation Set](https://arxiv.org/abs/1703.09580), already
stop from training-gradient statistics. Neural-collapse work already tracks class-mean,
classifier/proxy alignment, within-class collapse, and simplex geometry during terminal
training (Papyan et al., PNAS 2020; Zhu et al., NeurIPS 2021). A proxy-specific
combination might be benchmark-unoccupied, but “geometry indicates convergence” is not
novel by itself.

## Decision

Do not retrofit per-epoch logging or rerun SOP for candidate 271. Reconsider only if a
future independently required run already records proxy geometry over time and a
predeclared rule predicts a held-out run's final-quality epoch. Until then, the early
test curve is descriptive and cannot authorize a speed claim.
