# Pass 170 — Coalition-Marginal Retrieval (CMR), preregistration

## Gate 1 motivation

The corrected In-Shop measurements show a between:local retrieval-error ratio
of `3.05–3.22`, while CUB is near `1:1`. This says a pairwise label is not the
whole error object on In-Shop: the usefulness of an image depends on the
*coalition* of same-class gallery images it joins. CMR tests that claim rather
than assuming it.

## Mechanism

For an image `i` of class `c`, define its coalition marginal signature over
held-out same-class queries `q` as the Shapley value of adding `i` to a random
subset of the class gallery:

`phi_i(q) = E_S[max(sim(q,i), max_{j in S} sim(q,j)) - max_{j in S} sim(q,j)]`.

Use the vector `phi_i` over held-out queries as a data-derived signature. During
training, two different same-class images are positive-to-unknown only when
their signatures agree (cosine agreement above a preregistered threshold);
ordinary class positives remain positive for the base Proxy Anchor term. This
changes which supervision relations exist, not only their scalar weight.

## CPU gate (before GPU)

On the In-Shop training embedding pack only, hash-split identities' images into
support/query halves. Estimate signatures using 64 fixed random subsets per
class, then compare signature-gated pairs against distance-matched pairs. CMR
passes only if, on all four corrected seed packs or the available seed-0 pack
as a screening diagnostic, signature agreement predicts held-out query utility
with AUC ≥ `0.60` and improves support-to-query R@1 by ≥ `0.5` point over the
distance-matched control. Otherwise it is dead at Gate 1.

## Gate 2 prior-art check

Data Shapley (Ghorbani & Zou, ICML 2019) values training data for predictor
performance, but does not create a train-time DML positive relation from
same-class coalition marginal signatures. Group Loss, DSLL, and graph metric
learning use class/group or pair distributions, not Shapley marginal
contribution as an eligibility gate. This remains `LIVE-NARROW` only if the CPU
gate passes and a primary-source search finds no exact relation-level use.

## Forecast and falsifier

If admitted to In-Shop GPU screening, preregistered one-seed R@1 is `0.9190`
against `0.9153889`; it is falsified below `0.9175`. The mechanism is falsified
if a distance-matched gate or soft reweighting matches CMR within `0.2` point.
No GPU is authorized before the CPU gate and final Gate-2 audit.

## CPU result (2026-08-08)

The preregistered CPU diagnostic on the corrected In-Shop training pack
(`25,882` images, seed 0) produced signature-agreement AUC `0.814993`, but the
cross-fitted support-to-query retrieval control fell from `0.992133` using all
support images to `0.982017` after selecting by coalition marginal utility
(`Δ=-0.010115`, −1.01 points). The signature predicts a proxy utility scalar
but selecting/gating by it harms held-out retrieval. CMR therefore fails Gate 1
and receives no GPU run.
