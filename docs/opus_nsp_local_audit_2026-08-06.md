# Pass 30 local audit: Null-Space Provisioning

Date: 2026-08-06 UTC. This audit was written independently while the mandatory
frozen-proposal review `5e522fb2c2c14ee5` was still running.

## Verdict before reviewer reconciliation

**DEAD at Gate 1 and mechanism-occupied at Gate 2. No diagnostic,
preregistration, implementation, or GPU.** NSP's online projector is a concrete
wrapper, but its causal premise is an algebraic category error already rejected
in this repository. Its deployed supervision object is the occupied conjunction
of supervised DML, augmentation-based instance discrimination, task
decorrelation/subspace separation, and a spectrum/energy floor. The frozen
method also forecasts no direct frontier crossing.

## Frozen object

NSP computes an EMA table of normalized training-class centroids, takes the
99%-energy right-singular subspace `P`, and calls `Q=I-P` label-null. Proxy
Anchor trains the ordinary normalized descriptor. A second strongly augmented
view trains NT-Xent on `Q z_tilde`, treating all different source images,
including same-class images, as negatives. A hinge requires at least 25% of the
pre-normalization descriptor energy in `Q`. Separate auxiliary BatchNorm state
is discarded at inference; the full 512-D descriptor is deployed.

## Gate 0 and Gate 1: no eligible provenance

The verified packet contains no measurement that official-query errors are
caused by a `C-1`-rank identity channel, that a centroid complement is unused,
or that instance discrimination in that complement transfers to unseen
identities. It explicitly excludes low-rank shortcut monopoly from the current
evidence boundary. The corrected In-Shop augmentation-response relation is a
relation between *different same-class images*; it does not establish that
same-class images should be contrastive negatives, and NSP predicts its own
projector becomes inert in that large-class regime.

The proposal therefore has neither a measured cause nor the protocol's
corrected In-Shop screen. Its dose-response constant `kappa=2.4` points,
`r_99` values, PA/multi-proxy baselines, and additive PFML gain are forecasts,
not repository measurements.

This premise has already recurred independently:

- candidate 365, Blind-Subspace Allocation, used a supervised block plus a
  proxy-blind block and failed Gate 1;
- Pass 21 RIM used augmented-view information in the purported `d-C+1`
  complement and failed the same provenance claim; and
- blind Pass 7 asserted the universal `C-1` constraint and was explicitly
  rejected as confusing between-class-scatter rank with an optimization bound.

## The `C-1` causal premise is false

Centered class means do have rank at most `C-1`. That fact does not imply that
identity-labelled training has zero gradient or zero information outside their
span.

1. **Pair/tuple counterexample.** Let two class centroids lie on `e1`. Two
   same-class samples can be `e1+e2` and `e1-e2`. The positive-pair squared
   distance has gradient `4 e2`, orthogonal to the centroid span. Labels decide
   that this sample-dependent within-class direction is pulled; replacing all
   pair differences by their class expectation deletes the very gradients being
   optimized.
2. **Proxy span is not centroid span.** Proxy Anchor's local derivative with
   respect to its *normalized* embedding is a weighted combination of learned
   proxies, not empirical class centroids. The proxies move and need not equal or
   remain inside `span{mu_c}`. A multi-proxy objective weakens the count further.
3. **Normalization breaks the claimed raw-output span.** If `z=zt/||zt||`,
   then `grad_zt = (I-zz^T) grad_z / ||zt||`. Even when `grad_z` lies in a fixed
   proxy span, the `zz^T` term introduces the sample direction, which can have a
   centroid-orthogonal component.
4. **A feature gradient is not a parameter or function constraint.** The
   supervised and auxiliary losses update one nonlinear backbone. Output-space
   orthogonality at the current sample does not make their parameter gradients
   orthogonal and does not constrain the function change at other images.

`Q` is consequently centroid-orthogonal, not label-null. It can remove
directions actively shaped by within-class positives, sample-specific negative
weights, normalization, and the shared encoder.

## Additional algebraic failures

### The advertised exact gradients are not exact

For `u=v/(||v||+epsilon)`, the Jacobian is

```
I/(r+epsilon) - vv^T/[r(r+epsilon)^2],  r=||v||,
```

not `(I-uu^T)/||v||`; `u` is not unit norm when `epsilon>0`, and the displayed
formula is singular at `v=0`. For
`L_e=(gamma-||Qzt||^2/D)^2` with detached `D`, the active derivative is
`-4(gamma-e)Qzt/D`, not the proposal's `-2` expression. These factor/formula
errors do not change the broad direction, but they disprove the claimed exact
derivation.

More importantly, `Qzt=0` is a stationary failure: NT-Xent similarities are all
zero and their feature derivatives vanish when every projected vector is zero;
the energy-hinge gradient is also proportional to `Qzt` and vanishes. A positive
hinge value is not an escape direction. The energy floor therefore does not
provide the claimed proof-level collapse protection.

### The Welch bound is not a learned-rank guarantee

The regular-simplex equality case states the dimension needed to attain one
particular spherical-code optimum for `N` unit points. NSP neither establishes
that its finite-temperature NT-Xent solution approaches that equality nor
derives a loss-to-effective-rank bound. A lower-dimensional, augmentation-stable
instance or background code can reduce NT-Xent substantially. Thus “provable
lower bound on provisioned rank: 179” does not follow.

Treating same-class images as negatives likewise does not imply
`E[u|y=c]=0`. A representation may carry a common class component plus
instance-specific residuals and trade the resulting similarities against every
other negative. The objective has no equality constraint on conditional means.

### Instantaneous projection is not noninterference

For fixed detached `Q`, the auxiliary *feature* gradient lies in `range(Q)`
away from singular points. The shared-backbone parameter gradient is
`J_theta^T Q g`; it need not be orthogonal to the supervised parameter gradient,
and its update can change `P`-coordinates on every other image. Meanwhile the
EMA centroids and `Q` themselves move. The sentence that the auxiliary signal is
“structurally incapable of rotating the discriminative subspace” is therefore
false at the trained model.

## Gate 2: occupied supervision and action

DiVA (Milbich et al., ECCV 2020) already identifies the same causal story in
benchmark DML: class-discriminative training specializes to seen classes, so it
jointly learns class-shared, intra-class, and sample-specific features from the
official training data. Its sample-specific task is augmentation-based
contrastive learning; it explicitly decorrelates every auxiliary embedding from
the discriminative embedding, combines them into one deployed descriptor, and
reports that the resulting spectrum is less compressed. NSP replaces DiVA's
learned/fixed task decomposition and decorrelation with an online empirical
centroid projector plus an energy hinge. That is an estimator and allocation
wrapper around the same supervision object and action, not a new kind of
supervision.

Primary source:

- Milbich et al., *DiVA: Diverse Visual Feature Aggregation for Deep Metric
  Learning*, ECCV 2020:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530579.pdf

S2SD, BIER/A-BIER, non-isotropy/spectrum regularization, and generic gradient
projection are additional adjacent components. The exact online centroid-SVD
plus energy hinge may be unreported, but exact wrapper novelty cannot rescue an
occupied training relation whose distinguishing “label-null” semantics are
false.

## Controls, forecast, and protocol mismatch

- Full-vector NT-Xent, a fixed coordinate block, and an energy-only hinge do not
  isolate NSP from DiVA's learned task decorrelation or from a random projector
  matched for retained energy and rank.
- C8 (`Q=0`) is a sanity no-op, not a causal control.
- The proposed test probe uses unseen/test identity labels to decide whether the
  mechanism worked; it may be reported after a frozen evaluation but cannot be
  a train-side authorization measurement.
- The base and multi-proxy rows are invented recipes, not PFML reproductions.
  Matching PFML's proxy count does not reproduce its objective or training.
- The best directly specified forecast is `0.729` CUB and `0.908` Cars, both
  below PFML (`0.734`, `0.927`). The claimed crossing exists only after adding a
  guessed `+1.0` to an unimplemented faithful PFML reproduction, despite the
  proposal saying PFML's loss and recipe are unknown. This does not satisfy the
  standing objective.
- `docs/search_protocol.md` requires corrected In-Shop screening after Gates
  1--3. NSP predicts `Q` approaches zero and declines to target In-Shop, so it
  cannot follow the mandatory screen while preserving its own mechanism.

## Authorizing condition

There is none for the frozen proposal. Replacing the centroid complement by a
measured causal subspace, changing same-class-negative instance supervision,
or supplying a non-occupied supervision relation would be a new proposal and
must restart blind generation and freezing.
