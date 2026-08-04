# Candidate 371: supervised-rank versus demanded-rank measurement audit

Date: 2026-08-04.

## Frozen measurement proposal

A clean, repository-blind `claude-fable-5` pass was asked for one prospectively
specified train-side measurement rather than another method. It proposed:

1. train Proxy Anchor on 75 CUB training classes and reserve 25 training classes
   as a pseudo-unseen validation population;
2. define a supervised subspace from the centered proxy/class-mean span and
   measure full-, span-, and complement-only retrieval on the held-out classes;
3. fit rank-truncated linear metrics on held-out-class folds to estimate a
   "demanded rank" `r*`;
4. constrain proxies to fixed random subspaces of ranks 8, 24, 74, and 512 at
   fixed class coverage; and
5. compare that lever with a class-count lever over 24 three-seed runs.

Its target branch required demanded rank at least twice the proxy effective rank,
useful retrieval in the pretrained complement, subsequent complement loss, and a
three-point rank-lever effect. That branch would supposedly license an
architecture whose supervised loss has zero gradient support in the complement
while a deployed block preserves pretrained structure.

## Why the proposed cheap gate is tautological

For 75 proxies, the centered proxy matrix has algebraic rank at most 74. An SVD
returning `k <= 74` is guaranteed by matrix dimensions; it cannot establish
channel starvation or falsify the competing geometry explanation. An effective
rank much below 74 may describe collapse, but the proposal's central
`k approximately 74` result is not empirical evidence.

The demand statistic is also not a rank requirement of the benchmark. A linear
map fit on 12 held-out classes and evaluated on 13 different classes measures
transfer of one regularized metric estimator under one split. Its truncation
curve depends on estimator, shrinkage, finite samples, and coordinate gauge. It
does not identify the minimum intrinsic dimension needed by unseen classes.
The proposed `r*_x > 512` outcome is literally unobservable when the input
descriptor and rank sweep both stop at 512; failure to saturate by 512 is not an
estimate above 512.

## The causal lever does not isolate supervised gradient rank

Constraining proxy parameters to a rank-`r` subspace changes proxy geometry and
optimization, but it does not constrain the learned representation's gradient
support to `r` directions. For normalized descriptor `z = h / ||h||`, the raw
descriptor gradient includes both the projected proxy combination and a
sample-direction term. Both then propagate through a shared nonlinear backbone.
Candidate 365 recorded this exact correction. A retrieval change across random
proxy ranks could therefore be conditioning, margin capacity, normalization,
or optimization damage rather than mediation by a supervised-rank channel.

The `r=512` arm is not a null for the random-subspace parameterization unless its
optimization and initialization are proven identical to the ordinary proxy
matrix. Matching final dimensions alone does not establish byte-identical
training.

## Outcome-to-action collision

Even if the descriptive measurements were positive, no branch maps to an
unoccupied intervention:

- The primary action--make a supervised block invisible to the complement and
  preserve pretrained content in the other block--is candidate **365, Blind
  Subspace Allocation**. It is a direct-sum/shared-private representation, with
  the same proxy-nullspace diagnosis. Candidate 80 occupies weighted split
  descriptors and candidate 69 occupies shared/private decomposition.
- Allowing Proxy Anchor to see only a projected block while deploying the full
  vector changes the architecture/hypothesis space, but this does not make it a
  new supervision primitive. Deep disentangled metric learning and orthogonal
  subspace DML occupy the family.
- The alternate branch of overcomplete class-agnostic measurement functionals
  exported as one response vector is proxy/logit-response representation,
  Classemes, attribute/concept embedding, and conditional metric learning.
- A lane change to broader pretraining is not a novel in-lane method and must be
  reported as such.

The Fable pass tried to exclude regularization by defining it as any added term
whose coefficient can go to zero, then claimed the architecture cannot collapse
to regularization because it adds no term. That syntactic definition does not
establish mechanism novelty. The same preservation constraint can be expressed
as stop-gradient, a frozen/private branch, orthogonality, or a penalty; all
consume the same pretrained-model referent and preserve the same subspace.

## Prior art and repository anchors

- Garrido et al., *RankMe*, ICML 2023:
  https://arxiv.org/abs/2210.02885
- Roth et al., *Revisiting Training Strategies and Generalization Performance
  in Deep Metric Learning*, ICML 2020:
  https://proceedings.mlr.press/v119/roth20a.html
- Roth, Vinyals, and Akata, *Non-Isotropy Regularization for Proxy-Based Deep
  Metric Learning*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html
- `docs/fable_bsa_invention_collision_365_2026-08-04.md`;
- `docs/search_stopping_adjudication_353_357_2026-08-04.md`.

## Verdict

**DEAD before diagnostic execution; no preregistration, implementation, or
GPU.** The proposal does not supply the required new information-to-action map.
Its supply statistic is dimensionally guaranteed, its demand statistic is not
an identified intrinsic rank, its rank lever does not isolate gradient support,
and every action branch is already occupied. Spending 24 runs would produce an
interesting representation study but cannot reopen a novel method under the
standing protocol.
