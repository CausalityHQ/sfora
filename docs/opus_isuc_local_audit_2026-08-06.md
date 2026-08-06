# Pass 50 ISUC authoritative local audit — 2026-08-06

## Decision

**DEAD at Gates 1 and 2; no implementation and no GPU.** Identity-Split Update
Coherence (ISUC) has no verified repository measurement showing that update
directions from disjoint training-identity subsets predict corrected zero-shot
retrieval. More decisively, it independently recurs candidate 228, which this
repository killed on 2026-08-02: maximizing an inner product between gradients
from disjoint identity partitions is the occupied MLDG/Reptile/Fish gradient-
alignment mechanism. Restricting the statistic to the last-layer update field,
normalizing it to a cosine, and replacing domains/tasks by random identity
subsets changes the estimator and wrapper, not the training object.

The frozen proposal is
`docs/opus_isuc_proposal_pass50_2026-08-06.md`, SHA-256
`441bda5aeafc43e50831223e64f2e4ef174d0cdf523a64195ae93d39d7a6f367`.
It was committed before this audit. This verdict does not alter or repair that
proposal.

## Gate 0/1: the required causal premise was not measured

The current admissible evidence boundary is
`docs/current_evidence_reliability_audit_321_2026-08-03.md`. Its verified
corrected In-Shop evidence includes:

- final-artifact Proxy Anchor reference scores and exact corpus/scorer checks;
- cross-seed top-1 error overlap `0.7675401522`;
- exact top-1 gallery-row agreement `0.8084822057`; and
- same-wrong-identity agreement `0.6475770925` among jointly wrong queries.

None of these measures the proposal's `G(S)`, the held-out-identity update
coherence `A`, an identity-specific update residual `E(S)`, or a relation
between any such quantity and final retrieval error. Persistent query errors
can arise from shared visual ambiguity, label granularity, acquisition
shortcuts, or capacity; they do not identify incoherent training updates as
their cause. The frozen proposal instead cites published PFML point estimates
and the benchmark's disjoint-identity protocol. A protocol fact is not a
repository measurement of the proposed causal error mode.

The nearest historical measurement—17.94% negative cosine among nominally
same-class Proxy Anchor embedding gradients in candidate 223—is outside the
current verified packet and cannot be promoted without artifact-level
revalidation. Even taken descriptively, its mean cosine was `0.4162`, so it did
not establish that gradient incoherence caused zero-shot errors. Candidate 228
already asked the exact stronger question (whether an update learned from one
identity population transfers to another) and found no matched provenance
quantity. ISUC therefore fails Gate 1 before its forecast can authorize work.

## Gate 2: occupied gradient-alignment mechanism

For two losses on disjoint partitions, the standard one-step meta objective is

```text
L_2(theta - eta grad L_1(theta))
  = L_2(theta) - eta <grad L_2(theta), grad L_1(theta)> + O(eta^2).
```

MLDG explicitly trains so that a step improving virtual-train domains also
improves virtual-test domains and analyzes this as alignment of their
gradients. Reptile's first-order analysis shows that its update maximizes inner
products between gradients of different minibatches. Fish makes the object
fully explicit: it maximizes inter-domain gradient inner products, with direct
optimization requiring second derivatives and its first-order procedure
approximating that objective.

Primary sources:

- Li et al., [*Learning to Generalize: Meta-Learning for Domain
  Generalization*](https://ojs.aaai.org/index.php/AAAI/article/view/11596),
  AAAI 2018 (MLDG).
- Nichol, Achiam, and Schulman, [*On First-Order Meta-Learning
  Algorithms*](https://arxiv.org/abs/1803.02999), 2018 (Reptile).
- Shi et al., [*Gradient Matching for Domain
  Generalization*](https://openreview.net/forum?id=vDwBW49HmO), ICLR 2022
  (Fish).

ISUC directly optimizes

```text
A = <G_1, G_2> / (||G_1|| ||G_2|| + epsilon),
```

where each `G_k` is a gradient-derived last-layer update field from a disjoint
identity half. This is normalized gradient matching. The proposed distinctions
do not rescue novelty:

1. **Identity subsets instead of named domains.** Random identity subsets are
   pseudo-environments/tasks. Candidate 228 already used disjoint DML identities
   for the same transfer claim. The application is additionally adjacent to
   disjoint-label episodic/meta-DML, including Zheng, Lu, and Zhou,
   [*Deep Metric Learning With Adaptively Composite Dynamic
   Constraints*](https://doi.org/10.1109/TPAMI.2023.3234536), TPAMI 2023.
2. **Disjoint proxy parameters.** The head/backbone parameter `W` being aligned
   remains shared. Giving each partition separate label-specific proxies is the
   expected construction for disjoint label spaces; it does not change the
   `-<G_1,G_2>` regularizer on the shared representation.
3. **Closed-form last-layer factorization.** Writing a shared-parameter gradient
   as `sum_i d_i h_i^T` is a computational shortcut used in gradient-coherence
   and influence work. It changes which coordinates are measured, not the
   objective of making two partition-induced updates agree.
4. **Cosine rather than an unnormalized inner product.** Normalization removes
   scale. It remains a differentiable gradient-direction alignment penalty.
5. **No virtual inner step.** Fish already identifies direct gradient-inner-
   product optimization as the second-order object and introduces its first-
   order algorithm only to approximate it. Directly differentiating ISUC's
   alignment scalar is closer to that occupied object, not farther from it.

This is also an exact internal recurrence, not merely broad thematic overlap.
`docs/class_disjoint_meta_update_audit_228_2026-08-02.md` records the same
proposed causal quantity, the same disjoint-identity construction, and the same
gradient-alignment reduction. Under the search protocol, an estimator-level
variant of an already rejected mechanism stops at Gate 2.

## The proposal's transfer decomposition is not established

Even setting novelty aside, Proposition 1 does not prove the advertised
interpretation of `A`.

1. It moves from expectations of numerator and squared norms to an expectation
   of a normalized random ratio. In general,
   `E[X / sqrt(YZ)] != E[X] / sqrt(E[Y]E[Z])`; the approximation has no error
   bound and becomes least reliable when the proposal expects `A` near `0.05`.
2. The two balanced halves are complements of one finite batch, not independent
   identity samples. Sampling without replacement introduces dependence, while
   repeated `Q` splits reuse the same images and identities.
3. The restricted PFML energies remove all cross-half sample/proxy interactions.
   Therefore `G_1 + G_2` is not the update field of the full batch objective,
   and the two fields are not unbiased subdivisions of one SGD update.
4. A last-layer field need not preserve alignment of backbone gradients. The
   latter additionally depends on every sample's backbone Jacobian; alignment
   of `sum d_i h_i^T` does not identify alignment of
   `sum J_i^T W^T d_i`.
5. Shared update direction is not synonymous with useful transferable
   structure. A nuisance shared across identities can align both fields, which
   the proposal itself admits. The proposed held-out-identity correlation is
   mechanism tracking after training, not an exclusion of that shortcut.

The finite-batch centering description is also inaccurate. With half-specific
`dbar_k`, global `hbar`, and `n_k = |S_k|`, direct expansion gives

```text
sum_i (d_i-dbar_k)(h_i-hbar)^T
  = sum_i d_i h_i^T - n_k dbar_k hbar_k^T,
```

where `hbar_k` is the half mean. The global feature mean cancels because the
centered `d` values sum to zero. Thus `G_k` is not simply the exact `W` gradient
with a common global mode removed; it removes a half-specific rank-one term.
That makes the statistic still more dependent on the arbitrary split.

## Degeneracy and cost claims do not establish safety

- Bounding the *value* `1-A` in `[0,2]` does not bound its gradient. The cosine
  derivative can grow as either `||G_k||` approaches zero despite the
  `epsilon` term. Consequently P2's scalar energy comparison does not establish
  benign optimization dynamics.
- The `0.1998` argument considers one selected repulsive configuration against
  a `2 lambda` value bound. It does not prove that the full summed PFML-plus-
  ISUC objective excludes lower-rank, shared-nuisance, proxy-mediated, or
  trajectory-level shortcuts.
- `d_i = M h_i` can encode a shared nuisance or generic low-rank force map. Calling
  every shared linear map “the target” assumes the very transfer semantics that
  the method is supposed to identify.
- Four restricted differentiable energies for `Q=2`, graph retention through
  their error signals, and a double backward are not justified by the proposal's
  forward-FLOP arithmetic. Kernel launches, autograd graph storage, and
  Hessian-vector work require measurement; the stated `<0.2%` FLOPs do not prove
  the `<=1.05x` wall-clock condition.

## Protocol and frontier failures

The protocol requires corrected In-Shop screening first with a same-seed
current-digest reference. ISUC gives **no In-Shop forecast**, then specifies five
CUB seeds and all controls on CUB. It does not define raw-best versus
independently selected/final reporting or an out-of-sample confirmation split.
Its Cars forecast `0.931` is below its own `0.932` frontier bar. Its marginal CUB
forecast depends on a future PFML reproduction whose primary recipe remains
materially under-specified, as recorded in
`docs/pfml_prelaunch_fidelity_audit_2026-08-03.md`.

These protocol defects are independent of the Gate 1/2 death. Repairing them,
changing the partition statistic, or substituting a different update object
would be a new proposal and must restart blind generation.

## Mechanism-level verdict

ISUC asks the model to make two class-disjoint identity subsets induce aligned
last-layer updates. That is normalized gradient matching over pseudo-
environments: an occupied MLDG/Reptile/Fish object and an exact recurrence of
candidate 228. The repository has no verified evidence that this update
coherence is the cause of its corrected retrieval errors, and the proposal's
ratio, independence, decomposition, degeneracy, cost, and screening claims do
not close the gap. Stop before code or GPU.
