# Pass 40 local evidence-aware audit: CNW

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_cnw_proposal_pass40_2026-08-06.md`  
Blind-proposal job: `e0719db954234ebd` (Fable credit failure, same-job Opus fallback)  
Proposal source SHA-256: `7999301181313b0f5d58efadbe160b16dab3b230ed41f743650f81f1d3578c32`  
Committed copy SHA-256: `e42f8c67fac43f1d5a4e081b96cdbe49ae95fb5722bce7121ebcb2288717fb8f`
(the committed copy differs only by one terminal newline)

This audit was written before launching or reading the mandatory independent
review. It binds the blind proposal to the repository's corrected evidence and
checks the executable statistic against the claim made for it.

## Verdict

**DEAD at Gate 1 and Gate 2, independently dead because the loss has a much
weaker zero set than the advertised identity-conditional isotropy; no
preregistration, implementation, or GPU.**

The exact method is cheap and its pooled participation-ratio calculation is
rotation invariant. Those correct properties do not rescue its provenance,
novelty, or causal identification.

## Gate 1: the repository evidence is absent and the closest locked result is adverse

CNW requires three empirical premises:

1. corrected Proxy Anchor errors are caused by anisotropy of a pooled
   within-identity covariance;
2. a conditional participation ratio is currently only `O(10)` and predicts
   official-query errors after controlling for ordinary compactness; and
3. a nuisance geometry learned on training identities transfers to unseen
   identities strongly enough that flattening it helps.

None has been measured here. The `O(10)` value, the CUB/Cars/SOP ordering, the
overlap `o`, and every R@1 conversion in the proposal are forecasts, not
repository measurements.

The closest prospective measurement points the other way. Candidate 225
learned leading within-class directions on one corrected In-Shop identity fold
and evaluated them on disjoint identities. Its locked `rho_32` values were
`0.9312`, `0.9287`, and `0.9345`, all below the preregistered `1.15` threshold:
source within-class directions captured at least as much target
between-identity energy as target within-identity energy. This does not prove
that every covariance intervention is harmful. It does deny CNW positive
provenance for a transferable nuisance covariance.

Passes 22 (CINA), 24 (EFML), 27 (FRAME), and 32 (HIRE) already recorded the
same missing measurement. Rephrasing the target as a participation ratio does
not create a new observed error mechanism.

## The executed zero set is pooled private-subspace allocation, not per-identity whitening

For the selected residual directions, CNW forms one matrix

```
A = sum_i rhat_i rhat_i^T
```

over **all classes in the batch**. It contains no per-class penalty. Therefore
`A` can be flat while every class covariance is low-rank, mutually different,
and maximally anisotropic.

There is an exact construction at the frozen CUB/Cars batch geometry. Give each
of the 30 classes a private, mutually orthogonal 3-D subspace. Put its four
residual directions at the vertices of a regular tetrahedron in that subspace.
For every class, the residuals sum to zero and

```
A_c = (4/3) I_3,
participation_ratio(A_c) = 3.
```

Across the 30 disjoint subspaces,

```
A = blockdiag((4/3) I_3, ..., (4/3) I_3, 0_422),
participation_ratio(A) = 90 = D_max,
L_CNW = 0.
```

A pure-CPU arithmetic check reproduced `pooled_pr=90`, `pooled_rank=90`, and
`loss=0`, while every per-class PR was exactly 3. CNW thus certifies its global
optimum on a supervised private-subspace code that cannot define an isotropic
covariance for an unseen identity. This is the same mechanism lesson as HIRE,
but CNW reaches it through pooled self terms rather than HIRE's removed self
terms.

The proposal anticipates one fixed low-rank `V` and declares it benign. The
construction above is more damaging: `V_c` can be identity-specific. A training
identity can be assigned a private nuisance plane with no rule for choosing the
plane of an unseen identity. The pooled statistic cannot distinguish that code
from a common transferable isotropic law. Shuffled labels, marginal isotropy,
between-class isotropy, trace, sign flip, and WCCN controls do not isolate this
escape. The required control is a per-class shape statistic or a disjoint-
identity transfer test, which would be a different proposal and still faces
the adverse candidate-225 result.

## The named covariance and exact-gradient claims are false

CNW acts on the second moment of **normalized residual directions**, not on
the within-class covariance. Under anisotropy,

```
E[rr^T / ||r||^2] != Sigma / tr(Sigma).
```

Pass 32 already established this distinction for HIRE. Flattening the former
does not prove whitening of the latter, and equal weighting deliberately
deletes the class/radius information needed to make that identification.

The smoothing makes even the direction-only claim inexact. For
`rhat=r/sqrt(||r||^2+eps^2)`, the exact Jacobian is

```
J = I/s - rr^T/s^3 = (I-rhat rhat^T)/s,
s = sqrt(||r||^2+eps^2).
```

The proposal displays an additional factor `||r||^2/(||r||^2+eps^2)` inside
the rank-one term, so its Jacobian is wrong. More importantly,

```
r^T J = eps^2 r^T / s^3 != 0.
```

At the gate `||r||=0.05`, `eps=0.01`, the radial Jacobian eigenvalue is
`0.754293` and `||rhat||=0.980581`; even at norm 0.10 it is `0.098519`.
Consequently the loss is not exactly scale free, individual residuals are not
equally weighted, its gradient is not exactly orthogonal to residual magnitude,
and it does not exactly preserve `tr(Sigma_W)`. The combined nonlinear network
update would not conserve that trace even if the isolated infinitesimal
descriptor gradient did.

Residual collapse is not blocked by the gate. At collapse `K` is empty,
`tr(A)=||A||_F=0`, and the written loss is undefined. If an implementation
returns zero or skips the term for an empty set, the gate removes the very
gradient claimed to escape collapse. Declaring such a run void after training
is instrumentation, not a barrier.

## The Bayes/Jensen argument does not establish the claimed retrieval optimum

For the proposal's scalar surrogate

```
f(v) = Phi(-a/sqrt(v)),  x=a/sqrt(v),
```

the sign of `f''(v)` is the sign of `x^2-3`. Strict convexity begins at
`x>sqrt(3)=1.732`, not at the proposal's `x >= 1.5`. A CPU check gives the
curvature sign term `-0.75` at 1.5. More fundamentally, retrieval R@1 is not
the equal-prior error of two Gaussian prototypes with fixed `||Delta mu||`;
direction, separation, covariance, gallery size, and nearest-neighbour events
are correlated.

The minimax statement is correct for choosing a covariance with fixed trace
when the signal direction is entirely unknown. But CNW neither fixes the trace
exactly nor forces the covariance to that minimax solution. On normalized
descriptors, mean separation and feasible conditional covariance are also
coupled by sphere geometry.

The claim that no post-hoc map reaches unseen identities is false as written.
A WCCN matrix estimated on training identities is applied pointwise to unseen
descriptors without fitting the unseen gallery; the proposal's own legal C6
does exactly this. Whether it transfers is empirical, not a categorical
difference between a training loss and a training-fitted transform.

## Gate 2: the supervision target is occupied internally and publicly

Even repairing the pooled-zero-set defect does not yield a novel supervision
object.

- **DVML** (Lin et al., ECCV 2018) explicitly separates intra-class variance,
  drives it to one class-independent isotropic Gaussian, and evaluates the
  resulting DML system on CUB, Cars196, and SOP. Its decoder/generation
  machinery differs, but the target that intra-class variation share an
  isotropic distribution across identities is older and benchmark matched.
- **Cheng and Vasconcelos** (CVPR 2021) add a differentiable class-conditional
  Gaussianity constraint with different class means and a shared learned
  covariance for fine-grained unseen-class detection, including CUB. It is a
  stronger distributional version of a common conditional-covariance target.
- **NIR** (Roth, Vinyals, and Akata, CVPR 2022) is not the same statistic, but
  it studies the same proxy-local sample distribution on the same DML
  benchmarks and supplies directly adverse evidence: proxy-induced local
  isotropy loses semantic intra-class structure, and deliberately restoring
  non-isotropy improves generalization.
- WCCN/PLDA and learned retrieval whitening already establish the
  training-identity within-covariance-to-unseen-cosine action. Moving the same
  action inside encoder training is an implementation location, not by itself
  a new source of supervision.

Within this repository, CINA, EFML, FRAME, and HIRE have already exhausted
shared/homoscedastic/isotropic conditional-variation targets. CNW's pooled
participation-ratio wrapper is a new estimator with a weaker zero set, not a
new mechanism.

Primary sources:

- DVML: <https://openaccess.thecvf.com/content_ECCV_2018/papers/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.pdf>
- Cheng and Vasconcelos: <https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.pdf>
- NIR: <https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf>
- Whitening-loss analysis: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/c057cb81b8d3c67093427bf1c16a4e9f-Abstract-Conference.html>

## Protocol, controls, recipe, and frontier defects

1. The protocol requires a corrected paired In-Shop screen after Gates 1--3.
   CNW instead makes CUB five-seed gain its principal F1 gate and provides no
   standalone In-Shop A0/A1 forecast.
2. The project's verified corrected In-Shop reference is Proxy Anchor seed 0,
   raw best `0.916303` and independently exported final `0.913701`. CNW invents
   a PA+DADA reproduction at `0.926 +/- 0.004` and a composed `0.936` without
   repository evidence, implementation, or a raw/final selection correction.
3. The corrected In-Shop recipe has trainable BatchNorm. CNW freezes it and
   therefore does not specify a matched project screen.
4. `m_c=min(4,n_c)` is inconsistent with a fixed `C_b x 4` batch when classes
   have fewer than four distinct images. The sampling probability, variable
   batch size, and loss normalization are undefined.
5. Selecting the stronger of two baseline schedules must happen on a
   training-only validation split. As written, choosing after benchmark-test
   evaluation is test selection.
6. The proposed `z` values divide forecast mean differences by a root-sum of
   **sample standard deviations**, not the paired standard error of the mean.
   They are not paired significance tests or posterior crossing
   probabilities.
7. A batch Gram may be cheap, but `3e-6` counts one forward Gram against a
   forward-plus-backbone estimate and omits its backward and kernel overhead.
   The correct cost claim is “small and measurable,” not a guaranteed
   `1.00x`.

These are secondary because Gates 1 and 2 already stop the candidate.

## What survives

- The rank ceiling `rank(A) <= B-C_b` is correct despite residual
  normalization: nonzero column scaling preserves each class-centred residual
  span of dimension at most `m_c-1`.
- The participation ratio and its displayed gradient with respect to `rhat`
  are rotation invariant and algebraically correct.
- The Gram implementation is compact and deployment is byte-identical.
- A **diagnostic** measuring pooled directional PR, true pooled covariance PR,
  per-class low-rank allocation, and their association with corrected errors
  could test the missing premise. It would not revive CNW at Gate 2, because
  the repaired conditional-isotropy target is occupied.

Process lesson: a pooled conditional statistic is not a per-condition
constraint. Before treating effective rank as transferable supervision,
construct disjoint private subspaces and check the exact global minimum; then
separate normalized-direction moments from covariance moments.
