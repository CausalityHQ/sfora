# Pass 42 local evidence-aware audit: GEVS

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_gevs_proposal_pass42_2026-08-06.md`  
Frozen proposal SHA-256: `a3cc8a357a70f2b9baf54b7fc0ea1ae272fbb2e3e8af6450528cb05a1b0d50b4`  
Exact provider answer SHA-256 (before the repository terminal newline): `b687706373d54605e30d66232fd0e191a28db4f2006319e2dead467c212cbf90`  
Durable proposal consultation: `448f11df12ec4ddc` (Fable credit failure, same-job Opus fallback)

This audit was written before requesting or reading an independent review of
the frozen proposal.

## Verdict

**DEAD at Gates 1 and 2, independently, with false exact-reduction and
endpoint guarantees and no standalone frontier-crossing forecast. No
diagnostic, preregistration, implementation, or candidate GPU is warranted.**

## Gate 1: the repository has measured against the premise

GEVS assumes that a POT/GPD shape fitted to 24--36 upper-bulk, dependent
in-batch negative similarities transports to unseen identities and identifies
the negative maximum at a nominal training-set-sized exposure. There is no
prospective corrected repository measurement supporting that transport.
There is direct adverse evidence from the same statistical object.

Pass 31 XTail's frozen independent calculations found that a bulk-threshold
GPD shape added only `0.0087` incremental R-squared beyond threshold and mean
exceedance, while a tuned constant coefficient predicted the deep quantile
better (`0.0918` versus `0.1575` RMSE). Shape-estimator noise was 67--86
percent of the entire batch-to-SOP depth signal. In the hard-negative mixture
that best represents a hard rival identity, the fitted coefficient was
`13.38` against a required `5.09`. Only about 12.8 distinct classes supplied
36 nominal exceedances. GEVS uses fewer exceedances on CUB/Cars, adds shrinkage
and a label-count transform, but supplies no new measurement that reverses
those results.

The proposal's dataset-size ordering, claimed literature batch effects, and
forecast deltas are hypotheses, not measurements from this repository. It
explicitly says its load-bearing Multi-Similarity batch-ablation statement was
not checked against the paper. Its “familiarity bias” is also not repaired by
the written method. Gate 1 therefore fails before any new experiment.

## Gate 2: ninth internal gallery-tail recurrence

GEVS is the ninth blind return to the same supervision object:

1. Pass 9 EVPC fitted a negative-score GPD and proposed a deployment-size
   return-level repair.
2. Pass 14 RLM fitted a differentiable POT/GPD image/proxy tail and trained
   against its extrapolated maximum.
3. Pass 15 EGR-PFML added an image queue and full-gallery return-level margin.
4. Pass 23 PORTAL used PWM and differentiable train-set-sized gallery risk.
5. Pass 26 PORT used class-block negative tails and a population return level.
6. Pass 31 XTail used in-batch PWM and a training-set-sized deep quantile.
7. Pass 33 POTER used two GPD extreme layers and estimated gallery error.
8. Pass 35 CFEV pooled a negative tail, extrapolated its survival to
   `|D_train|`, and wrapped it as estimated R@1 error.
9. Pass 42 GEVS fits per-anchor PWM to the top 20 percent of batch negatives,
   extrapolates to `|D_train|`, and adds an ad hoc class-cluster correction.

The negative-shape clamp, detached threshold, EMA shrinkage, and proposed
“extremal index” are estimator and wrapper changes. The supervision referent
is unchanged: treat observed seen-identity upper-bulk similarities as a fitted
parametric tail that reveals the unobserved unseen-gallery maximum. Internal
Gate 2 rejects that rediscovery regardless of public novelty.

Primary literature independently occupies the broad mechanism. WEINCE uses
anchor-wise online extreme-value corrections for bounded normalized
contrastive scores. TriSim (CVPR 2026) fits generalized-Pareto similarity
tails and uses their probabilities in a retrieval-training loss. LDReg (ICLR
2024) differentiates an in-batch EVT/LID estimator through the learned
representation. Recall@k surrogate training, ranked-list objectives, XBM, and
large-batch DML occupy empirical operating-depth pressure. Post-hoc EVT image
search and relevance prediction use the same non-match upper-tail object.

Primary sources:

- WEINCE: <https://arxiv.org/abs/2606.00262>
- TriSim: <https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf>
- LDReg: <https://openreview.net/forum?id=oZyAqjAjJW>
- Recall@k surrogate: <https://arxiv.org/abs/2108.11179>
- XBM: <https://arxiv.org/abs/1912.06798>

## Frozen mathematical failures

### The endpoint guard is disconnected from the executable target

Before the guard, the displayed coefficient follows from
`sigma=(1-xi)*mean_excess`. The proposal then states

```
sigma <- min(sigma, (-xi)*(1-u))
```

but the boxed executable target remains

```
s_hat = u + c(xi,N,theta,zeta) * mean_excess,
```

where `c` contains no guarded `sigma`. If the minimum changes `sigma`, the
coefficient multiplying `mean_excess` must change with it. As frozen, the
guard changes an unused intermediate and does not enforce `s_hat <= 1`.
Consequently D1's finite-endpoint guarantee and its displayed derivative do
not describe the executable boxed loss.

### LSE is not recovered exactly

The log-sum-exp of a finite observed score set is not exactly a POT return
quantile. At `xi=0`, `theta=1`, and `N=M`, the GPD formula gives a
distributional extrapolation involving a fitted threshold, exceedance rate,
and scale. LSE remains a deterministic smooth maximum of all observed scores.
Equating `sigma=1/alpha` does not make these functions equal for arbitrary
score vectors. Thus the claimed exact baseline reduction and the statement
that every LSE loss implicitly assumes `N=M` are false.

### The “extremal index” is not an extremal-index estimator

`k / (# distinct labels in the top-k)` is mean exceedances per represented
class. Multiplying that value by training images-per-class and inverting it is
not a runs or inter-exceedance estimator of the extremal index. It uses no
ordering or cluster-separation rule, then transfers a class-balanced training
batch statistic to unseen query-gallery dependence. In-Shop makes the mismatch
visible: the training mean is 6.48 images per identity while the official
query/gallery structure and gallery identity counts differ. The correction is
an ad hoc label-multiplicity weight, not an EVT dependence estimate.

### The fitted sample is neither iid nor a tail sample

The top 20 percent is an upper-bulk threshold with only 24 or 36 exceedances.
Negatives come from a deliberately class-balanced `P x K` sampler, share the
anchor, repeat identities, and change every optimizer step. They are not iid
draws from the deployment gallery distribution. Replacing the sample size by
`|D_train|` cannot supply the missing unseen-identity law. The method itself
concedes that it leaves familiarity bias unresolved, which is one of its two
claimed causal biases.

### Exact normalized-cosine collapse is stationary

At identical normalized descriptors, every pairwise cosine is one, but the
tangent derivative of cosine between identical unit vectors is zero. A
positive scalar loss does not imply a first-order descriptor escape. The
claim that separating a class lowers the negative term “at first order” is
false on the normalized sphere; score changes begin at second order. D4 is
therefore not a strict-saddle proof.

The displayed per-score monotonicity is narrower than the claimed network
gradient as well: each descriptor also appears as a positive or negative in
other anchors' losses. Detaching the fitted statistics reduces the novel term
to an adaptive top-k negative penalty. It does not establish that the adaptive
coefficient estimates gallery risk; C1 is predicted by the earlier XTail
evidence to match or beat it.

## Arithmetic, controls, and protocol mismatch

The claimed CUB/Cars negative-class log ratios are arithmetically wrong. With
roughly 99/97 test negative identities and 29 batch negative identities, the
ratios are about `log(99/29)=1.23` and `log(97/29)=1.21`, not `0.81` and
`0.79`. This weakens the claimed four-dataset ordering.

C6 is internally contradictory as an attribution test. The proposal says
gallery exposure is the causal lever, yet predicts the method must be flat
over a 16-fold N sweep. If the sweep is flat, the train-count input is inert
over the tested range and a constant top-k coefficient is the immediate
explanation; flatness cannot distinguish EVT calibration from a margin.

The frozen primary screen is SOP (F1), contrary to this repository's
In-Shop-first protocol. It reports only five-seed means, not raw
best-over-training and independently selected/final values. The proposed
ResNet-50 recipe also does not match the corrected BN-Inception In-Shop lane
currently being reproduced here, and the claimed `0.913` baseline is a
forecast rather than a measured paired control.

Most decisively, standalone GEVS is forecast below PFML on SOP, below PA+DADA
on In-Shop, and far below PFML on CUB/Cars. Its only possible frontier claim
comes from adding a forecast delta to an unimplemented PFML reproduction or a
“DADA-strength” carrier. Those compositions are not frozen executable methods
and their additivity is unmeasured. Even if every standalone forecast landed,
the frozen object would not satisfy the standing objective.

The eight-combination `(lambda, weight_decay)` search plus several structural
hyperparameters creates substantial validation-selection capacity. Saying the
`lambda=0` baseline receives the same `(lambda,wd)` grid produces duplicate
baseline configurations, not a matched method search. A class-disjoint
training validation set is good practice but does not repair the unequal
effective search spaces.

## Correct pieces worth preserving

- R@1 is governed by a best-positive versus hardest-negative event, so
  batch-to-gallery operating depth is a legitimate measurement target.
- The detached-threshold change removes XTail's direct `(k+1)`-st-score
  attraction path.
- The unguarded fixed-shape coefficient has positive direct derivatives for
  selected top-k negative scores.
- Using training-only metadata avoids direct test-gallery leakage.
- Constant-coefficient, independence, large-batch/XBM, and carrier controls
  are directionally useful, although they cannot rescue Gates 1 and 2.
- The proposal plainly discloses source ambiguities, weak confidence, and its
  non-crossing standalone forecasts.

## Mechanism lesson

Nine blind rediscoveries show that “training sees fewer negatives than the
gallery, so extrapolate a GPD tail” is a language-model invention attractor,
not a repository-supported lead. More shrinkage, a bounded-shape clamp, an
ad hoc dependence factor, or a new soft-margin wrapper does not create new
supervision. Future candidates in this family remain pre-GPU dead unless a
prospectively frozen corrected measurement first demonstrates transport of
the exact tail object and the proposal supplies a materially different
supervision referent.
