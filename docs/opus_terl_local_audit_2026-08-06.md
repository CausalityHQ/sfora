# Pass 44 local evidence-aware audit: TERL

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_terl_proposal_pass44_2026-08-06.md`  
Frozen full-proposal SHA-256: `dca492ecaa6734f3fa405663f54083cc5772d11cad6f226a64266e8c2f44da1b`  
Exact provider answer SHA-256 (before repository terminal newline): `e067b8ee78c40a2d23342d2299eeba1ad75b4510e29bffb0d0b10dd252f28bb5`  
Durable proposal consultation: `b785dd99870c46e5` (Fable credit failure, same-job Opus fallback)

This audit was written after freezing the exact complete provider answer and
before requesting or reading an independent review.

## Verdict

**DEAD at Gates 1 and 2, independently.** Tail-Extrapolated Return-Level
Metric Learning is the tenth internal proposal to fit or approximate the
training negative-similarity tail and turn an extrapolated gallery-scale
extreme into supervision. The repository has direct adverse measurements for
this causal premise, and the supervision object is exactly occupied internally
and publicly. The proposal also forecasts no defensible frontier crossing. No
diagnostic, preregistration, implementation, or candidate GPU is warranted.

## Gate 1: the measured tail premise is adverse

TERL claims that a fitted GPD shape contains stable population information
about rare unseen-identity collisions that ordinary batch losses cannot see.
No corrected repository measurement supports that claim. Pass 31 XTail tested
the same premise prospectively: fitted shape added only `0.0087` incremental
R-squared beyond threshold and mean excess, a tuned constant had lower deep-
quantile RMSE (`0.0918` versus `0.1575`), and shape noise consumed 67--86
percent of the full depth signal. TERL supplies no new measurement that reverses
those results.

The proposal's example of two hypothetical tails differing by `0.253` cosine
is constructed arithmetic, not a repository observation. Consistency of POT
functionals under an asymptotic domain-of-attraction condition does not show
that one pooled GPD fits the finite, bounded, dependent, identity-mixture
similarities produced during training, nor that fitted parameters transport to
disjoint identities. The proposal itself calls a 344-times SOP extrapolation
and mixture misspecification its primary risk.

Its dataset-dose test is also malformed. For the frozen values,
`ln(M*zeta/r)` is about `3.52` CUB, `3.84` Cars, `5.01` In-Shop, and `5.84`
SOP; the table states `4.60` for In-Shop. More importantly, training image count
changes class count, class size, product duplication, and negative-mixture
composition simultaneously, so a four-dataset slope cannot identify gallery
depth as the cause.

## Gate 2: exact tenth recurrence

Internally, EVPC, RLM, EGR-PFML, PORTAL, PORT, XTail, POTER, CFEV, and Pass 42
GEVS already fit or approximate an upper negative-similarity tail and
backpropagate a train/gallery-scale extrapolated extreme. TERL is the tenth
instance. Its pooled index-flood PWM estimator, smoothed threshold, EMA shape,
and custom stop-gradient change the estimator and backward wrapper, not the
supervision object: **penalize a fitted extreme of the observed negative-score
tail at a larger nominal retrieval depth**.

Publicly, WEINCE supplies online EVT correction of bounded contrastive scores;
TriSim fits GPD similarity tails inside retrieval training; LDReg
differentiates an in-batch EVT/LID statistic; Recall@k surrogates, ranked-list,
top-k/CVaR losses, XBM, and large-batch DML occupy operating-depth pressure.
The proposal's novelty table omits WEINCE, TriSim, and LDReg even though they
were already established by the repository's primary-source audits. Its
distinctions from post-hoc OpenMax/EVM and generic pair weighting therefore do
not address the nearest supervision action.

Primary neighbours already audited:

- Erol et al., *WEINCE*, ICML 2026.
- Zheng et al., *TriSim*, CVPR 2026.
- Huang et al., *LDReg*, ICLR 2024.
- Patel et al., *Recall@k Surrogate Loss*, CVPR 2022.
- Wang et al., *Cross-Batch Memory*, CVPR 2020.

## Executable defects independent of recurrence

The collapse proof is false for normalized cosine. If every descriptor is the
same unit vector, every pairwise cosine has zero first derivative with respect
to either pre-normalized descriptor: the loss is positive, but the collapsed
state is stationary. This exact bug already recurred in GEVS.

The proposed backward is deliberately not the derivative of its stated loss:
the threshold participates in the forward return level but is detached inside
the exceedances specifically to remove an adverse derivative. That can define
a custom update, but it invalidates claims based on descent of the displayed
scalar objective and makes `no-sg` more than an ordinary ablation. The global
pooled shape also couples every query gradient to all other sites through a
high-variance ratio whose denominator can approach zero; a low-mean guard does
not guard that denominator.

The positive term is a soft **best-positive** (`logsumexp`), so TERL inherits
Easy-Positive-style neglect of other within-class modes. Its `O(B^2)` pairwise
matrix, per-query sorts, and global sort are not `10^-7` of a training step;
sort comparisons are not FLOPs-equivalent to a dense convolution, and the
claimed sub-1.5-percent overhead requires measurement.

The proposed composite changes the PA baseline to a ResNet-50/no-head-BN,
batch-180, 200-epoch recipe with unresolved source correspondence, bypasses
the corrected paired BN-Inception In-Shop reference, and gives every arm large
weight-decay/margin/temperature searches without a frozen raw-versus-final
selection plan. It therefore cannot inherit PFML or DADA frontiers.

## Standing-objective failure

TERL candidly forecasts no frontier crossing on any dataset: CUB `0.716`
versus PFML `0.734`, Cars `0.899` versus `0.927`, SOP `0.828` versus `0.829`,
and a statistically unresolved In-Shop `0.933` versus PA+DADA `0.930` with
unreported reference uncertainty. Even exact agreement with every forecast
would not fulfill the standing objective.

## Correct pieces worth preserving

- The GPD mean-to-scale conversion and continuous `xi -> 0` return-level limit
  are useful diagnostic mathematics.
- The proposal explicitly identifies the adverse threshold-gradient path and
  does not pretend its detached backward is conservative.
- C1 (no extrapolation), C3 (matched fixed coefficient), calibration, XBM,
  fixed-shape, and no-EMA controls are sensible for an EVT diagnostic.
- Deployment remains legal and unchanged; source ambiguities and non-crossing
  forecasts are reported candidly.

## Mechanism lesson

Ten independent gallery-tail rediscoveries are now a hard pre-GPU stop family.
Changing the tail estimator, pooling scheme, threshold smoother, or backward
surrogate does not reopen it. A future candidate must first supply a new
prospective corrected measurement that reverses XTail's shape-instability
result and must change what supervision exists, not merely how the same fitted
extreme is estimated.

## Frozen independent review and reconciliation

The independent cold review ran as durable consultation
`53e7fadb33c04a42`: Fable exhausted its credits and the same job completed
under Claude Opus. Its exact final answer is frozen at
`docs/opus_terl_review_2026-08-06.md`, file SHA-256
`19e3ee80d7f6370c182a59c98bf514d08b807190a9443d37ee0083197f1e9884`
and pre-terminal-newline provider-answer SHA-256
`4be2572bd0c1dbc7a67f05465317b3755eee6cdcef582838de4491a69c7a79b2`.

The reviewer independently returns **DEAD**, and finds an even earlier analytic
failure in the executable gradient. Because normalized exceedances divide by
a detached site mean, pooled `alpha0` is one. Differentiating the shared PWM
shape makes the shape path dominate the honest mean-excess path with the wrong
sign on most of the band: depending on `xi`, 36--72 percent of the top-band
negatives are pushed toward **higher** similarity. The exact bottom-band
shape-to-scale gradient ratio is reported as 2.5 at the `xi=-3` clamp, 6 at
`xi=-1`, and about 21 at `xi=0` and `0.4` on SOP. Thus the novelty-bearing
path recreates the threshold-gaming shortcut that the custom stop-gradient was
introduced to remove.

The reviewer's concrete population calculation starts from exponential
excesses with `u=0.40`, mean `lambda=0.05`, and SOP scale. Flattening all top
negatives upward to their own expected maximum raises the mean top-band
similarity from `0.45` to about `0.68`, yet lowers the fitted return target from
`0.855` to about `0.74` by driving shape to the lower clamp. This is analytic
model evidence, not a benchmark measurement, but it is a direct shortcut of
the frozen equations. At the clamp, the shape gradient dies and TERL reduces
to a fixed `u+(4/3)lambda` control; away from it, the estimator is gameable.

The reviewer also catches `ceil(0.15*176)=27`, not 26, so pooled `N=4860`, not
4680; reproduces the zero first-order cosine gradient at collapse; confirms
In-Shop's dose is `5.01`, not `4.60`; and finds the four-dataset slope
underidentified and non-decisive. It independently identifies WEINCE and LDReg
as closer train-time tail precedents, notes that WEINCE deliberately stops
gradients through fitted tail parameters, and concludes that pooling/EMA does
not reverse the supplied XTail evidence.

There is no conflict. The local audit kills TERL at repository Gates 1 and 2
from adverse measurements and exact recurrence; the cold review kills it even
before those gates because the executable update optimizes the estimator
shortcut. Correct PWM/return-level algebra, the threshold-path diagnosis,
index-flood variance reduction, legal deployment, and candid non-crossing
forecasts remain useful diagnostic pieces. TERL remains dead without GPU work.
