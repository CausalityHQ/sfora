# Pass 22 local evidence-aware audit: CINA

Date: 2026-08-05 UTC  
Frozen proposal: `docs/fable_cina_proposal_pass22_2026-08-05.md`  
Independent-review prompt: `docs/fable_cina_review_prompt_2026-08-05.txt`  
Review prompt SHA-256: `cb55dd10cfa26240c807d218e712ef720721ad9a7cd98b4df997a60a462d2e73`  
Durable independent review: `202f2f67b8b44602` (still running when this audit was frozen)

This audit was written without reading the independent review result. It binds
CINA to the repository's verified evidence packet and checks the proposal's
mathematics and its closest primary-source prior art.

## Verdict

**DEAD at Gate 1, independently DEAD at Gate 2, and not executable as frozen.**
No implementation or GPU run is warranted.

## Gate 1: provenance fails

CINA assumes that zero-shot retrieval is limited by identity-specific
within-class covariance shapes and that forcing a common cross-identity
nuisance geometry will transfer to unseen identities. The verified packet says
the opposite of “measured”: it explicitly lists **a shared cross-class nuisance
basis** among the premises that the repository does not support. None of the
reliable measurements estimates per-identity covariances, their shape
dispersion, their alignment with official-query errors, or a causal benefit
from making them proportional.

The reliable augmentation-response relation does not repair this gap. It shows
that a non-distance response graph exists on corrected In-Shop data; it does
not show that the response lives in a shared linear covariance basis, that
second moments are the relevant sufficient statistic, or that covariance
homogenization repairs unseen-identity errors. CINA's `+0.008/+0.006/+0.004`
forecasts are therefore invented rather than derived from a measurement in
this repository.

## Gate 2: the substantive target already exists

The proposal's nearest-work table omits a direct mechanism-level neighbour:

- Cheng et al., *Learning Deep Classifiers Consistent with Fine-Grained
  Novelty Detection* (CVPR 2021), define a Class-Conditional Gaussianity loss
  that trains class-conditional feature distributions toward different means
  and **one learned covariance shared by all classes**. The primary paper says
  the covariance need not be specified, is learned as a byproduct of the loss,
  and evaluates on CUB-200. This is stronger than CINA's proportional-shape
  target. CINA uses random identity pools, generalized spectra, proportional
  rather than equal scale, and a split null; those are estimator choices around
  an occupied supervision target, not a new kind of supervision.
- Deep CORAL (Sun and Saenko, 2016) already turns covariance alignment into a
  differentiable representation-learning loss. Its source/target-domain
  setting is not identical to CINA, but together with Cheng et al. it removes
  the defensible claim that using a covariance-alignment statistic as a deep
  training signal is an unexplored import.
- Conditional Bures Metric (Luo and Ren, CVPR 2021) further occupies learned
  alignment of conditional covariance/distribution operators. It is a domain
  adaptation method rather than DML, so it is adjacency rather than the
  decisive collision.

Primary sources:

- Cheng et al.: <https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.pdf>
- Sun and Saenko: <https://arxiv.org/abs/1607.01719>
- Luo and Ren: <https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Conditional_Bures_Metric_for_Domain_Adaptation_CVPR_2021_paper.pdf>

The novelty search was bounded and primary-source-first. A broader search could
find still closer work, but is unnecessary once the supervision target itself
is occupied.

## Executability and mathematical defects

These defects are independent of Gates 1 and 2.

1. **The frozen eigensolve is undefined at exactly the degeneracies it claims
   to handle.** `S` can have rank below `k=24`, especially as within-class
   residuals collapse. No positive ridge is added before `S^{-1/2}`. Clamping
   generalized eigenvalues after inversion and adding eigengap jitter do not
   make a singular inverse square root exist.
2. **The rank-one defence has the sign backwards.** If all identities share
   the same rank-one covariance shape, `Sigma_A` and `Sigma_B` are
   proportional, hence `D=0` (or the unregularized inverse is undefined).
   Shared low-rank over-invariance therefore perfectly satisfies CINA rather
   than making `D` large.
3. **Scale invariance leaves within-class collapse unopposed.** A base metric
   objective already rewards same-class contraction. A penalty blind to scale
   cannot prevent the covariance magnitude approaching zero, and the proposed
   split null merely closes the hinge there. This is a stationary escape, not
   a collapse defence.
4. **The executed finite-`k` method is not affine-invariant.** Selecting the
   Euclidean top-`k` eigenspace of `S` is not equivariant under a general
   invertible change of coordinates. Only the full-rank generalized spectrum
   has the claimed `GL(d)` invariance. Calling the finite method
   “affine-invariant” is therefore false.
5. **The Bayes-optimal cosine corollary does not follow.** Simultaneous
   whitening of proportional Gaussian covariances yields a shared spherical
   noise model, but a same/different Bayes test also depends on means, priors,
   norms, and the L2 normalization. It does not imply that cosine is the
   Bayes-optimal statistic.
6. **The null subtraction has no proved calibration.** The identity-subset
   statistic and same-identity sample-split statistic average different random
   objects. Proposition 2 is only a stated local approximation, so
   `D - stopgrad(D_null)` is not established as a debiased estimator of
   between-identity shape dispersion.
7. **CINA-Z exposes a radial shortcut invisible to deployment.** It penalizes
   raw unnormalized `z`, while the base loss and evaluation use normalized
   descriptors. Identity-dependent norm changes can alter CINA without
   changing cosine retrieval. Offering CINA-T as a validation-selected variant
   does not make the frozen CINA-Z claim valid.
8. **The noise-injection bound is invalid.** Proxy Anchor's attractive
   derivative is not globally bounded below by `alpha/4`; it can saturate and
   approach zero. The numeric lambda ceiling therefore does not structurally
   rule out the claimed degeneracy.
9. **The sampler is underdefined.** Requiring `m=5` or `m=9` distinct samples
   per identity needs a stated fallback for training identities with fewer
   images. Repetition changes the residual covariance and the purported null.

## Mechanism lesson

“Make within-class covariance shape common across training identities” is an
old homoscedastic representation target, not a new answer to what supervision
exists. More importantly, this proposal repeated the project's recurring
failure mode: it promoted a mathematically attractive latent error mechanism
without first measuring that mechanism in the corrected benchmark lane. The
next blind pass must bind its causal premise to one of the verified channels or
propose a prospective CPU diagnostic before claiming Gate 1.
