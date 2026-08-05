# Pass 21 local audit: Reliable-Interaction Maximization

Date: 2026-08-05 UTC. This audit was written independently while the mandatory
frozen-proposal review `06478848f06b4c24` was still running.

## Verdict before reviewer reconciliation

**DEAD at Gate 1 and mathematically non-authorizing at Gate 2.** No diagnostic,
preregistration, implementation, or GPU run. The narrow combination of
class-residualization with a repeat-reliability trace may remain bibliographically
distinct, but the frozen causal argument and objective do not survive algebra.

## Frozen object

RIM class-centres each of two augmented-view descriptor batches, forms pooled
within-view covariance and cross-view covariance of the residuals, applies a
stop-gradient inverse-square-root EMA whitener, and maximizes
`tr(W Sigma_12 W)` up to a target. It adds this term to a newly constructed
multi-proxy Proxy Anchor base and deploys the ordinary 512-D cosine descriptor.

## Gate 1: no eligible causal provenance

The verified packet establishes one corrected-corpus fact: controlled
augmentation responses define a non-distance-reducible relation between
*different same-class images* on In-Shop. It does not measure the rank,
repeat-reliability, covariance, or zero-shot utility of same-image augmented
residual descriptors. It explicitly says no artifact establishes a causal map
from the observed response relation to official-query repair.

RIM additionally requires three unmeasured premises: that proxy training deletes
augmentation-reliable within-class factors; that preserving those factors raises
unseen-identity R@1; and that the effect is governed by `max(0,d-C+1)`. None is in
the verified packet. The last premise is contradicted by RIM's own base: with 15
sub-proxies per class on CUB/Cars, the proxy-gradient span can be full 512-D, so
counting only `C-1` class-mean directions cannot motivate the forecast ordering.

## Gate 2: formal failures

### The claimed escape from class quantization has zero gradient

At the frozen class-quantized state, both residual views are zero. The proposal's
own derivative is `dT/dr_i^(1) = W W r_i^(2)/nu` (and symmetrically for view 2),
so every RIM gradient is exactly zero there. Calling this state a strict loss
maximum does not provide an optimization escape direction. The sentence
"strictly penalized, with nonzero gradient" is false, and the claimed
"unique maximizer-inverse" attack fails at the load-bearing degeneracy.

### The reliable-rank identity is stated too strongly

`T = sum lambda_s/(lambda_s+lambda_eta+epsilon)` in a shared eigenbasis requires
signal and noise covariances to commute. In general the invariant object is a
trace/generalized-eigenvalue expression. Finite-batch `Sigma_12` is not forced
symmetric or positive semidefinite, so `T` can be negative. Consequently the
frozen loss is not bounded in `[0,1]`: for `T < 0`,
`max(0,1-min(T,r*)/r*) > 1`.

### The target is almost the batch-rank ceiling

With 24 classes and four distinct images per class, class centring leaves only
`nu = 96-24 = 72` residual degrees of freedom. The CUB/Cars target `r*=64` asks
the noisy per-step cross-covariance to occupy almost its maximum possible rank,
not 64 independently evidenced semantic factors. No estimator-bias or sampling
analysis supports that target.

### Several shortcut and fixed-point claims do not follow

- A pooled whitener does not penalize augmentation-stable instance codes; it
  normalizes them and can make a shared high-rank code satisfy the objective.
- Independent augmentation parameters do not make all nuisance contributions
  independent across views of the same source image.
- A detached inverse covariance does not imply the asserted fixed point
  `barSigma proportional to I`; the base loss, L2 sphere, ridge, hinge, and EMA
  all alter the dynamics.
- The EMA update does not explicitly detach `hatSigma` before persistent state
  assignment, leaving the executable autograd/state transition underdefined.
- Five Newton--Schulz iterations are specified without the actual recurrence or
  a condition-number/convergence guard at an absolute ridge near `1e-4`.
- The population claim that every proxy/pair loss is minimized by a label-only
  descriptor is not proved by substituting a conditional mean on a normalized
  sphere through the complete positive and negative log-sum-exp objective.

### Baseline and forecast are not matched to the frontier

MP-PA is an invented smooth-max multi-proxy objective, not PFML. Matching proxy
count does not match PFML's objective or recipe. RIM changes augmentation,
batch structure, dataset passes, pooling/recipe assumptions, optimizer, and
schedule. Its CUB `0.741` frontier claim is therefore an unsupported absolute
forecast over an unvalidated `0.720` constructed baseline, not arithmetic from
repository measurements.

## Prior-art boundary

The core estimator is occupied mathematics. Dmochowski et al. (NeuroImage
2015) and Parra, Haufe, and Dmochowski, *Correlated Components Analysis*
(2018), maximize between-repetition covariance relative to within-repetition
covariance to extract repeat-reliable components. Barlow Twins (Zbontar et al.,
ICML 2021) trains a shared image encoder on two augmented views by driving their
normalized cross-correlation toward identity, jointly producing invariance and
non-redundant directions. Partial CCA classically applies the same covariance
machinery after regressing nuisance covariates from both views.

The exact package "apply a scalar reliable-rank trace to class-centred deployed
DML descriptors" was not located in this bounded search, so this audit does not
claim an exact bibliographic collision. But class residualization plus a hinge
does not repair the frozen algebra or supply Gate-1 provenance. The proposal is
dead without needing to overstate the novelty search.

Primary sources:

- Zbontar et al., *Barlow Twins*, ICML 2021:
  https://proceedings.mlr.press/v139/zbontar21a.html
- Dmochowski, Greaves, and Norcia, *Maximally reliable spatial filtering of
  steady state visual evoked potentials*, NeuroImage 2015:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6583904/
- Parra, Haufe, and Dmochowski, *Correlated Components Analysis -- Extracting
  Reliable Dimensions in Multivariate Data*, 2018:
  https://arxiv.org/abs/1801.08881

## Independent-review reconciliation

The mandatory frozen review returned **DEAD** through both provider paths. Its
additional decisive findings are accepted:

1. The five-step trace-normalized Newton--Schulz operator does not approximate
   the specified 512-D inverse square root. Even at the isotropic initialization,
   the scalar iterate begins near `1/sqrt(512)` and remains far from one after
   five steps; low-variance directions are under-whitened most severely. Thus
   the executed statistic trends toward reproducible variance rather than the
   advertised reliability rank. IterNorm's groupwise construction exists in
   part because full-width whitening converges poorly under this iteration.
2. Exactly, with `u_i=W r_i^(1)` and `v_i=W r_i^(2)`,
   `T = tr(W hatSigma W) - (1/(2 nu)) sum_i ||u_i-v_i||^2`. At stationarity the
   first term is ridge effective-rank/variance expansion and the second is
   whitened cross-view invariance. RIM is therefore a coupled version of its C5
   and C3 null mechanisms. F4 tests each null *individually* at a 70% threshold,
   so it cannot reject an additive combination in which neither arm reaches
   70% alone.
3. Ermolov et al.'s W-MSE already trains whitened augmented views through their
   cross-view MSE, and Zhang, Jayasuriya, and Berisha (NeurIPS 2023) explicitly
   add an intra-class-correlation repeatability regularizer to contrastive
   embedding training. MIC (ICCV 2019) already targets class-exogenous factors
   such as viewpoint and illumination for DML generalization. These do not make
   the literal class-residual hinge conjunction word-for-word identical, but
   they occupy its estimator, auxiliary-loss role, and causal motivation.
4. The actual 15-proxy base makes the `d-C+1` forecast mechanism void; the
   sampler is also not executable for classes with fewer than four images; the
   smooth-max score can exceed one while inheriting PA's calibration; and the
   proposed memory-bank variant changes the objective to current-versus-stale
   temporal consistency.

The review corrects one overstatement in the initial local audit: for the exact
population whitener, a shared ordinary eigenbasis is unnecessary to bound the
generalized reliability eigenvalues in `[0,1)`. The proposal's scalar
same-index formula still assumes simultaneous diagonalization, while the
finite-batch nonsymmetric cross-covariance can still make `T` negative and the
claimed `[0,1]` loss range false.

One reviewer assertion is explicitly **rejected**. PFML does use `M` learned
proxies per class to represent out-of-batch subpopulations; the CVPR 2025 paper
defines them in Sec. 3.2.2, and the local fidelity audit binds `M=15` on
CUB/Cars and `M=2` on SOP to the released recipe. PFML is not merely a
proxy-free continuous sample field. This correction does not rescue RIM:
matching PFML's proxy *count* still does not make MP-PA's smooth-max objective a
matched PFML reproduction.

Additional primary sources:

- Ermolov et al., *Whitening for Self-Supervised Representation Learning*,
  ICML 2021: https://proceedings.mlr.press/v139/ermolov21a.html
- Zhang, Jayasuriya, and Berisha, *Learning Repeatable Speech Embeddings Using
  an Intra-class Correlation Regularizer*, NeurIPS 2023:
  https://arxiv.org/abs/2310.17049
- Roth, Brattoli, and Ommer, *MIC: Mining Interclass Characteristics for
  Improved Metric Learning*, ICCV 2019: https://arxiv.org/abs/1909.11574
- Bhatnagar and Ahuja, *Potential Field Based Deep Metric Learning*, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf

## Authorizing condition

There is none for the frozen proposal. Fixing the zero-gradient state,
symmetrizing/constraining the estimator, changing the base, or replacing the
trace objective would be a substantive new method and must restart blind
generation and freezing under `docs/search_protocol.md`.
