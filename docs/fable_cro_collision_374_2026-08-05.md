# Candidate 374: Cross-identity Residual Orthogonality (CRO)

**Verdict: DEAD at Gates 1 and 2 on 2026-08-05. No diagnostic,
preregistration, implementation, or GPU.**

## Proposal and provenance audit

A repository-isolated Fable search, supplied with the audited numeric horizons
but no suggested method family, proposed adding a scale-free residual
decorrelation term to Proxy Anchor. For unit embedding `z_x` and its labelled
class proxy `w_c`, it defines

`r_x = (I - w_c w_c^T) z_x`, `u_x = r_x / ||r_x||`

and minimizes the squared inner product `E[(u_x^T u_y)^2 | c(x) != c(y)]`.
Fable called this Cross-identity Residual Orthogonality (CRO) and argued that it
removes low-rank nuisance factors shared across identities without permitting
class memorization.

Gate 1 fails. This repository has not measured the proposed mediator—test-class
cross-identity coherence of own-proxy residual directions—or shown that it
causes retrieval errors at fixed residual energy. The closest relevant
measurement points the other way: candidate 225's preregistered source-fold
within-class covariance transfer ratios were `0.9312, 0.9287, 0.9345`, all
below the locked `1.15` threshold and below the random-subspace reference. A
linear direction learned as within-class variation on one identity set captured
at least as much between-identity variation on another. That result does not
prove every nonlinear residual penalty is useless, but it does rule out the
claim that this repository supplies positive provenance for a transferable
shared-nuisance subspace.

Fable's numerical argument is also not repository evidence. Its assumed
baseline covariance statistics, in-house ViT baseline, and conversion from a
margin-noise variance ratio to Recall@1 were not measured here or supported by
a cited primary source. Its CUB forecast of `0.886` was compared to a misstated
PFML target of `0.878`, while its own prompt correctly supplied `0.878` only as
a separate higher-capacity observation and `0.766` as the broadly comparable
ResNet frontier. Changing tier does not repair missing matched measurements.

## Algebraic collapse of the novelty claim

The proposal's key equality is false under its own sampling rule. Let
`Sigma_c = E[u u^T | c]`, let class probability be `pi_c`, and let
`Sigma_bar = sum_c pi_c Sigma_c`. Conditional on drawing different classes,

`E[(u^T v)^2 | c != d]`

equals

`(tr(Sigma_bar^2) - sum_c pi_c^2 tr(Sigma_c^2)) /
 (1 - sum_c pi_c^2)`,

not `||Sigma_bar||_F^2`. Consequently the claimed unique isotropic minimizer
does not follow.

There is a direct counterexample to the stronger "memorization-proof"
statement. A training network can map every residual direction in class `c` to
a class-private unit vector `e_c`, with different classes assigned orthogonal
or low-coherence vectors and `e_c` chosen orthogonal to `w_c`. Then every
cross-class CRO term is zero (or small), while every within-class residual
distribution is rank one and maximally anisotropic. Normalizing residual
magnitudes prevents a shrink-to-zero shortcut but does not prevent a
class-conditional rotation/code shortcut. The use of labelled proxies makes
that shortcut particularly accessible. With more classes than dimensions, a
low-coherence frame or reused subspaces replaces exact orthogonality; the same
training-class coding mechanism remains.

There is a second geometric error. The residuals do not share one fixed
`(d-1)`-dimensional space: each class has a different proxy-orthogonal
hyperplane. Their pooled second moment lives in the ambient `d`-dimensional
space, whose isotropic Frobenius floor at unit trace is `1/d`, not `1/(d-1)`.
Subtracting the wrong constant does not alter gradients, but it invalidates the
stated optimum and mediator target. Moreover, unit-normalizing a nearly zero
own-proxy residual amplifies arbitrary direction noise, so the base loss
controlling residual magnitude does not make the CRO term control the deployed
similarity error in the claimed way.

## Prior-art neighbourhood

Even a repaired pooled or per-class decorrelation objective is occupied:

- Zhang et al., *Learning Spread-out Local Feature Descriptors*, ICCV 2017,
  introduce Global Orthogonal Regularization, matching the first two moments
  of inner products of non-matching unit descriptors, including the same
  squared-inner-product statistic and spherical target:
  <https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Learning_Spread-Out_Local_ICCV_2017_paper.pdf>.
- Shi et al., *Mimicking the Oracle: An Initial Phase Decorrelation Approach
  for Class Incremental Learning*, CVPR 2022, explicitly minimize the
  Frobenius norm of each class's normalized representation covariance to make
  its spectrum isotropic. This is the direct repaired form of CRO's claimed
  class-residual effect:
  <https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_Mimicking_the_Oracle_An_Initial_Phase_Decorrelation_Approach_for_Class_CVPR_2022_paper.pdf>.
- Roth et al., *MIC: Mining Interclass Characteristics for Improved Metric
  Learning*, ICCV 2019, explicitly identify viewpoint and illumination shared
  across classes and separate those cross-class characteristics from the
  identity embedding by mutual-information reduction:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html>.
- Roth, Vinyals, and Akata, *Non-isotropy Regularization for Proxy-based Deep
  Metric Learning*, 2022, already intervene on the class-local distribution
  around proxies and, importantly, find that preserving non-isotropic local
  structure improves zero-shot DML. CRO prescribes the opposite geometry
  without a repository measurement supporting that reversal:
  <https://arxiv.org/abs/2203.08547>.

Project candidate 69 had already rejected shared/private PCA leakage plus
orthogonalization as occupied disentanglement and candidate 70 had rejected a
cross-class isotropy tensor as an occupied geometry regularizer. CRO changes
the residual estimator and pair mask but not that method class; its only
purported escape—the cross-identity mask—creates the class-code degeneracy
above.

## Mechanism recorded

The candidate tried to remove nuisance-induced distractor similarity by making
normalized own-proxy residual directions incoherent across labelled training
identities. It dies because (1) no repository measurement establishes that
mediator, (2) the cross-class estimator is not the pooled covariance quantity
claimed, (3) class-private residual coding defeats the claimed
memorization-proof property, and (4) repaired decorrelation/disentanglement
forms are prior art. The DGX remains idle.
