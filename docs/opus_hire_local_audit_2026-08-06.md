# Pass 32 local evidence-aware audit: HIRE

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_hire_proposal_pass32_2026-08-06.md`  
Independent-review prompt: `docs/opus_hire_review_prompt_2026-08-06.txt`  
Review prompt SHA-256: `15a68ed8c5b24e67476fa4b38e0293d981ac5a13ae75457bc854f967d3ec1f78`  
Durable independent review: `377b8b30a2264746` (running when this audit was frozen)

This audit was written without reading the independent review's partial or
final answer. It binds HIRE to the repository's corrected measurements and
checks its estimator against the proposal's own equations.

## Verdict

**DEAD at Gate 1 and Gate 2, independently dead by the sign and zero set of its
central statistic; no preregistration, implementation, or GPU.**

## Gate 1: the causal premise has no positive provenance

HIRE assumes that heterogeneous, anisotropic identity-conditional covariance
causes corrected zero-shot retrieval error and that making training-identity
residuals isotropic and homoscedastic transfers to unseen identities. The
repository has measured no association between either proposed statistic and
official-query error, no matched intervention showing that conditional
whitening repairs retrieval, and no transfer of a common nuisance covariance
across disjoint identities.

The closest locked prospective measurement is adverse. Candidate 225's
source-fold within-class-subspace transfer ratios were `0.9312, 0.9287,
0.9345`, all below the preregistered `1.15` threshold: directions learned as
within-class variation on one corrected In-Shop identity set captured at least
as much between-identity energy on another. That does not prove every
second-moment penalty must fail, but it directly denies HIRE positive
repository provenance for a shared transferable nuisance geometry.

The proposal's assumed effective ranks, condition numbers, and conversion to
R@1 are forecasts, not measurements. BLenDeR's external-data augmentation gain
does not measure HIRE's covariance mediator and cannot anchor its magnitude.

## The cross-class statistic rewards the opposite of its stated target

For one batch define

```
C_c = (1/|A_c|) sum_{a in A_c} delta_a delta_a^T.
```

The executed statistic is exactly

```
U = 2/[P(P-1)] sum_{c<c'} tr(C_c C_c')
  = [||sum_c C_c||_F^2 - sum_c ||C_c||_F^2]/[P(P-1)].
```

With independently sampled classes its expectation can be written
`tr((E_c C_c)^2)`, as the proposal states. But this is a cross term with
every within-class self term deliberately removed. It neither estimates
`E_c ||C_c-I/d||_F^2` nor implies that every `C_c` is common and isotropic.

The cheapest counterexample is decisive. Assign each training class a private
unit residual direction `e_c`, mutually orthogonal when `P<=d`, and set
`C_c=e_c e_c^T`. Every class is maximally rank one, yet every cross-class
inner product is zero and therefore `U=0`. The proposed isotropic common
target gives `U=1/d`. Since HIRE **minimizes** `log(d U+1e-8)`, it strictly
prefers the class-private anisotropic code, driving the term from zero toward
`log(1e-8)`. Thus `U-1/d>=0`, the claimed effective-rank interpretation,
the unique isotropic optimum, and the advertised certificate are all false for
the statistic actually executed.

This is not a new defect discovered after a near miss. Candidate 374,
Cross-identity Residual Orthogonality, used the same cross-class
squared-inner-product construction and died on the same removed-self-term and
class-private-direction shortcut. HIRE changes how residuals and their scale
are defined, but preserves the failed shape statistic and its wrong zero set.

The ten overlapping pair differences within a class do not repair the issue.
They reduce effective sample size and make within-class terms dependent; the
cross-class factorization may remain unbiased over independently drawn
classes, but it is unbiased for the wrong population object.

## The written gradient is not the executed gradient

HIRE defines

```
delta = d / sg(max(||d||, rho_min)).
```

Because the entire denominator is detached, for `||d||>rho_min` the live
Jacobian is `I/||d||`, not the claimed tangent projector
`(I-delta delta^T)/||d||`. The tangent formula is the derivative obtained
when the norm is *not* stopped. HIRE therefore does not make shape and scale
gradients orthogonal; the stated reason for the stop-gradient has its sign
backwards.

In the clamped regime, `||delta||=||d||/rho_min<1`. Consequently
`tr(C_c)=1` fails, `1/d` is not the relevant floor, and `log(dU)` can be
reduced simply by shrinking residual magnitude. This adds a second direct path
toward the same low-`U` shortcut.

The collapse proof is not executable. At exact same-class collapse,
`R_c=0`, so the pseudocode evaluates `log(0)` in both scale terms and
produces infinities/NaNs. Adding an epsilon would make the scalar barrier
finite, but every squared pair distance has zero first derivative at identical
descriptors, so the claimed divergent derivative with respect to `log R_c`
does not imply a nonzero descriptor gradient at the symmetric state. Exact
normalized-cosine collapse is also stationary for the base loss. A high or
infinite scalar objective is not a proof of first-order escape.

## The causal discriminability proof overclaims

The displayed ratio

```
||w||^2 / sqrt[(w^T Sigma w)(w^T Sigma^{-1} w)]
```

is at most one. A condition-number Kantorovich inequality supplies a worst-case
lower bound, not an identity saying efficiency is determined entirely by
`kappa`. For any eigenvector `w` of an anisotropic `Sigma`, the ratio is
exactly one: Euclidean and Mahalanobis discriminants differ only by a positive
scale along that direction. Hence losslessness does **not** require
`kappa=1`, and equality in the lower bound is not “iff `kappa=1`.” The
proposal omits the alignment between between-identity mean differences and
covariance eigenvectors, which is load-bearing for its causal claim.

On normalized descriptors, a full-rank Euclidean isotropy target is additionally
not the same as tangent isotropy around a class mean. Sphere geometry couples
the permissible radial covariance to mean norm and within-class spread. HIRE
does not derive feasibility of one common `tau^2 I/d` for separated normalized
class distributions; its loss in any event constrains only pooled directional
cross terms and scalar pair distance, not those class covariances.

## Gate 2: both target and estimator family are occupied

The substantive target already appeared in this project's Pass 22 CINA audit.
Cheng et al., *Learning Deep Classifiers Consistent with Fine-Grained Novelty
Detection* (CVPR 2021), train class-conditional feature distributions toward
different means with one covariance shared across all classes and evaluate on
CUB. That is a stronger homoscedastic covariance target than HIRE's scalar
homogeneity plus pooled shape. Deep CORAL makes covariance alignment a
differentiable deep training loss, and Conditional Bures Metric supplies
conditional-covariance alignment. A cross-class U-statistic and a one-sided
scatter floor change the estimator and safeguards, not what supervision
exists.

The shape term independently repeats candidate 374 CRO. Its repaired
per-class form is occupied by Global Orthogonal Regularization and
class-wise decorrelation. Most adversely, Roth et al.'s NIR (CVPR 2022)
identifies proxy-induced local isotropy as destructive of transferable
intra-class structure and regularizes translations around proxies to preserve
non-isotropy. HIRE prescribes the opposite geometry without a new corrected
measurement supporting that reversal.

Primary sources:

- Cheng et al.: <https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.pdf>
- Deep CORAL: <https://arxiv.org/abs/1607.01719>
- Conditional Bures Metric: <https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Conditional_Bures_Metric_for_Domain_Adaptation_CVPR_2021_paper.pdf>
- Global Orthogonal Regularization: <https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Learning_Spread-Out_Local_ICCV_2017_paper.pdf>
- NIR: <https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf>

## Recipe, controls, and frontier

The fixed `P=30,K=5` sampler duplicates identities with fewer than five
images, common in SOP and In-Shop. Repeated images make pair differences zero,
activating the clamped non-unit regime and corrupting both the shape and log
scale statistics. The proposal does not specify a distinct-sample fallback.

The claimed frontier is not a frozen executable arm. HIRE defines and tunes
itself on an invented 200-epoch Proxy Anchor recipe, then forecasts additivity
on a PFML implementation whose disclosed recipe it says it could not retrieve.
Its own standalone PA+HIRE forecasts remain below PFML on CUB, Cars, and SOP.
The only frontier crossings therefore depend on an unimplemented PFML+HIRE
composition and unmeasured near-additivity. The standing search protocol also
requires an In-Shop screen before CUB; the proposal explicitly declines an
In-Shop forecast.

Controls C1--C9 cannot rescue a statistic whose global preference is a
class-private rank-one code. C1 does not actually guarantee the same
post-normalization `R_c`; C7 touches test labels for a report-only mechanism
diagnostic and cannot justify selection; C8 changes the PFML/Proxy Anchor base
mechanics while the central estimator already fails.

## Mechanism lesson

Cross-class orthogonality is not class-conditional isotropy. Removing
within-class self terms creates precisely the private class-code solution that
an isotropy certificate must penalize. The repository has now seen this exact
estimator identity fail twice and the broader homoscedastic-covariance target
at least twice. Future proposals must compare their statistic's exact finite-
batch zero set against the advertised population property before novelty
search or GPU.

## Reconciliation with the frozen cold review

The independent Opus review completed after this audit was frozen and returned
**DEAD**, with the same earliest failure: T1's zero set is cross-class
subspace separation, not class-conditional isotropy. It strengthens the local
counterexample in four useful ways.

First, exact degeneracy is dimensionally feasible on the two forecast
fine-grained datasets. Five samples give each empirical class residual
covariance rank at most four, while CUB has 100*4=400<=512 and Cars has
98*4=392<=512; all training-class residual subspaces can therefore be mutually
orthogonal in the deployed dimension. A direct 4,000-step CPU descent on the
frozen pseudocode drove d*U from 1.0043 to zero and L_shape from +0.0043 to
-11.5427, while per-class effective rank fell and condition number rose. Adding
a Proxy-Anchor term did not reverse the sign.

Second, direction normalization creates an additional identification error:
E[dd^T/||d||^2] is not Sigma/tr(Sigma) for an anisotropic law. In the
reviewer's four-dimensional Gaussian check, the leading directional moment
was 0.6205 where the normalized covariance prediction was 0.8182.
Consequently HIRE's reported r_eff is not the covariance effective rank named
in its causal story.

Third, finite-difference/autograd checks confirm the local derivative audit.
The stopped denominator gives Jacobian I/rho; the measured radial shape
gradient was +0.025418, exactly the non-tangent analytic value. Below the
clamp, contributions shrink as rho^4, furnishing another reward for
contraction. At exact collapse the executable log-scale terms yield NaN/Inf
rather than the claimed repulsive dynamics.

Fourth, the full proposed programme is about 1,206 GPU-hours, not merely the
near-zero per-step overhead, once its tuning, controls, and seed table are
counted. This makes the proof-level rejection especially consequential.

The review correctly resolves NIR as a normalizing-flow density intervention,
not literally a second-moment sign flip. Its published diagnosis remains
adverse evidence: Proxy Anchor's locally isotropic residuals lose semantic
context. The local Gate-2 verdict is stronger because the cold reviewer was
intentionally barred from repository history and therefore did not see the
exact candidate-374 recurrence or Pass-22 CINA/Cheng collision.

One reviewer conclusion is not adopted. Its fixed-tangent-noise calculation
shows that a common natural construction on the unit sphere has much smaller
radial than tangential variance; it does not prove that every spherical
distribution with nonzero mean and isotropic covariance is geometrically
impossible. A scalar radial mixture with appropriately coupled uniform tangent
directions can satisfy the first two moments when the moment inequalities are
feasible. The narrower conclusions remain: HIRE does not derive feasibility
for its trained distributions, its statistic does not enforce the target, and
Frobenius effective rank does not control the condition number used in its
causal bound.

Frozen review: docs/opus_hire_review_2026-08-06.md.
