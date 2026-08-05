# Failed pass-eight near-miss: Class-Conditional Rank Barrier

Date: 2026-08-05. The blind proposer consultation
`b5a40f14c63242d2` ended with status `failed`: Fable stopped, and the automatic
Claude Opus fallback also exited nonzero after writing a coherent `NONE`
response. The exact partial output is preserved at
`docs/fable_ccrb_failed_output_pass8_2026-08-05.txt`. This is not a numbered
candidate under `docs/search_protocol.md`; no diagnostic, implementation,
preregistration, or GPU is authorized.

A separate fresh review of the frozen text was started as durable consultation
`4bb4c918c3f34e7a`. Its verdict will be resolved below. The local audit does not
depend on model agreement.

## Frozen method

CCRB pools the within-class covariance of normalized embeddings, mixes the
current batch covariance into a stop-gradient EMA, trace-normalizes it to mean
eigenvalue one, and penalizes

```
L_rank = (kappa / d) tr((hat_Sigma + kappa I)^-1).
```

It adds one-way two-view invariance on 25% of samples and Proxy Anchor on the
base view. The claimed certificate says that if the embedding factors through
any `k`-dimensional label-sufficient shortcut, the within-class covariance has
rank at most `k`, so at least `(d-k)/d` of the barrier remains. The response
returned `NONE` because the barrier cannot choose which content fills the
new dimensions, its claimed certificate does not cover high-class-count
In-Shop/SOP, its forecasts do not clear two targets, and adjacent variance/
covariance regularizers occupy the mechanism.

## Formula audit

For a positive-semidefinite `hat_Sigma` with eigenvalues summing to `d`, the
stated scalar bounds are correct. Convexity makes the minimum
`kappa/(1+kappa)` at the isotropic spectrum. Rank at most `r` implies at least
`d-r` zero eigenvalues and therefore the lower bound `(d-r)/d`. The stated
trace-normalization gradient is also algebraically consistent for a symmetric
matrix.

Those facts do **not** establish the load-bearing theorem.

## Decisive algebraic failure: latent dimension does not bound nonlinear covariance rank

The implication

```
z = g(s), dim(s) = k  =>  rank Cov(z) <= k
```

is false for nonlinear `g`. A scalar can index `d+1` affinely independent
points on the unit sphere (for example, the vertices of a regular simplex in
`R^d`). A deterministic piecewise mapping from that scalar to those points has
full-rank covariance while adding no information beyond the scalar. Smooth
neural networks can approximate the same mapping or a curve passing through
affinely independent neighbourhoods.

Thus a one-dimensional shortcut can satisfy CCRB by a nonlinear full-rank code.
The rank bound is valid only when `g` is affine or its image is otherwise known
to lie in a `k`-dimensional affine subspace. CCRB imposes no such restriction on
the ResNet/head. Replicating a scalar linearly fails, as the frozen text says;
nonlinearly lifting it succeeds. This destroys the proposal's claimed
“rank-exclusion theorem,” not merely its effect-size forecast.

The remaining content-selection failure is also real. Even an affine-support
barrier can be filled by augmentation-stable background, viewpoint, sensor, or
arbitrary label-irrelevant codes. The one-way invariance term removes only
variation induced by its chosen augmentations and supplies no identity-specific
referent for the remaining dimensions.

## Gate 1: no eligible causal provenance

The corrected evidence packet establishes near-saturated In-Shop training
retrieval and a replicated fraction of disconnected within-class graphs. It
does not measure a beneficial relationship between within-class covariance
rank and corrected test retrieval. The older outcome association that motivated
variance-preservation candidates is outside the post-audit-321 evidence tier.
CCRB's own class-count argument says its certificate is vacuous on corrected
In-Shop (`3997 > 512`) and SOP (`11318 > 512`). No verified CUB/Cars artifact
identifies low within-class covariance rank as the causal error source.

## Gate 2: occupied mechanism even after removing the false theorem

What remains is class-conditional variance/covariance preservation plus
augmentation invariance:

- VCReg adapts VICReg-style variance/covariance regularization to supervised
  learning explicitly to improve transfer and address gradient starvation and
  neural collapse.
- NIR is benchmark-matched proxy DML that explicitly preserves class-local,
  non-isotropic structure around proxies for unseen-class generalization.
- *Deep Metric Learning Assisted by Intra-variance* preserves intra-class
  structure with self-supervised synthesis/ranking on the same four benchmark
  family.
- Repository candidate 156 already rejected a log-determinant/set-volume
  requirement as occupied variance preservation; changing log-determinant to
  a trace-inverse spectral barrier changes the scalar estimator, not the
  supervision relation.
- Candidate 186 independently rejected protecting current class-covariance
  directions as established non-isotropy preservation plus gradient surgery.

Primary sources:

- https://arxiv.org/abs/2306.13292
- https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html
- https://arxiv.org/abs/2304.10941
- https://arxiv.org/abs/2306.03440

## Forecast and cost

The frozen forecast was approximately `0.735 ± 0.008` CUB and
`0.910 ± 0.010` Cars. The corrected 512-D targets are PFML `0.734 ± 0.003`
CUB and `0.927 ± 0.003` Cars, not DADA's lower CUB/Cars rows. The mean forecast
barely clears CUB by 0.1 point and misses Cars by 1.7 points; it cannot satisfy
the two-dataset outcome even before the algebra and novelty failures.

The matrix solve itself is small relative to a ResNet batch, but a strong
second view on 25% of samples makes the claimed training cost approximately
1.25x, not approximately 1x. Inference is unchanged.

## Local verdict

**DEAD at Gates 1 and 2.** The only claimed substantive distinction—a proof
that the barrier excludes low-information shortcuts—is false under nonlinear
encoding. The executable residue is occupied within-class variance/covariance
regularization, lacks verified causal provenance, and does not forecast a
two-dataset crossing. No GPU follows.

## Independent frozen-text review

Pending consultation `4bb4c918c3f34e7a`.
