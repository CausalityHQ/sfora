# Candidate 225: contraction geometry and nuisance-transfer audit

Date: 2026-08-02. This document was written before reading the diagnostic
values. No GPU run is authorized by this registration.

## Method proposal and Gate-2 verdict

The motivating measurement is exact: at the In-Shop epoch-10 operating point,
Proxy Anchor's positive term predicts a same-acquisition cosine increase of
`7.77e-5` per unit step and a cross-acquisition increase of `4.16e-5`, while the
negative term slightly opposes the gap. The observed acquisition gap grows from
`0.0251` near initialization to `0.1804` at epoch 10.

Candidate 225 proposed changing the frame in which own-proxy attraction acts so
that its induced contraction is independent of acquisition alignment. It is
**DEAD at Gate 2**. For unit embeddings `z_i,z_j`, unit proxy `p`,
`a=z_i.p`, `b=z_j.p`, `c=z_i.z_j`, and positive PA weights `w_i,w_j`, the exact
first-order cosine change is

```
d c = eta [w_i (b - a c) + w_j (a - b c)].
```

At equal weights this is `eta w (a+b)(1-c)`. Both `(1-c)` and PA's hard-positive
weighting favour the less-similar cross-acquisition pairs. Their measured
contraction is nevertheless smaller, so joint proxy alignment `(a+b)` is the
rich-get-richer coordinate. Every available intervention changes either the
weights (general pair weighting), the attractor (multiple/moving proxies), or
the metric of the step (preconditioning/natural gradient). There is no fourth
slot. A frame change is therefore an optimizer/preconditioner, not new
supervision.

This gives an algebraic closure of the acquisition-drift branch rather than
another naming-based rejection.

## New CPU-only measurement: within-class nuisance transfer ratio

The repository has measured per-image and per-class nuisance structure, but not
whether a nuisance-heavy direction learned on one set of identities remains
nuisance-heavy and identity-light on disjoint identities. That property is the
minimum prerequisite for a class-exogenous supervision source that can cross a
zero-shot class split.

For each of the three digest-bound In-Shop PA epoch-10 training packs, normalize
the embeddings and retain classes with at least three examples. Sort the labels;
even-indexed labels form fold A and odd-indexed labels fold B. In both A->B and
B->A directions, estimate the pooled within-class covariance on the source
fold. For `k in {1,2,4,8,16,32,64}`, let `P_k` project onto its leading `k`
eigenvectors. On the disjoint target fold compute

```
w_k   = tr(P_k S_target_within) / tr(S_target_within)
b_k   = tr(P_k S_target_between) / tr(S_target_between)
rho_k = w_k / b_k.
```

The deciding statistic is `k=32`. Controls are 100 fixed-seed Haar-random
32-dimensional subspaces and a label-permuted source-fold null preserving class
sizes. Independent seed spaces are never compared by principal angles because
their embedding gauges need not align.

### Locked decision

- **PASS:** in all three seeds and both fold directions, `rho_32 >= 1.60`,
  `rho_32 / rho_perm_32 >= 1.35`, and `w_32 >= 0.25`.
- **FAIL:** in at least two of three seeds, either fold-averaged `rho_32 <= 1.15`
  or fold-averaged `rho_32 / rho_perm_32 <= 1.05`.
- Otherwise the result is **inconclusive** and authorizes nothing.

All `k` values and both directions will be reported, but thresholds will not be
moved. FAIL closes only the linear transferable-nuisance-subspace family. PASS
does not license projection: NAP, WCCN, PLDA, nuisance projection, and natural
gradient are prior art. It licenses a new Gate-2 search for an operator that
uses a transferable coordinate without projecting or reweighting it. Any PASS
must be reported alongside the acquisition-series alignment ARI `0.754--0.761`.

Primary prior art delimiting the interpretation: Hatch, Kajarekar and Stolcke,
WCCN (Interspeech 2006); Prince and Elder, PLDA (ICCV 2007); Amari, natural
gradient (Neural Computation 1998); Martens and Grosse, K-FAC (ICML 2015); Wang
et al., General Pair Weighting (CVPR 2019); Kim et al., Proxy Anchor (CVPR 2020).

## Result

The diagnostic **FAILS** the locked criterion in all three seeds. At `k=32`:

| seed | A->B rho | B->A rho | fold mean | fold-mean w | fold-mean rho / permuted rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9182 | 0.9442 | **0.9312** | 0.3685 | 1.3053 |
| 1 | 0.9073 | 0.9501 | **0.9287** | 0.3513 | 1.3331 |
| 2 | 0.9165 | 0.9525 | **0.9345** | 0.3597 | 1.3109 |

Every fold-mean `rho_32` is below the preregistered `1.15` falsifier. Random
32-dimensional subspaces give rho means `0.998--1.002`, so the result is not a
normalization artifact. The source-fold within-class leading subspace captures
roughly 35--37% of target within-class variance but 38--40% of target
between-class variance: it is not nuisance-selective on disjoint identities.

The permuted-label control has still lower rho (`0.688--0.719`), so real class
structure is detectable relative to that anisotropic null. That does not rescue
the candidate: the registered claim required nuisance to dominate identity, not
merely to be less identity-heavy than a deliberately broken partition.

This closes the **linear, class-exogenous nuisance-subspace** route on this
operating point. It does not claim that no nonlinear transferable nuisance
exists. No implementation or GPU run follows.

Reproducibility:

- analyzer commit: `3095cac`
- analyzer SHA-256: `db15713899fbbcddc3090cd65c53f84989d0227f3b6e6ada8e72ec8006335ddf`
- result SHA-256: `28979ad9ce5d48deab36e155955046247be02b7b224919236f50144f54a37864`
- seed pack SHA-256 values: `85e76245...`, `ff30ac7f...`, `dfb72dde...`

