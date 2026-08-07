# Pass 117 — Margin-conditioned path harmonization (DEAD at Gate 1)

## Candidate

MIPH would retain Proxy Anchor and add a Gaussian perturbation consistency
penalty only in the subspace orthogonal to the own-class proxy, weighted by the
detached proxy margin deficit. The aim was to suppress nuisance-sensitive local
path distortion without suppressing the identity direction. Plain Gaussian
Jacobian/embedding-drift regularization is occupied by PMH; the projected,
hardness-conditioned variant would have required separate controls.

## Gate 1 diagnostic

The seed-0 exploratory measurement on 1,200 corrected In-Shop epoch-10 images
gave weak drift/margin correlations (about -0.09 at normalized noise σ=.01)
and slightly higher drift on errors. I therefore repeated the exact diagnostic
at the same operating point for seeds 1 and 2, using their digest-bound epoch-10
embeddings and checkpoints, before any implementation.

The result was not stable: drift/margin correlation was **+0.00762** for seed 1
and **−0.17188** for seed 2 at σ=.01 (the σ=.02 values were +.00957 and
−.14701). Error-versus-correct drift also reversed direction on seed 1. The
registered premise is therefore not a reproducible property of the operating
point. MIPH is **DEAD at Gate 1**; no code, prior-art escalation, or GPU run is
authorized. The PMH paper remains a literature lead, not repository evidence
for a method here.
