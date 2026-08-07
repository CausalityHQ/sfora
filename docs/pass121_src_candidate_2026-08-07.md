# Pass 121 — Stoichiometric Residual Coalition (SRC)

Status: **LIVE-NARROW; no GPU authorization yet.**

## Gate 1: provenance

The corrected CUB decomposition attributed 51.9% of failed queries to
between-class centroid overlap. CIS addresses that component with a summed
union-labelled coalition. SRC is the falsifiable follow-up: for a bundle of
distinct-class images, supervise the normalized sum against the union of
member proxies and supervise every leave-one-out residual against the
corresponding complementary proxy set. The complementary equation tests
whether the learner can identify which class contribution is missing, rather
than merely learning that an arbitrary mixture is a valid multi-label object.

## Gate 2: distinction audit

DCML (Zheng et al., CVPR 2021) applies losses to learned composites of
sub-embeddings; HSE (Yang et al., ICCV 2023) uses mixed samples to create
additional supervision. Neither checked primary source trains a real-image
leave-one-out coalition against complementary class-proxy targets. This
distinction is narrow and must be demonstrated in code and controls; SRC is
not being called novel yet.

## Gate 3: preregistration (before any deciding GPU run)

The paired corrected In-Shop Proxy Anchor reference is 0.9163033. If the
coalition screen is authorized, SRC predicts corrected best R@1 **0.9192**.
The candidate is falsified if corrected best is **<0.9180**, or if it fails to
strictly beat both the CIS union-only arm and a same-compute no-residual
coalition control by 0.0010. Raw best-over-training and selection-corrected
values must both be reported.

Required CPU tests: finite loss/gradients, permutation invariance of the
bundle, changed targets under each omission, and a test that a two-member
residual cannot silently equal the ordinary single-image objective.

## Gate 4–7

Screen In-Shop first, one seed only if the controller authorizes it. A pass
requires independent confirmation, selection-bias correction, and replication
on CUB or Cars196. No SOTA claim follows from this screen alone.

