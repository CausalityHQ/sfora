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

Adversarially, Katsikas et al., *Your Dissimilarities Define You:
Complementary Learning Exploiting Class Diversities* (CVPR 2026), explicitly
supervises per-image distributions over non-target classes. That is a close
warning, not a mechanism-identical kill: it does not form a multi-image sum or
train one complementary target for each omitted coalition member. Any future
novelty claim must state this distinction and include a per-image
complementary-target control.

## Gate 3: preregistration (before any deciding GPU run)

The paired corrected In-Shop Proxy Anchor reference is 0.9163033. If the
coalition screen is authorized, SRC predicts raw best-over-training R@1
**0.9192**. The candidate is falsified if the independently frozen checkpoint
is **<0.9180**, or if it fails to strictly beat both the CIS union-only arm and
a same-compute no-residual coalition control by 0.0010. Report raw best and
frozen-checkpoint values; the existing leave-neighbour peak-gap output may be
reported only as a descriptive selection diagnostic, never as an identified
selection correction.

Required CPU tests: finite loss/gradients, permutation invariance of the
bundle, changed targets under each omission, and a test that a two-member
residual cannot silently equal the ordinary single-image objective.  The
deciding controls must include `pa_coalition_complementary`, which trains each
image independently against the other members' proxy labels; this separates
complementary targets from SRC's leave-one-out sum.

The residual operator and recipe are now implemented locally and the focused
objective/recipe suite passes (65 tests). This is only an implementation
readiness result; it is not a benchmark result and has not been copied into
the active DGX queue.

## Fresh Gate-2 adversarial audit (2026-08-08)

A second primary-source pass found no metric-learning paper that combines all
three defining operations: summing embeddings of distinct real images,
forming one leave-one-out coalition for each omitted member, and supervising
each residual against the complementary class-proxy target. Deep Sets and Set
Transformer are set encoders without this supervision; Proxy Synthesis uses
synthetic embeddings/proxies; multi-proxy methods use subproxies rather than
real-image coalitions; complementary-label learning assigns non-target labels
to a single image. SRC therefore remains **LIVE-NARROW**, not established
novel. It must die if its implementation collapses into per-image
complementary loss (Katsikas et al.), ordinary mixed-sample metric learning
(HSE/CIS), a learned compositor (DCML), or synthetic/union-only proxy
supervision (Proxy Synthesis).

The deciding controls are consequently: union-only coalition, no-residual
sum, per-image complementary target, and ordinary Proxy Anchor. This audit
does not authorize GPU work by itself.

## Implementation repair (2026-08-08)

The Gate-1 audit found that the dispatcher previously used residual-only
supervision. It now has an explicit `_stoichiometric_residual_coalition_loss`
composition containing both the union equation and all leave-one-out residual
equations, with regression tests for the two-member collapse and dispatcher
composition. This repairs the implementation mismatch only; the original
SRC benchmark prediction is void and requires a fresh preregistration before
any GPU screen.

## Gate 4–7

Screen In-Shop first, one seed only if the controller authorizes it. A pass
requires independent confirmation, selection-bias correction, and replication
on CUB or Cars196. No SOTA claim follows from this screen alone.
