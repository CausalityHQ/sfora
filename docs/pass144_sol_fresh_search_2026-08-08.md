# Pass 144 — Sol fresh invention review (2026-08-08)

## Scope

Read the handoff, search protocol, method ledger, and recent CIS/CEA/transport
audits. This was a read-only cross-field search across supervision, geometry,
optimisation, sampling, architecture, and activations. The active Pass133
In-Shop control run was deliberately excluded until its two controls finish.

## Verdict

**NONE at Gate 2.** No mechanism had both a new measured information source and
a defensible mechanism-level distinction from prior art. No GPU candidate is
authorised from this pass.

## Strongest rejected constructions

1. **Owner-resolved coalition credit.** CIS and plain Proxy Anchor both reached
   raw-best 0.9170; CIS finished 0.9149 versus 0.9158. The normalized coalition
   gradient gives every member the same proxy-space update. Linear leave-one-out
   or coded measurements therefore decode to ordinary per-image proxy logits.
   Nonlinear/context-conditioned variants become compositional or cross-image
   modelling. Collisions: Proxy Anchor, SRC, ECOC/vector-symbolic coding, DCML
   (CVPR 2021), and Cross-Image Attention (CVPR 2023).

2. **CEA/rival-conditioned conservative potential.** CEA was selective but
   collapsed in training (R@1 0.8738 -> 0.6156/0.5939 as its loss went to zero).
   Adding fixed attraction is balanced OT/soft pair weighting; retaining PA and
   adding contextual attraction is an auxiliary contextual regulariser.
   Collisions: Contextual Similarity Optimization, Contextual Similarity
   Distillation (CVPR 2022), Sinkhorn/OT DML, and Potential Field DML (CVPR
   2025).

3. **Gradient-feedback or transported-discrepancy supervision.** PA gradients
   contain 17.94% opposing same-class pairs, but shared proxy-coordinate signs
   make the natural compatibility mask inactive. Disjoint-identity transport
   was adverse (`rho32 = 0.9312, 0.9287, 0.9345`, below the preregistered 1.15).
   Executable forms collide with PCGrad/CAGrad, GRAD-MATCH, DML-ALA,
   influence/meta-weighting, Delta-Encoder, Meta Variance Transfer, Spherical
   Feature Transform, AdvRF (ICCV 2025), and VAPNet (NeurIPS 2023).

Activation and architecture variants likewise exposed no residual opening:
unseen features are inside training activation support, while routed heads,
local/global fusion, curvature heads, dendritic/KAN/state-space heads, and
multi-centre representations were either unmotivated or occupied.

## What would reopen the search

The next provenance measurement must discover a **class-disjoint transferable
information channel** before an operator is designed. Fit only on training
identities, then predict held-out-identity positive correctness after
residualising cosine distance, proxy margin, acquisition token, augmentation
dispersion, rival signature, and gradient coefficients. Require cross-fitted
`Delta AUC >= 0.05` with a bootstrap 95% lower bound above zero, independent-pack
or cross-seed reliability Spearman `>= 0.50`, and survival under
cross-acquisition-only evaluation. No currently persisted signal meets this
standard.

Fable and its automatic Claude fallback were unavailable at their weekly limit;
the Sol review is not evidence that the search is impossible.
