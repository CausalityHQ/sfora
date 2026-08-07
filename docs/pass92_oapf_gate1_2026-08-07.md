# Pass 92 — OAPF Gate 1 diagnostic (2026-08-07)

## Result: dead before retrieval training

OAPF was the only current candidate with a narrowly surviving Gate-2
mechanism: use each image's measured response to label-preserving
augmentations as that image's plateau radius in a potential-field positive
force.  The preregistered In-Shop training-only diagnostic required the radius
to reproduce across two independently seeded six-view augmentation packs
(global and within-class residual Spearman rho >= 0.50) before any retrieval
run.

The digest-pinned Proxy Anchor epoch-10 checkpoint and training split were
used.  No query/gallery images were loaded.  The diagnostic found:

* global radius Spearman rho = **0.3175929**;
* within-class residual radius Spearman rho = **0.1840571**;
* binary held-out compatibility prevalence = 0.4725820 (the prevalence
  condition itself passed).

Both reliability thresholds failed, so all downstream AUC, direction,
permutation, and distance-decile gates are false by construction.  OAPF is
**dead at Gate 1** and no OAPF retrieval training or GPU benchmark is
authorized.  The negative is mechanistic: augmentation-orbit radius is not a
stable per-image quantity at this operating point, so it cannot safely define
an endpoint-specific potential-field plateau.

Artifact: `reports/generated/oapf_gate1_seed0.json` on the DGX, generated from
checkpoint SHA-256
`31307c9e0ce816397e3d3b3ff3f0084dc84b3ef47e8e9847ecbc71fa3b97fcbd`.
