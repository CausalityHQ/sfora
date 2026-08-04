# OAPF Gate-0 artifact revalidation

Date: 2026-08-04.

## Question

Candidate 174 was one of the few mechanisms that narrowly survived prior-art
review before dying on a data diagnostic. Because historical negatives can be
invalidated by missing artifacts or implementation defects, this audit checks
whether its Gate-1 death remains usable under the current Gate-0 standard.

## Surviving immutable artifacts

The DGX retains all deciding sources:

- checkpoint `arcg_inshop_pa_epoch10_seed0.pt`, SHA-256
  `31307c9e0ce816397e3d3b3ff3f0084dc84b3ef47e8e9847ecbc71fa3b97fcbd`;
- training report SHA-256
  `e84aa1b7a0e3ee052b5bd4ce13a6a8e77396cb4f4738797a83853c4f4ded92cc`;
- deterministic view ledger SHA-256
  `b2f0d80e02d8ae3d9e41cc20016bf69788582cd53a7847514c00669b869facbb`;
- embedding pack SHA-256
  `17cc963e47de9ae89b88ab4aa535aff53da73a4ff14cc5db1db5d30bae6b51d3`.

The pack contains 25,882 canonical embeddings, labels, raw pre-normalisation
norms, and two independent packs of six 512-dimensional views. Canonical and
view vectors are L2-normalised to float precision.

## Independent recomputation

An inline NumPy/SciPy calculation imported no `sfora` code. It computed each
pack's endpoint radius as the linear 90th percentile of the six Euclidean
distances from the canonical L2-normalised embedding, then computed Spearman
correlation globally and after subtracting the mean radius within every class.

It reproduced the persisted report exactly:

- global independent-pack Spearman: **0.31759290322040923**;
- within-class-residual Spearman: **0.18405712123299486**.

Both are below the prospectively fixed `>= 0.50` requirements. Radius medians
were 0.58765 and 0.59093 for packs A and B, so the failure is not caused by an
empty or numerically degenerate radius distribution. The 17 focused OAPF unit
tests also pass in the current environment.

## Verdict

**Candidate 174 remains DEAD at Gate 1.** The negative is artifact-backed and
independently recomputable; no implementation or provenance defect reopens it.
The precise finding is unchanged: a q90 radius from six random crops is not a
stable per-image property at this operating point, especially after class
means are removed. No OAPF training or GPU rerun is justified.

