# Pass 103 — state-space/Mamba embedding head (killed before GPU)

## Gate 1

The candidate was suggested as a long-range spatial architecture for the
local-evidence portion of In-Shop errors. The repository does not measure a
long-range dependency bottleneck, so this is weak provenance rather than a
directly identified failure.

## Gate 2

State-space visual backbones are established (Vision Mamba), and Mamba has
already been applied to image retrieval/hashing (MambaHash, arXiv:2506.16353)
and retrieval feature fusion. Replacing BN-Inception's head/backbone with a
state-space scan would be a generic architecture swap, not a new DML
supervision mechanism; it also breaks matched-compute assumptions.

## Decision

**DEAD at Gates 1–2; no implementation or GPU run.**
