# MET-small ScaNN anisotropic-quantization screen result

## Result

At the same nominal 64-byte asymmetric-hashing payload, ScaNN's fixed
anisotropic threshold `0.2` passes the preregistered support gate against the
otherwise identical isotropic arm.

| Split | queries | isotropic mMP@5 / R@1 | anisotropic mMP@5 / R@1 | anisotropic delta |
|---|---:|---:|---:|---:|
| gallery-derived pseudo-query | 3,050 | 0.557519 / 0.498361 | **0.569694 / 0.500328** | **+0.012175 / +0.001967** |
| official validation | 129 | **0.685271** / 0.643411 | 0.680491 / 0.643411 | -0.004780 / 0.000000 |

The powered proxy improvement exceeds the frozen `0.002` mMP@5 threshold and
does not reduce R@1. On official validation, the mMP@5 loss remains inside the
frozen `-0.005` non-inferiority boundary and R@1 is exactly tied. Therefore
`anisotropic_supported=true`.

The two arms use the same 768 normalized dimensions, 128 LUT16 blocks, all
35,257 reduced-gallery rows for training, 20 training iterations, and no tree
partition or exact reordering. The comparison isolates ScaNN's score-aware
anisotropic quantization objective; it does not compare different storage,
candidate sets, or rerankers.

## Timing and implementation boundary

The isotropic and anisotropic indexes built in `3.75` and `4.29` seconds.
Searching all 3,050 pseudo-queries took `0.257` and `0.259` seconds; searching
all 129 official-validation queries took `0.0108` and `0.0110` seconds.
These are one-thread CPU ScaNN batch measurements, not Sfora production
latency measurements.

This positive result licenses a native Sfora algorithm spike that reproduces
the anisotropic objective and verifies the gain under Sfora's public codec and
scoring contracts. It does not yet license a custom CUDA kernel. The measured
search path is already small relative to fitting, and kernel work remains
conditional on end-to-end profiling showing a named operation at least 50% of
runtime with a credible twofold operation-level gain.

## Authority

- result:
  `docs/evidence/rank_finished_l14_336_met_small_scann_anisotropic_screen_v1.json`;
- result SHA-256:
  `3b35ddf0dfc49faba4359735fac1624a92de7e8319fc42e825aaef49bd194ab4`;
- repaired throwaway driver SHA-256:
  `f7ef6972091e09bec8ffc6339770d2ca022fabe84afc6da5967abd314ff1eca2`;
- ScaNN version `1.4.2`, Python 3.9 ARM extension SHA-256
  `6bf70cff2ac0d7646f91664cab0292117490ada30a6dddf8afaefa920ee79725`;
- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- the official MET test image remained sealed.

## Next discriminator

Implement the smallest faithful native anisotropic product-quantization
training/scoring spike and compare it with the same-byte isotropic native arm
on the powered pseudo-query protocol, retaining official validation as the
secondary non-inferiority check. Only after native quality is reproduced
should an end-to-end profile decide between optimized Torch/Faiss, CuTile, or
CUDA-Oxide execution.
