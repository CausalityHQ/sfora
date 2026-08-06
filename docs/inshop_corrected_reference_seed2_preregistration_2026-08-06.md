# Corrected In-Shop Proxy Anchor seed-2 reference preregistration

Date: 2026-08-06 UTC. Written before deployment or GPU work. This is reference
infrastructure, not a candidate or method result.

## Why this run is admissible

The corrected official-pixel reference currently has two independently
verified seeds:

| seed | raw best R@1 | frozen-final R@1 |
|---:|---:|---:|
| 0 | 0.9163032775 | 0.9137009425 |
| 1 | 0.9189056126 | 0.9167956112 |

Their raw/final means are `0.9176044451` and `0.9152482768`. Two points are not
a variance estimate. A third reference does not make a three-run SD reliable,
but it supplies another same-seed control for future screens, another
independently selected final artifact, and a third trajectory for prospective
error-persistence analyses.

This run is lower priority than any method that clears protocol Gates 1--3. It
may occupy an otherwise idle DGX while blind proposal/prior-art review is in
flight; it must be stopped before launch, rather than killed mid-run, if a live
candidate is ready. Free GPU pricing does not turn it into method evidence.

## Locked execution

Run `scripts/run_inshop_corrected_reference_seed2.sh` from a clean tracked-file
deployment at the deciding committed revision. The immutable recipe is
`proxy_anchor.inshop.official-51db570` on
`/home/riomus/datasets/inshop_official_standard`:

- official 256-pixel train/query/gallery partitions;
- BN-Inception GAP+GMP, 512-D normalized descriptor;
- Proxy Anchor, batch 180, trainable BatchNorm;
- AdamW, backbone/head LR `6e-4`, proxy multiplier 100, weight decay `1e-4`;
- one warm-up epoch, StepLR 20 / gamma 0.25, 60 epochs;
- value clipping at 10, drop-last training, raw Kaiming-normal proxies; and
- seed **2**, final step **8,580**.

Locked outputs:

- `reports/generated/inshop_corrected_pa_seed2.json`
- `reports/checkpoints/inshop_corrected_pa_seed2.pt`

The runner fails closed if either output already exists, if the corrected
dataset root is absent, if the report recipe/seed differs, if the checkpoint is
not `final_training_state` at step 8,580, or if checkpoint/report configs differ.

## Frozen prediction and integrity intervals

- Expected raw best-over-training R@1: **0.918**.
- Expected frozen-final R@1: **0.915**.
- Raw integrity interval: **[0.907, 0.929]**.
- Final integrity interval: **[0.905, 0.923]**.

These reuse the prospectively registered seed-1 integrity ranges rather than
being narrowed after seeing two seeds. Falling outside either range triggers
diagnosis, not tuning or exclusion.

After training, the existing generic final-state exporter must independently
re-encode all official splits and verify:

- 25,882 train / 14,218 query / 12,612 gallery images;
- train/test identity disjointness and zero path/content leakage;
- BN-Inception/512 checkpoint identity and exact config binding; and
- exact agreement among production, float64 cosine, float64 Euclidean, and
  tie-aware nearest-neighbour scorers.

Report raw best and frozen-final values together. Do not call their difference
a selection-bias correction, do not use the best-test epoch as an exported
training artifact, and do not treat `n=3` as a stable sigma estimate.

Estimated cost: about 2.2 GPU-hours plus final exports.

## Runner verification before deployment

`tests/test_run_inshop_corrected_reference_seed2_script.py` executes the real
shell runner against a temporary project and a behavioral fake trainer. It
verifies the complete CLI argument vector, corrected root, seed, locked recipe,
artifact names, and that the runner's inline final-state validator accepts a
config-identical step-8,580 artifact. The test was observed failing because the
runner did not exist, then passing after the minimal runner was added.
