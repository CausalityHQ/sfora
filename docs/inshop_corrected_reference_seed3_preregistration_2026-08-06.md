# Corrected In-Shop Proxy Anchor seed-3 reference preregistration

Date: 2026-08-06 UTC. Written before deployment or GPU work. This is reference
infrastructure, not a candidate or method result.

## Conditional queue rule

Seed 3 may start only after the preregistered seed-2 reference completes and
only while no candidate has cleared search-protocol Gates 1--3. A live
candidate takes DGX priority. Free GPU pricing motivates filling otherwise idle
time; it does not turn an additional baseline seed into method evidence or
permit bypassing novelty, provenance, or preregistration gates.

The purpose is to add another corrected same-seed control and final-state
trajectory. It does not make `n=4` a reliable variance estimate, restore the
withdrawn historical `0.12`-point sigma, or make a small unpaired delta
decisive.

## Locked execution

Run `scripts/run_inshop_corrected_reference_seed3.sh` from a clean tracked-file
deployment at the deciding committed revision. The immutable recipe is
`proxy_anchor.inshop.official-51db570` on
`/home/riomus/datasets/inshop_official_standard`:

- official 256-pixel train/query/gallery partitions;
- BN-Inception GAP+GMP, 512-D normalized descriptor;
- Proxy Anchor, batch 180, trainable BatchNorm;
- AdamW, backbone/head LR `6e-4`, proxy multiplier 100, weight decay `1e-4`;
- one warm-up epoch, StepLR 20 / gamma 0.25, 60 epochs;
- value clipping at 10, drop-last training, raw Kaiming-normal proxies; and
- seed **3**, final step **8,580**.

Locked outputs:

- `reports/generated/inshop_corrected_pa_seed3.json`
- `reports/checkpoints/inshop_corrected_pa_seed3.pt`

The runner fails closed if either output exists, if the corrected dataset root
is absent, if report seed/recipe differs, if the checkpoint is not
`final_training_state` at step 8,580, or if checkpoint/report configs differ.

## Frozen prediction and integrity intervals

- Expected raw best-over-training R@1: **0.918**.
- Expected frozen-final R@1: **0.915**.
- Raw integrity interval: **[0.907, 0.929]**.
- Final integrity interval: **[0.905, 0.923]**.

These retain the prospectively registered seed-1/seed-2 intervals. Falling
outside either interval triggers diagnosis, not tuning or exclusion.

After training, the generic final-state exporter must independently re-encode
all official splits and verify the registered image counts, train/test identity
separation, zero path/content leakage, checkpoint/config binding, and agreement
among production, float64 cosine, float64 Euclidean, and tie-aware scorers.

Report raw best and frozen-final values together. Do not call their difference
a selection-bias correction and do not use the best-test epoch as a retained
artifact.

Estimated cost: about 2.2 GPU-hours plus final exports.

## Runner verification before deployment

`tests/test_run_inshop_corrected_reference_seed3_script.py` executes the real
runner against a temporary project and behavioral fake trainer. The test was
observed failing because the runner did not exist, then passing after the
minimal runner was added. It locks the complete CLI, corrected root, seed,
recipe, output names, and final step/config validation.
