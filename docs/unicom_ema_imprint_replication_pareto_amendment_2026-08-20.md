# UniCOM imprint replication Pareto amendment

## Scope and chronology

This prospective amendment repairs only the six-seed confirmation summary and final
claim. It does not change either training arm, seeds `1..6`, epochs
`[4, 8, 12, 16]`, the selected cell `imprinted_raw`, the quality thresholds, or any
already-produced checkpoint or paired report.

It is written after the immutable seed-1 paired report was observed and before any
seed-2 training. The seed-1 report remains unchanged at
`reports/generated/unicom_ema_imprint_replication_c83cd96_seed1.json`, SHA-256
`0cfb888fdbc0e409048943a8d6e47635e571dec5830b86260e0730d80ebf4ab8`.
That report strictly validates and gives endpoint deltas `+0.01443298839283702`
mAP@R and `+0.008782936010037656` Recall@1. Its per-query bootstrap interval
`[0.007050684568871645, 0.02209061680646096]` is descriptive within-run evidence,
not a training-seed confidence interval.

Independent review `2946f2385b914ac9` found that the scientific quality result is
sound but the planned final cost claim has two defects: non-monotone control curves
make the registered seed-1 time-to-quality ratio `1.5`, not `2.0`, and raw wall time
is environmentally contaminated while still gating `claim_supported`.

## Frozen selection authority

The final summary must authenticate and embed this exact selection authority:

```json
{
  "path": "reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json",
  "sha256": "c0666a68e70990115d80e8dc06a9f94efe83156a3fddd50f36bdbf2b3b8cd217",
  "commit": "81f3f48c374d14b5a91bbeba7a1fec2fb0a4a2d6",
  "selected_cell": "imprinted_raw",
  "decision": "PROMOTE"
}
```

The summary CLI must receive that report explicitly, verify its exact bytes and gate,
and reject any other path, digest, commit, selected cell, or decision. This closes the
selection-lineage gap without changing the already-valid per-seed report schema.

## Canonical time-to-quality estimator

For each seed, the target is that seed's `random_raw` epoch-16 mAP@R. For each arm,
the first-quality epoch is the earliest registered epoch whose mAP@R is at least that
target. The reported speedup is:

```text
random_first_quality_epoch / imprinted_first_quality_epoch
```

This is the existing six-seed summarizer estimator and is now the sole confirmation
estimator. It handles non-monotone controls. For seed 1 the exact registered values are
`random_raw=12`, `imprinted_raw=8`, `speedup=1.5`. The earlier `16/8=2.0` statement is
withdrawn as a fixed-endpoint shorthand and must not appear in the final report.

## Pareto cost estimator

Raw end-to-end training wall time remains mandatory evidence but is descriptive only.
It must not gate `pareto_cost_noninferior`, `pareto_nondominated_against_random_raw`,
or `claim_supported` because seed-1 wall-time overhead is asymmetric and dominated by
environmental non-step delay.

The prospective cost gate uses a conservative, symmetric compute proxy:

```text
random_proxy_seconds = random_first_epoch * 161 * random_step_wall_seconds

imprinted_proxy_seconds =
    imprinted_first_epoch * 161 * imprinted_step_wall_seconds
    + 25_882 * inference_latency_ms_per_image / 1_000
```

`161` is the registered optimizer-step count per epoch. `25_882` is the complete
In-Shop training image count and therefore an upper bound on the smaller optimization
subset encoded by imprint initialization. Charging every train image at the measured
per-image inference latency conservatively includes the one-time imprint pass instead
of pretending it is free.

For every seed, Pareto cost non-inferiority requires:

- finite first-quality epochs for both arms and
  `imprinted_proxy_seconds <= random_proxy_seconds`;
- imprinted peak GPU memory no greater than random peak GPU memory;
- imprinted checkpoint storage no greater than random checkpoint storage;
- identical deployment storage (the classifier is not deployed).

Inference latency is one shared final-backbone measurement per paired report and is
reported, not attributed to classifier initialization. The A-B-B-A profiler kernel
gate remains unchanged at `0.1`; seed 1 measured about `0.00045`, so custom-kernel work
remains closed.

## Initialization evidence limitation

The seed-1 checkpoints bind the reviewed trainer bytes and declare
`classifier_init`, but they do not contain the epoch-0 classifier bytes. That historical
fact cannot be repaired without replacing a valid result, which is forbidden. The final
report must state this limitation exactly and may not claim cryptographic recovery of
seed-1 initialization.

Before seed-2 training, the trainer will add an atomic initialization receipt containing
the seed, arm, trainer SHA-256, algorithm name, classifier tensor SHA-256, shape, dtype,
and the post-initialization RNG-state hashes. Seeds 2..6 must bind those receipts in their
measurement evidence. This adds evidence only; it does not alter initialization or the
training trajectory.

## Decision semantics

The quality gate remains the preregistered all-six paired training-seed gate. The exact
sign-test p-value is reported as the mathematical consequence of all six nonzero deltas
having one sign, not as independent evidence. The Student-t interval generalizes over
training seeds on the fixed holdout only.

`claim_supported` is true only when the unchanged quality gate and the amended Pareto
cost gate both pass. No official-protocol or global-SOTA claim is authorized by this
summary; an exact one-shot official In-Shop test readout requires a separate prospective
protocol.
