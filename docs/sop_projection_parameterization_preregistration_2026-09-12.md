# SOP Projection-Parameterization Preregistration

## Question

The matched loss-control panel showed that equal positive pressure matters, but changing the
positive aggregator does not explain most of the remaining gap. This experiment tests whether
adapting only inside the deployed 128-dimensional subspace discards useful directions that remain
available in the frozen 768-dimensional teacher representation.

This is a diagnostic on the SOP official-train class-disjoint validation split. It does not touch
the SOP official test and is not, by itself, a publication claim.

## Frozen comparison

Both arms start from the same deployed affine map and emit one normalized 128-dimensional code:

- `restricted_adapter` trains an identity-initialized 128-by-128 linear adapter on frozen base
  codes, then folds it into the frozen 768-to-128 affine head for scoring and deployment.
- `direct_projection` trains the 768-to-128 affine head directly, initialized from the frozen base
  weight and bias.

Both arms use the matched `mean_logit` objective, deterministic class schedule, positive rows,
once-mined frozen 47,704-by-256 hard-negative table, temperature, margin, anchor weight, clipping,
update count, and seed. Both training geometries are computed on the same CUDA device. The
restricted learning rate is `1e-4`; the direct rate is `1e-4 * sqrt(128 / 768)` to match
first-order deployed-map motion across unequal parameter widths. There is no weight decay or
hyperparameter search.

Quality is computed from each arm's actual single folded/deployed affine head, then unit
normalization and the established symmetric-int8 packed evaluator. The receipt records relative
Frobenius displacement from the base head and restricted-fold code equivalence. Each checkpoint
is self-contained: restricted stores adapter and base head; direct stores affine weight and bias.

## Authorities

- Source snapshot SHA-256:
  `6e315945ba4b005baabaa81f7051de23fef2ec8658db09298694ac92438f3471`
- Teacher snapshot SHA-256:
  `b0da9f6097646ffad78c21c84970751ae9a7da003e0eb2979e28f72009785264`
- Base checkpoint SHA-256:
  `9365e6135e6e44f826f99cba2734bbaccd973c8a0a211dc703291a6d3db426ac`
- Base affine-parameter SHA-256:
  `acd49a3854a238a84d760c5499ab5f07e53a48c9ef995253a636ad3276f8b6ee`
- Base receipt SHA-256:
  `df3064bbef5b53a0de15cd45dd03f8c9bc32ae5a82af8512bd5d39a5b933f5a5`
- Expected packed base mAP@R / Recall@1:
  `0.5556011035874895 / 0.8111758251034017`
- Scientific shape: frozen 768-dimensional teacher to deployed 128-dimensional code, with a
  128-dimensional bias.

The executable authenticates all inputs, source revision, driver digest, base score, partition,
schedule, negative table, arm states, and canonical receipt before publishing COMPLETE artifacts.
Execution is explicit-only and every receipt is `claim_eligible=false`.

## Decision rule

Run seed 0 as an engineering screen. If execution is valid and direct training is neither
unstable nor clearly harmful, run the already fixed seeds 1 through 4 without altering parameters.

Direct projection passes the capacity screen only if:

- packed mAP@R gain over restricted is at least `0.003`;
- packed Recall@1 gain is at least `0`;
- the one-sided paired class-cluster bootstrap lower bound for per-query packed AP exceeds `0`.

The five-seed conclusion uses paired seed evidence and the same class-clustered analysis. Failure
means lost teacher directions are not a sufficiently large explanation under this optimizer; it
does not prove all higher-capacity heads useless. A pass promotes direct projection to a
fresh-dataset replication and systems measurement, not to the already observed SOP official test.

## Next boundary

After this diagnostic, the generic-library path is a representation frontier rather than another
positive-loss variant: test trainable projection capacity, then nested quantization-aware
16/24/32/128-dimensional codes and a real integer retrieval backend. Class-name semantics remain
a separate optional adapter with real-name, shuffled-name, and no-name controls; the label-only
core must remain generic.
