**GO to implement one prospective signed-concatenation trial.** No current engineering blocker prevents implementation. Native admission remains the root’s decision; neither cost nor quality success is established.

The authenticated [selection receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/selection-score-v3-receipt.json) reproduces all three reported metrics from 1734 per-query rows. Preserve both historical Pareto heads. The quadratic procedure remains KILL, with no intervals or validation admission.

1. **Freeze one objective and one common penalty.**

   For each independently reconstructed full TRAIN6355/1008 fit:

   ```text
   x  = original TRAIN normalization(raw_cache)
   H0 = unchanged source061_head(x)
   Z  = source061_head.down(normalize(x) - source061_head.center)

   P[c] = mean(H0[labels == c])          # raw, member-inclusive
   E    = P[labels] - H0

   V_linear = Z
   V_concat = concat(Z, H0)             # order fixed: 32 then 128

   Zc = Z - mean(Z)
   reference_scale = trace(Zc.T @ Zc) / 32
   lambda = 0.1 * reference_scale       # identical FP32 scalar in both arms

   Phi = V - mean(V)
   Y   = E - mean(E)
   W   = solve(Phi.T @ Phi + lambda*I, Phi.T @ Y)
   A   = W.T

   inference_raw = H0(input) + (V(input) - source_mean) @ A.T
   ```

   Preserve CPU FP32, qualified flags, canonical row order and TRAIN versus direct-FIT normalization. No squared basis, H0 normalization, balancing, intercept, multiplier change or search. Retain `target_mean` as a numerical witness; never add it during inference.

   The existing [solver](/home/rb/worktrees/sfora-positive-causality/src/sfora/foundation_adapter.py:505) scales its penalty by the fitted feature width. Calling it with `regularization=0.1` unchanged would therefore violate common λ. Use a procedure-owned adapter of its authenticated AST: add a `reference_scale` keyword and replace **only** the scale assignment. Preserve all original validation, centering and `torch.linalg.solve` statements. Authenticate the original full file and AST, freeze the transformed AST, and compile into a fresh namespace. Do not mutate the original module or globals.

   Match the original solver’s operation order: `0.1 * (trace(GZ)/32)`. The historical control witness gives reference energy `210074.703125` and λ `656.4834594726562`; freshly derive these, never substitute constants.

   In exact arithmetic, `[W_linear; 0]` is feasible for the candidate under the same penalty, so its minimum **penalized objective** cannot increase. This establishes neither retrieval improvement nor finite-precision dominance. H0 adds available output directions; their independent rank is not guaranteed.

2. **Make the smallest explicit source and schema change.**

   | Boundary | Change |
   |---|---|
   | [Fitter](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:371) | Keep `fit_prototype_residual(raw_train, targets, base, arm) -> dict`; arms become `linear`, `concat`. Add common-scale adapter and dimension-aware witnesses/reload validation. |
   | New `scripts/prototype_residual_readout.py` | One shared signed-basis arithmetic helper. Reuse authenticated source validators; preserve `raw_features(features, base, A, means, arm)` behavior with explicit 32/160 widths. |
   | [Evaluator](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:288) | Admit only the new matched endpoints; replace quadratic labels with concat throughout costs, deltas, wires and decisions. Preserve scoring, retention and four-load scheduling. |
   | Existing two bounded check scripts | Extend algebra, schema, origin, tamper and gate checks; no new framework. |

   Freeze a new three-file fitter closure and two-file evaluator closure. Keep archived source closures, original factories, solver bytes and historical artifacts unchanged.

   Use new schemas:

   ```text
   siglip2-prototype-residual-ridge-v2
   siglip2-prototype-residual-launch-v2
   siglip2-prototype-residual-evaluation-v2
   siglip2-prototype-residual-evaluation-authority-v2
   ```

   Keep the complete payload key set:

   ```text
   schema, identity, source, encoder, config, buffers, head, classifier,
   warm_payload, partition, original_rows, target, features,
   A, means, source_mean, target_mean, prototypes, counts, fit_witness,
   output_witness, cpu_rng, numerical_flags
   ```

   Explicitly discriminate these shapes by schema and arm:

   | Member | Linear | Concat |
   |---|---:|---:|
   | Feature width | 32 | 160 |
   | `A`, FP32 | `[128,32]` | `[128,160]` |
   | `source_mean`, FP32 | `[32]` | `[160]` |
   | Coefficients | 4096 | 20480 |

   Both payloads carry independently fitted `means={'linear': FP32[32], 'concat': FP32[160]}`. Preserve `target_mean[128]`, `prototypes[1008,128]`, normalized `features[6355,1152]`, int64 counts/labels/ordered rows, complete warm payload, classifier and encoder composition.

   Extend the exact fit-witness schema with feature width, reference energy and reference scale; retain actual feature energy, λ and stationarity fields. Bind dimensions, basis order and penalty rule in `identity.method.recipe`; fingerprint every fitted member.

   Reject v1 payloads in the v2 consumer and v2 payloads in historical consumers. Never infer a new basis from tensor shape or reinterpret an accepted 32-column head.

3. **Require one bounded correctness falsifier, then matched cost admission.**

   First extend the existing stdlib oracle checks to falsify wrong concat order, candidate-specific λ scaling, added target mean, dimension/schema confusion and altered solver AST. These checks do not qualify native execution.

   The root’s smallest native falsifier is **one fresh CPU300 qualification** using actual full TRAIN: two independent fits per arm, complete strict reload, zero-weight raw/unit/packed source parity, unchanged frozen state, unchanged-version `.data` tamper rejection, origin checks and full uncached exit authentication.

   Require finite positive reference energy/λ, finite nonzero coefficients, exact independent fitted-state agreement, and:

   ```text
   ||M @ W - B||F / (||M||F * ||W||F + ||B||F) <= 1e-5
   M = Phi.T @ Phi + common_lambda*I
   B = Phi.T @ Y
   ```

   Verify both arms’ actual λ bits agree. Recompute the witness against common λ, not a candidate-specific scale.

   After qualification, run fresh linear-fit300 and concat-fit300 endpoints, each with two independent fits. Admit their original terminal outcomes before enforcing:

   ```text
   concat whole-service / linear whole-service <= 1.50
   sum(concat two fit cores) / sum(linear two fit cores) <= 1.50
   ```

   Preserve the contiguous fit-core scope, including normalization, source arithmetic, prototypes, means, Gram/RHS, solve and witnesses. Include admission, serialization, reload and exit auditing in whole-service cost. No padding, old denominators or cached sufficient statistics.

   Any integrity, resource, numerical, reload, timeout or cost failure stops this frozen trial before quality. No automatic retry or cap expansion.

4. **Use the unchanged quality gate and explicit public boundary.**

   Only after accepted fitter CPU, both cost endpoints and fresh evaluator CPU may selection1734q/1715g/498 be read. Replay source per-query results and wires exactly. Candidate-minus-linear R1 must be positive, mAP nonnegative, and both candidate metrics must meet the source floor. GO additionally requires **both gains ≥0.002 fraction** and both paired-product 95% lower bounds positive, with unchanged shared5000 draws/seed179019 and reported query intervals.

   Selection GO admits only sealed VAL1749q/1730g/498 under the same gates and its own source floor.

   The current [public loader](/home/rb/worktrees/sfora-positive-causality/src/sfora/siglip2_compact_serving.py:152) constructs a 1024→128 linear head; it cannot load this 1152-wide source composition. After the sealed gates, add an explicitly named So400 prototype loader that consumes the v2 payload, authenticated original source factory and immutable 448-tensor encoder. Reuse the signed-basis helper and existing 130-byte packed wire. Require public B1/B32 parity and matched image-to-top-k measurements before any speed claim; full TRAIN and official SOP+InShop confirmation remain required.

**Cost assessment:** whole-service passage is plausible, not measured. Added coefficient storage is only 64KiB, but Gram work grows 25×, RHS work 5× and cubic solve work 125×. Against the historical linear sum2 core of `0.248912s`, the illustrative allowance is `0.373367s`—only about 62ms extra per fit. Actual fresh endpoints must decide feasibility.

The score cap is also tight: `286.466s` leaves `13.534s`, and that historical KILL computed no bootstrap intervals. A surviving candidate’s interval work has not been qualified within that margin.

All proposed units retain300s/8GiB/noSwap/zero events/CUDA hidden/both locks, full actual SHA checks at every boundary, and encoder-only read-only retention capped at2GiB. Maximum five-unit service allowance is1500s, stopping at the first failed gate.

No files were edited, and no native experiments, quality jobs, children or reviews were launched.
