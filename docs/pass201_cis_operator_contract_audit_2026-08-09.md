# Pass 201 — CIS operator-contract and gradient-scale audit

## Status and timing

This audit was frozen while the corrected Pass-120 controller was training the
`pa_coalition_dropout` arm, before the corrected dropout or full-coalition
results were available.  The completed corrected arms visible at that point
were only:

- `pa_coalition_single`: raw best `0.9167252778`, final `0.9132789422`;
- `pa_coalition_complementary`: raw best `0.9184132789`, final
  `0.9150372767`.

An exploratory reviewer accidentally encountered the older Pass-151 table, so
its empirical interpretation is not a clean preregistration.  The conclusions
below are instead source-level identities independently reproduced by a second
reviewer who was explicitly forbidden from reading empirical results.

## Contract defect: the required no-coalition control is absent

The Gate-2 audit requires a **single-image multi-label / no-coalition field
control**: every selected image is classified independently against the full
union `U` of labels in its bundle.  For selected row `i` and proxy row `c`, its
target must be

`Y[i,c] = 1[proxy_label[c] in U]`.

No current `coalition_mode` implements that object:

- `single` uses only the row's own one-hot target;
- `single_complementary` uses `U \ {y_i}` and therefore marks the row's own
  proxy negative;
- `union` and `dropout` classify one summed descriptor rather than individual
  rows.

Consequently, `pa_coalition_single` does not implement the control described in
`docs/pass120_cis_gate2_audit_2026-08-07.md` and
`docs/pass120_cis_candidate_2026-08-07.md`.  The existing corrected Pass-120
run can still measure the configured full CIS arm, but **cannot clear its
mechanism gate even if the headline threshold is crossed**.  A correct
per-image union-field control is mandatory before attributing any gain to the
summed coalition.

## Exact reduction-scale confound

Let `m` be the number of unique classes selected from the batch, `K` the number
of proxy rows, `u_i` the normalized representative embedding, and `p_c` the
normalized proxy.  PyTorch BCE uses its default mean reduction.

For the atomic `single` and `single_complementary` modes,

`L_atomic = (1/(mK)) sum_i sum_c BCE(u_i dot p_c, Y_ic)`, so

`dL_atomic/du_i = (1/(mK)) sum_c (sigmoid(u_i dot p_c)-Y_ic) p_c`.

For `union` and `dropout`,

`b = (1/sqrt(m)) sum_i u_i` and
`L_sum = (1/K) sum_c BCE(b dot p_c, Y_c)`, so

`dL_sum/du_i = (1/(K sqrt(m))) sum_c (sigmoid(b dot p_c)-Y_c) p_c`.

The scalar reduction-prefactor ratio is therefore exactly `sqrt(m)`.  This is
not a claim that realized gradient norms differ by exactly `sqrt(m)`: logits,
targets, sigmoid residuals, normalization Jacobians, and the shared base Proxy
Anchor term differ.  It is nevertheless a deterministic scale confound that
must be measured.  At most one stable-index representative per unique class
receives the auxiliary gradient; other rows receive only the base loss.

The proxy table is learnable and is not detached in the current auxiliary.
This does not change the embedding-side prefactors, but it means the screen also
changes proxy optimization.  The summed coalition is divided by `sqrt(m)` but
is not normalized after summation.

## What the complementary result can and cannot mean

Conditional on the selected label set and proxy table, row `i` of
`single_complementary` depends on `u_i` alone.  Other image embeddings never
enter its logit and its cross-image embedding Hessian blocks are zero.  Its
observed trajectory therefore cannot by itself evidence cross-image coalition
interference.  It is compatible with complementary-label regularization,
seen-proxy de-anchoring, or attraction toward randomly co-batched class proxies.

## Frozen hill-climb decision

The running Pass-120 thresholds remain unchanged; this audit must not rescue a
failed configured arm by retuning its benchmark threshold.

Before any new CIS training arm, run one train-only, checkpoint-bound operator
diagnostic on deterministic same-class-set pairs `S` and `S'`:

1. compare ordinary PA, atomic one-hot, atomic complementary, the missing
   per-image full-union control, summed union, and summed dropout;
2. report auxiliary-to-PA parameter-gradient norm, summed-to-atomic norm ratio,
   gradient cosine, owner-margin directional derivative, and foreign-proxy-mass
   directional derivative both at configured weight and equal parameter-update
   norm;
3. take a stateless virtual update on `S` and evaluate the two train-only
   outcomes on the disjoint-image, same-class-set `S'`;
4. separately test whether shared foreign-proxy activation exceeds a fixed
   independent per-row proxy-column permutation null.

Decision rules, to be frozen numerically in a dedicated preregistration before
values are computed:

- if aligned shared-confuser excess is non-positive, the coalition premise is
  dead;
- if norm-matched summed union does not beat the per-image full-union control on
  held-out `S'`, coalition-specific action is dead and any benchmark effect is
  ordinary multi-label/proxy regularization;
- if the ordering disappears after equal-norm comparison, gradient scale is the
  sufficient explanation;
- if foreign suppression has a negative owner-margin derivative, the operator
  is not a viable repair;
- only a mechanism survivor may receive a single prospectively frozen
  norm-calibrated weight and another GPU screen.  No weight sweep is allowed.

This is a hill-climb on the measured operator, not a new novelty claim.

