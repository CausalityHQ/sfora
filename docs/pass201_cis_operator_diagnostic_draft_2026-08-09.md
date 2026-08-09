# Pass 201 — CIS operator-isolation diagnostic draft

> **PENDING_SOURCE / NOT YET A PREREGISTRATION / NO COMPUTATION AUTHORIZED**
>
> The fresh current-code ordinary Proxy Anchor checkpoint, its report, its
> resolved source revision, and its integrity digests have not landed.  Every
> source-dependent field in this draft therefore remains locked.  This document
> cannot authorize artifact inspection, diagnostic computation, training, or a
> GPU run.

## Non-authority and scope

This is a proposed train-split-only, checkpoint-bound local operator diagnostic.
It cannot change, reinterpret, rescue, or retune the running Pass 120 experiment,
regardless of any Pass 120 outcome.  No CIS report, result, log, table, checkpoint,
or benchmark metric may be read while activating or executing this design.

If eventually activated and passed, this diagnostic may authorize only the
writing of a separate prospective preregistration for at most one
norm-calibrated GPU arm.  It cannot authorize that GPU arm directly.  The future
arm would require its own frozen source, weight, seed, run command, and success
threshold.  A failure or unresolved result here closes the mechanism screen; it
does not permit a sweep or revised Pass 120 interpretation.

## Source activation gate

The sole admissible source is the fresh current-code, seed-0, ordinary Proxy
Anchor control produced by the matched corrected In-Shop controller.  Its
actual source revision must be read from that control's report after it lands.
This draft deliberately does not assume that the source revision is the current
repository HEAD.

Before this draft can become a preregistration, a separate activation revision
must commit a `pass201-source-v1` manifest whose values were obtained without
reading any CIS output:

```json
{
  "schema_version": "pass201-source-v1",
  "status": "frozen",
  "source_report_path": null,
  "source_report_sha256": null,
  "source_revision": null,
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "checkpoint_bytes": null,
  "checkpoint_epoch": null,
  "objective": "proxy_anchor",
  "seed": 0,
  "resolved_config_path": null,
  "resolved_config_sha256": null,
  "train_manifest_path": null,
  "train_manifest_sha256": null,
  "diagnostic_source_sha256": null,
  "activated_preregistration_sha256": null,
  "torch_version": null,
  "numpy_version": null
}
```

The `null` values are intentional activation locks, not values that may be
filled during a scored run.  The activation revision must replace every `null`
with a literal value, freeze the exact production-derived diagnostic config,
and be committed before any checkpoint tensor, transformed input, gradient, or
outcome is computed.  The runner must recompute all digests and abort on any
mismatch.

The source checkpoint must be an ordinary-PA checkpoint.  Any coalition,
single, complementary, dropout, residual, selected-on-CIS, older-code, or
externally published checkpoint is forbidden as a substitute.  If the fresh
ordinary-PA report/checkpoint pair never lands or cannot be authenticated, the
diagnostic remains `BLOCKED`; no alternative source may be chosen.

## Frozen production-context constants

The operating point is the official corrected In-Shop shuffled training loader,
not a balanced P-K construction:

```text
batch_size = 180 rows
context_pairs = 32
null_replicates = 256
bootstrap_replicates = 20_000
s_prime_rank_seed = 2010809
null_seed = 2010810
bootstrap_seed = 2010811
model_forward_seed = 2010812
```

All other runtime values, including source revision, worker count, transform
configuration, learning rate, coalition weight, proxy learning-rate multiplier,
and proxy count, must come literally from the activated, digested resolved
configuration.  They may not be inferred from this draft or from the current
checkout.

## Deterministic official training contexts

Instantiate the exact production indexed training dataset and DataLoader using
the activated optimization split, official training transform, batch size,
shuffle flag, generator seed, worker count, pin-memory setting, and drop-last
setting.  The indexed dataset must expose the original stable training-row index
used by the production coalition helper.  No class filtering, balanced sampler,
P-K sampler, hard mining, or diagnostic-specific shuffle is allowed.

Traverse only the first production epoch.  Rejected contexts still consume the
production sampler and transform RNG streams.  Accept the first 32 batches that
satisfy the disjoint-image counterpart rule below.  Preserve, without sorting
or deduplication, all 180 row tensors, labels, example IDs, stable indices, and
their production order.

For candidate context `b`, let `n[b,y]` be the multiplicity of class `y` among
its 180 rows.  It is feasible only if the frozen training manifest contains at
least `n[b,y]` distinct same-class images absent from this candidate context for
every represented class.

Construct `S_prime[b]` as follows:

1. Preserve the exact 180-row label sequence and class multiplicities of `S[b]`.
2. For each class, exclude every image ID appearing anywhere in `S[b]`.
3. Rank the remaining same-class training examples by
   `SHA256("pass201-sprime|2010809|" + example_id)`, breaking hash ties by the
   UTF-8 bytes of `example_id`.
4. Assign the first `n[b,y]` ranked alternatives to the class's row positions in
   their original order.
5. Require 180 distinct `S_prime[b]` image IDs and require
   `images(S[b])` and `images(S_prime[b])` to be disjoint.

Class or image reuse across different context pairs is allowed because
forbidding it would alter the official sampler's operating distribution.  All
cross-pair reuse must be reported.  The bootstrap unit remains the complete
180-row context pair and its uncertainty interval is a stability interval, not
a claim that overlapping class identities are statistically independent.

If the first epoch yields fewer than 32 feasible context pairs, return
`UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS`.  Do not inspect a later epoch,
reduce the context count, change the batch size, prefilter the sampler, or use a
second augmentation of the same image as an alternative.

For each accepted context define

```text
m_b = number of unique labels among the 180 rows
```

and reproduce production representative selection exactly: sort unique labels
and select, for each label, the row having the minimum stable dataset index.
The auxiliary operators consume those `m_b` representatives.  Ordinary PA and
the train-mode network/BN graph consume all 180 rows.  Nonrepresentatives must
remain in the graph and receive their configured total gradient through PA and
shared network computation.

## Frozen official input transforms

`S[b]` must use one deterministic materialization of the official production
training transform per row, including the official random resized crop and all
other resolved training transformations.  It must be produced through the
exact production DataLoader and its worker-seeding behavior.  Independently
reseeding or calling the transform per image is forbidden.

Before any operator scoring, persist the 32 accepted transformed batches and
record a SHA-256 over each contiguous CPU tensor's bytes, dtype, shape, example
ID, and stable index.  A second preparation pass in a fresh process must
reproduce every batch membership and tensor hash.  Failure is
`INVALID_NONDETERMINISTIC_TRAIN_INPUT`.

`S_prime[b]` uses the activated deterministic clean evaluation transform.  Its
input tensors and metadata receive the same hashing and replay check.  The
declared transfer question is therefore:

> Does one production-train-mode virtual update on augmented `S` improve clean,
> eval-mode outcomes on disjoint images `S_prime` having the same 180-row label
> sequence?

## Bufferless train-mode gradient graph

For every `S[b]`:

1. Hash all checkpoint parameters, persistent buffers, and module training
   flags.
2. Clone every persistent buffer, including BN running means, running variances,
   and counters.
3. Put the functional model in training mode without freezing BN affine
   parameters.
4. Before the first scored context, seed the CPU and every available accelerator
   generator once with `2010812`.  Before and after each context, capture the
   resulting RNG states in the integrity record.
5. Execute exactly one shared functional forward of all 180 transformed rows
   using checkpoint parameters and the disposable cloned buffers.
6. Allow BN to use current-batch statistics and to mutate only the disposable
   buffer clones exactly as a production training forward would.
7. Normalize the resulting embeddings through the same production path.
8. Construct all operator losses from this single shared embedding graph.
9. Discard the cloned buffers and restore every module training flag.  Do not
   rewind RNG state between contexts; stochastic train-mode layers, if present,
   must consume the single frozen sequential RNG stream.
10. Verify byte-identical checkpoint parameter and persistent-buffer hashes.

This is “bufferless” only in the sense that no checkpoint buffer survives a
diagnostic forward mutation.  Eval-mode BN on `S`, frozen BN affine values,
microbatching, per-operator forwards, or distinct BN statistics for different
operators are invalid operating-point substitutions.  If the architecture has
another stochastic train-mode layer, all operators still consume the same
single captured forward graph.

For `S_prime[b]`, put the functional model in evaluation mode and use the
original checkpoint BN buffers before and after every virtual parameter update.
Disposable `S` buffers are never transferred.  Restore and verify training
flags and hashes after every call.

## Exact operator panel

Let `u_i` be a normalized selected representative embedding, `p_c` a normalized
proxy row, `lambda_c` its class label, `U_b` the unique class set, `K` the number
of proxy rows, and

```text
BCE(a, y) = softplus(a) - y*a
```

Every auxiliary BCE call must explicitly use `reduction="mean"`, with no
element weight and no positive-class weight.  Evaluate exactly six operators:

1. **Ordinary PA.** The activated production `_proxy_anchor_loss`, including
   the resolved `alpha` and `delta`, over all 180 rows.
2. **Atomic one-hot.** On the `m_b` representatives,
   `mean_[i,c] BCE(u_i dot p_c, 1[lambda_c = y_i])`.
3. **Atomic complementary.** On the `m_b` representatives,
   `mean_[i,c] BCE(u_i dot p_c, 1[lambda_c in U_b minus {y_i}])`.
4. **Per-image full-union.** On the `m_b` representatives,
   `mean_[i,c] BCE(u_i dot p_c, 1[lambda_c in U_b])`.
5. **Summed union.** Form
   `b = sum_i u_i / sqrt(m_b)` without post-sum L2 normalization and compute
   `mean_c BCE(b dot p_c, 1[lambda_c in U_b])`.
6. **Summed dropout.** Use the same `b`, but remove the numerically largest label
   in sorted `U_b` from the target.  Its embedding remains in `b`, matching the
   production operator.

The diagnostic full-union implementation must be a standalone direct formula.
It must not alias complementary mode, enter the production recipe registry, or
modify any training arm.

Before scoring, pure tests must establish exact target matrices, tensor shapes,
mean reductions, deterministic representative selection, permutation
invariance under aligned stable indices, deterministic dropout, finite nonzero
gradients, and equality with hand-computed formulas.  Test failure is `INVALID`.

For context `b`, the atomic and summed scalar coefficients are

```text
atomic: 1 / (m_b*K)
summed per selected member: 1 / (K*sqrt(m_b))
```

so the deterministic coefficient ratio is `sqrt(m_b)`.  No fixed `m` may be
used.  This identity does not imply an exact realized gradient-norm ratio.

## Parameter and proxy panels

Obtain gradients over the complete set of checkpoint parameters that the
activated training configuration would mark trainable.  Derive two panels from
the same autograd results:

- `network_only`: every trainable model parameter except `metric_proxies`; proxy
  values participate in logits but their update component is set to zero.
- `joint_faithful`: the same network parameters plus the learnable proxy table.

`network_only` is the primary causal panel because transfer to disjoint images
must pass through shared network parameters.  A joint-only success is classified
as proxy-table regularization and cannot rescue a network-only failure.
`joint_faithful` remains mandatory because the production operator does update
proxies.

Define the stateless update multiplier `A` from the activated resolved config:

- ordinary network/head parameter multiplier: `1`;
- proxy multiplier: the literal resolved `proxy_learning_rate_multiplier`;
- no Adam moments, weight decay, clipping, EMA, BN-buffer update, or optimizer
  state.

All flattened norms, dot products, and cosines must be accumulated in float64
after application of `A`.  This is an explicitly stateless diagonal update
geometry, not a claim to reproduce an AdamW step without optimizer state.

## Gradient measurements

For every context, operator, and parameter panel, report:

- raw loss;
- raw gradient norm;
- update-space norm `||A*g_o||_2`;
- auxiliary-to-PA update-norm ratio;
- summed-union-to-each-atomic update-norm ratio;
- cosine with PA;
- cosine with per-image full-union;
- cosine between summed union and summed dropout.

Also report, for each atomic operator `o`,

```text
scale_residual[o,b]
  = ||A*g_summed_union,b||
    / (sqrt(m_b) * ||A*g_o,b||)
```

as a descriptive statistic only.  Different logits, targets, sigmoid residuals,
and tangent projections prevent an expectation that it equals one.  Any missing,
zero, NaN, or infinite required gradient invalidates the entire run.

## Clean held-out outcomes

After a train-mode gradient is obtained on `S[b]`, evaluate both pre-update and
post-update clean `S_prime[b]` in eval mode over all 180 rows.  Let
`a[i,c] = u_i dot p_c` on those rows.

Foreign mass excludes the complete bundle class set:

```text
F = mean_[i,c: lambda_c not in U_b] sigmoid(a[i,c])
```

Abort if there is no proxy outside `U_b`.

With fixed temperature `tau = 0.05`, define

```text
owner_i = tau * logmeanexp_[c: lambda_c = y_i](a[i,c] / tau)
hard_foreign_i = tau * logmeanexp_[c: lambda_c != y_i](a[i,c] / tau)
M = mean_i(owner_i - hard_foreign_i)
```

Abort if any represented class lacks an owning proxy.  Positive outcome
conventions are

```text
R_F = (F_before - F_after) / max(F_before, 1e-6)
Delta_M = M_after - M_before
```

## Configured and equal-norm stateless virtual updates

At activation, freeze literal values

```text
eta = resolved learning_rate
lambda = resolved coalition_weight
```

from the authenticated diagnostic config.  They cannot be selected from any
diagnostic result.

Configured directions are

```text
g_cfg[PA] = g_PA
g_cfg[o] = g_PA + lambda*g_o
Delta_cfg[o] = -eta*A*g_cfg[o]
```

where PA uses all 180 rows and every auxiliary uses the production-selected
`m_b` representatives from the shared 180-row train graph.

For equal-norm operator isolation, define the same-context reference norm

```text
rho_b = ||-eta*A*g_PA||_2
Delta_eq[o] = -rho_b * (A*g_o) / ||A*g_o||_2
```

for every pure operator, including pure PA.  Thus every equal-norm operator has
exactly the PA parameter-update norm for that production context.  No global
norm target, adaptive step, line search, optimizer-state reconstruction, or
post-outcome step-size change is permitted.

Use a stateless functional parameter call; never mutate the checkpoint.  For
each configured and equal-norm update, report the actual `R_F` and `Delta_M` plus
the first-order predictions

```text
D_F = -(grad(F) dot Delta) / max(F, 1e-6)
D_M = grad(M) dot Delta
```

where positive values predict foreign suppression and owner-margin improvement.

## Shared-confuser permutation null

Evaluate the coalition premise on the `m_b` selected `S[b]` representatives,
not on all nonrepresentative rows.  For proxy rows outside `U_b`, set

```text
q[i,c] = clamp(sigmoid(u_i dot p_c), 1e-12, 1)
A_aligned = mean_c exp(mean_i log(q[i,c]))
```

For null replicate `r`, independently permute the outside-`U_b` proxy columns
within every representative row using NumPy
`PCG64(null_seed + 100000*b + r)`.  Preserve every row's marginal activation
values and recompute the same geometric-mean score.  With 256 replicates,

```text
E_shared
  = (A_aligned - mean_r(A_null[r]))
    / max(mean_r(A_null[r]), 1e-12)
```

The null destroys shared proxy-column identity while preserving each row's
foreign-activation distribution.

## Aggregation and bootstrap

The complete 180-row `(S, S_prime)` pair is the only resampling unit.  Do not
pool rows, representatives, classes, or proxy columns as independent units.

For every metric, report the mean, median, standard deviation, quartiles, and
the distribution of `m_b` over the 32 context pairs.  Generate 20,000 paired
bootstrap samples of the 32 context indices with NumPy `PCG64(2010811)`.  Use
the same resampled indices for every operator and paired difference.  Report
one-sided percentile bounds:

```text
LCB = quantile(bootstrap_means, 0.005, method="linear")
UCB = quantile(bootstrap_means, 0.995, method="linear")
```

No context may be removed after construction.  Any invalid context invalidates
the entire audit.

For both parameter panels in the equal-norm regime define

```text
A_F = R_F[summed_union] - R_F[atomic_full_union]
A_M = Delta_M[summed_union] - Delta_M[atomic_full_union]
```

Configured-weight differences are reported identically but cannot rescue an
equal-norm failure.

## Frozen PASS / FAIL / UNRESOLVED thresholds

A `PASS` requires every following 99.5% lower bound:

| Component | Required LCB |
|---|---:|
| Shared-confuser relative excess `E_shared` | `>= 0.010` |
| Network-only equal-norm union foreign advantage `A_F` | `>= 0.001` |
| Network-only equal-norm union margin advantage `A_M` | `>= 0.001` |
| Network-only equal-norm union foreign suppression `R_F` | `>= 0.001` |
| Network-only equal-norm union margin change `Delta_M` | `>= 0.000` |
| Network-only equal-norm predicted suppression `D_F` | `>= 0.001` |
| Network-only equal-norm predicted margin change `D_M` | `>= 0.000` |
| Joint equal-norm union foreign advantage `A_F` | `>= 0.000` |
| Joint equal-norm union margin advantage `A_M` | `>= 0.000` |
| Joint equal-norm union foreign suppression `R_F` | `>= 0.000` |
| Joint equal-norm union margin change `Delta_M` | `>= 0.000` |

Declare `FAIL` immediately if any corresponding 99.5% UCB is `<= 0`, except
that owner-margin harm uses `UCB < 0`.  Assign the most specific applicable
reason codes:

- `FAIL_NO_SHARED_CONFOUNDER` if `UCB(E_shared) <= 0`.
- `FAIL_NO_COALITION_SPECIFIC_ACTION` if either network-only equal-norm
  superiority UCB is `<= 0`.
- `FAIL_NOT_VIABLE` if union foreign suppression has `UCB <= 0` or union
  owner-margin change has `UCB < 0`.
- `FAIL_SCALE_SUFFICIENT` if configured joint union beats full-union on both
  outcomes with `LCB >= 0`, but either corresponding joint equal-norm advantage
  has `UCB <= 0`.
- `FAIL_PROXY_ONLY` if the joint panel clears its thresholds but the
  network-only panel fails.
- `FAIL_OWNER_DAMAGE` if equal-norm union has positive foreign-suppression
  directional effect but `UCB(D_M) < 0`.

If no failure condition holds but any PASS lower-bound threshold is missed, the
result is `UNRESOLVED`.  The smaller number of expensive 180-row production
contexts may widen intervals; it never permits relaxed thresholds, more
contexts chosen after inspection, or a changed bootstrap rule.

Only `PASS` may set the next action to
`write_separate_gpu_preregistration`.  It still authorizes no run.

## Theoretical complexity

Let `P` be trainable parameter count, `K` proxy rows, `Q=32` context pairs,
`N=180` rows, `O=6` operators, `R=256` null replicates, and `m_b <= 180` the
context-specific representative count.

Each context requires one shared 180-row train forward, six loss-gradient
reverse passes, one clean 180-row outcome graph with two outcome-gradient
reverse passes, and at most `2*O` stateless clean forwards for configured and
equal-norm updates.  The permutation null is streamed.

```text
time = O(Q * [O*C_backward(N,P)
              + 2*O*C_forward(N,P)
              + R*m_b*(K-m_b)])
memory = O(one 180-row train graph + P + m_b*K)
```

Microbatching is prohibited because it changes train-mode BN semantics.  This
is a bounded local diagnostic, not an epoch-scale training arm.

## Output schema

An activated runner must emit exactly one JSON artifact with this top-level
shape:

```json
{
  "schema_version": "pass201-cis-operator-v1",
  "status": "PASS|FAIL|UNRESOLVED|BLOCKED|INVALID",
  "reason_codes": [],
  "source": {
    "source_report_path": null,
    "source_report_sha256": null,
    "source_revision": null,
    "checkpoint_path": null,
    "checkpoint_sha256": null,
    "checkpoint_bytes": null,
    "checkpoint_epoch": null,
    "resolved_config_sha256": null,
    "train_manifest_sha256": null,
    "diagnostic_source_sha256": null,
    "activated_preregistration_sha256": null
  },
  "constants": {
    "batch_size": 180,
    "context_pairs": 32,
    "null_replicates": 256,
    "bootstrap_replicates": 20000,
    "s_prime_rank_seed": 2010809,
    "null_seed": 2010810,
    "bootstrap_seed": 2010811,
    "model_forward_seed": 2010812,
    "learning_rate": null,
    "coalition_weight": null,
    "proxy_learning_rate_multiplier": null,
    "owner_margin_temperature": 0.05
  },
  "contexts": [
    {
      "context_index": 0,
      "production_epoch": 0,
      "production_batch_index": 0,
      "batch_size": 180,
      "m_unique": 0,
      "row_example_ids": [],
      "row_sample_indices": [],
      "row_labels": [],
      "class_multiplicities": {},
      "representative_row_indices": [],
      "representative_sample_indices": [],
      "s_tensor_sha256": null,
      "s_prime_example_ids": [],
      "s_prime_sample_indices": [],
      "s_prime_tensor_sha256": null,
      "cross_context_reuse": {},
      "foreign_proxy_rows": 0,
      "shared_confuser": {},
      "operators": {
        "proxy_anchor": {},
        "atomic_one_hot": {},
        "atomic_complementary": {},
        "atomic_full_union": {},
        "summed_union": {},
        "summed_dropout": {}
      }
    }
  ],
  "aggregates": {
    "m_unique": {},
    "configured": {},
    "equal_norm": {},
    "shared_confuser": {},
    "bootstrap": {}
  },
  "decision": {
    "thresholds": {},
    "component_decisions": {},
    "overall": null,
    "authorized_next_action": "none|write_separate_gpu_preregistration"
  },
  "integrity": {
    "accepted_context_count": 0,
    "rejected_context_count": 0,
    "invalid_context_count": 0,
    "input_replay_verified": false,
    "parameter_hash_before": null,
    "parameter_hash_after": null,
    "buffer_hash_before": null,
    "buffer_hash_after": null,
    "training_flags_restored": false,
    "all_finite": false
  }
}
```

The `null` source-dependent fields shown in this draft schema must be literal
frozen values in any activated output.  A runner that emits a scored result
while any such field is null is invalid.

## Invalidation conditions

Return `INVALID_OPERATING_POINT_MISMATCH` for any of the following:

- a source revision inferred from the current checkout rather than authenticated
  from the fresh PA report;
- any CIS artifact read before or during activation or execution;
- a batch size other than 180;
- balanced, P-K, class-filtered, or diagnostic-specific sampling;
- row deduplication or removal of production nonrepresentatives;
- a fixed unique-class count rather than context-specific `m_b`;
- eval-mode BN or an eval transform on `S`;
- persistent BN-buffer mutation;
- microbatching or different train forwards across operators;
- non-disjoint `S` and `S_prime` images within a context;
- a production or recipe edit made to supply the diagnostic full-union formula;
- adaptive step size, threshold, bundle count, source, proxy treatment, or null
  chosen after inspecting an operator value;
- any attempt to reinterpret or rescue Pass 120 with this diagnostic.
