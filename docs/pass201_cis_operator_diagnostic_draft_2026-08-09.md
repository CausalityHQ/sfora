# Pass 201 — CIS operator-isolation diagnostic draft

> **PENDING_SOURCE / NOT YET A PREREGISTRATION / NO COMPUTATION AUTHORIZED**
>
> The fresh current-code ordinary Proxy Anchor checkpoint, report, resolved
> configuration, and post-run integrity digests have not landed. Its prelaunch
> source identity is frozen separately, but every artifact-dependent field in
> this draft remains locked. This document
> cannot authorize artifact inspection, diagnostic computation, training, or a
> GPU run.

> **Prospective implementation-review repair:** before any source activation or
> candidate value, review found that the draft required an exact `BLOCKED`
> reason without naming one. This revision freezes that reason and the complete
> non-scored reason-code domains below; it changes no metric, threshold,
> operator, context, or decision predicate.

## Non-authority and scope

This is a proposed train-split-only, checkpoint-bound local operator diagnostic.
It cannot change, reinterpret, rescue, or retune the running Pass 120 experiment,
regardless of any Pass 120 outcome.  No CIS report, result, log, table, checkpoint,
or benchmark metric may be read while activating or executing this design.
The result must declare `uses_test_data="artifact_binding_only"`. Query/gallery
metrics may authenticate the PA report/checkpoint pair, but no query/gallery
image, label, descriptor, metric, or curve may enter contexts, constants,
thresholds, outcomes, control choice, aggregation, or the decision.

If eventually activated and passed, this diagnostic may authorize only the
writing of a separate prospective preregistration for at most one
norm-calibrated GPU arm.  It cannot authorize that GPU arm directly.  The future
arm would require its own frozen source, weight, seed, run command, and success
threshold.  A failure or unresolved result here closes the mechanism screen; it
does not permit a sweep or revised Pass 120 interpretation.

## Source activation gate

The sole admissible source is the fresh current-code, seed-0, ordinary Proxy
Anchor control produced by the separately source-bound In-Shop controller. The
trainer report and checkpoint format do not record the executing Git revision,
so they cannot be their own source authority.  The prelaunch authority is the
committed `docs/pass201_pa_source_prelaunch_manifest.json`, frozen while the
report, checkpoint, and log were all absent.  It binds the complete executed
`src/sfora` Python tree, launcher, entry point, dependency files, environment,
exact argv, official partition, and the path/content Merkle root of all 52,712
dataset images.  Its source tree is byte-identical to local revision
`f42ba573aa86080fd13b62ed19b4669eca1af5f7`; the unrelated dirty legacy Git
state of the remote checkout is recorded but is not treated as authority.

The launcher must recompute every prelaunch source and dataset digest before
training and after process exit.  A mismatch invalidates the PA artifact.  This
is a repair made before the PA artifact existed, not a post-result choice.

Before this draft can become a preregistration, a separate activation revision
must commit a `pass201-source-v1` manifest whose values were obtained without
reading any CIS output:

```json
{
  "schema_version": "pass201-source-v1",
  "status": "frozen",
  "prelaunch_source_manifest_path": "docs/pass201_pa_source_prelaunch_manifest.json",
  "prelaunch_source_manifest_sha256": "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803",
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

The activation revision must verify that the prelaunch manifest itself was
committed before the report, checkpoint, and log were created, then bind their
post-run hashes and resolved configuration.  The source checkpoint must be an
ordinary-PA checkpoint.  Any coalition,
single, complementary, dropout, residual, selected-on-CIS, older-code, or
externally published checkpoint is forbidden as a substitute.  If the fresh
ordinary-PA report/checkpoint pair never lands or cannot be authenticated, the
diagnostic remains `BLOCKED` with the sole reason code
`BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE`; no alternative source may be chosen.

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

All other runtime values, except the already frozen source revision, including
worker count, transform
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

`cross_context_reuse` is causal-prefix metadata so context 0 never depends on a
future context. It has exactly
`prior_context_indices_sharing_s_ids`,
`prior_context_indices_sharing_s_prime_ids`,
`prior_context_indices_sharing_any_ids`, `reused_s_image_count`,
`reused_s_prime_image_count`, `reused_any_image_count`, and
`reused_label_count`. The three index fields are ascending lists of accepted
context indices `< b` whose corresponding ID set has nonempty intersection
with the named current set. The four counts are cardinalities of the current
context's `S` IDs, `S_prime` IDs, union of both, and label set respectively that
also appeared in the union of accepted contexts `0..b-1`. Context 0 therefore
has three empty lists and four zeros in every process. Future-context reuse is
reported when that future context is emitted; no symmetric look-ahead field is
permitted.

If the first epoch yields fewer than 32 feasible context pairs, return status
`UNRESOLVED` with reason code
`UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS`. Do not inspect a later epoch,
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
record the exact `input-context-digest-v1` below for each. The
`integrity_replay_a` preparation pass in a fresh
process must reproduce every one of the 32 batch memberships and tensor hashes;
it scores the operator panel only for context 0.  The independent
`integrity_replay_b` process independently reproduces all 32 input contexts but
also scores only context 0.  Failure is
`INVALID_NONDETERMINISTIC_TRAIN_INPUT`.

`input-context-digest-v1` has exactly `context_index`, `s_tensor_sha256`,
`s_prime_tensor_sha256`, `metadata_sha256`, and `combined_sha256`.
For either tensor digest, convert the tensor to CPU C-contiguous storage without
changing dtype and hash this frame: little-endian `uint32` length of the exact
NumPy dtype string, its UTF-8 bytes, little-endian `uint32` rank, every dimension
as little-endian `int64`, little-endian `uint64` payload-byte length, and the
literal C-order payload bytes. `metadata_sha256` hashes canonical JSON of
exactly `context_index`, `production_epoch`, `production_batch_index`,
`row_example_ids`, `row_sample_indices`, `row_labels`,
`class_multiplicities`, `representative_row_indices`,
`representative_sample_indices`, `s_prime_example_ids`,
`s_prime_sample_indices`, and `cross_context_reuse`, using the canonical JSON
function frozen below. `combined_sha256` hashes canonical JSON of exactly the
other four fields. Each process record stores an ordered list of these complete
digest records, not a bare digest string or an implementation-specific pickle.

`S_prime[b]` uses the activated deterministic clean evaluation transform.  Its
input tensors and metadata receive the same hashing and replay check.  The
declared transfer question is therefore:

> Does one production-train-mode virtual update on augmented `S` improve clean,
> eval-mode outcomes on disjoint images `S_prime` having the same 180-row label
> sequence?

## Deterministic process and first-context replay

The runner must start in a fresh process with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python imports torch or initializes
CUDA. It must enable `torch.use_deterministic_algorithms(True)`, set cuDNN
benchmarking off and deterministic mode on, disable TF32 for both matmul and
cuDNN, use FP32 with autocast disabled, and fail rather than warn when an
executed operation lacks a deterministic implementation. Record all settings,
the accelerator model, CUDA/cuDNN versions, and every initial RNG-state hash.

Before hashing any initial RNG state, every process must execute exactly
`random.seed(2010812)`, `numpy.random.seed(2010812)`,
`torch.manual_seed(2010812)`, and `torch.cuda.manual_seed_all(2010812)` after
visible-device enumeration. The legacy NumPy global RNG is initialized only for
integrity and is forbidden for the null/bootstrap algorithms, which must use
their separately frozen `PCG64` generators. CUDA RNG hashes are recorded as a
mapping from visible integer device index to state SHA-256 in ascending index
order; the visible PCI-bus/device identity list is also frozen in that order.

The first fresh integrity-only child, `integrity_replay_a`, must start from the
frozen initial RNG state, prepare and hash every accepted input context 0–31 in
order, and compute the complete operator panel only for context 0.  The second
fresh integrity-only child, `integrity_replay_b`, starts from the same frozen
initial state, independently prepares and hashes all 32 input contexts in the
same order, and likewise scores only context 0. Requiring both integrity
processes to perform the complete preparation keeps the model-forward RNG state
identical even if input construction consumes a main-process RNG stream. After
their input lists and context-0 records agree, a third fresh `scientific`
process starts from that same frozen initial state, reconstructs and verifies
all 32 input-context hashes against both integrity processes, and executes
contexts 0–31 sequentially without rewinding RNG. Its context-0 record must
agree with both integrity-only records before it
is retained in the aggregate; neither integrity-only record enters the
scientific aggregate.  This is exactly three processes: no fourth input-replay
process and no in-process replay is allowed.

Require identical row IDs, stable indices, labels, transformed-tensor hashes,
representative rows, `S_prime` rows, and null streams across all three records.
For every gradient/update tensor, require maximum absolute difference `<=2e-6`;
for every norm, cosine, directional derivative, and outcome scalar, require
`abs(a-b)/max(abs(a),abs(b),1e-12) <=1e-5`. A failure is
`INVALID_NONDETERMINISTIC_OPERATOR_REPLAY` and no remaining context may be
scored. Persist process role and PID with every replay record.

## Bufferless train-mode gradient graph

For every `S[b]`:

1. Hash all checkpoint parameters, persistent buffers, and module training
   flags.
2. Clone every persistent buffer, including BN running means, running variances,
   and counters.
3. Put the functional model in training mode without freezing BN affine
   parameters.
4. The process-start initialization above is the sole RNG reset.  Before and
   after each scored context, capture the resulting CPU and every visible
   accelerator RNG state in the integrity record.  Do not reseed or rewind
   after input preparation or before the first scored context.
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
- `joint_including_proxies`: the same network parameters plus the learnable
  proxy table.

`network_only` is the primary causal panel because transfer to disjoint images
must pass through shared network parameters.  A joint-only success is classified
as proxy-table regularization and cannot rescue a network-only failure.
`joint_including_proxies` remains mandatory because the production operator
does update proxies; the name claims parameter membership only, not optimizer
faithfulness.

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

## Configured-loss and equal-norm stateless virtual updates

At activation, freeze literal values

```text
eta = resolved learning_rate
lambda = resolved coalition_weight
```

from the authenticated diagnostic config.  They cannot be selected from any
diagnostic result.

Configured-loss stateless directions are

```text
g_cfg[PA] = g_PA
g_cfg[o] = g_PA + lambda*g_o
Delta_cfg[o] = -eta*A*g_cfg[o]
```

where PA uses all 180 rows and every auxiliary uses the production-selected
`m_b` representatives from the shared 180-row train graph.

For equal-norm operator isolation, define a separate same-context reference norm
for each parameter panel `j in {network_only, joint_including_proxies}`:

```text
rho[b,j] = ||-eta*A_j*g_PA[b,j]||_2
Delta_eq[o,b,j]
  = -rho[b,j] * (A_j*g_o[b,j]) / ||A_j*g_o[b,j]||_2
```

for every pure operator, including pure PA. Thus every equal-norm operator has
exactly the PA parameter-update norm for the same production context and the
same parameter panel. A network-only update may never use the joint PA norm or
vice versa. No global norm target, adaptive step, line search, optimizer-state
reconstruction, or post-outcome step-size change is permitted.

Use a stateless functional parameter call; never mutate the checkpoint.  For
each configured-loss stateless and equal-norm update, report the actual `R_F`
and `Delta_M` plus the first-order predictions

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

Configured-loss stateless differences are reported identically but cannot rescue an
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
- `FAIL_SCALE_SUFFICIENT` if configured-loss stateless joint union beats
  full-union on both
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
`N=180` rows, `O=6` operators, `J=2` parameter panels, `R=256` null
replicates, and `m_b <= 180` the context-specific representative count.

Each context requires one shared 180-row train forward, six loss-gradient
reverse passes, one clean 180-row outcome graph with two outcome-gradient
reverse passes, and exactly `2*J*O = 24` stateless clean forwards for the two
update regimes, two parameter panels, and six operators. The permutation null
is streamed.

```text
time = O(Q * [(O+2)*C_backward(N,P)
              + (2+2*J*O)*C_forward(N,P)
              + R*m_b*(K-m_b)])
memory = O(one 180-row train graph + P + m_b*K)
```

Microbatching is prohibited because it changes train-mode BN semantics.  This
is a bounded local diagnostic, not an epoch-scale training arm.

## Output schema

The following reusable records are part of the frozen schema. `$ref` below is
specification notation and must not appear in the emitted result.

`operator-record-v1` has exactly:

- `name`: one of the six frozen operator names;
- `loss`: one finite float;
- `representative_count`: integer `m_b`;
- `panels`: exactly `network_only` and `joint_including_proxies`.

Each panel has exactly `parameter_count`, `gradient_sha256`,
`raw_gradient_norm`, `update_space_norm`, `auxiliary_to_pa_norm_ratio`,
`cosine_with_pa`, `cosine_with_atomic_full_union`,
`cosine_with_summed_dropout`, `scale_residual_to_summed_union`, and `updates`.
`parameter_count` is the integer scalar-element count and
`representative_count` is an integer row count; all other numeric fields are
finite floats. `scale_residual_to_summed_union` is a
finite float only for `atomic_one_hot`, `atomic_complementary`, and
`atomic_full_union`; it is literal JSON `null` for the other three operators.
`gradient_sha256` is the digest of the ordered float64-flattened gradient after
panel membership is applied but before `A`: panel-member parameters are ordered
lexicographically by the UTF-8 bytes of their exact checkpoint state-dict keys,
and each parameter contributes the following unambiguous byte frame: little-
endian `uint32` UTF-8 name-byte length, exact UTF-8 name bytes, little-endian
`uint32` rank, each dimension as little-endian `int64`, little-endian `uint64`
payload-byte length, then the C-contiguous little-endian float64 tensor bytes.
The SHA-256 input is the concatenation of those frames in the frozen parameter
order. Excluded parameters are absent, not zero filled. `update_sha256` uses
the identical framed panel-only order and encoding after `A` and regime
scaling.

`updates` has exactly `configured_loss_stateless` and `equal_norm`. Each update
has exactly `update_sha256`, `parameter_update_norm`, `R_F`, `Delta_M`, `D_F`,
and `D_M`. `equal_norm` additionally has `reference_pa_norm` and
`norm_match_absolute_error`; `configured_loss_stateless` has those two fields
set to literal JSON `null`. The equal-norm error must be `<=1e-10 *
max(reference_pa_norm,1e-12)`.

`summary-record-v1` has exactly `n`, `mean`, `median`, `sample_sd`, `q25`,
`q75`, `lcb_0_005`, and `ucb_0_995`. Every regime/panel/operator combination
must contain one such record for each of `R_F`, `Delta_M`, `D_F`, and `D_M`.
Every regime/panel must also contain paired `A_F` and `A_M` summary records.
`m_unique` and `E_shared` use the same summary record. The bootstrap record has
exactly `seed`, `replicates`, `quantile_method`, `joint_context_index_sha256`,
and `distribution_sha256_by_metric`. Missing, extra, duplicate, nonfinite, or
wrong-null fields invalidate the whole artifact; the validator must have a
mutation test for every field family.

`joint_context_index_sha256` hashes the complete `(20000,32)` resample-index
matrix encoded as little-endian int64, C-contiguous, in replicate order.
Every `distribution_sha256_by_metric` value hashes its complete length-20,000
bootstrap-mean vector encoded as little-endian float64, C-contiguous, in that
same replicate order. NaN canonicalization is irrelevant because any nonfinite
value invalidates the artifact before hashing.

`regime-aggregate-v1` has exactly the two panel keys. Each panel has exactly
`operators` and `paired_advantages`. `operators` has exactly the six frozen
operator keys, each containing exactly four `summary-record-v1` values named
`R_F`, `Delta_M`, `D_F`, and `D_M`. `paired_advantages` has exactly `A_F` and
`A_M`, each one `summary-record-v1`.

`frozen-thresholds-v1` has exactly these eleven machine keys with the literal
values in the PASS table above:

```text
shared_confuser_excess
network_equal_union_advantage_foreign
network_equal_union_advantage_margin
network_equal_union_foreign_suppression
network_equal_union_margin_change
network_equal_union_predicted_suppression
network_equal_union_predicted_margin_change
joint_equal_union_advantage_foreign
joint_equal_union_advantage_margin
joint_equal_union_foreign_suppression
joint_equal_union_margin_change
```

`component-decisions-v1` has exactly those eleven threshold keys; every value
is one of `PASS`, `FAIL`, or `UNRESOLVED`. The six frozen failure predicates
`FAIL_NO_SHARED_CONFOUNDER`, `FAIL_NO_COALITION_SPECIFIC_ACTION`,
`FAIL_NOT_VIABLE`, `FAIL_SCALE_SUFFICIENT`, `FAIL_PROXY_ONLY`, and
`FAIL_OWNER_DAMAGE` are reason codes only and may appear only in the top-level
`reason_codes` list when their documented predicate holds. The overall decision
must be recomputed from the eleven component decisions, summaries, and reason
predicates with the documented failure precedence and must reject an
inconsistent supplied status or reason-code set.

`distribution_sha256_by_metric` has exactly the canonical dot-path keys from
the cross-product of the two regimes, two panels, six operators, and four
metrics (`R_F`, `Delta_M`, `D_F`, `D_M`), plus both paired advantages for each
regime/panel and the two roots `m_unique` and `shared_confuser.E_shared`.
Canonical keys are sorted by UTF-8 bytes before JSON serialization; no alias,
missing key, or extra key is accepted.

An activated runner must emit exactly one JSON artifact with this top-level
shape:

```json
{
  "schema_version": "pass201-cis-operator-v1",
  "status": "PASS|FAIL|UNRESOLVED|BLOCKED|INVALID",
  "reason_codes": [],
  "candidate_values_computed": true,
  "uses_test_data": "artifact_binding_only",
  "source": {
    "prelaunch_source_manifest_path": "docs/pass201_pa_source_prelaunch_manifest.json",
    "prelaunch_source_manifest_sha256": "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803",
    "source_report_path": null,
    "source_report_sha256": null,
    "source_revision": null,
    "checkpoint_path": null,
    "checkpoint_sha256": null,
    "checkpoint_bytes": null,
    "checkpoint_epoch": null,
    "resolved_config_path": null,
    "resolved_config_sha256": null,
    "train_manifest_path": null,
    "train_manifest_sha256": null,
    "diagnostic_source_sha256": null,
    "activated_preregistration_sha256": null,
    "python_version": null,
    "torch_version": null,
    "numpy_version": null,
    "cuda_version": null,
    "cudnn_version": null
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
      "shared_confuser": {
        "A_aligned": null,
        "null_mean": null,
        "E_shared": null,
        "null_distribution_sha256": null
      },
      "operators": {
        "proxy_anchor": {"$ref": "operator-record-v1"},
        "atomic_one_hot": {"$ref": "operator-record-v1"},
        "atomic_complementary": {"$ref": "operator-record-v1"},
        "atomic_full_union": {"$ref": "operator-record-v1"},
        "summed_union": {"$ref": "operator-record-v1"},
        "summed_dropout": {"$ref": "operator-record-v1"}
      }
    }
  ],
  "aggregates": {
    "m_unique": {"$ref": "summary-record-v1"},
    "configured_loss_stateless": {"$ref": "regime-aggregate-v1"},
    "equal_norm": {"$ref": "regime-aggregate-v1"},
    "shared_confuser": {"$ref": "summary-record-v1"},
    "bootstrap": {"$ref": "bootstrap-record-v1"}
  },
  "decision": {
    "thresholds": {"$ref": "frozen-thresholds-v1"},
    "component_decisions": {"$ref": "component-decisions-v1"},
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
    "deterministic_process_verified": false,
    "first_context_operator_replay_verified": false,
    "deterministic_settings": {
      "cublas_workspace_config": ":4096:8",
      "deterministic_algorithms": true,
      "cudnn_benchmark": false,
      "cudnn_deterministic": true,
      "matmul_tf32": false,
      "cudnn_tf32": false,
      "autocast": false,
      "dtype": "float32"
    },
    "process_records": [
      {
        "role": "integrity_replay_a|integrity_replay_b|scientific",
        "pid": null,
        "accelerator": null,
        "python_version": null,
        "torch_version": null,
        "cuda_version": null,
        "cudnn_version": null,
        "visible_cuda_devices": [],
        "initial_python_rng_sha256": null,
        "initial_numpy_rng_sha256": null,
        "initial_torch_cpu_rng_sha256": null,
        "initial_torch_cuda_rng_sha256_by_device": {},
        "prepared_context_count": null,
        "input_context_digest_records": [],
        "context0_record_sha256": null
      }
    ],
    "replay_residuals": {
      "pair_count": 3,
      "tensor_max_absolute": null,
      "scalar_max_relative": null,
      "tensor_tolerance": 0.000002,
      "scalar_tolerance": 0.00001,
      "scalar_denominator": "max(abs(a),abs(b),1e-12)"
    },
    "all_finite": false
  }
}
```

The shape above is the **scored** payload. Conditional payloads are frozen as
follows:

- Scored `PASS`, `FAIL`, or ordinary `UNRESOLVED` must set
  `candidate_values_computed=true`, contain exactly 32 complete contexts, and
  include every aggregate, bootstrap, component decision, and integrity field.
  Every per-context `shared_confuser` value and every operator-record value must
  be finite and non-null except the explicitly allowed scale/reference nulls.
- Early `UNRESOLVED` is permitted only for
  `UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS`. It must set
  `candidate_values_computed=false`, contain only the partial
  context-construction audit described below (no tensors, operators, outcomes,
  or candidate scores), and omit `aggregates` entirely (the bootstrap record is
  nested inside `aggregates` and therefore is also absent).
- `BLOCKED` or `INVALID` must set `candidate_values_computed=false`, contain no
  `contexts`, `aggregates`, or scientific component decisions, and
  include only the common provenance/constants, binding/integrity evidence,
  exact reason codes, `overall`, and `authorized_next_action="none"`.

Every early-`UNRESOLVED` partial `contexts` entry has exactly
`context_index`, `production_epoch`, `production_batch_index`, `status`,
`rejection_code`, `row_example_ids`, `row_sample_indices`, `row_labels`,
`class_multiplicities`, `representative_row_indices`,
`representative_sample_indices`, `s_prime_example_ids`, and
`s_prime_sample_indices`. `status` is exactly `accepted` or `rejected`.
`rejection_code` is literal JSON `null` for an accepted entry and exactly
`INSUFFICIENT_DISJOINT_S_PRIME` for a rejected entry. Rejected entries retain
the original 180-row IDs, indices, labels, multiplicities, and representative
indices but have empty `s_prime_example_ids` and `s_prime_sample_indices`.
Accepted entries contain the complete 180-row `S_prime` IDs and indices. No
other rejection code or partial-context field is legal. The list contains every
production batch consumed in epoch 0 through the end of that epoch, in sampler
order; `context_index` is the zero-based consumed-batch index, not the accepted
pair rank.

The full scored `decision` record has exactly `thresholds`,
`component_decisions`, `overall`, and `authorized_next_action`. For early
`UNRESOLVED`, `BLOCKED`, and `INVALID`, the reduced `decision` record has
exactly `thresholds`, `overall`, and `authorized_next_action`; `thresholds` is
still the literal `frozen-thresholds-v1`, `overall` must equal the top-level
status, and `authorized_next_action` must be `none`. A reduced payload may not
smuggle in `component_decisions`.

The displayed `integrity` object is `scored-integrity-v1` and is legal only for
a scored 32-context payload. Non-scored payloads instead use exactly
`reduced-integrity-v1`, with keys `stage`, `accepted_context_count`,
`rejected_context_count`, `invalid_context_count`, `input_replay_verified`,
`deterministic_process_verified`, `process_records`,
`failure_evidence_sha256`, and `all_finite`. `stage` is one of
`source_activation`, `context_construction`, `integrity_replay_a`,
`integrity_replay_b`, or `scientific`. Counts are nonnegative integers;
`failure_evidence_sha256` is the 64-lowercase-hex SHA-256 of canonical JSON
containing exactly the top-level status, sorted reason-code list, stage, counts,
and the last reached process record. When no process launched, that last
record is literal JSON `null`. The two verification fields and `all_finite` are
literal booleans, never null.

Every reduced process entry uses the same exact `process-record-v1` fields shown
in the scored schema and audits a launched process even if that process is the
one that fails. It may set `context0_record_sha256` to literal JSON `null`
if and only if no complete context-0 operator panel was produced. For every
launched process, `pid` is a positive integer; `accelerator`, all four version
fields, and every visible-device identity are nonempty strings;
`visible_cuda_devices` is a nonempty ascending list; all three scalar initial RNG
digest fields and every CUDA-state mapping value are 64-lowercase-hex;
`prepared_context_count` is a nonnegative integer equal to the length of
`input_context_digest_records`; and every digest record validates exactly.
No other process field may be null. The process list
is always the launched prefix of
`[integrity_replay_a, integrity_replay_b, scientific]`, never a subset or a
reordered list. Stage `source_activation` requires zero process records and all
three counts zero. Early `UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS` requires
stage `context_construction`, exactly one `integrity_replay_a` process record,
`accepted_context_count` in `0..31`,
`rejected_context_count >= 1`, `invalid_context_count=0`, both verification
booleans false, and `all_finite=true`; that process record contains exactly the
accepted input digest records and no context-0 operator digest. For `INVALID`,
the process-record count is exactly 0 for `source_activation`, exactly 1 for
`context_construction` or `integrity_replay_a`, exactly 2 for
`integrity_replay_b`, and exactly 3 for `scientific`;
`invalid_context_count=0` at `source_activation` because no context exists, and
`invalid_context_count >= 1` at every later stage. A `BLOCKED` payload is legal only at
`source_activation`, with zero process records/counts and all three booleans
false. Any other null, process count, stage/status combination, or extra scored
integrity field is invalid.

The common top-level key set is exactly `schema_version`, `status`,
`reason_codes`, `candidate_values_computed`, `uses_test_data`, `source`,
`constants`, `decision`, and `integrity`, plus only the conditionally authorized
keys above. Validators must mutate each status family to prove that fabricated
numeric fields and missing required fields both fail closed.

The complete non-scored reason-code domains are now exact. Early `UNRESOLVED`
has exactly `["UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS"]`. `BLOCKED`
has exactly `["BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE"]`. `INVALID` has a
nonempty, duplicate-free list drawn only from
`INVALID_OPERATING_POINT_MISMATCH`,
`INVALID_NONDETERMINISTIC_TRAIN_INPUT`, and
`INVALID_NONDETERMINISTIC_OPERATOR_REPLAY`, sorted by UTF-8 bytes. A source-
activation `INVALID` can contain only `INVALID_OPERATING_POINT_MISMATCH`.
Scored `PASS` has an empty list; scored `FAIL` uses only the six frozen failure
predicate codes; scored ordinary `UNRESOLVED` has an empty list. No alias,
free-form explanation, or additional reason code is legal.

The `null` source-dependent fields shown in this draft schema must be literal
frozen values in any activated output.  A runner that emits a scored result
while any such field is null is invalid.

`process_records` must contain exactly three entries in this order:
`integrity_replay_a`, `integrity_replay_b`, and `scientific`.
Every one of the three records has `prepared_context_count=32` and the same
complete ordered list of exactly 32 `input-context-digest-v1` records. The
first two score only context 0 and never contribute an aggregate row. The
scientific record owns the retained context 0 and then the uninterrupted
contexts 1–31. `context0_record_sha256` hashes the complete operator-containing
scientific-context object at conceptual path `contexts[0]`, constructed with
the exact same schema in all three processes. It includes no process metadata
and excludes no field from that context object. Hash UTF-8 of
`json.dumps(context_object, sort_keys=True, separators=(",",":"),
ensure_ascii=False, allow_nan=False)` under the bound Python version. This
canonicalization is used for all three records.

## Invalidation conditions

Return `INVALID_OPERATING_POINT_MISMATCH` for any of the following:

- a source identity inferred from the post-run checkout/report rather than
  authenticated by the committed prelaunch source manifest and post-run digest
  replay;
- any CIS artifact read before or during activation or execution;
- a batch size other than 180;
- balanced, P-K, class-filtered, or diagnostic-specific sampling;
- row deduplication or removal of production nonrepresentatives;
- a fixed unique-class count rather than context-specific `m_b`;
- eval-mode BN or an eval transform on `S`;
- persistent BN-buffer mutation;
- microbatching or different train forwards across operators;
- missing deterministic settings, a deterministic-operation warning, or a
  context-0 replay residual above its frozen tolerance;
- any query/gallery value used outside artifact binding;
- non-disjoint `S` and `S_prime` images within a context;
- a production or recipe edit made to supply the diagnostic full-union formula;
- adaptive step size, threshold, bundle count, source, proxy treatment, or null
  chosen after inspecting an operator value;
- any attempt to reinterpret or rescue Pass 120 with this diagnostic.
