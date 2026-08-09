# Pass 200 RSTA adjoint-integrity-prefix amendment — 2026-08-09

## Status and evidentiary boundary

This amendment is frozen before any further RSTA candidate computation. It
repairs the adjoint reduction and makes all-seed integrity a strict prefix of
scientific scoring. It changes no RSTA candidate formula, image, checkpoint,
batch, role, support, Proxy Anchor loss, cotangent, field, control, statistic,
bootstrap, threshold, prediction, or decision rule. The candidate remains
**LIVE-NARROW at Gate 2; Gate 1 unresolved**.

The H6 handoff is full commit
`e836d5861154c2a7674f366adf496b90c38b9da4`. Its manifest binds scientific
source commit `b6f41a4c82ddf96f5cda238ed1db7a0d47fc7279` and diagnostic SHA-256
`5fae385dd25115ba817349321d27757c21b462997fc08f66c7ed0459dbb4a3f3`.
The H6 integrity smoke passed. The H6 four-seed scientific process later exited
`INVALID` on a later-seed full-model adjoint check and created no output file.
Before that failure, the current seed-local schedule had computed partial seed-0
receiver rows in memory. Those rows were never persisted, printed, inspected,
aggregated, bootstrapped, or used for a decision. They are discarded and are not
RSTA evidence. No Stage-A result or decision exists.

## Measured implementation defects

The H6 adjoint identity forms both inner products in model FP32. Summing millions
of signed products in FP32 can make the registered identity fail from reduction
roundoff even when the FP32 JVP and VJP are mutually adjoint. This amendment
changes only the multiply/reduction arithmetic used to audit the identity.

The H6 scientific loop also violates the intended all-seed integrity prefix. It
runs first-batch integrity for one seed and then immediately calls
`score_rsta_batch` for that seed's primary and alternate batches. A later seed can
therefore fail integrity after earlier-seed candidate rows have already been
computed. Per-seed integrity-before-own-score is insufficient: every seed must
pass every registered integrity gate before any seed may enter candidate scoring.

## Frozen adjoint identity

The following objects and operations remain unchanged:

- the output direction is the C-order standard-normal draw from a fresh NumPy
  `Generator(PCG64(domain_seed('rsta-stage-a-v1|adjoint-u|', str(seed))))`;
- the parameter direction is the C-order standard-normal draw from a separate
  fresh NumPy
  `Generator(PCG64(domain_seed('rsta-stage-a-v1|adjoint-v|', str(seed))))`,
  flattened once and partitioned in exact trainable named-parameter order;
- `domain_seed(domain, text)` is SHA-256 of domain ASCII bytes, one NUL byte,
  and text UTF-8 bytes, interpreted from the first eight digest bytes as an
  unsigned big-endian integer;
- seed order is exactly `0,1,2,3`, batch size is exactly `180`, the deployed
  model and both consumed direction tensors are FP32, and the same exact
  `torch.func.jvp` and `torch.func.vjp` evaluate `Jv` and `J^T u`;
- the denominator is exactly
  `max(abs(lhs), abs(rhs), float64(1e-12))`, the relative error is
  `absolute_error / denominator`, and the tolerance is exactly `5e-4`.

Only the two inner products change. Cast both factors of every product to
float64 before multiplication. Compute `lhs` with a float64 sum. Compute each
named-parameter RHS inner product with a float64 sum, stack the terms in exact
named-parameter order, and sum that stack in float64. The model, primal forward,
JVP, VJP, and directions remain FP32. No direction normalization, resampling,
tolerance widening, alternate denominator, compensated summation, or float64
model execution is authorized.

The exact persisted adjoint object has these keys and no others, in this order:

```text
direction_domain
output_direction_seed
parameter_direction_seed
output_direction_sha256
parameter_direction_sha256
output_shape
parameter_name_order_sha256
parameter_count
model_dtype
reduction_dtype
lhs
rhs
absolute_error
denominator
relative_error
tolerance
passed
```

`direction_domain` is exactly `rsta-stage-a-v1`. The two seed fields are the
unsigned big-endian integers derived above. `output_direction_sha256` hashes the
C-contiguous bytes of the actual FP32 output-direction tensor consumed by the
VJP. `parameter_direction_sha256` hashes the ordered concatenation of the
C-contiguous bytes of the actual FP32 parameter-direction tensors consumed by
the JVP, in exact trainable named-parameter order. These hashes bind the already
frozen directions after their existing float64 PCG64 arrays are cast; they do
not alter the draws. `output_shape` is the integer shape list of the B=180 model
output. `parameter_name_order_sha256` uses the repository's existing
`_ordered_text_sha256` framed encoding. `parameter_count` is the total number of
elements in that ordered trainable parameter tuple. `model_dtype` and
`reduction_dtype` are exactly `torch.float32` and `torch.float64`.

`lhs`, `rhs`, `absolute_error`, `denominator`, `relative_error`, and `tolerance`
are finite JSON numbers, with `absolute_error=abs(lhs-rhs)`, the denominator and
relative error defined above, and `tolerance=0.0005`. `passed` is exactly
`relative_error <= tolerance`. A nonfinite input, product, reduction, or derived
scalar is structural `INVALID`, not a recorded tolerance failure.

## Candidate-free four-seed adjoint audit

Add CLI mode `--integrity-all-seeds-only`. It remains mutually exclusive with
`--smoke-only` and `--scientific` and requires the same exact manifest, binding
receipt, output path, deterministic configuration, provenance validation,
training-only loaders, B=180 transform cache, models, and immutable artifacts.
It never calls `exact_contextual_rsta_fields`, `score_rsta_batch`,
`decide_stage_a`, `joint_bootstrap`, `scientific_payload`, or any receiver-row
serializer. It computes no RSTA field or candidate statistic.

The output has exactly these top-level keys:

```text
schema_version
diagnostic
mode
candidate_values_computed
stage_a_verdict
uses_test_data
execution_audit
manifest
environment
binding
integrity
```

Their fixed scalar values are `schema_version=1`,
`diagnostic='pass200-rsta-adjoint-integrity'`,
`mode='integrity_all_seeds'`, `candidate_values_computed=false`,
`stage_a_verdict='NOT_COMPUTED'`, and
`uses_test_data='artifact_binding_only'`. There are no `rows`, `fields`,
`scores`, `decision`, `aggregation`, or `bootstrap` keys at any depth.

`binding` has exactly these keys:

```text
receipt_sha256
receipt_producer_commit
historical_manifest_sha256
seeds
```

`seeds` has exact string keys `'0','1','2','3'`. Each seed value has exactly:

```text
checkpoint_sha256
train_pack_sha256
first_batch_ordered_id_sha256
transform_cache_order_sha256
transform_tensor_set_sha256
```

The first two values are the digest-bound checkpoint and final-training-pack
SHA-256 values. The ordered-ID hash binds all 180 first-batch example IDs. The
cache hash binds the deterministic transform-cache order. The tensor-set hash
uses the existing ordered framed hash over each ordered
`<example_id>\0<tensor_sha256>` string. All four checkpoint and training-pack
hashes must be present even if their bytes happen to match across seeds.

`integrity` has exactly `dense_fixture`, `bn_fixture`,
`deterministic_global_max`, `seeds`, and `all_passed`. Its `seeds` mapping has
exact string keys `'0','1','2','3'`; each value has exactly
`zero_jacobian_classifier` and `adjoint`. The zero-Jacobian object is the
already-frozen `pass200-zero-jacobian-last-linear-v1` audit. The adjoint value is
the exact object above. `all_passed` is true exactly when all four adjoint
objects have `passed=true` and every preceding fixed audit passed.

A finite relative error above `5e-4` records `passed=false`, continues through
the remaining seeds, writes the complete four-seed audit atomically, and makes
`all_passed=false`. This exception to fail-fast behavior exists only so a
candidate-free diagnostic can expose the complete numerical pattern. Receipt,
manifest, source, artifact, binding, configuration, shape, parameter topology,
hash, zero-Jacobian, fixture, deterministic-pool, nonfinite, serialization, or
atomic-publication failure remains immediately `INVALID` and produces no output.

## Scientific all-seed integrity prefix

Scientific execution is split into two phases after common provenance, binding,
cache, dense-fixture, BN-fixture, and deterministic-pool validation:

1. In seed order `0,1,2,3`, construct one seed model and its registered first
   B=180 context; complete that seed's zero-Jacobian audit, exact field
   repeatability, exact adjoint object, and rotation audit. Persist the audit in
   memory, then release all field tensors and derivative graphs before the next
   seed. Any failure exits `INVALID`, makes zero calls to candidate scoring or
   decision/bootstrap code, and creates no output.
2. Only after all four seeds pass, enter scoring. Reconstruct each seed model
   from the same immutable checkpoint bytes, restore the exact zero-Jacobian
   exclusion, and recompute every contextual field used by scoring. Never reuse
   a field or graph from the integrity phase. Score the unchanged primary and
   alternate panels serially, release each graph before the next batch, then run
   the unchanged aggregation, bootstrap, and decision.

This schedule preserves one full B=180 derivative graph at peak. It also proves
that every candidate row belongs to a run whose entire four-seed integrity prefix
already passed. Scientific output retains the exact full adjoint object for every
seed instead of the former scalar `adjoint_relative_error`.

## Provenance transition and interpretation

Implementation uses a new independently reviewed source commit `S7`. The
receipt-backed manifest gains an exact `adjoint_integrity_amendment` entry with
`path`, `sha256`, and the commit containing this document. A manifest-only
handoff `H7` binds that entry and every current scientific source file to `S7`.
The base preregistration, prior amendments, historical receipt, historical source,
artifact schema, and seed domains remain byte-semantically unchanged.

The candidate-free all-seed audit is executability evidence only. It cannot pass,
fail, or resolve RSTA Gate 1. A green audit authorizes one fresh four-seed
scientific process at the same H7 source and configuration. A failed scientific
integrity prefix remains `INVALID`; only a fully persisted, independently
validated scientific result may be adjudicated by the original Stage-A rules.
