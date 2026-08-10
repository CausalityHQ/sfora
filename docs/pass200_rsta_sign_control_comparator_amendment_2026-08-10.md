# Pass 200 RSTA signed-control comparator amendment — 2026-08-10

## Status and scientific boundary

This is a prospective, non-scientific integrity amendment. It is committed
after the registered synthetic CPU calibration passed, after the prospective
normwise amendment and its reviewed production implementation, and before any
new DGX execution, real-data audit, or candidate computation. It changes only
how the two production adjoint sign controls prove exact signed action
relations. It does not change an RSTA direction, checkpoint, image, batch,
model operation, baseline action, rebuild action, reversed-order action,
field, receiver, score, aggregate, bootstrap, prediction, decision rule, or
tolerance.

The candidate-free H8 audit remains failed under its registered legacy gate.
H8 produced no candidate. The registered synthetic calibration remains passed.
The normwise amendment, its production implementation, and its source-boundary
repair remain prospective authorities in their original chronology. This
amendment does not recompute, reinterpret, repair, erase, or supersede any
prior artifact or result. A later audit under this amendment will be new
prospective evidence.

No DGX command, new real-data audit, or candidate computation was run while
reviewing or authoring this amendment. No RSTA field, receiver row, candidate
score, aggregate, bootstrap value, scientific verdict, or scientific result
was computed or inspected.

## Exact chronology and defect disclosure

The chronology is exact:

1. H8, commit `cc7b0a102d938db0cf49e756fc4d18410186bf4d`,
   failed its registered candidate-free all-seed integrity gate. Its artifact
   SHA-256 is
   `234cc3055b0209bcd095f2932867ff09252ca1c2d4b5e5080a988c77fcd5e74c`.
   It had `all_passed=false` and computed no candidate.
2. The registered synthetic CPU normwise-adjoint calibration passed. Its
   complete result is
   `reports/generated/pass200_rsta_receipt/0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2-normwise-adjoint-calibration.json`,
   SHA-256
   `5fcb09a1e3a6eedddd05ef49bd22bc9920656089aa401a5aae2c5704a9d9dc50`,
   committed at `95525af61d66b063983dc55a6015168d9aafd12b`.
3. The prospective normwise-adjoint amendment was fixed and committed at
   `6ddf1db20e75a47e40726d223827cd3f1a8968e3`. Its path is
   `docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md` and its
   SHA-256 is
   `416fdd6af90fa2e54ace61fcd72721713aae84dc0dd2010bde91037bf0eccbd4`.
4. The reviewed normwise production source was committed at
   `718cda41bfbf49234a75a1c8763e39e57a5eb246` with subject
   `implement RSTA normwise adjoint integrity`.
5. Its source-boundary repair was committed at
   `a456ac620999e823cf8534cf877ae9647f7c458e` with subject
   `fix RSTA normwise source boundaries`.
6. A fresh, read-only signed-zero and dead-ReLU review then found that the
   current production comparator derives each sign trial's `exact_relation`
   from raw SHA-256 equality against a separately negated baseline action. That
   raw-negated-hash relation can be false when the registered mathematical
   relation is true under direct `torch.equal`.
7. The initial prospective comparator amendment was committed docs-only at
   `a27dd7b3c8ff089c7cb80821c43658b975985a34`, and its initial implementation
   plan was committed docs-only at
   `3df1f9571c910fb1240b82c4f8addb5b2a8c5dce`.
8. A subsequent independent read-only documentation review returned
   `NOT READY`, with no Critical finding and two Important findings. First, the
   initial plan incorrectly routed the live comparator through the calibration
   helper `_tensor`, whose registered contract rejects CUDA tensors. Second,
   its equality sentinel counted calls without proving the exact provenance of
   either operand, so signed-zero canonicalization or detached reconstruction
   could create false confidence. This repair resolves both documentation
   defects before source work.

IEEE signed zero explains the mismatch. A dead ReLU or another exact-zero
action can produce `+0.0` in one fresh derivative evaluation while explicit
negation of a separately retained baseline produces `-0.0`. Their raw FP32
bytes, and therefore their SHA-256 values, differ. PyTorch direct
`torch.equal`, however, correctly accepts `+0.0` and `-0.0` as elementwise
equal for this registered exact tensor relation. Hashes remain exact byte
evidence, but a hash of an explicitly negated tensor is not the authority for
signed numerical equality.

Both review rounds were read-only. They produced no source, test, manifest,
result, GPU, real-audit, or candidate change, and no new real audit, DGX result,
or candidate value. They do not establish how any prior seed would behave under
the repaired comparator and do not alter the published calibration artifact,
whose complete bytes and original schema remain historical evidence.

## Frozen design A: same-graph target and reference

The original production baseline, exact-rebuild, and reversed-action-order
trials remain unchanged in construction, directions, action order, metrics,
hashes, schema, predicates, and lifetime. Only the `parameter_sign` and
`output_sign` trials change.

Each sign control uses one fresh derivative graph constructed from the same
immutable functional model state and inputs as the baseline. That graph
executes the registered target action pair and one ephemeral baseline-reference
action pair. The target and reference share the graph, functional state,
parameter-name order, primal tensors, and VJP closure. Neither pair redraws or
normalizes a direction.

For each sign-control graph, the literal call order is:

```text
1. construct exactly one VJP closure
2. execute the target JVP
3. execute the target VJP through that closure
4. execute the reference JVP
5. execute the reference VJP through that same closure
```

Thus each sign-control graph has exactly one `torch.func.vjp` closure
construction, two `torch.func.jvp` action calls, and two calls to the one VJP
closure, in the order above. No second closure, reordered call, cached action,
analytic action, or separate reference graph is permitted.

The exact target and reference action inputs are:

| Control | Target `(parameter direction, output direction)` | Reference `(parameter direction, output direction)` |
|:---|:---|:---|
| `parameter_sign` | `(-v, u)` | `(v, u)` |
| `output_sign` | `(v, -u)` | `(v, u)` |

Here `u` and `v` are the unchanged registered baseline direction tensors. The
functional model state and input tensors are immutable across both pairs.

## Device-agnostic live comparator contract

The authenticated normwise helper adds one separate function named exactly
`exact_live_sign_control_relation`. Its exact interface is:

```text
exact_live_sign_control_relation(
    control_name: str,
    target_jvp: torch.Tensor,
    target_vjp: Mapping[str, torch.Tensor],
    reference_jvp: torch.Tensor,
    reference_vjp: Mapping[str, torch.Tensor],
    parameter_names: Sequence[str],
    *,
    expected_device: torch.device,
) -> bool
```

This function is device-agnostic. It must operate on the live action device,
including CUDA in production, and must not call or route an operand through the
calibration helper `_tensor`. The definition and behavior of `_tensor` remain
unchanged: it continues to accept only finite CPU FP32 tensors for the existing
calibration and detached diagnostic paths. No existing calibration constructor,
trial, metric, hash, validator, CLI, artifact, or result path may import, call,
or depend on `exact_live_sign_control_relation`.

Before the first equality call, the live comparator performs structural
validation on device. It accepts only actual `torch.Tensor` instances with
exact dtype `torch.float32` whose values are finite. `parameter_names` must be a
nonempty exact sequence of unique nonempty strings. Both VJP mappings must have
exactly those keys in that insertion order and no others. Target and reference
JVPs must have identical shape and device, and that device must equal
`expected_device`. For every name in registered order, target and reference VJP
tensors must have identical shape and device; that device must equal the JVP
device and `expected_device`. A wrong control name, tensor type, dtype,
finiteness, topology, name/order, shape, or device is structural and raises
before any equality call.

The production caller derives `expected_device` from the live functional model
actions and first proves that all live functional parameters/actions use that
same device. The comparator never transfers an action to satisfy the device
contract.

While all raw target and reference tensors are still live on that validated
device, compute `exact_relation` directly with `torch.equal` as follows:

```text
parameter_sign:
    comparison_results[0] = torch.equal(target_jvp, -reference_jvp)
    for each name in exact named-parameter order:
        append torch.equal(target_vjp[name], reference_vjp[name])

output_sign:
    comparison_results[0] = torch.equal(target_jvp, reference_jvp)
    for each name in exact named-parameter order:
        append torch.equal(target_vjp[name], -reference_vjp[name])

exact_relation = all exact Python booleans in comparison_results
                 after every comparison has executed
```

The equality-call schedule is exact and never short-circuits: call
`torch.equal` once for the JVP first, then exactly once for every VJP tensor in
registered trainable named-parameter order, with no additional equality call.
Collect the exact Python boolean results and conjoin them only after every
registered comparison has executed.

For every call, the left operand is the identical raw target action object
yielded by `torch.func`, not a view, clone, detached tensor, CPU copy,
contiguous copy, reconstruction, or canonicalized value. For an unchanged
relation, the right operand is the identical raw reference action object
yielded by `torch.func`. For a negative relation, the right operand is the
immediate result of direct unary negation of that live raw reference tensor at
the equality call: the production expression is exactly
`torch.equal(raw_target, -raw_reference)`. No detached, copied,
canonicalized, reconstructed, precomputed, cached, or multiply-by-minus-one
substitute is permitted.

All structural validation and every direct equality call occur before any
target, reference, or negated action is detached, cloned, moved with `.cpu()`
or any `.to` call, made contiguous, converted to NumPy, hashed, zero-canonicalized,
or reconstructed. Direct `torch.equal` on the live device is normative:
signed-zero equality is accepted, and no raw hash comparison, `allclose`,
tolerance, scalar reduction, canonicalization, zero-sign rewrite, or post-hoc
reconstruction may substitute for it.

## Exact evidence retention and graph lifetime

For each sign control, compute the existing normwise metrics only for the
registered target pair. Persist only the target JVP/VJP raw hashes, target
`beta_norm`, the two reference raw hashes, and the registered booleans. Do not
compute or persist a reference `lhs`, `rhs`, norm, error, denominator,
`eta_norm`, `beta_norm`, cancellation factor, metric object, or any other
reference metric.

`reference_jvp_sha256` hashes the C-contiguous raw FP32 bytes of the ephemeral
reference JVP. `reference_vjp_sha256` hashes the ordered concatenation of the
C-contiguous raw FP32 reference VJP parameter tensors in exact trainable
named-parameter order. The target hash encodings remain unchanged.

Both reference hashes must exactly equal the original baseline raw hashes from
the unchanged baseline graph:

```text
reference_jvp_sha256 == baseline jvp_sha256
and
reference_vjp_sha256 == baseline vjp_sha256
```

This independent byte-exact reference requirement prevents a mutually
consistent target/reference drift from masking reference drift. It does not
require the sign-changed target hash to equal a raw hash computed by explicitly
negating baseline or reference tensors.

The direct tensor relation and every structural check must be completed while
the identical raw target and reference actions are live. Only after the final
named VJP equality returns may production detach or transfer tensors and
compute the registered JSON-ready target metrics, booleans, and raw hashes.
Release the VJP closure, functional parameters, primal outputs, graph actions,
raw target tensors, raw reference tensors, directions derived for the trial,
detached CPU tensor copies, and every immediate unary-negated reference
temporary before constructing the next graph. Only JSON scalars, exact Python
booleans, and lowercase SHA-256 strings may survive. At most one full
derivative graph exists at peak.

The complete per-seed production action schedule is therefore exactly:

```text
baseline:
    construct VJP closure; baseline JVP; baseline VJP
rebuild:
    construct VJP closure; rebuild JVP; rebuild VJP
reversed_action_order:
    construct VJP closure; reversed VJP; reversed JVP
parameter_sign:
    construct one VJP closure;
    target(-v,u) JVP; target(-v,u) VJP;
    reference(v,u) JVP; reference(v,u) VJP
output_sign:
    construct one VJP closure;
    target(v,-u) JVP; target(v,-u) VJP;
    reference(v,u) JVP; reference(v,u) VJP
```

Every trial remains a fresh graph. The baseline/rebuild/reversed trials still
make one JVP and one VJP action each. Each sign trial makes exactly two JVP and
two VJP actions through one closure. The full per-seed total is exactly five
VJP closure constructions, seven JVP action calls, and seven VJP-closure action
calls in the literal order above.

## Exact nested production schema and predicates

The top-level production adjoint object and its exact field order remain
unchanged. `controls` retains exactly these ordered keys:

```text
rebuild
reversed_action_order
parameter_sign
output_sign
```

The `rebuild` and `reversed_action_order` schemas and semantics remain
unchanged. Each continues to have exactly:

```text
jvp_sha256
vjp_sha256
beta_norm
exact_action_hash_match
passed
```

Each `parameter_sign` and `output_sign` object is extended to have exactly the
following keys in this order and no others:

```text
jvp_sha256
vjp_sha256
reference_jvp_sha256
reference_vjp_sha256
beta_norm
reference_exact_action_hash_match
exact_relation
passed
```

The meanings and exact predicates are:

- `jvp_sha256` and `vjp_sha256` are the unchanged raw hashes of the registered
  target actions.
- `reference_jvp_sha256` and `reference_vjp_sha256` are the raw hashes of the
  ephemeral same-graph reference actions.
- `beta_norm` is the target-only normwise metric under the existing frozen
  arithmetic and corner encoding.
- `reference_exact_action_hash_match` has exact Python/JSON boolean type and is
  true if and only if both reference hashes equal the corresponding original
  baseline raw hashes.
- `exact_relation` has exact Python/JSON boolean type and is exactly the
  conjunction of the direct live `torch.equal` comparisons registered above.
- `passed` has exact Python/JSON boolean type and is true if and only if
  `reference_exact_action_hash_match is True`, `exact_relation is True`,
  `type(beta_norm) is float`, and `beta_norm <= 0.0005`.

Equivalently, the exact predicate is:

```text
type(reference_exact_action_hash_match) is bool
and reference_exact_action_hash_match is True
and type(exact_relation) is bool
and exact_relation is True
and type(beta_norm) is float
and beta_norm <= 0.0005
```

An authorized `beta_norm="infinity"` makes `passed=false`. No integer, NumPy
boolean, string, truthy substitute, missing field, reordered field, or extra
field is accepted.

`integrity_passed` remains exact Python/JSON boolean type. Its semantics are
updated only through the extended sign-control predicates: it is true exactly
when `normwise_passed is True` and the unchanged `rebuild`, unchanged
`reversed_action_order`, extended `parameter_sign`, and extended `output_sign`
`passed` values each have exact boolean type and value `True`. The legacy
adjoint `passed` remains evidence and is not an input to
`integrity_passed`.

## Validation, failure, and output ordering

Production validators must require the exact nested order above, exact types,
and exact predicate derivations. They must recompute
`reference_exact_action_hash_match` from the two persisted reference hashes and
the two persisted baseline hashes. They must bind `exact_relation` as the exact
boolean produced by the registered direct live-tensor conjunction and bind
`passed` and `integrity_passed` to their complete predicates. They must never
infer a sign relation by hashing an explicitly negated baseline or reference
tensor. Direct `torch.equal` is the signed-relation authority, and signed zero
is accepted by that authority.

Tests must instrument direct `torch.equal` and prove operand provenance, not
only call count. In exact call order, each left operand must be the identical
raw target action object yielded by fake or real `torch.func`; each unchanged
right operand must be the identical raw reference object; and each negative
right operand must be the recorded immediate unary-negated live-reference
temporary on the same device. The tests must prove the comparator runs before
any detach, clone, CPU/device transfer, contiguity conversion,
canonicalization, or reconstruction. Sentinel tensor subclasses or wrappers
must raise if one of those operations occurs before the final comparison.

Explicit comparator mutants that canonicalize signed zero before comparison
and that detach/reconstruct actions post hoc must both fail the provenance
oracle. A direct signed-zero dead-ReLU case must pass; a mutually
target/reference-consistent reference drift from baseline must still set
`reference_exact_action_hash_match=false`, `passed=false`, and
`integrity_passed=false`. CUDA reachability must be exercised when CUDA is
available, with only that execution test skipped when it is not; an always-run
fake/device contract test must independently prove that neither the helper nor
production routes live comparator actions through CPU-only `_tensor`.

Weak-reference evidence must include every raw target, raw reference, and
captured unary-negated reference temporary. All must be dead before the next
graph. Per sign comparator, the sentinel must observe exactly one JVP equality
followed by exactly one VJP equality per named parameter and no extra call.

Topology, name/order, shape, dtype, device, nonfinite factor, helper
authentication, action-call count/order, extra closure, graph-lifetime, schema,
or invalid exact-type failure is structural and fail-fast immediately, before
the next graph is constructed. An authorized finite target `beta_norm` or
boolean-control failure retains the existing record-and-continue behavior for
candidate-free all-seed execution and the existing no-candidate/no-output
scientific prefix. No partial or reordered adjoint object may be published.

The top-level adjoint field order is unchanged. Within each sign control, JSON
serialization preserves the exact eight-key order above. Rebuild and
reversed-order serialization is unchanged. The candidate-free top-level,
environment, binding, integrity, and seed ordering remain unchanged except for
the exact nested sign-control extensions and the manifest authority insertion
below.

## Calibration and prior-domain preservation

The published synthetic calibration result is not rewritten, migrated, or
reinterpreted. Its correct-fixture sign-control objects retain their original
five-key schema and their original `6.25e-5` calibration ceiling. This
amendment governs the prospective production sign controls only. The new
device-agnostic live comparator is a separate addition: `_tensor` and every
existing calibration constructor, trial, control, metric, hash, validator, and
CLI code path retain their existing definition bytes and semantic behavior.
No calibration path calls the new comparator. The published result validator
continues to accept and authenticate the original result bytes under the
original calibration protocol.

Every prior receipt, historical object, artifact object, seed object,
base-preregistration object, calibration protocol, calibration result,
amendment object, and previously frozen scientific field remains
byte-semantically identical. No prior pass or failure is recalculated from the
new nested schema.

## Manifest authority and provenance transition

Add one new manifest authority key named exactly:

```text
normwise_adjoint_sign_control_amendment
```

Its value has exactly these nested keys in this order and no others:

```text
path
sha256
commit
```

They bind this document's committed path, exact SHA-256 bytes, and full
lowercase 40-hex commit. The future receipt-backed manifest top-level key order
is exactly:

```text
schema_version
base_preregistration
amendment
deterministic_pool_amendment
zero_jacobian_classifier_amendment
adjoint_integrity_amendment
normwise_adjoint_calibration_protocol
normwise_adjoint_calibration_result
normwise_adjoint_amendment
normwise_adjoint_sign_control_amendment
binding_receipt
historical
current_scientific_source
artifact_schema
seeds
```

The new authority is inserted immediately after `normwise_adjoint_amendment`
and before `binding_receipt`.

The candidate-free projected `manifest` audit inserts the same authority in
the same relative position. Its complete exact key order is:

```text
path
sha256
base_preregistration
amendment
deterministic_pool_amendment
zero_jacobian_classifier_amendment
adjoint_integrity_amendment
normwise_adjoint_calibration_protocol
normwise_adjoint_calibration_result
normwise_adjoint_amendment
normwise_adjoint_sign_control_amendment
binding_receipt
historical
artifact_schema
source
```

The existing `current_scientific_source.files` mapping retains exactly the
following 31 paths in this order:

```text
scripts/diagnose_pass159_cotangent_stage_a.py
scripts/diagnose_pass200_rsta_stage_a.py
scripts/rsta_normwise_adjoint.py
src/sfora/__init__.py
src/sfora/ablation.py
src/sfora/api.py
src/sfora/arcg.py
src/sfora/benchmark.py
src/sfora/bn_inception.py
src/sfora/catalog.py
src/sfora/cea.py
src/sfora/cem.py
src/sfora/cli.py
src/sfora/compose.py
src/sfora/data.py
src/sfora/encoder_ablation.py
src/sfora/encoder_training.py
src/sfora/evaluation.py
src/sfora/experiments.py
src/sfora/image_benchmark.py
src/sfora/image_end_to_end.py
src/sfora/image_recipes.py
src/sfora/ipsr.py
src/sfora/losses.py
src/sfora/method.py
src/sfora/oapf.py
src/sfora/publication.py
src/sfora/remote.py
src/sfora/report.py
src/sfora/text_baselines.py
src/sfora/training.py
```

The reviewed source revision and hashes change after the reviewed repair; path
membership and order do not. No test path or calibration CLI path enters the
scientific source mapping. Apart from the one exact authority insertion and the
reviewed source revision/hash transition, all prior manifest domains remain
byte-semantically identical.

## Implementation and execution authority

Implementation is authorized only after this committed amendment receives an
independent full review and every finding is repaired in the amendment before
source work. Production work is test-first in exactly these four planned
source/test files:

```text
scripts/rsta_normwise_adjoint.py
scripts/diagnose_pass200_rsta_stage_a.py
tests/test_rsta_normwise_adjoint.py
tests/test_diagnose_pass200_rsta_stage_a.py
```

The implementation must begin with failing tests for signed-zero dead-ReLU
behavior, reference drift despite target/reference consistency, exact target
and reference call count/order, direct-`torch.equal` operand identity and
provenance, unchanged and immediate-negated right operands, CUDA reachability,
the fake/device contract, `_tensor` non-reachability, signed-zero
canonicalization and detach/reconstruction mutants, every nested schema
mutation, weak-reference release of raw/negated actions, one-graph peak,
target-only metrics, and immediate structural fail-fast. Minimal source changes
then make those tests pass. The real manifest is not edited in the source
commit.

Manifest and source-provenance validator changes are also RED before GREEN and
must authenticate the new amendment bytes, Git blob, ancestry, exact insertion
order, unchanged 31-path source membership/order, and reviewed source hashes.
After full local assurance, commit the reviewed source/test implementation with
exact subject `implement RSTA sign-control comparators`.

A fresh independent full-source review follows that commit. Every Critical or
Important finding is repaired test-first, the full assurance gate is repeated,
and focused independent review repeats until clean. Only then may a later
handoff edit the manifest alone, using already-green validators, to bind this
amendment and the final reviewed source revision/hashes.

After the manifest-only refreeze, the first permitted DGX execution is one new
candidate-free all-seed integrity audit. It must contain no candidate field at
any depth and make zero field, scoring, decision, bootstrap, scientific-payload,
or receiver-row serialization calls. A structural or finite integrity failure
keeps RSTA blocked. Even a green candidate-free audit is executability evidence
only. Scientific execution remains forbidden until that audit is independently
authenticated as green and a separate existing authorization process permits
science.
