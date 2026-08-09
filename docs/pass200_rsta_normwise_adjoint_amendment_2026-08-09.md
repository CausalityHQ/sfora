# Pass 200 RSTA normwise-adjoint-integrity amendment — 2026-08-09

## Status and evidentiary boundary

This is a prospective integrity amendment committed after the registered
synthetic CPU calibration passed and before any new RSTA real-data audit or
candidate computation. It changes the numerical conditioning and
reproducibility evidence used by the full-model adjoint integrity gate. It does
not change an RSTA direction, checkpoint, image, batch, model operation, JVP,
VJP, field, receiver, score, control, aggregate, bootstrap, prediction, or
decision rule. It neither adjudicates Gate 1 nor authorizes scientific
execution by itself.

The candidate-free H8 audit remains failed under its registered legacy gate.
This amendment does not recompute, reinterpret, erase, or repair H8. A later
audit under this amendment is new prospective evidence.

## Complete H8 disclosure

H8 is commit `cc7b0a102d938db0cf49e756fc4d18410186bf4d`. Its
manifest bytes have SHA-256
`65eb69939dfca56ab96308472dbece804574910f410274bac56112472994e40d`
and bind reviewed source `ea7100110816b1eafb5f309d85272019b3b78ada`.
The bound diagnostic source SHA-256 is
`45fd0659f59a165701e9819f204092d04fdcd9a51ad3afd75f18f108837d14f1`.
Its
candidate-free all-seed artifact has SHA-256
`234cc3055b0209bcd095f2932867ff09252ca1c2d4b5e5080a988c77fcd5e74c`.
The exact legacy adjoint relative errors were:

| Seed | Legacy relative error | Legacy result |
|---:|---:|:---|
| 0 | `0.0002978400759949888` | pass |
| 1 | `0.00031774657235941295` | pass |
| 2 | `0.0003494177665848447` | pass |
| 3 | `0.0010248181825567289` | fail |

For seed 3, the inspected candidate-free values were exactly
`lhs=38.44344988333859`, `rhs=38.404052336897934`, and
`absolute_error=0.03939754644065374`. H8 therefore had
`all_passed=false`. It computed no RSTA field, receiver row, candidate score,
aggregate, bootstrap value, scientific verdict, or scientific result. No such
candidate value was inspected while designing, calibrating, or authoring this
amendment.

The H8 scalars cannot be converted to the new normwise metric because H8 did not
persist the required JVP/VJP action norms. Seeds 0--2 passing the old gate and
seed 3 failing it do not predict the outcome of a new audit.

## Pre-calibration structural and publication record

The calibration protocol was committed before fixture execution at
`171a3fe24386dbab4eb361c04cbf252da4f4e0bb`; its bytes have SHA-256
`2f4d52fd6c69588248f1b27acbcd5503b0e53dc3c5bd6b5e0755564017dc21db`.
The protocol path is
`docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md`.

One pre-fixture invocation used reviewed calibration source
`0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2` but passed the then-plan's
relative output path. It exited exactly `2` with `output must be absolute`.
Path authentication rejected it before CPU configuration, fixture construction
or execution, temporary-file creation, artifact creation, or production of any
calibration value. It did not consume the registered calibration run.

Before the valid run, plan-only review fixes required the absolute normalized
repository-root output path, verified or created only the real non-symlink
`reports/generated/pass200_rsta_receipt` leaf beneath an existing real
`reports/generated`, and bound the interpreter directly to the physical
non-symlink environment `/home/rb/worktrees/sfora-emafactorial/.venv`. No
repository `.venv` symlink was used. Those review corrections exposed no fixture
or candidate value and changed no calibration source or protocol constant.

After the valid PASS artifact was published, ordinary `git add` rejected its
ignored `reports/generated` path. That staging attempt changed neither the
index, artifact bytes, path, filesystem mode, validation state, nor provenance;
it made no `.gitignore` change and no commit. A later plan-only authorization
permitted `git add -f` for the one literal artifact path only after repeating
the exact path, SHA-256, filesystem mode `0600`, VALID PASS, source/execution,
and no-temp checks. The result commit has the artifact as its sole added path at
Git mode `100644`; the worktree and Git-blob bytes have the same registered
SHA-256.

## Calibration authority and validation

The complete calibration artifact is
`reports/generated/pass200_rsta_receipt/0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2-normwise-adjoint-calibration.json`,
SHA-256 `5fcb09a1e3a6eedddd05ef49bd22bc9920656089aa401a5aae2c5704a9d9dc50`,
committed as the sole change in
`95525af61d66b063983dc55a6015168d9aafd12b`. Its parent is plan-fix commit
`f739652c0d322c65545ea0c8a41a735d9dec5c02`.

The artifact binds:

- reviewed calibration source
  `0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2`;
- executing plan-fix commit
  `6e2bf99b443fd24320f9f1ea461266cc1bd3871d`;
- helper `scripts/rsta_normwise_adjoint.py`, SHA-256
  `1da6ac8bb205ac66e5fb998ffe382b4c5c04548a31e6857a6263f9d0e02b98bf`;
- CLI `scripts/calibrate_pass200_rsta_normwise_adjoint.py`, SHA-256
  `076811ed0bdb71eebe976c088b54fc38b63ef0e4847e293b57f358937cbeacb9`;
- test `tests/test_rsta_normwise_adjoint.py`, SHA-256
  `689428e63546ec5980e9c975f6aa11fa0bbc33a38929567526cc3c15de79a93c`;
- Python `3.12.3`, PyTorch `2.12.1+cu130`, NumPy `2.5.0`, CPU, one intra-op
  and one inter-op thread, deterministic algorithms, autocast off, FP32
  operators, and FP64 diagnostic reductions.

The executing runtime had literal `sys.prefix` and physical environment path
`/home/rb/worktrees/sfora-emafactorial/.venv`; its `pyvenv.cfg` SHA-256 was
`37c4098131a586cddac2548c8c4b64d47c51614454367bee9f83fdd0f9b3a8fb`.
The resolved executable was real executable `/usr/bin/python3.12`, SHA-256
`a7d56a8a764faf7bbf5c164055a48fd072be52287bdeb523a9e07b2042f4e7e1`;
the imported PyTorch and NumPy module paths were beneath the pinned
`lib/python3.12/site-packages` directory.

The exact top-level values are
`diagnostic="pass200-rsta-normwise-adjoint-calibration"`,
`mode="cpu_synthetic_calibration"`, `candidate_values_computed=false`,
`stage_a_verdict="NOT_COMPUTED"`, `uses_test_data="synthetic_only"`, and
`all_passed=true`.

Independent validation reauthenticated the protocol Git blob/worktree bytes,
source and executing ancestry, all three source Git blobs/worktree hashes, exact
recursive schemas and metadata, every derived scalar and action hash, all six
correct fixtures, all rebuild/order/sign controls, and all three registered
faults. It then repeated VALID PASS, no-temp, literal SHA, byte equality, sole
commit scope, and Git mode checks after result commit.

The principal observed calibration values were:

| Fixture | `lhs` | `rhs` | Legacy relative error | `beta_norm` | Result |
|:---|---:|---:|---:|---:|:---|
| `zero_corner` | `0.0` | `0.0` | `0.0` | `0.0` | pass |
| `affine_scale_2m12` | `-0.028861635721179862` | `-0.02886160526283066` | `1.0553230419143364e-6` | `3.259503718370092e-8` | pass |
| `affine_scale_1` | `-118.21725991395272` | `-118.21713515655438` | `1.0553230419143364e-6` | `3.259503718370092e-8` | pass |
| `affine_scale_2p12` | `-484217.8966075503` | `-484217.38560124673` | `1.0553230419143364e-6` | `3.259503718370092e-8` | pass |
| `smooth_parameter_tree` | `7.020457748092397` | `7.020440029784281` | `2.5238109468735353e-6` | `1.412469446769033e-8` | pass |
| `paired_cancellation` | `0.0009765625` | `0.0009765625` | `0.0` | `0.0` | pass |
| `zero_map_forward_injection` | `0.03053091811446108` | `0.0` | `1.0` | `2.0` | detected |
| `identity_reverse_scale_fault` | `4072.6412713216096` | `4056.732512537118` | `0.003906250937571276` | `0.0039138952660976506` | detected |
| `identity_reverse_pair_sign_fault` | `4024.956095153726` | `0.0` | `1.0` | `0.9999999999999999` | detected |

All correct fixtures were below the preregistered calibration ceiling
`6.25e-5`. Every registered fault was at or above `5e-4` with at least the
registered separation. Baseline, exact rebuild, and reversed-order action
hashes matched exactly for every correct fixture; both sign relations were
exact and their normwise metrics passed. The paired-cancellation fixture's
left and right cancellation factors were each exactly
`5470267643.793289`, demonstrating that a well-formed adjoint may be extremely
cancellation-sensitive under the legacy scalar denominator.

The artifact's complete literal metadata, norms, absolute-product sums,
cancellation factors, action hashes, control hashes, and booleans are
incorporated here by path, SHA-256, and commit. No value may be omitted,
substituted, or selectively used when authenticating the calibration.

## Frozen metric and arithmetic

For the actual FP32 tensors used by the audit, let `u` be the output direction,
`v` the exact named-parameter direction tuple, `a=Jv` the FP32 JVP action, and
`b=J^T u` the FP32 VJP action tuple. Retain the legacy values

```text
lhs = <u,a>
rhs = <v,b>
absolute_error = abs(lhs-rhs)
denominator = max(abs(lhs), abs(rhs), float64(1e-12))
relative_error = absolute_error / denominator
tolerance = 0.0005
passed = relative_error <= tolerance
```

unchanged, including the legacy meaning of `passed`. Add:

```text
normwise_denominator = ||u||_2 * ||a||_2 + ||v||_2 * ||b||_2
eta_norm = absolute_error / normwise_denominator
beta_norm = 2 * absolute_error / normwise_denominator
normwise_tolerance = 0.0005
normwise_passed = beta_norm <= normwise_tolerance
```

The factor two makes `beta_norm` coincide with the legacy relative error in the
well-conditioned equal-scale case. The threshold `5e-4` was fixed before any
fixture value and is not fitted to H8 or the calibration result.

Cast every FP32 factor to float64 before multiplication. Compute `lhs`, every
named RHS term, each sum of squares, norm, absolute-product sum, and final RHS
stack sum in float64. Traverse parameter directions and VJP actions in exact
trainable named-parameter order. The norms bind the actual FP32 action and
direction tensors; they do not promote the model, primal, JVP, or VJP.

For `normwise_denominator==0` and `absolute_error==0`, define
`eta_norm=beta_norm=0`. For zero denominator and positive error, encode both as
JSON string `"infinity"` and fail. Cancellation factors use `1.0` for zero
absolute-product sum and zero scalar, and `"infinity"` for positive sum and
zero scalar. Every other scalar must be a finite JSON number.

## Exact production adjoint object

The production adjoint object retains the original seventeen fields first,
with their original values and semantics, and appends the following fields. Its
exact key order is:

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
output_direction_l2
parameter_direction_l2
jvp_l2
vjp_l2
normwise_denominator
eta_norm
beta_norm
lhs_absolute_product_sum
rhs_absolute_product_sum
lhs_cancellation_factor
rhs_cancellation_factor
jvp_sha256
vjp_sha256
controls
normwise_tolerance
normwise_passed
integrity_passed
```

The first seventeen fields remain exactly as frozen by the adjoint-integrity
amendment. In particular, `direction_domain="rsta-stage-a-v1"`; the direction
seeds, hashes, shape, parameter-name-order hash, count, and dtypes retain their
existing derivations; `tolerance=0.0005`; and `passed` remains the legacy
scalar-relative boolean. A legacy `passed=false` is persisted and must not be
rewritten even when the prospective normwise gate passes.

`output_direction_l2`, `parameter_direction_l2`, `jvp_l2`, and `vjp_l2` are
finite nonnegative JSON numbers from the exact FP64 norm reductions.
`lhs_absolute_product_sum` and `rhs_absolute_product_sum` are finite
nonnegative JSON numbers. The normwise and cancellation fields follow the
corner encodings above. `jvp_sha256` hashes the C-contiguous bytes of the actual
baseline FP32 JVP tensor. `vjp_sha256` hashes the ordered concatenation of the
C-contiguous bytes of the actual baseline FP32 VJP parameter tensors in exact
named-parameter order. Both are lowercase 64-hex strings.

`controls` has exactly these keys in this order:

```text
rebuild
reversed_action_order
parameter_sign
output_sign
```

`rebuild` and `reversed_action_order` each have exactly:

```text
jvp_sha256
vjp_sha256
beta_norm
exact_action_hash_match
passed
```

Their hash fields use the same actual-action encodings as the baseline.
`exact_action_hash_match` is true exactly when both hashes equal their baseline
hashes. Their `passed` is true exactly when the hash match is true and their
`beta_norm <= 0.0005`.

`parameter_sign` and `output_sign` each have exactly:

```text
jvp_sha256
vjp_sha256
beta_norm
exact_relation
passed
```

For `parameter_sign`, rerun on a fresh graph with `-v` and unchanged `u`;
require the JVP action to equal elementwise `-a` and the VJP action to equal
baseline `b` under `torch.equal`. For `output_sign`, rerun with `-u` and
unchanged `v`; require the JVP action to equal baseline `a` and the VJP action to
equal elementwise `-b`. `exact_relation` is that exact tensor relation. Sign
control `passed` is true exactly when the relation is exact and
`beta_norm <= 0.0005`.

`normwise_tolerance` is exactly `0.0005` and `normwise_passed` is exactly the
inclusive comparison `beta_norm <= normwise_tolerance`. `integrity_passed` is
true exactly when `normwise_passed` and all four control `passed` values are
true. The legacy `passed` value is evidence but is not an input to prospective
`integrity_passed`.

## Frozen graph schedule and directions

For every real seed, use the unchanged output and parameter directions already
registered by the prior amendment: separate fresh PCG64 streams derived from
`domain_seed("rsta-stage-a-v1|adjoint-u|", str(seed))` and
`domain_seed("rsta-stage-a-v1|adjoint-v|", str(seed))`, float64 C-order draws
cast once to the actual FP32 tensors, and exact trainable named-parameter order.
Seed order remains `0,1,2,3`; batch size remains `180`; model, primal, JVP, and
VJP remain FP32; TF32 and autocast remain off. No resampling, normalization,
model promotion, tolerance change, or alternate direction is allowed.

For each seed, execute sequentially and release each graph before the next:

1. baseline: construct a fresh VJP closure, then JVP followed by VJP;
2. rebuild: reconstruct the functional graph from the same immutable model,
   inputs, and exact direction tensors, then repeat baseline;
3. reversed order: on another fresh graph, execute VJP before JVP;
4. parameter sign: on a fresh graph, use `-v` and unchanged `u`;
5. output sign: on a fresh graph, use `-u` and unchanged `v`.

No control redraws a direction or changes the primal. Only detached JSON
scalars, booleans, and hashes survive a graph. This preserves one full B=180
derivative graph at peak.

## Candidate-free and scientific gates

The existing candidate-free `--integrity-all-seeds-only` top-level schema,
fixed values, provenance, bindings, fixtures, zero-Jacobian evidence, and
candidate-forbidden call set remain unchanged. Each seed's `adjoint` value is
the exact extended object above. Its global `all_passed` is true exactly when
every seed `0,1,2,3` has `integrity_passed=true` and every preceding registered
integrity audit passed. It deliberately does not use the legacy adjoint
`passed` boolean.

A finite normwise or control failure records the complete seed object,
continues candidate-free execution through the remaining seeds, and produces
`all_passed=false`. Nonfinite data, provenance, topology, shape, dtype,
serialization, or publication failure remains structural and fail-fast. The
candidate-free mode still never calls `exact_contextual_rsta_fields`,
`score_rsta_batch`, `decide_stage_a`, `joint_bootstrap`, `scientific_payload`,
or a receiver-row serializer and contains no candidate field at any depth.

Scientific execution retains the strict all-seed integrity prefix. It may not
initialize scoring or candidate-row state until all four extended objects and
all other registered integrity gates pass. Any seed's
`integrity_passed=false`, including a later seed, exits `INVALID`, makes zero
score, decision, bootstrap, or scientific-payload calls, and leaves no
destination or sibling temporary file. Only after the entire prefix passes may
the unchanged scoring phase reconstruct fresh graphs and compute the unchanged
candidate.

## Provenance transition and implementation authority

The receipt-backed manifest must add exact objects with only `path`, `sha256`,
and `commit` for:

```text
normwise_adjoint_calibration_protocol
normwise_adjoint_calibration_result
normwise_adjoint_amendment
```

The protocol and result objects bind the exact authorities above. The amendment
object binds this document's eventual committed bytes. The existing
`adjoint_integrity_amendment` remains present and authenticated because its
legacy object and all-seed prefix remain historical authority. The future
reviewed scientific-source mapping must include the shared normwise helper and
every modified validator/source file.

Implementation is authorized only test-first under the committed plan. Every
manifest/source-validator RED and GREEN must land before the reviewed source
commit. After independent source review, a subsequent handoff changes only the
manifest and uses already-green validators. Only then may one new DGX
candidate-free all-seed audit run. A failed audit leaves RSTA blocked and
forbids scientific execution. A green candidate-free audit is executability
evidence only; it computes no candidate value and cannot retroactively alter H8.
