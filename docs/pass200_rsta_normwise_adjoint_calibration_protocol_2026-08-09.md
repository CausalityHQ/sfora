# Pass 200 RSTA normwise-adjoint calibration protocol — 2026-08-09

## Status and disclosure boundary

This protocol is committed before implementing or running any calibration
fixture and before any further RSTA candidate computation. It is a prospective,
candidate-free calibration of a possible replacement for the scalar-relative
adjoint gate. It does not amend the candidate, authorize a real-data rerun, or
reinterpret the H8 outcome.

H8 is commit `cc7b0a102d938db0cf49e756fc4d18410186bf4d`. Its candidate-free
all-seed audit has SHA-256
`234cc3055b0209bcd095f2932867ff09252ca1c2d4b5e5080a988c77fcd5e74c`.
That audit reported the following exact legacy relative errors:

| Seed | Legacy relative error | Legacy result |
|---:|---:|:---|
| 0 | `0.0002978400759949888` | pass |
| 1 | `0.00031774657235941295` | pass |
| 2 | `0.0003494177665848447` | pass |
| 3 | `0.0010248181825567289` | fail |

For seed 3 only, the inspected candidate-free scalars were
`lhs=38.44344988333859`, `rhs=38.404052336897934`, and
`absolute_error=0.03939754644065374`. No RSTA field, receiver row, candidate
score, aggregate, bootstrap value, scientific decision, or scientific result
was computed or inspected in forming this protocol. The H8 audit remains a
failed integrity audit, and RSTA remains blocked.

## Question and frozen interpretation

For the actual FP32 tensors consumed by the audit, write

```text
u = output direction
v = ordered parameter-direction tuple, flattened only for reductions
a = FP32 JVP action Jv
b = FP32 VJP action J^T u, in exact named-parameter order
L = <u,a>
R = <v,b>
r = L-R
```

The legacy metric

```text
abs(r) / max(abs(L), abs(R), float64(1e-12))
```

is retained and reported. It is a valid scalar forward-relative discrepancy,
but it is cancellation-sensitive because neither scalar magnitude measures the
norm of the two action problems. The prospective primary metric is exactly

```text
beta_norm = 2 * abs(r) / (||u||_2 * ||a||_2 + ||v||_2 * ||b||_2)
```

The factor two makes `beta_norm` agree with the legacy relative discrepancy in
the well-conditioned, equal-scale case `abs(L)=||u||||a||` and
`abs(R)=||v||||b||`. The threshold is frozen now at exactly `5e-4`. It is not
fit to H8 and must not be changed after observing calibration values. The
correct-fixture acceptance ceiling is exactly `threshold/8 = 6.25e-5`.

The related exact max-relative normwise backward error is
`eta_norm=beta_norm/2`. It is the minimum value of

```text
max(||delta_a||_2/||a||_2, ||delta_b||_2/||b||_2)
```

over perturbations satisfying
`<u,a+delta_a>=<v,b+delta_b>`. Both `eta_norm` and `beta_norm` are persisted.

Every product is formed only after both FP32 factors are cast to float64. Each
sum, sum of squares, absolute-product sum, RHS per-parameter sum, and final RHS
stack sum uses `dtype=torch.float64`. Norms are square roots of those float64
sums. The parameter tuple is traversed in exact named-parameter order. Moving
the cast after multiplication, using an FP32 partial reduction, concatenating a
parameter tensor in a different order, or normalizing a direction is forbidden.

The following secondary quantities are retained:

```text
legacy_denominator = max(abs(L), abs(R), float64(1e-12))
legacy_relative_error = abs(r) / legacy_denominator
normwise_denominator = ||u||||a|| + ||v||||b||
eta_norm = abs(r) / normwise_denominator
lhs_absolute_product_sum = sum_i abs(u_i * a_i)
rhs_absolute_product_sum = sum_j abs(v_j * b_j)
lhs_cancellation_factor = lhs_absolute_product_sum / abs(L)
rhs_cancellation_factor = rhs_absolute_product_sum / abs(R)
```

For a cancellation factor, a zero numerator and zero scalar gives `1.0`; a
positive numerator and zero scalar gives JSON string `"infinity"`. No other
nonfinite JSON value is permitted. If `normwise_denominator==0` and
`absolute_error==0`, define `eta_norm=beta_norm=0`. If the denominator is zero
and the error is positive, define both as `"infinity"` and fail. Otherwise all
primary and derived values must be finite JSON numbers.

## Frozen CPU process

Calibration runs locally on CPU only in a fresh process. It sets PyTorch and
NumPy seeds only through the literal PCG64 streams below, sets PyTorch intra-op
and inter-op thread counts to one before tensor work, enables deterministic
algorithms without warn-only, disables autocast, and rejects CUDA tensors. The
operator and every stored action/direction tensor are `torch.float32`; only the
diagnostic products, reductions, and norms are `torch.float64`. NumPy generates
C-order float64 arrays, which are cast once to C-contiguous FP32 CPU tensors.
The result records exact Python, PyTorch, and NumPy version strings.

The calibration executable must be unable to import or call
`exact_contextual_rsta_fields`, `score_rsta_batch`, `decide_stage_a`,
`joint_bootstrap`, `scientific_payload`, any receiver-row serializer, any model
loader, or any checkpoint/data/manifest loader. It accepts only `--output` and
refuses an existing destination.

The reviewed calibration source revision is a full lowercase 40-hex commit.
At process start, record repository HEAD as `executing_git_commit`. Require the
reviewed calibration source revision to be an ancestor of or equal to the
executing commit. The normalized destination must equal repository-root path
`reports/generated/pass200_rsta_receipt/<source>-normwise-adjoint-calibration.json`,
where `<source>` is exactly the 40-hex reviewed source revision parsed from the
basename; there is no separate source-revision argument or inference from file
contents. The worktree bytes and Git blob at the reviewed source
revision must be byte-identical for exactly these paths, in this order:

```text
scripts/rsta_normwise_adjoint.py
scripts/calibrate_pass200_rsta_normwise_adjoint.py
tests/test_rsta_normwise_adjoint.py
```

Independently SHA-256 each byte string and require equality with the result's
source mapping. For all three paths, the Git blob at `executing_git_commit` must
also equal the reviewed-source blob and worktree bytes. The executing CLI path
and digest must equal the second source entry. Dirty, missing, extra, symlinked,
differently hashed, non-ancestor, or non-Git source is a structural failure
before fixture construction.

Publication is exact no-clobber. Let `destination` be the absolute normalized
`--output` path and let the sole temporary path be
`destination.parent / ("." + destination.name + ".tmp." + decimal_pid)`.
`decimal_pid` is the process ID encoded as unsigned base-10 ASCII with no leading
zero.
Reject a missing parent directory, non-regular pre-existing destination, any
pre-existing destination, or any pre-existing temporary path. Open the temporary
path with binary exclusive creation (`xb`), mode `0o600`, and no symlink
following; the process owns it only after that open succeeds. Write the complete
validated UTF-8 JSON bytes plus one LF, flush, and `fsync` the file. Publish with
`os.link(temp, destination)`, which must fail rather than replace if a race
created the destination. `fsync` the parent directory, unlink the owned temp,
and `fsync` the parent directory again. Success requires the destination bytes
to equal the validated buffer and the temp to be absent.

On any error before the hard link, remove only a temp successfully created by
this process and `fsync` the directory; never remove a pre-existing temp or
destination. On any error after the hard link, first verify destination and temp
refer to the same inode, unlink that owned destination link, then unlink the
owned temp and `fsync` the directory. If inode identity cannot be proved, do not
unlink the destination. Report structural failure and preserve evidence. Tests
must cover a pre-existing destination, pre-existing exact temp, destination
creation between the initial check and `os.link`, short write, file-fsync,
hard-link, directory-fsync, and temp-unlink failures. No `rename`, `replace`, or
overwrite fallback is permitted.

## Correct-fixture matrix

All shapes, streams, scales, tensor construction, and fixture order below are
frozen. `N(s,k)` means a fresh `Generator(PCG64(s)).standard_normal(k)` float64
C-order draw, cast once to FP32. A matrix draw uses the listed row-major shape.
Hexadecimal seeds are literal unsigned 64-bit integers.

1. `zero_corner`, input/output dimension `17`. Draw `x`, `u`, and `v` from
   seeds `0x4e4f524d00000001`, `0x4e4f524d00000002`, and
   `0x4e4f524d00000003`. Evaluate `f(x)=x*float32(0)`. The JVP and VJP are
   obtained from `torch.func.jvp` and `torch.func.vjp`; the normwise denominator
   and residual must both be zero, invoking the frozen zero/zero convention.
2. `affine_scale_2m12`, `affine_scale_1`, and `affine_scale_2p12`, input
   dimension `193`, output dimension `257`. Draw row-major `A`, `x`, `u`, and
   `v` from seeds `0x4e4f524d00000101` through
   `0x4e4f524d00000104`. Cast once to FP32 and evaluate
   `f_s(x)=s*(A@x)` for exact binary scales `s` in
   `[2**-12, 1, 2**12]`, in that order. Obtain both actions only through
   `torch.func`.
3. `smooth_parameter_tree`, batch `17`, input dimension `11`, hidden dimension
   `23`, output dimension `19`. Exact named-parameter order is
   `w1,b1,w2,b2`, with shapes `[23,11]`, `[23]`, `[19,23]`, `[19]`. Draw those
   tensors from seeds `0x4e4f524d00000201` through
   `0x4e4f524d00000204`; multiply weights by `2**-3` and biases by `2**-4`.
   Draw the fixed `[17,11]` input from `0x4e4f524d00000205` and multiply by
   `2**-2`. Draw one flat parameter direction from
   `0x4e4f524d00000206`, then partition it in the exact named order. Draw the
   `[17,19]` output direction from `0x4e4f524d00000207`. Evaluate
   `h=tanh(x@w1.T+b1)`, `y=h@w2.T+b2`, and
   `f=normalize(y,dim=1,eps=1e-12)` through a functional parameter mapping.
4. `paired_cancellation`, input/output dimension `8193`. Draw `x` from
   `0x4e4f524d00000301`, `q[4096]` from
   `0x4e4f524d00000302`, and `p[4096]` from
   `0x4e4f524d00000303`. Set each adjacent output-direction pair to
   `[q_k,q_k]`, each adjacent parameter-direction pair to `[p_k,p_k]`, and set
   both final direction elements to exactly `1.0`. Evaluate `f(x)=d*x`, where
   every adjacent diagonal pair is exactly `[2**10,-2**10]` and the final
   diagonal value is exactly `2**-10`. In real arithmetic the large pairwise
   contributions cancel and both adjoint scalars equal `2**-10`; the absolute
   product sums remain large. Actions must still come only from `torch.func`.

Each of these six correct fixtures passes only if `beta_norm <= 6.25e-5` and
all reproducibility controls below pass. No correct fixture may be dropped,
replaced, or rerun under a different seed or scale.

## Rebuild, action-order, and sign controls

For every correct fixture, run these exact trials on fresh graphs:

1. baseline: construct VJP closure, then execute JVP followed by VJP;
2. exact rebuild: reconstruct every input and callable from the frozen PCG64
   streams and repeat baseline;
3. reversed action order: construct a fresh graph, execute VJP before JVP;
4. parameter-sign trial: use `-v` with unchanged `u` and require the JVP action
   to equal `-a` elementwise and the VJP action to equal baseline `b`;
5. output-sign trial: use `-u` with unchanged `v` and require the VJP action to
   equal `-b` elementwise and the JVP action to equal baseline `a`.

Hash C-contiguous bytes of every actual FP32 JVP tensor and the ordered
concatenation of every actual FP32 VJP tensor. Baseline, rebuild, and reversed
order hashes must be byte-identical. For sign trials, `torch.equal` must hold
against the specified baseline or elementwise-negated tensor; their own hashes
are persisted. Every sign-trial `beta_norm` must also be at most `6.25e-5`.
Any nondeterminism, order dependence, nonlinear action, shape/order drift, or
nonfinite quantity fails calibration.

## Registered non-adjoint faults

Faults are evaluated after all correct fixtures. They do not change a random
seed in response to an observed value. Each construction has a real-arithmetic
lower separation fixed before execution.

1. `zero_map_forward_injection` reuses `zero_corner` directions and substitutes
   `a_fault=2**-10*u`, `b_fault=0`. Its mathematical `beta_norm` is exactly
   `2`, independent of the draw.
2. `identity_reverse_scale_fault` uses dimension `4096` and one vector
   `q=N(0x4e4f524d00000401,4096)` for both `u` and `v`. The correct identity-map
   actions are `a=q`, `b=q`; substitute only
   `b_fault=(255/256)*q`. Its mathematical `beta_norm` is exactly `2/511`,
   which is greater than `5e-4`.
3. `identity_reverse_pair_sign_fault` draws
   `q=N(0x4e4f524d00000402,2048)`, expands both directions and the forward action
   as adjacent pairs `[q_k,q_k]`, and substitutes reverse-action pairs
   `[q_k,-q_k]`. Its mathematical `beta_norm` is exactly `1`.

The harness first proves the unmodified zero and identity fixtures meet the
correct-fixture ceiling, then applies the registered substitutions. Every fault
must satisfy `beta_norm >= 5e-4`. It must also be separated from its correct
control by at least `7*threshold/8 = 4.375e-4`. A fault that does not cross the
threshold falsifies the calibration; its amplitude, seed, dimension, or
construction must not be tuned.

## Exact fixture metadata objects

The `fixture_id` and `kind` values are JSON strings. Every `seeds` value is a
JSON object whose values are lowercase strings consisting of `0x` followed by
exactly sixteen hexadecimal digits; they are decoded as unsigned 64-bit
integers. Every `dimensions` value is the literal JSON object below, using only
positive JSON integers and integer arrays. Every `scales` value is the literal
JSON object below; scale values are exact ASCII strings interpreted only by the
fixture-specific constructor, never parsed as decimal floating-point. Object
key order is normative.

```json
{
  "zero_corner": {
    "fixture_id": "zero_corner",
    "kind": "zero_linear",
    "seeds": {"x": "0x4e4f524d00000001", "output_direction": "0x4e4f524d00000002", "parameter_direction": "0x4e4f524d00000003"},
    "dimensions": {"input": 17, "output": 17},
    "scales": {"operator": "0"}
  },
  "affine_scale_2m12": {
    "fixture_id": "affine_scale_2m12",
    "kind": "affine_linear",
    "seeds": {"matrix": "0x4e4f524d00000101", "input": "0x4e4f524d00000102", "output_direction": "0x4e4f524d00000103", "parameter_direction": "0x4e4f524d00000104"},
    "dimensions": {"input": 193, "output": 257},
    "scales": {"operator": "2**-12"}
  },
  "affine_scale_1": {
    "fixture_id": "affine_scale_1",
    "kind": "affine_linear",
    "seeds": {"matrix": "0x4e4f524d00000101", "input": "0x4e4f524d00000102", "output_direction": "0x4e4f524d00000103", "parameter_direction": "0x4e4f524d00000104"},
    "dimensions": {"input": 193, "output": 257},
    "scales": {"operator": "1"}
  },
  "affine_scale_2p12": {
    "fixture_id": "affine_scale_2p12",
    "kind": "affine_linear",
    "seeds": {"matrix": "0x4e4f524d00000101", "input": "0x4e4f524d00000102", "output_direction": "0x4e4f524d00000103", "parameter_direction": "0x4e4f524d00000104"},
    "dimensions": {"input": 193, "output": 257},
    "scales": {"operator": "2**12"}
  },
  "smooth_parameter_tree": {
    "fixture_id": "smooth_parameter_tree",
    "kind": "smooth_parameter_tree",
    "seeds": {"w1": "0x4e4f524d00000201", "b1": "0x4e4f524d00000202", "w2": "0x4e4f524d00000203", "b2": "0x4e4f524d00000204", "input": "0x4e4f524d00000205", "parameter_direction": "0x4e4f524d00000206", "output_direction": "0x4e4f524d00000207"},
    "dimensions": {"batch": 17, "input": 11, "hidden": 23, "output": 19, "parameter_shapes": {"w1": [23, 11], "b1": [23], "w2": [19, 23], "b2": [19]}},
    "scales": {"weight": "2**-3", "bias": "2**-4", "input": "2**-2", "normalization_eps": "1e-12"}
  },
  "paired_cancellation": {
    "fixture_id": "paired_cancellation",
    "kind": "paired_cancellation_linear",
    "seeds": {"input": "0x4e4f524d00000301", "output_pair_base": "0x4e4f524d00000302", "parameter_pair_base": "0x4e4f524d00000303"},
    "dimensions": {"input": 8193, "output": 8193, "pairs": 4096},
    "scales": {"positive_pair": "2**10", "negative_pair": "-2**10", "final": "2**-10"}
  }
}
```

`correct_fixtures` has exactly the six keys above in that order. The fault
metadata objects are exactly:

```json
{
  "zero_map_forward_injection": {
    "fixture_id": "zero_map_forward_injection",
    "kind": "injected_forward_action",
    "seeds": {"x": "0x4e4f524d00000001", "output_direction": "0x4e4f524d00000002", "parameter_direction": "0x4e4f524d00000003"},
    "dimensions": {"input": 17, "output": 17},
    "scales": {"operator": "0", "forward_injection": "2**-10"}
  },
  "identity_reverse_scale_fault": {
    "fixture_id": "identity_reverse_scale_fault",
    "kind": "injected_reverse_scale",
    "seeds": {"shared_direction": "0x4e4f524d00000401"},
    "dimensions": {"input": 4096, "output": 4096},
    "scales": {"operator": "1", "reverse_action": "255/256"}
  },
  "identity_reverse_pair_sign_fault": {
    "fixture_id": "identity_reverse_pair_sign_fault",
    "kind": "injected_reverse_pair_sign",
    "seeds": {"pair_base": "0x4e4f524d00000402"},
    "dimensions": {"input": 4096, "output": 4096, "pairs": 2048},
    "scales": {"operator": "1", "forward_pair": "[1,1]", "reverse_pair": "[1,-1]"}
  }
}
```

`registered_faults` has exactly those three keys in that order. The result
validator requires recursive equality with these literal metadata objects; it
does not accept equivalent numeric encodings, reordered keys, extra metadata,
or constructor-supplied substitutions.

## Result contract and decision

The JSON result has these top-level keys in this order and no others:

```text
schema_version
diagnostic
mode
candidate_values_computed
stage_a_verdict
uses_test_data
protocol
execution_audit
source
environment
correct_fixtures
registered_faults
all_passed
```

Fixed values are `schema_version=1`,
`diagnostic="pass200-rsta-normwise-adjoint-calibration"`,
`mode="cpu_synthetic_calibration"`, `candidate_values_computed=false`,
`stage_a_verdict="NOT_COMPUTED"`, and `uses_test_data="synthetic_only"`.
`protocol` has exactly `path`, `sha256`, and `commit`, binding this document's
committed bytes and Git blob. Its path is exactly
`docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md`, its
digest is lowercase 64-hex, and its commit is full lowercase 40-hex. The commit
must be an ancestor of or equal to `source.git_revision`; the protocol bytes in
that Git commit, the executing worktree, and the recorded digest must agree.
`execution_audit` has exactly these keys in order:

```text
executing_git_commit
calibration_source_commit
calibration_cli_path
calibration_cli_sha256
```

The commits are full lowercase 40-hex strings with the ancestor relationship
defined above. `calibration_cli_path` is exactly
`scripts/calibrate_pass200_rsta_normwise_adjoint.py`; its digest is lowercase
64-hex and equals the executing worktree, executing Git blob, and source entry.
`source` has exactly `git_revision` and `files`. `git_revision` equals
`execution_audit.calibration_source_commit`; `files` has exactly the three
ordered path keys listed in **Frozen CPU process**, each mapped directly to its
lowercase 64-hex SHA-256 string. No path wrapper object or alternate encoding is
permitted.

`environment` has exactly `device`, `torch_threads`,
`torch_interop_threads`, `deterministic_algorithms`, `autocast`, `model_dtype`,
`reduction_dtype`, `python_version`, `torch_version`, and `numpy_version`, with
fixed values `cpu`, `1`, `1`, `true`, `false`, `torch.float32`, and
`torch.float64` before the three nonempty version strings.

`correct_fixtures` has the exact ordered keys listed in the correct-fixture
matrix. `registered_faults` has the exact ordered keys listed in the fault
matrix. Every entry has exactly:

```text
fixture_id
kind
seeds
dimensions
scales
lhs
rhs
absolute_error
legacy_denominator
legacy_relative_error
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
threshold
passed
```

`seeds`, `dimensions`, and `scales` equal the literal objects above recursively.
For faults, hashes and norms bind the substituted action tensors and `controls`
has the single exact key `unmodified`, whose value has exactly
`jvp_sha256`, `vjp_sha256`, `beta_norm`, and `passed`. For correct fixtures,
`controls` has exact keys `rebuild`, `reversed_action_order`, `parameter_sign`,
and `output_sign`. The first two values have exactly `jvp_sha256`,
`vjp_sha256`, `beta_norm`, `exact_action_hash_match`, and `passed`; the sign
values have exactly `jvp_sha256`, `vjp_sha256`, `beta_norm`, `exact_relation`,
and `passed`. `threshold` is always `0.0005`.

`all_passed=true` exactly when every correct fixture and sign trial is at most
`6.25e-5`, every rebuild/order hash is exact, every sign relation is exact,
every registered fault is at least `5e-4` and has the required separation, and
every schema/provenance/finite check passes.

After successful publication, exit code is exactly `0` when `all_passed=true`
and exactly `1` when `all_passed=false`. Structural, provenance, validation, or
publication failure exits exactly `2`; it must not fabricate an `all_passed`
value or overwrite any destination.

If any numerical condition fails, calibration is falsified. Publish and commit
the complete structurally valid candidate-free result with `all_passed=false`,
record its SHA-256 and commit, keep RSTA blocked, and make no threshold, seed,
scale, dimension, fixture, or fault adjustment. This durable negative artifact
is evidence and must not be deleted or replaced. A new theory and new
prospective protocol would be required. A structural/provenance/publication
failure may have no valid result to commit; preserve its external execution log
and stop without a rerun that changes any registered constant.

## Permitted transition after calibration

Only a passing result from an independently reviewed harness may support a new
prospective RSTA amendment. That amendment must bind this protocol and the
complete calibration result by path, SHA-256, and commit; disclose all H8 values
above; retain the legacy scalars; add the normwise scalars and action hashes; and
state explicitly that `beta_norm <= 5e-4` replaces, rather than retroactively
repairs, the prior scalar-relative gate. It must be committed before production
implementation or any new real-data audit.

After implementation, independent review, and a manifest-only refreeze, the
first real execution is candidate-free all-seed integrity only. A structural,
reproducibility, or normwise failure leaves RSTA blocked and forbids scientific
execution. This calibration itself can neither pass Gate 1 nor authorize Stage A.
