# Pass 200 RSTA zero-Jacobian classifier amendment — 2026-08-09

## Status and scope

This amendment is prospective. It is written after the third integrity smoke
failed, but before any RSTA candidate statistic, Stage-A output, or decision
exists. Every failed smoke created no output file; scientific Stage A has never
started.

It repairs one exact parameter-topology mismatch. It changes no image, checkpoint
tensor, forward value, batch, role, support, loss, cotangent, nonzero Jacobian
column, field, control, statistic, threshold, prediction, or decision.

## Observed integrity failure

After the deterministic global-max replacement passed, the first full-batch
field construction rejected exactly two trainable parameters as missing from the
embedding graph:

```text
model.last_linear.weight
model.last_linear.bias
```

The bound BN-Inception source defines this legacy ImageNet classifier as
`last_linear = Linear(1024,1000)`, but its embedding `forward` uses
`features -> gap + gmp -> embedding -> l2_norm` and never calls `last_linear`.
The smoke found gradients for every other registered trainable encoder parameter.

Treating arbitrary missing gradients as zero or enabling broad `allow_unused`
would weaken the integrity gate and is forbidden.

## Exact zero-Jacobian exclusion

On the strict diagnostic clone only, after checkpoint loading and deterministic
global-max replacement, require exactly these named parameter contracts:

```text
model.last_linear.weight: shape [1000,1024], dtype float32, requires_grad true
model.last_linear.bias:   shape [1000],      dtype float32, requires_grad true
```

No other missing or unused parameter is accepted. Set `requires_grad_(False)` on
exactly those two tensors before constructing the ordered diagnostic parameter
tuple. Their values and checkpoint state remain unchanged.

This is an exact zero-column removal: if `z` does not depend on parameter `q`,
then `partial z / partial q = 0`; deleting that zero Jacobian column cannot change
`J J^T dbar`, any receiver self field, or any registered statistic.

## Mandatory runtime audit

Both smoke and scientific processes run the same audit before any RSTA field or
score. On their first registered B=180 tensor batch, before freezing the two
parameters:

1. compute the model output once under the strict deterministic process;
2. request gradients of `output.sum()` with respect to only the two exact
   `last_linear` tensors using `allow_unused=True`, and require both results are
   exactly `None`;
3. clone both tensors, fill weight with float32 `0.125` and bias with float32
   `-0.25`, recompute the same full-batch output, restore original tensor bytes in
   a `finally` block, and require `torch.equal` with the original output;
4. require SHA-256 of both restored tensors equals the pre-mutation SHA-256;
5. freeze exactly those two tensors and require the final ordered trainable
   parameter names exclude them and every remaining parameter produces a non-None
   gradient in the existing full-field gate.

The literal audit identifier is
`pass200-zero-jacobian-last-linear-v1`. Persist an integrity object named
`zero_jacobian_classifier` with exactly:

```text
audit_id
parameter_names
parameter_shapes
parameter_dtypes
pre_sha256
restored_sha256
gradients_none
mutated_output_equal
frozen_requires_grad
```

Exact values are the identifier above; the two names in weight,bias order; shapes
`[[1000,1024],[1000]]`; dtypes `['torch.float32','torch.float32']`; SHA mappings
keyed by the two names; `gradients_none=[true,true]`;
`mutated_output_equal=true`; and `frozen_requires_grad=[false,false]`. Missing or
extra fields, another name/shape/dtype, a non-None dependency, changed output,
failed restoration, or another missing gradient makes the run `INVALID` before
candidate scoring.

The mutation audit is stateless with respect to scientific computation: original
bytes are restored and verified before the tensors are frozen. No altered output
is used by a field, control, or statistic.

Smoke persists `integrity.zero_jacobian_classifier` as the exact audit object for
seed 0. Scientific execution repeats the audit independently for every seed before
that seed's first derivative field and persists
`integrity.zero_jacobian_classifier` as an exact object with string keys
`'0','1','2','3'`, each mapping to that seed's exact audit object. There is no
cross-seed reuse or seed-selective acceptance.

## Provenance transition

Implementation uses a new reviewed source commit `S4`. The receipt manifest gains
one exact `zero_jacobian_classifier_amendment` reference containing `path`,
`sha256`, and this document's commit. A new manifest-only handoff `H4` binds that
reference and every current source file to `S4`. All prior preregistration,
amendment, receipt, artifact, and seed domains remain byte-semantically unchanged.

Any audit or implementation failure remains `INVALID`; no Stage-A value is run.
