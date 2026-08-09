# Pass 200 RSTA deterministic global-max amendment — 2026-08-09

## Status and scope

This amendment is prospective. It is written after an integrity smoke failed,
but before any RSTA candidate statistic, Stage-A output, or decision exists. The
failed smoke created no output file and the scientific Stage-A process was not
started.

The amendment repairs one implementation incompatibility only. It changes no
checkpoint, image, batch, role, support, loss, cotangent, parameter set, field,
control, statistic, threshold, prediction, or decision rule. Deterministic
algorithms remain required without warning-only mode; TF32 remains disabled.

## Observed integrity failure

The TF32-disabled deterministic smoke reached the first real BN-Inception
derivative and failed in CUDA backward:

```text
adaptive_max_pool2d_backward_cuda does not have a deterministic implementation
```

The model uses `AdaptiveMaxPool2d((1, 1))` only to compute a global maximum over
the final spatial axes. Disabling deterministic algorithms, using warning-only,
or ignoring the failure is forbidden.

## Exact compatibility operator

On the diagnostic model clone only, replace the final `gmp` module after strict
checkpoint construction. From the diagnostic root returned by
`_torchvision_model_factory`, the fully qualified submodule path is exactly
`model.gmp`. Require its exact type to be `torch.nn.AdaptiveMaxPool2d` and its
stored `output_size` to equal the integer `1`. The tuple `(1,1)` describes the
semantic output shape only; it is not the stored constructor value in the bound
model. Replace that module with an audited stateless module whose forward is exactly:

```python
x.flatten(-2).max(dim=-1, keepdim=True).values.unsqueeze(-1)
```

For an input `[B,C,H,W]`, this returns `[B,C,1,1]`, takes the same global maximum,
and uses the same first-maximum tie rule as `AdaptiveMaxPool2d((1,1))`. No other
pool or model module is changed. The checkpoint schema and parameter state are
unchanged because both modules are stateless.

Before any receipt-backed smoke may continue, one exact fixture constructs a
NumPy `PCG64(200)` generator, draws a C-order array of shape `[2,3,5,7]` with
`standard_normal`, casts it once to float32, and derives four named inputs:

- `random`: the cast array;
- `relu`: elementwise `maximum(random, 0)`;
- `zeros`: an all-zero float32 array of the same shape;
- `tie`: a copy of `random` with spatial positions `[0,0]` and `[0,1]` in every
  batch/channel set to float32 `100.0`.

For each input, copy identical C-order bytes into CPU and CUDA tensors. Strict
deterministic mode is already enabled as the first torch action and is never
toggled. Both devices are mandatory on DGX.

On CPU, compare reference and replacement outputs and the input derivatives of
`output.sum()`. Require `torch.equal` outputs and gradients and both maximum
absolute differences exactly float `0.0`.

On CUDA, do not call the unavailable reference backward. Call
`torch.nn.functional.adaptive_max_pool2d(input, (1,1), return_indices=True)` and
compare both its output and flattened argmax indices with
`input.flatten(-2).max(dim=-1, keepdim=True)`. Construct the expected input
gradient by scattering float32 `1.0` at those identical flattened indices into
an all-zero tensor. Run only the replacement `output.sum()` backward under the
unchanged strict deterministic setting and require its gradient to equal that
one-hot expected gradient bit-for-bit, with maximum absolute difference `0.0`.
This proves the replacement selects the same derivative as the reference without
executing the forbidden reference CUDA backward. The four exact cases are:

- seeded random values;
- ReLU outputs with repeated zeros;
- all-zero inputs;
- explicit equal positive maxima at two spatial positions.

The model loader must reject any architecture where exact path `model.gmp` is
absent, has another exact type, or has `output_size != 1`; it may not silently
substitute another pool.

The literal replacement identifier is
`pass200-global-max-flatten-first-v1`. Smoke records an integrity object named
`deterministic_global_max` with exactly these keys:

```text
replacement_id
module_path
reference_type
reference_output_size
fixture_seed
fixture_generator
fixture_shape
fixture_dtype
derivative
input_sha256
cases
deterministic_cuda_backward
```

Their exact scalar values are the identifier above, `model.gmp`,
`torch.nn.modules.pooling.AdaptiveMaxPool2d`, integer `1`, integer `200`,
`numpy.PCG64`, `[2,3,5,7]`, `float32`, and `output.sum()`. `input_sha256` maps
the four case names to SHA-256 of their C-order float32 bytes. `cases` has exact
device keys `cpu` and `cuda`. CPU maps each case name to exact objects
`{output_equal:true, gradient_equal:true, max_abs_output_difference:0.0,
max_abs_gradient_difference:0.0}`. CUDA maps each case name to exact objects
`{output_equal:true, index_equal:true,
replacement_gradient_equal_expected:true,
max_abs_output_difference:0.0,
max_abs_replacement_gradient_difference:0.0}`. `deterministic_cuda_backward` is exactly
`{enabled:true, warn_only:false, completed:true}`. Missing CUDA, any missing or
extra key, a false boolean, or a nonzero difference makes smoke `INVALID`.

## Pre-amendment feasibility evidence

The deciding smoke computed no candidate value. Separate tiny operator-only
checks on DGX established CPU bit-exact output/gradient equality before the
strict process issue was identified. Under strict deterministic mode, all four
CUDA cases then had bit-exact outputs, identical flattened argmax indices, and
replacement gradients bit-exact to the one-hot gradients implied by those
indices, with maximum absolute difference `0.0`. The replacement CUDA backward
completed successfully without toggling deterministic mode. The unavailable
reference CUDA backward is not part of the authorized smoke.

This evidence establishes executability only. It is not RSTA Gate-1 evidence and
does not alter any registered threshold.

## Provenance transition

Implementation uses a new reviewed source commit `S2`. The receipt-backed
manifest schema gains one exact `deterministic_pool_amendment` reference with
`path`, `sha256`, and the commit containing this prospective document. A new
manifest-only handoff `H2` binds that reference and every current source file to
`S2`. The historical receipt, historical diagnostic, original preregistration,
binding-receipt amendment, artifact schema, and four seed domains remain byte-
semantically unchanged.

Any implementation or equivalence failure is `INVALID`; no Stage-A value is run.
