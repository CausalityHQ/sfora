# UniCOM official readout structural erratum

Date: 2026-08-22 UTC

The first official-readout process used source
`4223251ab4aa2af47546b6e31fe3bfeefe764b7a` and exited 2 before the first
evaluation row with:

`official evaluation failed: evaluation model must be BatchNorm-free`

The process produced no registered result, atomic temporary, row progress, retrieval
metric, or partial aggregate. The registered output
`reports/generated/unicom_ema_imprint_official_4223251.json` remained absent.

The failure is a structural source-contract defect. The authenticated UniCOM source at
revision `d71992ed969e6c271436ac0a0ee1f3ca61474ac0` defines the projection head with
`feature.1 = BatchNorm1d(1024, eps=2e-5)` and
`feature.3 = BatchNorm1d(768, eps=2e-5)`. Both modules are part of the official model
and their checkpointed running statistics are required for eval-mode inference. Their
`num_batches_tracked` buffers are scalar `int64`; all parameters and floating-point
buffers are FP32 after the already-registered conversion.

A metric-free DGX structural probe instantiated that pinned model and independently
confirmed exactly those two module paths/types, `track_running_stats=True`,
`affine=True`, `training=False`, `eps=2e-5`, finite FP32 affine/running-state tensors, and equal
nonnegative int64 counters. A retained raw checkpoint contained 281 model tensors:
279 FP32 tensors, two int64 tensors, and zero nonfinite floating tensors. The corrected
validator then accepted the real pinned model. The probe loaded no official images and
computed no embedding or retrieval metric. The root cause was an architectural
assertion written without first instantiating the authenticated model, despite the
repository's existing BatchNorm-aware checkpoint-soup evaluator.

For the single replacement attempt, the model validation contract is prospectively
corrected as follows:

- load the exact raw `checkpoint["model"]` with `strict=True`, then call `eval()`;
- require the only BatchNorm modules to be the two exact `BatchNorm1d` modules above,
  with affine parameters, tracked running statistics, and `eps=2e-5`;
- require both modules to remain in eval mode, require finite FP32 affine/running-state
  tensors, nonnegative running variance, and nonnegative scalar `int64`
  `num_batches_tracked` buffers;
- require every model parameter and every floating-point model buffer to be FP32 and
  finite; and
- perform no training-data recalibration, mutation, or batch-statistics inference.

No dataset, checkpoint, seed, ordering, batch size, transform, embedding geometry,
metric, bootstrap, threshold, decision rule, or scientific branch changes. The source
fix must be test-first, independently reviewed, and committed before a replacement run
configuration is frozen. That configuration must use `attempt=2`, preserve prior exit
status `[2]`, bind the new reviewed source commit, and use a fresh source-addressed
output path. A second structural failure closes the gate without another run.

Only a failure after the evaluator has authenticated and opened the registered
scientific inputs consumes an attempt number. A source/config/checkout preflight abort
before that boundary is corrected without advancing the attempt, is disclosed in the
operator report, and still exposes no scientific values. The first failure occurred
after all 48 checkpoint hashes and the official partition had been opened, so it is
attempt 1 and the replacement is attempt 2.
