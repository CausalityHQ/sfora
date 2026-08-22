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
