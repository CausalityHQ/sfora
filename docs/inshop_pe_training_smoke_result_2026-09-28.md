# PE/Large training mechanics: STOP before optimizer

The sole DGX job `sfora-pe-training-smoke-v4`, invocation
`9b3fd272614043b9b329e1ae71308f72`, exited1 at the Large control's initial
precision/cache guard. No optimizer update ran; PE never ran. This closes the
fixed BF16/cache mechanics configuration. It is no learned-quality result and
does not establish that PE cannot train.

Frozen implementation `e3b9cfd3`; [gate](inshop_pe_training_smoke_gate_2026-09-28.md).
Official In-Shop **TRAIN-fit fixture** only:1024 images/512 products initialize
each arm, scheduled512 distinct images/256 products/1024 presentations, seed179032.
No TRAIN-held or official query/gallery score was read.

| Evidence | Verified measurement | Decision |
|---|---|---|
| Final CPU preflight |8.89s whole process;2,438,676KiB peak RSS; exit0|Original FP32 models, pixels, objective VJPs, inventories, duplicates and startup authorities pass|
| Sole GPU job |14.27s whole process;3,041,396KiB peak RSS; exit1|Large first-four cosine guard fails before optimizer|
| CPU alignment falsifier |6.36s;2,549,004KiB peak RSS; exit0|Four source rows/pixels have no gross alignment deficit|

The GPU assertion combines fresh BF16-versus-FP32 and cached FP16-versus-fresh
BF16 cosine≥0.999 on rows0,1,2,5. The code did not persist either cosine array
before asserting. **The failing branch and values are unknown.** No CUDA peak,
training images/s, step median, PE/control cost ratio, checkpoint or final route
integrity measurement exists. Do not reconstruct or project them. The180s/8GiB
host limits were enforced; neither timeout nor OOM caused this exit. Shared GPU
lock was inside the service; after exit the GPU is idle.

One bounded CPU-only falsifier used the same original FP32 Large checkpoint,
processor and source rows. Single-image versus grouped pixels were byte-exact;
cached GPU FP16 versus fresh CPU FP32 cosines were
`[0.9999925494,0.9999858141,0.9999765158,0.9999227524]`.
This excludes a gross row/processor/cache mismatch for those rows. A numerical
profile mismatch is the leading **inference**; the diagnostic cannot identify
the GPU branch or reproduce GPU BF16 behavior. No GPU rerun, precision change,
threshold relaxation or cap increase occurred.

The failed configuration remains closed. Next: qualify the exact numerical
profile before any new image-training proposal; first record each cosine branch
and exact processing/checkpoint authority, then use a separately declared
precision mechanism with an appropriate packed-output/quality validation gate.
No automatic full-fit cache, optimizer, held score or official evaluation follows.

The repository's probe now logs both cosine arrays **before** its guard for
future diagnostics. The original DGX executable/manifest remain untouched and
are reproduced by checking out `e3b9cfd3`; the logging change was not executed on
GPU and does not backfill this result. Both reviews were completed and mandatory
startup/provenance/workspace fixes applied before the job. Optional known limits:
key-bias gradient can be mathematically null; guarded timing includes per-tensor
finite-check synchronizations; host RSS is cumulative. These yield no claims here.

Raw launch/terminal/preflight/CPU alignment/review evidence:
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/pe-training-smoke-v1/`.
CPU alignment invocation `5c19141b3a9c41c9a1c98a7ef0aada9f`.
The production joint quality-and-speed goal remains unmet and active.
