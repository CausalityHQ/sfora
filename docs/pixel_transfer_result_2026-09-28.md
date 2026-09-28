# Native FP16 pixel transfer: stop before production

The deployed encoder copies float32 CPU pixels to CUDA and then converts them
to FP16. One combined `to(device='cuda:0', dtype=torch.float16)` call looked
like a cheaper equivalent. A bounded DGX GB10 test rejects that change.

The source froze a conjunction for each batch before launch: candidate transfer
p50≤0.95×original, p95≤original, peak Torch allocation≤original, and identical
FP16 pixel bits. One process,60-second timeout, nonblocking GPU file lock,
20 CPU threads,5 warmups/arm,25 alternating ABBA/BAAB blocks yielded50 timed
transfers/arm/batch. Every timed output was checked outside its timer.

| Synthetic processor-level pixels | Original p50/p95 (ms) | Combined call p50/p95 (ms) | Original→candidate peak Torch allocation (bytes) |
| --- | ---: | ---: | ---: |
| Batch1,3×256×256 |0.030616/0.109805|0.098632/0.858646|1179648→393216|
| Batch32,3×256×256 per image |0.633295/0.673881|1.539430/1.849468|37748736→12582912|

Both latency gates fail despite exact bits and lower allocation. Keep the
two-step production path. No follow-up encoder, full-call timing, tuning or
10,000-call gate for this fixed combined-call arm. These are synchronized
transfer measurements, not image-to-top-k latency, retrieval quality or QPS.
Inputs enumerate all256 uint8-derived normalized processor levels; they are
synthetic tensors, not a SOP/In-Shop dataset or split. They do not qualify
arbitrary float inputs or a different runtime/device.

Root cause established by this experiment: the combined Torch API call is
slower for this pageable CPU→GB10 FP16 transfer configuration. Fewer API calls
and lower GPU allocation do not imply lower latency. Which internal copy/cast
implementation explains that result was not profiled; do not invent that cause.

Process exited0 in1.75s GNU wall,918588KiB maximum host RSS, Torch2.12.1+cu130.
Source SHA4beb4c31e4e3887e7a80be0c95229f84946bcc54a00db202cf0b7b469b2aa23f.
[Raw source, samples and terminal log](evidence/compact_metric/sop-siglip2-substrate-v1/pixel-transfer-v1/).
No model, gallery, held/official result or production source changed. The joint
quality-and-speed goal remains active and unmet.
