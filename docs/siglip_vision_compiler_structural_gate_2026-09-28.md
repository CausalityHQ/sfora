# Fixed-shape vision compiler structural gate

Frozen before running the diagnostic. Candidate: native Torch compiler fusion
of the existing FP16 SiglipVisionModel batch-32 encoder. Existing batch-32 CUDA
graph replay is closed; this tests a different compiler route. No production
option, model/loss change, checkpoint selection, or quality read is authorized
by this structural result alone.

First falsifier: one CPU-only strict `torch.export` of the canonical config,
meta weights and `(32,3,256,256)` FP16 meta input, with a 120-second external
timeout. Require `(32,1024)` FP16 meta pooled output and no export exception.
No weights, pixels, CUDA, dependencies, or real embedding outputs are needed.
Export failure closes this route pending a specific demonstrated cause; do not
work around it by silently changing model or attention semantics.

A structural pass licenses a separately frozen GPU compiler pilot only. That
pilot must use the existing pinned SOP TRAIN checkpoint/gallery, unchanged
head/packing/native scorer, exact packed codes/norms and top-10, and unchanged
preprocessing. Minimum useful paired full image-to-top-k p50 AND p95 reduction:
5%. Compile plus parity smoke must finish within 120 seconds; reject on timeout,
output mismatch, or gross latency deficit. Predeclare the timing sample and
resource cap before launching. A promising short pilot can license one frozen
tail gate; no 10,000-call run otherwise. Meta export cannot measure speed or
predict GB10 compiler support. The full quality-and-speed target remains unmet.

## Original terminal

Strict export and exported-module meta execution passed: 5.644441 seconds,
Torch 2.12.1+cu130 / Transformers 5.12.1, SDPA attention, 32×1024 FP16 meta
output. Original native session 39474 exited 0. The framework warned about
Transformers output-capture side effects; this did not fail strict export but
actual compilation still needs checking. Text-config BOS/EOS warnings concern
the unused text tower. No GPU, real-output parity, quality or latency measured.
[Raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/siglip-vision-meta-export-v1/receipt.json).

## Frozen GPU pilot (before launch)

One unit `sfora-sop-compiler-smoke-v1` under the shared GPU lock. Use
`probe_sop_siglip2_compiler.py`: pinned existing seed179024 checkpoint and
59,551-row SOP TRAIN gallery, first32 archive rows (not promised unique images),
current public source SHA ef454200ca17b80917fe7aa52cf232703779d21af7e8ffe0d25f8e811724bc62.
Native `torch.compile`, default mode, fullgraph/static, no math substitutions.
One index switches only its vision module. Compilation and first packed/top10
parity together capped120sec; CUDA allocation cap16,000,000,000 bytes; external
unit cap300sec/hostMemoryMax24G. Kill on compiler error, timeout, mismatch or cap.
If exact, two warmups per arm then10 alternating ABBA/BAAB blocks, two calls
per entry:40 decoded-image-to-top10 calls per arm at batch32. Fixed selected
TRAIN images, warm caches,20CPUthreads, TF32off, same scorer/head/gallery.
Require compiled p50 ANDp95 <=95% eager. These40 calls are a kill/promising
screen, not p99 certification or general quality evidence. No p99 claim, new
quality read, checkpoint training, public API option or package build.
