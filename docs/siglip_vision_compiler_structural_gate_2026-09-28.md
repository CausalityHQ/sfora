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
