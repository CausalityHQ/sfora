# Native FP16 image-training mechanics: PASS

PE and Large completed the same16x64 native-image updates, with original FP32
weights, FP16 autocast and GradScaler128. Both unchanged0.999 precision floors,
all finite/integrity/update/scaler/resource checks passed. This qualifies the
training route and a cost guard; **no retrieval quality or serving win was measured**.

Frozen implementation `41e2f63b`; [gate](inshop_pe_fp16_training_gate_2026-09-28.md).
DGX Spark unit `sfora-pe-fp16-smoke-v1`, invocation
`061a25094f0647469c4059b864a8940c`, exit0. Official In-Shop **TRAIN-fit fixture**:
1024 source images/512 products initialize each own head/proxies/bank; scheduled
512 distinct images/256 products/1024 presentations, seed179032. No augmentation,
TRAIN-held, official query/gallery, SOP, CUB or Cars evaluation in this job.

| Measured cost/precision | SigLIP2 Large/256 control | PE-Core-B16/224 |
|---|---:|---:|
| Guarded total-step median3–16 |793.6618ms|385.8370ms|
| All16 guarded-step throughput |77.7214 images/s|164.4939 images/s|
| Peak allocated CUDA, including initialization |11,453,047,296B|3,643,665,408B|
| Minimum fresh FP16 versus FP32 cosine |0.9999095201|0.9999739528|
| Minimum cached FP16 versus fresh FP16 cosine |0.9999963045|0.9999967217|
| Completed updates; scale on every update |16;128|16;128|
| Allocated CUDA after arm cleanup |0B|0B|

Median-step PE/control ratio **0.4861478846** passes the predeclared≤0.8 guard.
This is a single serial guarded mechanics comparison: finite checks and scaler
operations are included, first two warmup steps excluded from the median, Large
then PE order. It is not model-only throughput, a paired-seed CI, full learning
cost, QPS, image-to-top-k latency or a p99 claim. The two architectures differ in
pretraining, width/depth and native resolution; this is a source/system contrast.

Whole process39.96s (in-script36.7540s), peak host3,618,344KiB; hard180s/8GiB
host limit and shared lock were enforced. Host figures are cumulative process
maxima. Binary checkpoints remain on DGX for replay (about1.2GiB/359MiB), not in
Git. Both model/optimizer/cache arms were released serially; CUDA cleanup0B.

Independent CPU checkpoint audit passed in8.76s, peak3,659,600KiB, exit0:
reload original strict FP32 models and saved final checkpoint bytes; independently
reconstruct initial/final frozen state and every tail/pool/norm/proj/head/classifier
group digest; check saved foreign PE rotary frequency and warmed grid; verify
complete first/last gradient inventories, exact12 Large null key-bias exceptions,
16 scales128, normalized bank, initializer/checkpoint hashes and all raw cost gates.
No score or gradient replay beyond the captured first/last norms is claimed.

Run `scripts/audit_inshop_pe_training_checkpoint.py` with the frozen root/output,
dataset/snapshot paths and `--vision-precision fp16`, CUDA hidden. Raw receipt,
preflight, launch, terminal journal, logs/cost and independent checkpoint audit:
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/pe-fp16-mechanics-v1/`.
CPU audit invocation `d56a254aa6eb4009afceb2ea7907cf81`.

Next: separately freeze full-fit source-cache acquisition and the matched actual
image-training quality-feasibility gate. Use complete fit-only PCA/proxy/bank
initialization, matched augmentations and frozen TRAIN-held evaluation/uncertainty.
The small unaugmented1024-image fixture is not that trainer or quality recipe.
No automatic official evaluation or release follows. The joint production goal
remains active and unmet.
