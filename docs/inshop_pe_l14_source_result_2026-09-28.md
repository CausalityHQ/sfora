# Fixed224 PE-L14 native source qualified

The sole public acquisition/CPU qualification passed. The exact checkpoint
bytes/hash, pinned native source and executing helpers were authenticated.
Native FP32 CPU loading matched all307 visual keys,317,151,232 parameters.
The pretrained336 configuration stayed intact; existing native interpolation
produced finite nonzero2×1024 outputs from two authenticated In-Shop TRAIN-fit
images at224, with registered source weights unchanged. No optimizer, CUDA,
held images or quality read occurred. This is source qualification only.

| Dataset / split | Relevant controls R@1 / mAP@R (%) | L14 quality | Cost | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| In-Shop TRAIN-fit first2images, oneproduct; subsequent quality target TRAIN-held6354q/6245g/1993products | Historical100update densePE95.0739692/76.3915922; Large95.6720176/78.6237120, prior verified measurements | Unmeasured | Verified acquisition+CPUforward68.81s whole;4,439,496KiB process RSS; zero updates/no swap | Unmeasured | Entire quality/speed gap remains; freeze FP16/native hardware cost prerequisite before full-fit initialization and one useful training pilot |

DGX user service `sfora-pe-l14-acquisition-v1`, invocation
`ddc494a42a574220a37a5bbbe17db11f`, original launcher23041 exited0.
The600s whole cap (599+1shutdown),8GiB host/no swap and CUDA-hidden constraints
held. Report process RSS from `/usr/bin/time`; systemd's2.5G aggregate report
is a different measure and does not replace it. The unauthenticated HF rate
notice was informational; public acquisition succeeded without credentials.
DGX was idle afterward. No GPU/training job is active.

Independent stdlib receipt/metadata/cost replay and post-run remote checkpoint,
script, native-manifest and helper hashes passed. Receipt SHA256
`6038ad876a209b3ae29925427ef6d54c7f23ff084be3863bb8fb74158bb052f7`.
The two-row forward is not retrieval evidence. No model-card benchmark is
reported as a fixed224 Sfora measurement. No source/resource blocker remains;
FP16/source parity, actual training cost, own full-fit features/head/bank,
TRAIN quality, updated serving and fresh official/transfer qualification
remain open. Preserve negative B16 LoRA/Muon procedures; do not rescue them.
The whole SOP+In-Shop production goal stays active and unmet.

[Original receipt](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-source-v1/l14-acquisition-v1.json),
[cost](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-source-v1/l14-acquisition-v1-time.txt),
[execution log](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-source-v1/l14-acquisition-v1.log),
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-source-v1/l14-acquisition-v1-journal.txt),
[post-run hashes](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-source-v1/source-post-audit.txt),
[audit](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-source-v1/source-audit-v1.json)
and [frozen procedure](inshop_pe_l14_source_gate_2026-09-28.md).
