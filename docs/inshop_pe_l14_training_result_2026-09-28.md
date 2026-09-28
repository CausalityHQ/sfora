# Uncheckpointed L14 native mechanics: terminal CUDA resource KILL

The actual CPU source/own initializer/native half12 qualification passed.
The sole17step GPU attempt then stopped at the first update's allocated-CUDA
resource guard. Close this uncheckpointed execution procedure; no100pilot,
trained-state continuation, input/prefix/LR/precision/budget rescue or
quality selection. This is a reproducible mechanics blocker, not a measured
L14 retrieval-quality negative. The whole product goal remains active.

| Dataset / split | Relevant controls R@1 / mAP@R (%) | L14 candidate R@1 / mAP@R | Training cost / resource | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| In-Shop TRAIN-fit13283images/2004products; first augmentedB64 update; TRAIN-held6354q/6245g/1993products excluded | Historical100update densePE95.0739692/76.3915922; Large95.6720176/78.6237120, prior verified | Unmeasured | Whole15.78s/3,999,816KiB processRSS/no swap; CUDA peak **>=10,000,000,000B**, verified guard lower bound; exact peak and guarded update median unavailable | Unmeasured | Native training resource gate fails; next one non-reentrant activation-checkpoint execution intervention with same representation/parameters/objective, real output/gradient parity and unchanged resource/cost/quality gates |

CPU invocation284016b22f3b4524b08d0d19bf5a7836 exited0 in11.30s with
5,758,356KiB processRSS. Source317,151,232registered parameters, native
half12 inventory150frozen/158trainable tensors including16foreign rotary
scalars. Trainable encoder164,800,512elements; frozen152,350,736 including
foreign rotary. Own full-fit PCA/head/2004proxies/bank authenticated; cached
objective4.01690769 had finite positive source/head/bias/proxy gradients.
Actual native source/frozen prefix/foreign state, disjoint empty AdamW
coverage and first augmented64RGB224pixels matched frozen authority.

GPU invocationbeff79d85a4e45238c4e88c6bb6d531d, original7527 collectedexit1,
15.78s whole. FP16/source minimum cosine.9998070598 passed. The traceback
is the CUDA allocation assertion after first update, bank refresh and finite
model/optimizer/bank checks, before its timing/diagnostic record. No recorded
step completed; no17step median, images/s or quality number exists. This
was an explicit allocated-memory guard, not an OOM exception or host cgroup
termination. The service retained120s whole/8GiB/no swap and both GPU locks.
No updated checkpoint was created, no held image decoded, no100pilot directory
exists; process memory/training state was discarded and DGX is idle.

The exact allocator counter was not logged before the shared assertion.
Report only the verified >=10GB bound; do not invent a peak or infer whether
activations, optimizer state or temporaries alone caused it. Post-run code/
initializer/preflight hashes match the CPU authority. Raw trace, guard code,
input/source/initializer authority and exact invocation make the blocker
reproducible without silently rerunning the closed attempt.

## One next execution intervention

Select **non-reentrant activation checkpointing of the12trainable native
blocks**, keeping source/config224 input/half12 inventory/objective/AdamW/
scaler/control exposure unchanged. This addresses the observed memory
blocker through recomputation. It does not change model capacity or select
on unobserved quality; memory savings and update speed are unmeasured.
No method has qualified L14 training or official quality yet.

The pinned native Transformer already has checkpoint support, but calls
Torch checkpoint without explicit `use_reentrant`; installed Torch defaults
that toTrue. Frozen-prefix inputs have no required data gradient, so blindly
enabling that flag cannot be assumed to preserve trainable encoder gradients.
Use explicit non-reentrant behavior confined to trainable blocks, preserve
native state keys/rotary aliases and use normal native execution for serving.
Qualify actual native outputs, loss and every trainable gradient against
uncheckpointed small-batch TRAIN-fit execution before a new budgeted B64
mechanics attempt. A tiny fixture alone cannot qualify native model parity.
Record allocator values and phase before guards in that next execution; do
not relaunch this failed job to recover its missing peak.

This is the one next selected intervention, not yet implemented or launched.
Frozen120s/300s,8GiB/no swap/<10GB, median<=.71769696s, same TRAIN quality/
uncertainty floors and mechanics-state discard remain. No architecture,
optimizer, scalar, prefix, source or input sweep. A survivor must advance to
updated serving and fresh official/transfer confirmation. No operator
decision is required; overall joint quality/speed remains unmet.

[Terminal decision/replay](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-training-v1/mechanics/terminal-decision.json),
[raw trace](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-training-v1/mechanics/l14-mechanics-v1.log),
[whole-process cost](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-training-v1/mechanics/l14-mechanics-v1-time.txt),
[post-run authority/service state](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-training-v1/mechanics/terminal-post-audit.txt),
[CPU native authority](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-training-v1/cpu-preflight.json),
[frozen procedure](inshop_pe_l14_training_gate_2026-09-28.md).
