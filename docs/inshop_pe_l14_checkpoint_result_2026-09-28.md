# Deterministic L14 activation replay: memory PASS, training-cost KILL

Actual native CPU and deterministicGPU output/loss/every161trainable-gradient
parity passed. One17step B64 mechanics attempt then solved the CUDA memory
blocker but failed the frozen training-cost ceiling. Close this execution
procedure without100pilot, checkpoint continuation or scalar/budget rescue.
No L14 quality or public latency was measured. The full product goal stays
active and unmet.

| Dataset / split | Historical controls R@1 / mAP@R (%) | Candidate quality | Training cost | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| In-Shop TRAIN-fit13283images/2004products, first17B64 augmented updates; TRAIN-held6354q/6245g excluded | densePE95.0739692/76.3915922; Large95.6720176/78.6237120, prior verified100update measurements | Unmeasured | Verified median3–17 **1.816015209s**,35.242guardedimages/s; ceiling.71769696s FAIL;44.31s whole;4,067,444KiB processRSS/no swap;CUDApeak5,379,930,624B PASS | Unmeasured | Training cost2.5303× ceiling/2.0243× archivedLarge.8971212s; next one supervised compact-readout learning procedure on frozen L14, actual source forward included in update cost, actual TRAIN quality pilot only on mechanicsPASS |

## Native parity and the backward reproducibility prerequisite

CPU native parity original23576/inva833b770c6ce44a0869f53bf49058e49 exited0
in16.63s/3,975,088KiB RSS: actual317m source/compact/loss/ALL161gradients
exact with frozen input, unchanged state/rotary/eval. DefaultGPU candidate
original6656/inv5fb3755f3b644ab2a4358866d89aaf4c exited1 in17.43s: outputs
and loss exact, finite nonzero gradients, gradientbits unequal. Its magnitude
was not saved, so no candidate error bound can be reconstructed.

Separate unchanged-baseline AA original59607/inv5280df0acc9547c0932003cef21803ec
exited0 in20.03s/4,151,164KiB RSS:133/161native encoder gradients differ
between identical uncheckpointed passes, maxabs.03125 **scaled128**,
maxrelativeL2.0016199522. Source/readout/loss exact; no output-head/proxy
gradient differed. Thus bitwise backward repeatability of the default GPU
baseline was invalid; the earlier result does not isolate checkpoint error.
No specific nondeterministic kernel was identified.

One separately frozen deterministicGPU execution (Torch deterministic
algorithms, pre-CUDA CUBLAS_WORKSPACE_CONFIG=:4096:8) original22038,
inv8e2d7da324534ee2a237add85d3b3c2d exited0 in14.80s. BaselineAA and
checkpointAB outputs/loss/ALL161gradients now EXACT, state/foreign/eval exact,
CUDApeak2,055,316,480B. The original gradient predicate, precision, source,
workload, resource/cost/quality floors were never loosened. No optimizer,
held or quality read in any parity diagnostic. Small-batch parity is not a
B64 speed/memory projection. The default-backend exact-gradient procedure
remains closed; deterministic execution is separately recorded.

## Actual B64 mechanics and independent replay

Sole deterministic checkpoint mechanics original40712 exited0 with valid
costKILL in44.31s (service44.321s); frozen120s8GiB/no swap/<10GB and both
lifetime locks held. Exact allocator/phase values were logged before every
forward/backward/optimizer/resource guard. Peak5,379,930,624B versus previous
uncheckpointed verified lower bound>=10GB demonstrates the memory blocker
was resolved; it does not isolate checkpointing alone from kernel determinism.
All17RGB hashes match archived dense control, scales128/no skips, finite
model/optimizer/bank and positive first/last data gradients. Every upper/
output group moved; frozen prefix/foreign hash stayed exact against CPU.
Independent stdlib median/scales/controlRGB/group/source/allocator-phase/
resource/cost/post-state replay passed. Original step3–17 median1.816015209s
exceeds unchanged.71769696s ceiling. Guarded cost includes actual decode/
augmentation/preprocessing/forward/backward/optimizer/integrity checks.
No stage timings isolate which portion dominated; do not assign the entire
slowdown to activation recomputation or GPU kernels alone.

All trained17state was discarded in process; no checkpoint was created before
costPASS, no updated reload/held export/100pilot ran. DGX idle afterward.
Execution authority SHA2f13bda3db9b702fa6c6dc798844900c86f78557a71517c90bb2ae81e2d34386;
17receipt SHA4c431e0816e570fc84892972b348792d7b3072af1123171a5dfadd83d84ef3fd.
No quality KILL or quality gain is inferred from this cost negative.

## One next useful learned-model procedure

Select **supervised128-D compact-readout/proxy learning on the frozen original
L14 representation**, using its own qualified full-fit initializer/bank,
unchanged actual augmented RGB exposures/loss/head-proxy AdamW prescription.
This changes the learning allocation from expensive full native encoder
updates to metric geometry. It is not a prefix-depth/LR/source/dtype sweep
or continuation of discarded state. No claim that frozen features suffice
for the quality target; that is the next real TRAIN quality question.

Measure actual native forward inside each B64 update; do not report cached
head-only time as complete training cost or amortize preprocessing away.
First qualify all-source frozen role/inventory, head/proxy-only data gradients,
source state/foreign invariance and original GPU/processor/control authority.
Then one17mechanics under unchanged120s/8GiB/no swap/<10GB/.71769696s ceiling,
state discarded, fresh100pilot300s only on PASS. Preserve exact updated
head/native serving reload and packed quality, both metric floors and product
uncertainty requirements. Use immutable encoder identity to qualify serving
without pretending it was fine-tuned. No checkpoint or new job is launched
for this next procedure. A survivor advances to updated serving/fresh
SOP+In-Shop/CUB/Cars and full matched latency gates.

[Execution/qualification authority](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/checkpoint-execution.json),
[deterministic native comparisons](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/native-deterministic-parity.diagnostics.json),
[baseline backward diagnostic](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/native-gradient-repeatability.json),
[17training](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/mechanics/training.json),
[terminal receipt](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/mechanics/receipt.json),
[independent replay](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/mechanics/cost-replay.json),
[raw allocation/cost log](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-checkpoint-v1/mechanics/checkpoint-mechanics-v1.log),
[frozen gate](inshop_pe_l14_checkpoint_gate_2026-09-28.md).
