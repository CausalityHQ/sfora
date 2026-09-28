# Native PE hidden-matrix optimizer gate

One selected next intervention after the rank32 procedure's verified quality
KILL: change the native encoder matrix optimizer while keeping all dense
native coordinates available. This tests the encoder-adaptation layer
implicated by saved-state attribution. Optimization versus capacity is still
unresolved; no PE quality, memory or speed gain is predicted.

The scope audit found no prior native PE hidden-matrix Muon run. The earlier
cached-feature ProxyMuon candidate and head-only proposals remain closed;
no proxy/head Muon, LR retuning or precision rescue is included. Use the
installed native PyTorch2.12.1 implementation, pinned implementation SHA
4d003aba2d0c7fcc24875845802e45edb4475a51a037e2caaa7b46c3944c0dae.
Its [primary documentation](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Muon.html)
defines hidden2D matrices and the RMS-adjustment option; this is algorithm
provenance, not In-Shop evidence.

## Fixed complete procedure

Authenticate original FP32 PE-B16-224 source/environment and old source/cache/
initializers/100-batch preflight. Freeze native blocks0–5 and foreign rotary
exactly as the dense control. Optimize only the24 dense hidden matrices in
blocks6–11 (QKV,output,MLP;42,467,328elements) with native Muon:
LR1e-5,decay.05,momentum.95,Nesterov,coefficients(3.4445,-4.775,2.0315),
eps1e-7,5NS steps,`adjust_lr_fn=match_rms_adamw`, unchanged native update
precision. This documented adjustment does not guarantee calibrated function
steps or isolate preconditioning from the rest of the optimizer change.
AdamW remains on native norms/biases/pool/projection at1e-5 and head/proxies
at1e-4,decay.05. No parametrization or inference-model change.

Exactly partition all89 trainable parameter tensors between the two
optimizers, with no missing/duplicated/frozen parameter. FP16 native forward,
scaler128; unscale both optimizers, clip the combined parameter vector to1
with nonfinite rejection before either step, step both and update scaler once.
Reject any skipped update/nonfinite gradient/state/weight or frozen prefix/
rotary change. Every24 matrix must actually move; preserve each initial/final
digest. Same seed179032, RGB/native pixels, objective/bank/singleton behavior
and first100 exposure as authenticated controls. No loss, LR or prefix search.

CPU prerequisite passed: the small two-optimizer fixture made two updates at
scale128 and verified an injected nonfinite Adam gradient leaves both
optimizers' parameters and states untouched. Actual native constructor/source/
control/coverage qualification then passed6.43s,1,718,732KiB RSS,89 trainable
parameters/24 matrices,0 optimizer updates/held images/CUDA. Both native code
and new code are hash bound in the execution authority. This qualifies
construction/partition only; actual native gradients and GPU cost remain open.
[CPU receipt](evidence/compact_metric/sop-siglip2-substrate-v1/native-muon-v1/native-muon-cpu.json).
Isolated source root `/home/riomus/runs/sfora-pe-muon-v1`; preserve old roots.

## Ordered decisive gates

1. One17-update native DGX mechanics attempt,120s whole incl shutdown,8GiB
   host/no swap,<10GB allocated CUDA. Both existing shared lock paths held
   for the whole service. Exclusive attempt/output/log reservation. Same
   first17 batches; no held images or quality. All gradients, both optimizer
   states, scales, bank, matrix movement and frozen state must pass. Actual
   saved strict-loaded updated native model must reproduce fit-only GPU
   outputs exactly. Median guarded steps3–17<=.71769696s (historical Large
   .8971212s×.8). Discard this trained state. Any integrity/cost/resource
   failure closes this fixed procedure; no100 run or rescue.
2. Only after authenticated17-step pass, one fresh100-update pilot,300s whole
   incl initialization/serialization/full-held parity/score,8GiB host/no swap,
   <10GB allocated CUDA, same locks. Restart original source/initializers.
   Strict saved native+head reload and all12599 in-process/loaded vectors must
   have cosine>=.999999 and packed per-query hit/AP difference<=1e-6.
   Score strict-loaded candidate on TRAIN-held6354query/6245gallery/1993products.
   ALL quality floors: R1>=95.1720176%,MAP>=77.6237120%,median3–100<=.71769696s,
   and5000-resample paired product95% lower bounds>0 for BOTHmetrics against
   archived dense PE95.0739692/76.3915922%. Historical Large control is
   95.6720176/78.6237120%. Preserve raw per-query outcomes and product/query
   intervals. Exposure and scorer are matched; historical nondeterminism,
   unmatched compute and repeated TRAIN-held method selection remain caveats.
3. Terminal negative closes this procedure only. No optimizer/LR/NS precision/
   steps/epoch/prefix/budget/official rescue. A survivor goes to actual updated
   serving qualification and fresh confirmation; neither TRAIN pass nor
   unchanged architecture proves a public latency gain or production quality.

Concrete executable: `train_pe_native_muon.py`, helper`pe_native_muon.py`,
launcher`run_pe_native_muon.sh`. The launcher has only fixed17/100 phases,
119/299s runtime+1s shutdown,8GiB/no swap and both lifetime locks. The pilot
requires an authenticated mechanics receipt. Native model-quality/GPU/public
latency measurements do not yet exist for this procedure. Full production
SOP/In-Shop quality/speed/transfer/frontier requirements stay active and unmet.

## Native17 measured checkpoint

The sole native17 attempt passed:20.06s whole process,3,674,024KiB RSS,
median3–17 .555988556s<=.71769696s,5,050,932,736B peakCUDA. All24 native
matrices moved;17RGB/scaler records, gradient/state/frozen checks and actual
updated strict-loaded GPU output parity passed. Trained checkpoint deleted;
no held images/quality read. Independent stdlib record replay passed.
The first shell launch stopped before systemd/GPU because copied historical
log names existed; unique Muon logs fixed the guard, preserving originals.
Full code hash binding required CPU receipt refresh after launcher-name edits;
final CPU6.356s passed with unchanged numerical source/constructor. No new
research/critique or alternative method ran. Fresh100 is the next gate, with
the17 pass receipt hash bound in its authority; no new quality number yet.
