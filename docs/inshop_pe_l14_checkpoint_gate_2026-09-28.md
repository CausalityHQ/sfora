# Fixed native L14 activation-checkpoint execution

Uncheckpointed L14 stopped at the first update's >=10GB allocated guard.
Change only training activation storage: explicit non-reentrant replay of
12trainable blocks. Native source/config/input/half12 parameters/initializers/
loss/AdamW/scaler/RGB schedule/cost/quality floors stay frozen. No architecture,
prefix/LR/input/precision/budget sweep or continuation of failed state.

Wrap upper block forwards only, Torch checkpoint(use_reentrant=False,
preserve_rng_state=True), bypass for eval/no-grad. Preserve native classes,
state keys, parameter objects and rotary aliases. Keep underlying native
Transformer checkpoint flag disabled: its default reentrant calls cannot
be assumed correct for frozen-prefix inputs. No global monkeypatch.
Old qualified source/files/initializers stay unchanged in a new isolated DGX
code root /home/riomus/runs/sfora-pe-l14-checkpoint-v1.

Runnable tiny CPU fixture: no input gradient, exact source output/every
parameter gradient, same state keys/weights/eval output, duplicate-wrap
rejection. RED missing implementation then GREEN; Ruff PASS. This is not
native model or quality evidence.

ONE native CPUFP32 parity attempt then ONE native GPUFP16 parity only if
CPU passes. Each120s whole/8GiB/no swap; GPU<10GB allocated and both locks.
Exact own initializer/preflightbe5a8bdcfd1749340f220d5669cebfb1c0d8b30d3c08667a0b72a4fb38701277,
qualified source/cache/control authority, actual317m L14 frozen half12 model.
Only first2 authenticated TRAIN-fit images/own labels/unchanged loss with
full detached fit bank. No optimizer update or held decoding. Compare
uncheckpointed vs checkpointed source+compact outputs, loss and ALL161
trainable parameter gradients (158encoder+2head+1proxy) EXACTLY, finite and
nonzero, with inputrequires_gradFalse. GPU uses FP16 autocast/scaled128;
CPU FP32. Source/frozen/foreign state unchanged, normal serving output exact.
Record cost/peakCUDA, code/helper/installed Torch-checkpoint hashes. This
small-batch baseline is a parity qualification, not another closed B64 run,
training cost forecast or retrieval measurement. Any parity/resource failure
stops this method, no threshold/dtype/workload/retry rescue.

Only real native CPU+GPU parity permits separately frozen ONE new17stepB64
mechanics: exact same own initializer/control exposure/AdamW/loss/scaler,
120s/8GiB/no swap/<10GB and guarded median3–17<=.71769696s. Log exact
allocator counters and phase BEFORE every CUDA resource assertion. Discard
all mechanics trained state, strict updated native GPU reload parity on PASS.
Then ONE fresh100TRAIN pilot only on PASS,300s/8GiB/no swap/<10GB,
median<=.71769696s, R1>=95.1720176%,MAP>=77.6237120% and both paired product
95% lower bounds>0 vs archiveddensePE95.0739692/76.3915922%. No exact budget
or scientific floor changes. A survivor advances to updated serving/fresh
SOP+In-Shop official and CUB/Cars confirmation/full matched latency gates.
The full product goal remains active and unmet.

## Actual native CPU parity PASS

SoleCPU inva833b770c6ce44a0869f53bf49058e49 original23576exit0,16.63s whole.
Actualnative317m L14 source/compact/loss and ALL161trainable parameter
gradients EXACT with inputrequires_gradFalse. Native state/frozenforeign
rotary/normal serving output exact, zerooptimizer/held/quality. Rawreceipt
and cost retained. GPUFP16 native parity is still unmeasured.
