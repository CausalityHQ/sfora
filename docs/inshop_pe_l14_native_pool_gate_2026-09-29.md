# Native L14 attention-pool adaptation after fixed-readout quality KILL

One selected intervention follows the [measured fixed-readout failure](inshop_pe_l14_prefetch_result_2026-09-29.md).
Learn the EXISTING native `attn_pool` query/attention/layernorm/MLP parameters
and the same128-D affine head/2004proxies. Freeze all24 transformer blocks,
patch/positions/ln_pre/ln_post/output projection and foreign rotary state.
Native PE class/config/forward/state keys stay unchanged. Pool parameter
count/inventory must be measured on the actual loaded source, not inferred.
This is native nonlinear token aggregation, not a new head/module, pooling
architecture, prefix-depth sweep or extension of discarded mechanics state.
No novelty or forecast. The best adaptation layer remains unknown.

## Frozen method and execution

Same authenticated PE-Core-L14-336 source/own PCA/head/proxies/bank,
native224 input with original336 positional config, first100of1000schedule
seed179032, original actual augmentedRGB/files/roles, ArcFace.3/64 +8 detached
bank ranking, singleton policy. Vision pool AdamW1e-5, head/proxy1e-4,
decay.05/globalclip1/scaler128, FP32 weights/FP16 encoder autocast/TF32off/
deterministicTorch/pre-CUDA CUBLAS_WORKSPACE_CONFIG=:4096:8. One bounded
CPU nextB64 worker remains; no main CPU RNG use while worker fork runs.
Actual native forward/learned-pool backward inside every update cost;
no cached-only timing. Whole training wall includes fill/drain.

CPU source qualification120s/8GiB/no swap/CUDA hidden: authenticate original
source/cache/initializer/control plus executing code; exact pool-only native
parameter inventory/frozen complement/foreign state; unchanged initial native
output and keys; actual first2TRAIN-fit source→head/proxy/fullbank loss and
finite positive pool/head/proxy gradients, no frozen gradient; disjoint empty
AdamW coverage. Prove frozen-complement state unchanged. Hook qualification
uses source-fit pixels only, no held/quality/optimizer updates; an isolated
temporary pool perturbation tests exact strict-loaded updated pool/source/
head outputs and restoration, with checkpoint roundtrip and native keys.
Reuse unchanged all17 worker/serial input authority and actual source files.

GPU fit-only qualification before mechanics: original FP16/F32/cached source
cosine>=.999; baseline native output bitwise equal with qualification hook;
temporary pool perturbation save/strict native reload and learned pool/final
normalized readout bitwise equal on actual native token inputs, source/core
state/foreign aliases exact. No optimizer/held read; frozen GPU/code receipt.
This is a SEPARATE120s whole-process qualification with8GiB/no swap/<10GB
and BOTH lifetime GPU locks. Require changed pool state AND changed fit-only
output under actual FP16, then exact roundtrip and restoration from independent
cloned tensors. A vacuous perturbation fails. Record actual pool attention
autograd backend nodes and deterministic warning/error evidence.
Then ONE17mechanics120s/8GiB/no swap/<10GB under both existing lifetime GPU
locks; median3–17<=.71769696s, all controls/gradients/scaler/group movement/
frozen complement checks. Always discard17state/checkpoint.
Only completePASS permits ONE fresh100TRAINpilot300s under same authority.
Median3–100 must remain<=.71769696s BEFORE held decoding. Successful
terminal process/resource evidence is mandatory for every PASS receipt.
Failure closes this fixed procedure without rate/budget/precision/depth rescue.

## Exact updated native serving parity with an immutable trunk

The old whole-encoder-immutable argument cannot be reused: pool tensors now
learn. Freeze only the complement of `attn_pool.*` plus rotary/projection.
After native save and strict reload, prove ALL live/loaded state keys/tensors
and foreign/grid/aliases exact; prove frozen complement equal original CPU.
Last actualB64 native encoder and normalized compact outputs must be exact.

For full-held evaluation, the strict-loaded ORIGINAL native class runs its
full normal forward on every real image. A removable forward hook ONLY on
its `attn_pool` observes its actual post-ln_post trunk-token input and native
pool output. Run the in-process updated native pool on that SAME token input
under the same inference/autocast context; require pool outputs bitwise equal.
Compute live source via the unchanged live projection, require equality to
the full strict-native output, then require live/strict-loaded normalized
compact head outputs bitwise equal. Check every held image, remove hook in
finally, reject missing/multiple pool calls or unexpected shape/state.
No global monkeypatch, alternate forward, cached token source or replacement
class. Prove hook leaves baseline native output/state unchanged first.
Immutable-trunk state identity + actual shared token inputs + exact learned
pool/source/readout equality covers the changed boundary while avoiding a
redundant expensive immutable-trunk pass under the original300s cap.

This is an explicitly registered equivalence procedure, not permission to
skip serving qualification. Failure of any predicate is a terminal integrity
failure; do not change parity/precision or enlarge the budget after observing
data. Reference packed hits/AP must agree for actual live/loaded vectors.
Retain independently constructed live vectors (live pool→live proj→live
head), never `loaded.copy()` as their provenance. Compact-head arithmetic
stays OUTSIDE encoder autocast. LastB64 live-vs-strict whole-native output
is the only directly measured trunk-token equivalence; full-held shared-token
boundary equivalence follows frozen state/code/configuration/context.
Do not claim the hook separately ran the live trunk.
Score the actual strict-native loaded candidate; retain per-query outcomes.

## Quality and promotion

Same exploratory TRAIN-held6354q/6245g/1993products,100update history:
densePE95.0739692/76.3915922% and Large95.6720176/78.6237120%, prior verified,
not concurrent reruns. Require packed R1>=95.1720176%,mAP>=77.6237120% and
5000resample paired product95lower>0 for BOTH versus densePE. Preserve query
uncertainty too. Cost/resources are measured, never projected from parameter
count. Independent CPU saved-native state/packed-score/CI/decision replay
before promotion. Repeated TRAIN-held development remains exploratory.
Survivor advances to updated public serving/fresh confirmation rather than
more TRAIN tuning. Full SOP+InShop/transfer/frontier/matched10kpairedp99 goal
remains active and unmet. No pool job has launched at this design checkpoint.

## Independent critique and actual native CPU qualification

Group50236cf153ff48d0 completed: separate Opus704739590a244f48 and
Astrae38ab0ce154d4980 both conditionalGO for this procedure, not production.
Both accept the shared-token boundary under registered exact predicates.
Implementing the new optimizer/clipping/autograd/pool movement checks and
independent live vectors remains mandatory before any mechanics run.
Foreign nonpersistent buffers/grid/device/rotary aliases now participate in
qualification. Pool-input guards prove graph-free fp32 trunk tokens.
No new paid review/research is needed to restate these conditions.

Actual originalCPU qualification v1 passed18.04s. Strengthened v2 after
review adds frozen-token/runtime/nonpersistent-buffer guards and passed
18.06s/5,125,436KiB/no swap, original32841 collectedexit0,
invbc5760213f244da98c40bdf52bb97625. Source317m native parameters remains
authenticated; exactly11 pool tensors/12,595,200parameters trainable,
all finite positive actual source-fit gradients, no frozen gradients, initial
native output unchanged. CPU pool state/output perturbation is observable;
strict native save/reload/pool→proj→head outputs exact, restoration exact,
no optimizer update/held/quality read. Actual tokens2×257×1024 fp32/no graph;
native pool8heads, grid16×16, positional source24×24, all rotary aliases exact.
Foreign grid/frequency/dummy buffer device is CPU for both native copies;
do not assume `.cuda()` recursively moves the plain Rope2D object.
CPU v2 authoritySHA
1f11069e3675890b75b474095c01a135078c346cc02e69979823ccc60899d154.

Tiny role/boundary/changed-pool rejection/hook-cleanup check REDmissingmodule
then GREEN; Ruff passed. These are qualification evidence, not model-quality
gains. No native-pool GPU qualification/mechanics/pilot had launched at that checkpoint.

## Actual FP16 GPU qualification PASS and training execution freeze

The separate native GPU qualification original47211 collectedexit0 in18.05s,
5,296,808KiB processRSS/no swap, allocatedCUDApeak3,333,430,272B. Actual
pool backward nodeScaledDotProductFlashAttentionBackward0 with Torch
determinism enabled; all11 native pool gradients and head/proxy gradients
finite/positive, frozen input nograd, no frozen parameter gradient. No
determinism warning/error appeared in the retained log. Source FP16/FP32
cosine min.9998948 and cached/fresh min.9999973 passed. Temporary query
perturbation changed actual FP16 source output; strict native pool/proj and
normalized head reload bitwise equal, original pool/native output/state
restored, optimizer empty. Zero optimizer update/held/quality read.
GPU qualification receiptSHA
81968cb8f1384262bf8c1e71fe9aa8e60234918d2971d80b6cca91d9d09517c1.

New train_pe_l14_native_pool.py enables exactly the qualified native pool,
includes all14 pool/head/proxy tensors in disjoint AdamW and global clipping,
keeps native autograd forward and graph-free trunk-input guards, pool movement
hashes and actual backend evidence. Hooks are removed in finally. Full-held
callback computes separately retained actual live pool/proj/head vectors and
strict-loaded vectors, then independently packed-scores both actual arrays.
Whole native live/reloaded state equality and frozen-complement/runtime
identity replace the earlier all-encoder-immutable assertion. CPU startup
authenticates receipts/source/initializers/data/code without CUDA.
Training execution manifestSHA
92c6e940a86498f89a69b0ea87102247e72d327058b36850500c0e81bffe1747.
Only ONE17mechanics next; no new cost/quality gain inferred from qualification.
