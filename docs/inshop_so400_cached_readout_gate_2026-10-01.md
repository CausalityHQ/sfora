# So400 fixed-view cached readout: prospective gate

Status: source implementation underway; native qualification, runtime and quality
are unmeasured. The production SOP/InShop quality-and-speed goal remains open.

The full-image native256 TRAIN100 procedure failed its 300-second whole-unit
engineering gate. Parallel and batched state snapshots were slower on identical
payloads; both are closed. Successful v5/v6 mechanics and useful historical
checkpoints remain valid. No failed partial checkpoint initializes this method.

## One matched intervention

Freeze the authenticated So400 native256 encoder and its normalized pooled
FIT features (13,283 images, 2,004 products, width 1,152). Train a compact
128-dimensional rank-32 residual readout and the classifier. Both arms use
the same PCA affine primary, FIT center, isolated seed179034 Kaiming down matrix
scaled by its FIT preactivation standard deviation, and zero up matrix:

`raw(u) = P*u + b + U*phi(D*(u-center))`, with `u=normalize(cached_row)`.

Control uses `phi(z)=0.5*z`; candidate uses exact GELU. Each head has188,544
scalars; with the2004x128 classifier this gives445,056 trainable scalars in
exactly five optimizer tensors. The frozen encoder is an authenticated static
dependency, not an optimizer member. Original208-member qualification is neither
changed nor claimed for this new method.

This reuses the architecture that failed on the older Large inner-FIT split
([result](inshop_nonlinear_head_result_2026-09-28.md)): R1−0.2111pp and
mAP@R−0.6650pp. That negative stands. The new hypothesis concerns materially
different So400 features; it is not an architecture novelty or success claim.

Predeclare1,000 cached updates per endpoint, B64/micro16, schedules179032/179041
whose first100 rows match their original authenticated schedules. Use FP32
head/classifier/bank, CE margin0.3/scale64 plus8 times the corrected valid-anchor
rank loss, AdamW LR1e-4/decay0.05/original defaults, clip1/scaler128, and original
detached bank refresh with last duplicate winning. These are fixed cached views:
no random image augmentation and no encoder learning.

## Admission, execution and terminal decisions

Bind the original source/extraction/FIT/PCA receipts, logs, execution closures,
configuration, processor, runtime origins and buffer facts. So400 FIT SHA is
`c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716`;
PCA-v2 initializers SHA is
`dcb92082a1e5ad1f4087a3e044ae245b6d32432d035a3214178d2ad72a494c72`.
Freeze the exact new source closure, authority and launch commands before native
execution. Complete changing state includes all head tensors/buffers, classifier,
bank, targets, positive ordinals, schedules, optimizer state/member order,
scaler, CPU/CUDA RNG, counters, flags and static bindings. Preserve original
typed serial fingerprint and full uncached dependency checks at exit.

1. CUDA-hidden CPU120: actual frozen source construction and independent reload,
   initial arm/PCA parity, two-stage normalization, complete state, nonzero up
   gradients and subsequent down learning (initial down gradient is zero).
2. One discarded mechanics300 per arm, seed179032: full uninterrupted17 versus
   independently restored8+9, complete state/diagnostic equality, strict final
   raw/unit/packed reload and complete exit integrity. Discard mechanics states.
3. Only both complete passes admit four fresh TRAIN1000 endpoints, each300s
   whole-unit: control032, candidate032, candidate041, control041. Admission,
   all updates, serialization, independent reload and exit count against the cap.
4. After all four terminal accepted receipts, qualify updated reloads underCPU120;
   then export TRAIN-held6354queries/6245gallery/1993products underexport300.
   A shared frozen B32 encoder pass may feed all four independently loaded heads;
   verify full raw/unit/packed parity and save the authenticated held1152 cache.
5. CUDA-hidden score300, exact packed scorer and tie semantics. Each seed requires
   candidate-minus-control R1>0 and AP>=0; equal-seed mean gains both>=0.002
   (+0.20pp); both paired product95% lower bounds>0. Use the same5,000 shared
   bootstrap draws, seed179019; report query intervals. Each seed's whole-service
   and median-update candidate/control ratios must be<=1.50. Report optimization
   wall separately. No tuning branch, best-seed selection or partial-state reuse.

Every unit retains8GiB host, no swap, zero disallowed memory events, CUDA
allocation<10GB, both lifetime locks and complete-unit peaks without reset.
Any runtime/resource/integrity failure stops this procedure. A negative closes
this fixed So400 rank32 recipe and budget, not all nonlinear readouts.

Report prior reusable preparation separately from new-method costs. Cached
optimization throughput is not image-training throughput or public latency.
Record residual affine-reconstruction energy; a dead/affine branch cannot support
a nonlinear-mechanism claim and never permits weaker gates or a retry.

A survivor advances to a separately frozen full-TRAIN cached comparison, then
explicit1152/nonlinear public serving, fresh confirmation and matched image-to-top-k
B1/B32 timing. A p99 claim requires10,000 interleaved paired calls and uncertainty.
No official quality or matched public latency has been measured for this method.

| Dataset/split | Matched control/candidate R1 + mAP@R | Training cost | Public latency | Remaining gap | Next decisive test |
|---|---|---|---|---|---|
| InShop TRAIN-held6354/6245/1993, prospective So400 | Unmeasured/unmeasured | Unmeasured; each endpoint<=300s | Unmeasured | Joint official SOP/InShop quality and speed unproven | Actual CPU qualification, then full paired mechanics |

Read-only plan f7323a6e88524ef0 completed exit0/447s atb91545a3;
root independently checked the archived Large negative. Existing scientific
Opus/Astra review findings remain in force; no duplicate review was launched.

Prospective CPU freeze: source commit29bc6a6e, original trainer worker1942aee;
new separate two-file execution16b29bc94b3ef74f72cc8047f3304ec719e66742a1f72d3b2f8ae72e57cb3c64,
CPU authoritye0ff2e1ffa286ec4fb85ac899eabc6160eb28027a4a557fb0548c9a7ebd8ede7,
command275e2dd53185990028baf821ea21e967912f5cb9936d213e1fd2c2dff3389b48.
Original serial v6 reference five-file closure remains separate and unchanged.
[Immutable launch evidence](evidence/compact_metric/sop-siglip2-substrate-v1/so400-cached-readout-v1/).
Native CPU result pending; no mechanics or TRAIN endpoint admitted.

Original CPU session70852/invocationdf03638a49304717bc9a6e213c53b2aa passed normalexit0/service66.308s, hostpeak5924012032B, RSS4426696KiB, events0/swap0/CUDAhidden. Actual source strictreload, initial matched raw/unit/packed arm parity, nonzero-up/subsequent-down witnesses and uncached exit pass. Five optimizer members/445056 scalars. Receiptce066d0e5a31782ac4a9e01866a62969ea1bbeb409c339b8c70ad42426369280. Prospective discarded mechanics authorities and commands are frozen next; no TRAIN or quality admitted.

Original control mechanics87711/invocationa7d361b90b31430fa322caf4dfb39b1d failed normalexit1/service37.495s after17updates at calibration: pinned CPU-only packer rejected a CUDA tensor. No8+9/finalreload/acceptedreceipt; candidate/TRAIN/quality remain unadmitted. Preserve this failed v1 gate. Root traced existing original calibration CPU-transfer contract; stdlib execution stand-in reproduced RED before correction, then GREEN with only unit.cpu() at the wire boundary. Original CPUv1 pass is preserved; corrected source requires a NEW prospective CPU/source freeze, never relabeling or unchanged retry.
