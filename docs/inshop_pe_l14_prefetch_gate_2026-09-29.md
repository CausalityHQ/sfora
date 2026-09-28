# Frozen L14 readout with one pending CPU input

One execution change follows the [serial cost KILL and measured input cost](inshop_pe_l14_readout_result_2026-09-29.md):
stdlib ThreadPoolExecutor, one worker and one pending nextB64 batch. No
method/source/processor/threads/precision/loss/optimizer/exposure change.
Current native GPU update overlaps actual next-image decode/augmentation/
file hashing/native pixels. Every update still runs the native L14 encoder.
No feature cache or preprocessing amortization; no speed forecast.

CPU qualifier completed11.54s/1,842,176KiB/no swap, original42922,
inv2e83003c846343019ccd5e910e3cda51, exit0. ALL17 actual serial/worker
pixels and strides exact, RGB exact originalcontrol, no CUDA, callerCPU RNG
unchanged. All17 pixel hashes also equal the prior independent serial profile.
Queue order/bound/exception check REDmissinghelper→GREEN. Ruff/bash/compile
checks passed. All67 old frozen executing files remain byteidentical in the
new isolated root; original actual317m native CPU/source/gradient/role gate
remains valid and is reused, not replaced by a small fixture.

New CPU execution authoritySHA
234d7ece495c74a294ff18e3cc7f12cf301244d6cc322783d8e26a9db0e2c273
pins five new files plus original67 and native CPU receipt. Worker CPU RNG
fork is safe only because this fixed main training graph has no CPU random
operation while it runs; assert caller RNG unchanged after worker close.
Native drop_path default0/attention dropout0; random parameter initialization
occurs before worker start or after worker drain. Do not use this helper as
a generic prefetcher for concurrently stochastic main CPU work.

ONE17mechanics, unique output, both GPU locks inside120s/8GiB/no swap/<10GB
unit, median3–17<=0.71769696s. Whole training wall includes first fill and
last drain. Per-step cost includes waiting for unfinished next input and the
complete native GPU update/integrity work. Log worker input times separately;
these overlapping times must not be added to wall or reported as savings.
Runtime files/RGB verified every batch, exact first17 pixel authorities in
worker, source/foreign hash immutable, three head/proxy gradients finite/
positive and both groups move, scaler128/no skip. Source calibration unchanged.

On costPASS strict native updated encoder/head GPU reload compares exact
source outputs AND actual normalized compact readouts. Always delete17
checkpoint/discardstate. Only full PASS permits ONE fresh100TRAINpilot300s
same authority, original source/owninitializer, same methods and unchanged
[quality/packed/full-held gates](inshop_pe_l14_readout_gate_2026-09-28.md).
Pin mechanicsPASS receipt; never extend17state. No source/input/scalar/budget
rescue after negative. All quality/public latency currently unmeasured L14.
Survivor proceeds to actual updated serving and fresh confirmation; full
production SOP/InShop joint quality+matched public speed target remains open.

[CPU raw authority](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-prefetch-v1/cpu-preflight.json),
[actual CPU qualification](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-prefetch-v1/prefetch-cpu-v1.log).

## Sole17 mechanics PASS

Original70007 collectedexit0: median3–17 **0.6378182559274137s**, training
wall including first fill/last drain11.336585931945592s, whole26.96s,
5,609,044KiB hostRSS/no swap, allocatedCUDApeak3,810,310,656B including
native strict reload. All17RGB/scales128/frozen source/data gradients/group
movement/CPU RNG/updated encoder AND compacthead strict GPU parity pass.
All72 executing files unchanged;17 checkpoint deleted/state discarded.
No held/quality read. Full production goal remains unmet.

MechanicsPASS receiptSHA
02fb166be9ca2ef26fd7af59f6fea14e6f93b461e16e305d2ad234948956ec6b
licenses ONE fresh100TRAINpilot300s under the same frozen execution authority.

## Terminal100 outcome

One fresh100 completed: cost0.6362088094465435s/update PASS, strict native
encoder/head/full-held packed parity PASS, TRAIN-held R1 87.2836009% and
mAP@R61.8878907% QUALITY KILL. Independent CPU saved-state/score/interval/
decision replay passed. Both quality floors and paired bounds failed.
No official/transfer/public latency or promotion. Procedure closed.
[Measured convergence row, raw evidence and next intervention](inshop_pe_l14_prefetch_result_2026-09-29.md).
