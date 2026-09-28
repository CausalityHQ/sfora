# Paired augmented native image training: STOP

The frozen100-update PE replacement **fails the predeclared TRAIN-held quality
gate** despite lower guarded training cost. Both actual native training arms
and independent saved-checkpoint/packed-score replay completed successfully.
No1,000-update extension, recipe/threshold rescue or official evaluation follows.
The full production joint quality-and-speed goal remains active and unmet.

Frozen implementation `1ae0d29f`, [gate](inshop_pe_augmented_training_gate_2026-09-28.md).
DGX Spark user unit `sfora-pe-augmented-100-v2`, invocation
`603d0b05235b4a188109f9a8a797fddc`, original timed-process exit0.
Matched source/system contrast: original FP32 SigLIP2 Large/256 control versus
native PE-Core-B16/224, qualified FP16/scaler128, half-prefix freezes, each
own full fit-only PCA128/proxies/detached bank. Same loss/optimizer, fixed first100
of1000 seed179032 balanced batches, same augmented RGB before native processing.
Official In-Shop TRAIN-fit13,283 images/2,004 products initializes each arm;
updates cover6,192 distinct images/1,600 products/6,400 presentations. Nine
singleton-containing batches disable rank loss in both arms; CE/refresh still
run. No source weights, image ordinals, loss, precision, caps or floors changed
in response to measured results.

Evaluation: **official In-Shop TRAIN-held identities**,12,599 images/1,993
products, fixed asymmetric **6,354 queries / 6,245 gallery**, unaugmented native
inputs. Packed signed-int8 codes and f16 inverse norms, inherited stable score
ordering, R@1 and AP truncated at each query's gallery-positive count R. These
are exploratory TRAIN results, not official14,218-query/12,612-gallery results,
SOP, CUB/Cars transfer, or a deployed image-to-top-k test.

| Verified measure | Large/256 control | PE-Core-B16/224 |
|---|---:|---:|
| Packed held R@1 |95.6720176%|95.0739692%|
| Correct queries /6,354 |6,079|6,041|
| Packed held mAP@R |78.6237120%|76.3915922%|
| Guarded step median3–100 |897.1212 ms|559.6408 ms|
| All100 guarded-step throughput |70.8882 images/s|112.4694 images/s|
| Sum guarded training-step time |90.2830 s|56.9044 s|
| Sum CPU input time |6.7435 s|14.3202 s|
| Allocated CUDA peak |12,938,073,088 B|5,222,738,944 B|
| CUDA allocated after arm cleanup |0 B|0 B|
| Arm time, including initialization/checkpoint/held inference/score |223.4717 s|112.3440 s|

PE/control median-step ratio **0.6238184584** passes≤0.8. Guarded times include
file rehash, decode, RGB augmentation, native preprocessing, encoder/backward,
optimizer/scaler/finite checks and bank refresh. CPU input is measured separately;
remaining time includes GPU work and CPU bookkeeping, not pure GPU time.
The parameter-dependent finite checks favour the smaller model; serial Large→PE
order and warm file cache are limitations. These are single-seed training costs,
not model-only throughput, QPS, full image-to-top-k latency, p95/p99 or a product
speed claim. Whole original process **343.76 s**, cumulative peak RSS
**3,566,116 KiB**; in-script post-startup whole **337.7960 s**. Native600s/8GiB
host/shared-lock envelope enforced; per-step/eval CUDA guards passed.

| PE minus control, percentage points | Point delta | Paired product bootstrap95% CI | Paired query bootstrap95% CI |
|---|---:|---:|---:|
| R@1 |-0.5980485|[-1.1278284,-0.0628923]|[-1.1020617,-0.1259049]|
| mAP@R |-2.2321198|[-2.8561202,-1.5916740]|[-2.6998124,-1.7773005]|

Both negative point deltas exceed the frozen allowed drops (R@1 0.5pp,
mAP@R 1pp), so STOP. Both product intervals exclude zero; mAP harm is clear at
this checkpoint. These5,000-resample fixed-seed intervals describe query/product
sampling for this one trained pair, not variation across independently trained
seeds. The point-rule decision is not relaxed using its interval. This half-epoch
prefix mostly reflects the starting sources as well as learning; no step-0 held
forward was run, so the experiment does not isolate training-induced change or
prove that PE cannot work under every other recipe.

All100 scales128, no skipped steps; first/last complete gradient inventories,
finite parameters/Adam/bank, frozen native prefix/foreign rotary/grid, every
trainable group changed, same100 augmented RGB hashes, calibration floors.999,
resource and0B cleanup checks passed. The actual saved checkpoint/group/bank
and score audit is independent of the GPU process: original source reload,
strict final state load, initial/final frozen/group digest reconstruction,
finite normalized13,283-row bank/2,004 proxies, complete grad/scaler/cost checks,
actual held-vector SHA/shape/unit checks, CPU replay of all per-query packed
hits/AP, bootstrap intervals and STOP decision. Audit **12.66 s**, cumulative
peak RSS **3,760,552 KiB**, exit0, invocation
`6789a627688b440a92a29902158e2665`. It replays the captured precision values;
it does not recompute image inference or verify a native search kernel. No
kernel was changed, and this gate used the inherited packed-score evaluator.

CPU preflight v2 passed12.83s/3,336,728KiB, initializers byte-identical to v1.
Combined Opus/Astra review and its execution-origin, model-authentication,
partial-artifact and external resource/lock safeguards were closed before launch.
The fixed external preflight hash and executing code are recorded. Checkpoints
(about1.2GiB/366MiB) and held matrices remain on DGX at
`/home/riomus/runs/sfora-pe-augmented-100-v2/`. Raw per-arm receipts, exact
per-query hits/AP, gradients/timings/inputs, terminal journal/cost, review,
preflight/startup/launcher and independent audit are archived in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/pe-augmented-100-v1/`.
Receipt SHA`d02a1c176f023e81bd9a0bb526ff3294e677e88dcc0724b9306e1ca884177dbb`.
Run `audit_inshop_pe_pair.py` with both trusted external receipt/preflight SHA
arguments and recorded root/cache/output/model/dataset paths, CUDA hidden.
All jobs and reviews terminal; DGX idle, protected Rust unchanged/unpromoted.

Next: diagnose the responsible quality layer using a fixed CPU counterfactual
on the authenticated complete fit caches: native1024-D source versus its own
frozen PCA128 projection, identical fit query/gallery roles and packed scorer,
plus retained variance. This changes projection only and needs no new image
inference or optimizer. It is a fit diagnostic, not unseen held generalization,
and cannot reopen the failed100-update recipe or license a longer run.
Use the result to choose one causally distinct next action; preserve this STOP.
The dated published references and older official/held results remain unchanged;
no newer global frontier or production quality/speed requirement is satisfied.
No operator decision or credential is needed for that CPU diagnosis.
