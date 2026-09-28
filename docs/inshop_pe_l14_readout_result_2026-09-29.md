# Frozen L14 supervised readout: terminal training-cost KILL

The sole native17-update mechanics run completed all updates and failed the
unchanged cost ceiling. No100-update pilot, checkpoint, held image read,
retrieval quality or public latency ran. Discarded mechanics state cannot be
continued. This closes the serial-input execution procedure; the complete
SOP/In-Shop production quality-and-speed goal remains active and unmet.

| Dataset / split | Historical controls R@1 / mAP@R (%) | Candidate R@1 / mAP@R | Training cost | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| In-Shop TRAIN-fit13,283 images/2,004 products; first17B64 updates. TRAIN-held6,354q/6,245g excluded | densePE95.0739692/76.3915922; Large95.6720176/78.6237120, prior verified100-update results | Unmeasured | Verified median3–17 **0.771269893s**,82.980 guarded images/s; ceiling0.71769696s FAIL;26.13s whole;4,071,400KiB processRSS/no swap;3,052,520,960B allocatedCUDA peak | Unmeasured | Cost exceeds ceiling by0.053572933s/update,7.46456%; next one bounded CPU-input prefetch execution procedure, exact same learned method and gates |

Original GPU session22201/invb0efff4dacc645ad9c36e97e7f93ef0f collected
exit0 with a valid scientific KILL. Both lifetime GPU locks and120s/8GiB/no
swap/<10GB bounds applied. All17RGB hashes match original densePE control;
scales128/no skips; all source/foreign tensors stayed exactly unchanged;
all three readout/proxy data gradients were finite and positive at first/last
steps, and both head/proxy groups moved. Native FP16/FP32 and cached/fresh
source cosines passed. Actual decode/augmentation/hash/native pixels/native
source forward/loss/backward/optimizer/integrity work remained inside cost.
Updated checkpoint parity is unmeasured because the cost gate stopped first.

Independent stdlib receipt/median/RGB/scales/source/group movement/gradient/
resource/terminal replay passed. ReceiptSHA
118910e8653e1ab177020bad8a06cc46d7ba36a7789f2e1297def8e34540326a;
CPU authoritySHA82c68382ec0e73598756553e78795d7d20b533e6bfcf9b440a77176593474c10.
The earlier informal0.760s log estimate is superseded by the exact receipt.

## Narrow input-stage attribution and one next execution intervention

No GPU stage timing isolates the encoder/loss share. Two CPU-only diagnostics
used actual frozen augmented images and native processor, zero optimizer or
held/quality reads. Original47773/inv770d4253004643398cb316b50b88c996 exited0
in7.22s: same warmed firstB64 pixel preparation median0.032949352s; tensor
conversion/normalization/stack0.013974732s. This alone is below the cost gap;
do not implement batch normalization as the proposed rescue.

Original24626/inv03e6b923a8dd4839a796fda6322d5bd5 exited0 in9.82s with
1,164,432KiB RSS/no swap. Each actual first17 batch was decoded/augmented/
hashed/preprocessed once, with exact originalRGB and first-pixel authority.
Median3–17 complete CPU input0.130837592s; decode/augmentation/hash
0.041126844s and native pixels0.089923660s. Separate medians need not sum.
These are CPU diagnostic measurements, not concurrent training timings or
latency/quality gains. The fresh-batch and warmed same-batch measurements
have different workloads; do not substitute one for the other.

Select ONE execution intervention: prepare the next actualB64 input in one
bounded CPU worker while the current GPU update runs. Reuse the unchanged
native processor and augmentation helper, verify all files/RGB/pixels/strides,
keep at most one pending batch, and propagate worker exceptions. No lookup
table, precision/thread/source/optimizer/LR/prefix/loss/exposure change or
offline feature cache. This addresses measured serial input work without
claiming it caused all slowdown. Its capacity to close the gap is UNKNOWN;
CPU/GPU contention and overlap must be measured.

Before GPU, qualify worker-versus-serial exact pixels/strides/RGB for all17
actual batches, unchanged caller CPU RNG state, no CUDA in worker, bounded
queue, and actual source/role authority unchanged. Pin new executing code and
qualification receipt in an isolated root without modifying qualified files.
Then ONE17mechanics under original120s/8GiB/no swap/<10GB and0.71769696s
median gate, including first fill/last drain in reported total training wall.
Measure actual source forward every update; no cached head-only timing.
Only PASS allows discarded17state followed by ONE fresh100TRAIN pilot300s,
strict native updated serving reload/full-held packed parity, original
quality floors and bootstrap bounds. A survivor advances to updated serving,
fresh confirmation and full matched public latency; no endless TRAIN extension.
No new prefetch code or GPU job has launched at this checkpoint. DGX idle.

[Frozen learning and quality gates](inshop_pe_l14_readout_gate_2026-09-28.md),
[terminal receipt](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-readout-v1/mechanics/receipt.json),
[training evidence](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-readout-v1/mechanics/training.json),
[complete CPU input profile](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-readout-v1/input-profile/readout-input-profile-v1.json),
[pixel stage profile](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-readout-v1/pixel-profile/readout-pixel-profile-v1.json),
[stdlib replay](../scripts/check_pe_l14_readout_cost.py).
