# Frozen L14 compact readout: measured TRAIN quality KILL

The bounded-input execution passed CPU qualification and17mechanics, then
completed ONE fresh100-update pilot with actual native strict-loaded full
TRAIN-held inference. Cost and serving parity passed; both quality floors
and both paired improvement bounds failed. This closes the fixed frozen
readout learning procedure without more epochs, LR/prefix/precision/source
retuning or official evaluation. The full SOP/InShop production goal stays
active and unmet. No model was promoted.

| Dataset / split | Historical matched controls R@1 / mAP@R (%) | Candidate R@1 / mAP@R (%) | Training cost | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| In-Shop TRAIN-fit13,283 images/2,004 products; TRAIN-held6,354 queries/6,245 gallery,1,993 products; exploratory repeated development | densePE95.0739692/76.3915922; Large95.6720176/78.6237120, prior verified100-update controls, not concurrent reruns | **87.2836009 / 61.8878907**, verified GPU packed score and independent saved-state CPU replay | Verified median3–100 **0.636208809s**,100.596 images/s by median;100updates including fill/drain **63.879182s**,100.189 images/s by wall;262.87s whole process;5,611,528KiB RSS/no swap;3,810,310,656B peak allocatedCUDA including reload/held | Unmeasured | −7.7903683 R1pp/−14.5037016 MAPpp versus densePE; next one native attention-pool adaptation with frozen transformer trunk, unchanged gates |

Original100 GPU session86711 collectedexit0, application255.8171985s, whole
262.87s, original299s+1s/8GiB/no swap/<10GB unit and both lifetime locks.
All100 actual augmentedRGB match archived control, scales128/no skips;
whole native/foreign encoder unchanged; all three readout/proxy gradients
finite/positive and groups moved. Training time includes worker fill/drain;
actual next-image preparation overlaps actual native source forward, no
precomputed feature substitute.17mechanics state was discarded before100.
The17 losses and final head/proxy hashes were EXACTLY equal to serial17,
so that paired execution change preserved this observed training trajectory.

The updated native encoder AND normalized head outputs matched the strict
reload exactly on actual lastB64. The strict-loaded native model encoded all
12,599 held images; live and loaded head outputs matched bitwise on every
native feature after whole immutable source identity. Packed per-query hit/AP
parity passed. Checkpoint and held matrix retained on DGX; no official,
transfer or public image-to-result latency read.

Paired5000-resample product95 intervals versus densePE, in percentage points:
R1 **[−8.6087504,−6.9756615]**, mAP **[−15.3491809,−13.6377238]**.
CPU query95 intervals: R1[−8.5143217,−7.0821530],
mAP[−15.1031434,−13.9116159]. All interval limits are negative. These are
query/product uncertainty for one exploratory seed and reused historical
control, not seed/method-selection uncertainty or an official generalization
claim. Quality floors95.1720176/77.6237120% both failed.

## Independent replay and responsible-layer evidence

CPU saved-state/source/foreign/head/proxy/bank/norm/100RGB/scaler/cost/packed
score/product/query interval/decision audit passed, original98442,12.17s,
3,674,508KiB RSS/no swap/CUDA hidden. First CPU attempt81028 failed6.90s
on an auditor schema assumption: final receipt omits preflightSHA, which is
recorded in attempt/training. The corrected audit binds those records to
the exact original authority. Failed log/source retained; no native job,
checkpoint, scientific threshold or score changed.

One CPU diagnostic on the same saved128-D vectors, zero native inference/
optimizer updates, measured float R1 **87.4409821%**, mAP **61.9656245%**.
Float minus packed **+0.1573812/+0.0777339pp**. Float also fails both
floors substantially; compression is insufficient to explain the deficit.
The fixed frozen-source/affine-readout allocation failed. This does not
identify whether nonlinear pooling adaptation or transformer updates are
necessary, nor prove an architectural capacity limit.

## One next useful learned-model intervention

Select **adaptation of the existing native L14 attention pool**, retaining
the original pretrained transformer trunk/positions/rotary/output projection
and native architecture. Train its existing query/attention/MLP/norm tensors
plus the same compact head/proxies; retain original vision1e-5 and
head/proxy1e-4 AdamW, loss/data/native224/precision/exposure and quality/cost
floors. No new module, token representation, input scale, prefix-depth sweep,
cached head-only timing or rescue of the closed frozen-readout arm.

This addresses the observed failure of fixed pooled features with a learned
linear metric by allowing native nonlinear token aggregation. Freezing the
expensive transformer trunk preserves the measured native forward path;
added pooling backward cost/memory and quality are UNKNOWN until measured.
This is a selected next intervention, not a claimed gain or launched job.
CPU role/source/actual native pooling gradients and exact parity first;
one17discard mechanics120s, one fresh100TRAINpilot300s only after all gates.
For full-held strict serving parity, freeze the immutable-trunk identity and
compare the live versus strict-loaded updated pool on EVERY same actual
native token input, plus exact final readouts. Do not silently reuse the
previous all-encoder-frozen parity argument when pool tensors now change.
Freeze the concrete native hook/parity procedure before GPU execution.
Survivor advances to updated serving and fresh confirmation, not more TRAIN
development. This next procedure is not implemented or launched yet.

ReceiptSHA0710914c9184eecfa47cfab8433126fb20fcafd8b78b751bbe3be6e82b344f7a;
checkpointSHAc380b1a68eea8536c2fccca681de31e07f7ca983f548bdbc0d7e0d9d7f0db09b;
heldSHA942c220fdc95342eba29504d9914c95d3a2a6d64b1c28d592f52e72c82c91850.
DGX idle after terminal collection. Protected Rust changes untouched.

[Frozen gate](inshop_pe_l14_prefetch_gate_2026-09-29.md),
[GPU receipt](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-prefetch-v1/pilot/receipt.json),
[training evidence](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-prefetch-v1/pilot/training.json),
[CPU audit](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-prefetch-v1/pilot/cpu-audit.json),
[float diagnostic](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-prefetch-v1/pilot/float-quality-diagnostic.json).
