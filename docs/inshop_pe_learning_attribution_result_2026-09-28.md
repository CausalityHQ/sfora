# Saved-state held attribution

The quality reversal emerges during encoder adaptation in this matched recipe.
PE starts ahead of Large on the existing In-Shop TRAIN-held split but ends
behind after 100 updates, including when scored directly at raw1024 without a
compact head. Source starting quality and PCA compression do not account for
the observed reversal. This locates a responsible layer; it does not identify
optimization versus capacity, or license extending the stopped training arm.

All measured values below use the same 12,599 held images / 1,993 TRAIN products,
6,354 queries and 6,245 gallery images, unaugmented native224/256 inputs, and
inherited int8/f16 packed emulation with stable ties. The eight fixed states
combine original/final encoder (including native pooling and projection) with
the fit-initialized/final 128-D head. No fitting, optimizer, checkpoint selection,
official evaluation or production serving measurement occurred.

| Encoder/head state | Large R@1 (%) | PE R@1 (%) | Large mAP@R (%) | PE mAP@R (%) |
|---|---:|---:|---:|---:|
| Original / initial (00) | 75.5429651 | 80.7994964 | 48.0553886 | 52.3273234 |
| Original / final (01) | 76.3141328 | 81.9326409 | 49.0546840 | 53.8161140 |
| Final / initial (10) | 95.6877557 | 95.0267548 | 78.3835023 | 76.0781527 |
| Final / final (11) | 95.6720176 | 95.0739692 | 78.6237120 | 76.3915922 |

Primary paired product-bootstrap 95% intervals, in percentage points:

| Contrast | R@1 delta [95% interval] | mAP@R delta [95% interval] |
|---|---:|---:|
| PE−Large initial00 | +5.25653 [4.18763, 6.32975] | +4.27193 [3.49547, 5.07178] |
| Large total11−00 | +20.12905 [18.90631, 21.42284] | +30.56832 [29.53418, 31.57848] |
| PE total11−00 | +14.27447 [13.19756, 15.32117] | +24.06427 [23.12655, 24.98851] |
| PE−Large total change | −5.85458 [−7.04296, −4.69475] | −6.50405 [−7.38171, −5.65497] |

Readout-only changes at encoder0 improve PE by 1.13314 R@1 points and Large by
0.77117. At encoder100, the head change is only +0.04721 for PE and −0.01574 for
Large. This supports inspecting encoder adaptation rather than another head
sweep. Hybrid components are coadapted: these are parameter substitutions,
not training-causal interventions. Interaction terms and 32 marginal intervals
are descriptive, without multiplicity correction. Query-bootstrap intervals
and every per-query value are retained in raw receipts.

The predeclared secondary raw1024 read removes the compact-head mismatch:

| Raw encoder | Large R@1 (%) | PE R@1 (%) | Large mAP@R (%) | PE mAP@R (%) |
|---|---:|---:|---:|---:|
| Original | 79.4302801 | 83.2861190 | 50.8816485 | 54.1461241 |
| Final100 | 95.9395656 | 95.2943028 | 78.6055210 | 76.4724845 |

PE−Large raw R@1 moves from +3.85584 points ([2.85115, 4.87191]) to −0.64526
([−1.15348, −0.12787]); its relative learning gain is −4.50110
([−5.59075, −3.44611]). Raw mAP@R moves from +3.26448 to −2.13304, with a
relative change of −5.39751 ([−6.21184, −4.56423]). These are secondary TRAIN
diagnostics, not proof of a new model's generalization or a serving claim.

The sole GPU service `sfora-pe-attribution-gpu-v1`, invocation
`bc9d7b1ef18d402a93a71b44af761b98`, exited 0 in 370.88 seconds with
4,824,380 KiB peak host RSS (internal 363.76638s). Encoder-state wall costs were
119.141/119.905s for Large original/final and 54.252/53.476s for PE; these include
loads, hashing, preprocessing, inference, export and state audit, not serving
latency. Peak allocated CUDA was 1,593,080,832 bytes for Large and 583,852,544
for PE; cleanup was zero after every state. All four FP16/FP32 calibrations,
immutable registered/foreign rotary/grid state digests, source/code/input
authority, and resource caps passed.

Final/final vectors reproduced the saved training exports with minimum cosine
0.9999999999999998 for both architectures; every packed query hit/AP matched
within 1e-6. Tiny CPU/CUDA mAP reduction differences remain far below that floor.
Local stdlib replay verified every per-query decomposition and contrast mean
and all eight quality files. The secondary CPU service
`sfora-pe-attribution-raw-v1`, invocation `a15645d0300a499e91dad33fd49d4af4`,
rechecked all four matrix SHAs and final golden vectors/scores before raw scoring.
It exited 0 in 8.80 seconds with 1,385,904 KiB peak host RSS (internal 6.07845s).
Its original cost log and numeric receipt are archived separately. No native
kernel output parity or independent full image re-encoding is claimed.

[Fixed gate and interpretation limits](inshop_pe_learning_attribution_gate_2026-09-28.md).
Raw [main evidence](evidence/compact_metric/sop-siglip2-substrate-v1/pe-learning-attribution-v1/)
and [secondary evidence](evidence/compact_metric/sop-siglip2-substrate-v1/pe-raw-attribution-v1/).
The next intervention must change the encoder-adaptation mechanism, be bounded
and distinct from closed loss/head/source/crop/teacher arms. No automatic hybrid
selection or 1,000-update continuation. The full production goal remains active
and unmet; published frontier, official confirmation, transfer and matched
end-to-end p99/QPS gates are unchanged.
