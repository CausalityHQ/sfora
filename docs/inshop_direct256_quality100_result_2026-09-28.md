# Direct256 actual-training quality: positive recovery, failed execution gate

Frozen source `12a1931d`; original sole DGX unit
`sfora-inshop-direct-width-quality100-v1`, invocation
`1bce9c592ebf4ba68c99bf0badc493d2`, terminal MainPID0/exit1/Resultexit-code.
Whole395.257988seconds, within600-second cap. Both fresh models completed
100×64 BF16 updates and nativeFP16 holdout export. No retry or budget extension.
Original failed receipt SHA
`4a0c2dd476fd151d26ceedf4fa823dc3753c64ccea9a2f1e70f5a3dc08c00c2b`.

## Measured result and decision

In-Shop official TRAIN only: original fit13283images/2004products, disjoint
outer held12599images/1993products, fixed6354queries/6245gallery. Seed179024,
SigLIP2 Large24/256, freeze first12blocks/embeddings, same native objective,
same100pixel/target hashes and leading128 PCA initialization. Candidate retains
256 dimensions; matched control128. NativeFP16 vision/FP32 head, TF32off,
batch32 export. This is a previously observed TRAIN panel, not official quality.

| Verified saved-tensor measurement | Native128 | Direct256 |
|---|---:|---:|
| Packed asymmetric Recall@1 (%) | 95.231350 | 96.033994 |
| Packed asymmetric mAP@R (0–1) | 0.77995431 | 0.79678527 |
| Training wall, excludes bank init (s) | 86.229278 | 86.311176 |
| Presentations/s, training wall only | 74.220731 | 74.150305 |
| Median training step (s) | 0.839991 | 0.843632 |
| Peak training CUDA allocated (bytes) | 12,988,997,120 | 13,002,782,208 |
| Training+bank init wall (s) | 87.092033 | Not recorded |
| Full image-to-top-k p50/p95/p99/QPS | Unmeasured | Unmeasured |

R@1 delta **+0.802644pp**, paired product-bootstrap95% interval
**[+0.466121,+1.161145]pp**. mAP@R delta **+0.01683096**, interval
**[+0.01410713,+0.01959777]** (5000draws, seed179019). Both declared quality
floors pass. This single seed is exploratory, not a multi-seed claim or a
comparison to UNICOM's different official split. Capacity change is known,
not proof of a novel learning algorithm. More dimensions double packed codes:
130→258bytes/row; full serving cost remains unmeasured.

## Failure and concrete production fix

After saving the candidate checkpoint and exported held_values, the auxiliary
legacy symmetric `score_packed_full_gallery` rejected256 because its geometry
guard required128. The parent therefore never received the candidate final
receipt. This is an integration bug, not a negative quality measurement.

Fix the shared scorer with an explicit `output_dim=128` default, accepting
only128/256; direct trainer passes its authorized width. Existing callers retain
128. A CPU regression compares native128 and zero-padded256 scores, requiring
exact full result equality/R1=1/AP=1. Also persist direct training/reload guards
before held export, so a later evaluation failure cannot lose them again.
Authority/export/timeout/scorer regression exit0; compilation/diff checks pass.
No production public256 format/kernel/default is enabled by this fix.

## Independent recovery and limits

CPU recovery scored actual saved256 embeddings without another encoder run
or training,1.874675s before CI calculation. Both arms' actual saved tensors
were then independently repacked and exhaustively scored using NumPy float32
integer-exact dot products, declared norm multiplication order and stable ties.
All6354 top1 outcomes per arm match. Max AP discrepancies1.073e-7/9.934e-8;
repacked codes/norms exactly match. Independent replay8.030054s. These are
verification costs, not serving latency. Original receipt remains immutable;
the recovered diagnostic is separately named and cannot pass the failed gate.

Candidate raw checkpoint training evidence independently confirms100 finite
losses/positive vision, head and classifier gradient norms; all100 input hashes,
fit rows and all20 loaded source files match control and frozen Git source.
Measured median-step/peak-memory/training-wall-excluding-bank guards pass.
Candidate final geometry, frozen/trainable hashes, returned private-reload
checks and bank initialization wall were lost with the missing final receipt.
Source execution reached export after those checks, but this is not a complete
durable gate attestation. Do not fabricate missing fields or relabel original
execution as GO. **Decision: positive recovered quality; execution FAIL;
retain production128.**

Both FP32 checkpoints/private fixtures and original exports remain on DGX.
Native checkpoint SHA
`fe56d0941618a0fad5d3395dd48d1988edc2d9a496a0329b57bde4160576d2a1`;
candidate
`ebccc08788a15b6b9536310dc2516ee0f8aa275ae6465b19ba6ecd17927bd811`.
Raw/recovery/audit/verifier artifacts are in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-width-quality100-v1`.
Actual float/packed arrays stay at
`/tmp/sfora-inshop-direct-width-quality100-v1/{128,256}` and original DGX root.

## Ordered next product gates

1. Push scorer fix, regression, pre-export durable guard checkpoint and these
   unchanged original/recovered receipts. No repeat training to fill bookkeeping.
2. Freeze a bounded saved-checkpoint audit that independently tests missing
   frozen/trainable/finite-state and exact private reload properties. New audit
   must name any replacement geometry panel explicitly; it cannot retroactively
   certify the original gate. Preserve original source/weights/tensors.
3. If the recovered candidate remains qualified, design the smallest optional
   native/public256 path with exact tie/score parity and a cheap matched serving
   pilot before long confirmation. Current product is128; a wider wire format
   is consequential and requires review. Do not launch new loss/model search.
4. Only promising quality+serving evidence warrants paired-seed confirmation
   and later independent SOP/In-Shop/transfer gates. No SOTA claim from this
   previously observed TRAIN panel. No operator decision is needed now.
