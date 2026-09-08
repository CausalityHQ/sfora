# UniCOM rank-finish confirmation and standard result

The fixed Smooth-AP rank-finishing method passed its three-seed train-fold
confirmation and its single preregistered official In-Shop standard readout.
All artifacts remain `claim_eligible=false`; this is release-candidate evidence,
not an unrestricted state-of-the-art claim.

## Three-seed confirmation

The baseline was 0.8975116742 mAP@R, 0.9861982434 Recall@1, and
0.9974905897 Recall@10. Epoch-8 results were:

| Seed | mAP@R | mAP@R delta | Recall@1 | Recall@10 | Gate |
|---:|---:|---:|---:|---:|:---|
| 0 | 0.9106583524 | +0.0131466781 | 0.9899623588 | 1.0000000000 | pass |
| 1 | 0.9119965211 | +0.0144848469 | 0.9912170640 | 1.0000000000 | pass |
| 2 | 0.9092679053 | +0.0117562311 | 0.9912170640 | 1.0000000000 | pass |

The independent seed-1/seed-2 confirmation mean delta was +0.0131205390,
above the frozen +0.010 gate. Every per-seed mAP@R and recall gate passed, so
the recomputed confirmation status was `CONFIRM`. The canonical confirmation
SHA-256 is
`5b8408d924019edb8b524e2bb0ea27294a245753ad7aa51f3a1d4db3e51f5fc9`.

## Official standard readout

The evaluator loaded the seed-1 inference artifact selected before the standard
readout and compared it with its authenticated parent checkpoint over the full
official In-Shop query/gallery partition.

| Metric | Parent baseline | Rank-finished | Delta |
|:---|---:|---:|---:|
| mAP@R | 0.7605731457 | 0.7760330990 | +0.0154599532 |
| Recall@1 | 0.9388802926 | 0.9447179631 | +0.0058376706 |
| Recall@10 | 0.9898719932 | 0.9912786609 | +0.0014066676 |
| Recall@20 | 0.9934589956 | 0.9950063300 | +0.0015473344 |
| Recall@30 | 0.9950766634 | 0.9963426642 | +0.0012660008 |

All frozen standard gates passed and the terminal status was `RELEASE`. The
canonical result is 370,088 bytes with SHA-256
`93ed2130fd1f8e8e03c84f3c9850d04f40f694b6702d8ae5eb86d50b3c23b911`.
The paired baseline/candidate evaluation took 676.333 seconds and reported
5,954,364,416 peak CUDA-allocated bytes.

## Authority repairs discovered before the accepted readout

Two failed attempts produced no result and are not quality evidence. The first
showed that the evaluator could authenticate a B/16 filename while the pinned
loader selected L/14 from the containing directory. Commit
`7f4192af0b349856679c68ee16ab5642ab0f06fa` now requires the exact
`FP16-ViT-L-14-336px.pt` identity. The second showed that the historical parent
stores its model state as an `OrderedDict`; commit
`54996d19cc83bd7ca90bc2778a7b1d8dedfe359a` accepts that authenticated mapping
while rejecting non-model payloads. The accepted readout authenticated the L/14
checkpoint at SHA-256
`3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`.

No training or evaluation process remains on the DGX after the terminal result;
memory PSI full avg10 was 0.00 at clearance.
