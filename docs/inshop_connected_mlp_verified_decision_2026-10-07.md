# Connected last-MLP: verified decision

Status: first-seed selection **CONTINUE**, both second-seed mechanics gates and both fresh second-seed TRAIN128 gates passed; the first full-stage CPU500 timed out and is ineligible for export. The production joint quality-and-speed objective remains unmet. All numbers below are measured and receipt-verified; no official result or speed result is inferred from TRAIN selection.

| Dataset / split / seed | Control | Candidate | Baseline | Decision and uncertainty |
|---|---:|---:|---:|---|
| In-Shop exposed TRAIN selection, 1734 queries / 1715 gallery / 498 products, 179061, packed R@1 | 96.770473% | 97.116494% | Original source 96.309112%; accepted CONCAT 96.482122% | +0.346021 percentage points vs control; CONTINUE only, no confidence interval in first gate |
| Same panel and seed, packed mAP@R | 82.765722% | 85.151734% | Original source 80.572295%; accepted CONCAT 81.777540% | +2.386012 percentage points vs control; previously exposed panel |
| In-Shop selection, paired seeds 179061 / 179069 | Pending | Pending | Same source / CONCAT replay required | Full same-four endpoint gate and product uncertainty pending |
| In-Shop sealed TRAIN VAL | Unread for this method | Unread for this method | Frozen source / CONCAT | Full selection GO required before opening |
| In-Shop official query/gallery | Unmeasured for this method | Unmeasured for this method | Protocol-matched published reference must be verified at confirmation | No official qualification |
| SOP official TEST | Unmeasured for this method | Unmeasured for this method | Protocol-matched published reference must be verified at confirmation | No SOP qualification from this experiment |
| CUB / Cars transfer TEST | Unmeasured for this method | Unmeasured for this method | Frozen matched control | No transfer qualification from this experiment |

Both arms use the same 6355-row / 1008-product CONTROL scope, genuine two-view pixels, seed schedule, B64/micro16, 128 updates and A/C readout. Candidate additionally updates exactly four layer-26 MLP tensors. Selection uses independently re-encoded query and gallery endpoints and actual packed scores.

| Measured cost on DGX Spark, seed 179061 | Control | Candidate | Interpretation |
|---|---:|---:|---|
| Whole fresh TRAIN service | 2207.053 s | 2295.117 s | Candidate 1.039901×, 3.99% slower |
| Training core including preprocessing, gallery work and integrity | 1703.061908 s | 1757.753537 s | Candidate 1.032114×, 3.21% slower |
| Query-view presentations / training-core second | 9.620320 | 9.320988 | Derived from actual 16,384 query-view presentations and measured core seconds |
| Query-view presentations / whole-service second | 7.423474 | 7.138634 | Same numerator; whole denominator includes admission, reload, qualification and exit |
| Peak CUDA allocated | 2,021,260,288 B | 2,351,303,168 B | Whole-run peaks; not device-resident-memory totals |
| Complete two-pass selection export service | 1264.694 s | 1289.351 s | Qualification workload, not public image-to-top-k latency |

Both fresh TRAIN ratios pass the frozen ≤1.50 admission threshold; neither demonstrates faster training. Throughput above counts 128 updates ×64 query rows ×2 actual pixel views, verified against every microbatch membership in both original receipts. It excludes extra reload/oracle images from the numerator and is not a raw encoder-kernel throughput measurement. Public decode-to-native-top-k B1/B32 p50/p95/p99, QPS, throughput and a matched end-to-end speed win remain unmeasured for these endpoints. A product p99 claim still requires 10,000 interleaved paired calls and a confidence interval.

The original first score exited normally in 617.915 s, with 1,667,022,848 B host peak, zero swap/events and complete source, archived per-query replay, wire, resource and uncached exit checks. Historical exports were authenticated through their original source owner; they were not relabelled as exports from the new evaluator source.

Second-seed control mechanics passed in 925.653 s: all17 versus independent8+9, strict updated reload and complete exit checks; host peak 6,155,317,248 B, CUDA peak 2,021,260,288 B, zero swap/events. Candidate mechanics passed normally in 943.903 s, invocation `cc546b2dcf2e4b9ea023c8b073ce893d`, with the same full checks; host peak 6,143,934,464 B, CUDA peak 2,349,907,968 B, zero swap/events. Both used unchanged 1200 s / 8 GiB / zero-swap / CUDA<10 GB / both-lock limits. No mechanics state is eligible for TRAIN reuse.

Fresh control069 TRAIN128 passed normally in 2188.712 s (training core 1693.385571 s), with host peak 6,110,826,496 B, CUDA peak 2,021,260,288 B and zero swap/events. All128 updates, exact first17 admitted mechanics, strict public reload and complete source exit were verified. This is engineering qualification, with no new quality read.

Fresh candidate069 TRAIN128 passed normally in 2345.673 s (core 1793.307852 s), with host peak 6,103,474,176 B, CUDA peak 2,349,298,688 B, zero swap/events and complete first17 mechanics replay, strict updated bundle reload and uncached exit. Relative to fresh control069, core ratio is 1.059007 and whole-service ratio 1.071714; both pass ≤1.50, while training is slower. No new quality values were read. Derived from every actual canonical/augmented microbatch membership (16,384 query-view presentations), control069 throughput is 9.675292 presentations/core-s and 7.485681 presentations/service-s; candidate069 is 9.136189 and 6.984776 respectively. These include the same core/whole cost definitions as seed061 and exclude extra oracle images from the numerator; they are not pure encoder throughput.

Next, in order: qualify the full evaluator CPU500 with ordered endpoints control061, candidate061, candidate069, control069 and the accepted first061 CONTINUE; run four fresh current-stage exports; apply the frozen same-four two-seed quality and product-LCB decision. Failure stops the affected arm. Full selection GO alone permits sealed VAL. No checkpoint state reuse, automatic cap rescue or official quality claim.

Independent serving research completed while native gates ran. Source inspection finds 3,423,104,512 parameter-occurrence bytes inspected per public request, but exclusive latency attribution remains unmeasured. The evaluator's 1941-file audit is not the public request loop. Existing fresh-copy batching lost its measured timing comparison (43.637 ms baseline / 45.224 ms proposed) despite parity; that arm stays KILL. After a quality survivor, the next serving gate is one discarded public-request attribution diagnostic preserving fresh-byte, packed-output, native-ID/tie and cleanup predicates.

Evidence:

- [First score and scientific summary](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-first-selection-score-v2/summary.json), with original receipt, log and parent verifier in the same directory.
- [Candidate TRAIN cost verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-candidate-179061-v1/verification.json) and [control verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-control-179061-v1/verification.json).
- [Receipt-derived training throughput accounting](evidence/compact_metric/sop-siglip2-substrate-v1/connected-training-throughput-accounting-v1.json).
- [Control069 mechanics verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-mechanics-control-179069-v1/verification.json) and [prospective candidate069 freeze](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-mechanics-candidate-179069-v1-freeze/mechanics-candidate-179069-v1-freeze.json).
- [Candidate069 mechanics verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-mechanics-candidate-179069-v1/verification.json) and [fresh control069 TRAIN freeze](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-control-179069-v1-freeze/train-control-179069-v1-freeze.json).
- [Control069 TRAIN verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-control-179069-v1/verification.json) and [prospective candidate069 TRAIN freeze](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-candidate-179069-v1-freeze/train-candidate-179069-v1-freeze.json).
- [Serving research decision](evidence/compact_metric/sop-siglip2-substrate-v1/connected-serving-critical-path-plan-v1/parent-decision.json).

- [Candidate069 TRAIN verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-candidate-179069-v1/verification.json) and [full-stage CPU prospective freeze](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1-freeze/full-cpu-v1-freeze.json).

- [Second-seed receipt-derived throughput accounting](evidence/compact_metric/sop-siglip2-substrate-v1/connected-training-throughput-accounting-seed179069-v1.json).

Full-stage CPU-v1 original invocation `4e74777ab12642969ad612b495cf326d` timed out at 500.114 s (frozen 500 s), host peak 1,571,721,216 B, zero memory events/swap. No receipt or admitted-progress marker was published. The exact admission/native-start subphase is unproven; footer command status0 does not override killed/TERM/timeout. No fresh full-stage GPU export or new held quality read was admitted. [Original failure evidence](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1/failure.json) is preserved. Next: bounded admission-path audit and one source-only correction or a precisely scoped diagnostic, then a new prospective qualification.
