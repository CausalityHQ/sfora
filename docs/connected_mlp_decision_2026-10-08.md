# Connected MLP decision — 2026-10-08

The production joint quality-and-speed goal remains unmet. Quality and training numbers below are verified measurements from the linked original receipts. Prospective gates and runtime estimates are labelled separately.

In-Shop previously exposed TRAIN selection: seed179061, 1734 query images, 1715 gallery images, 498 products. This is not the official query/gallery protocol.

| Model | Packed R@1 (%) | mAP@R (%) |
|---|---:|---:|
| Frozen source | 96.309112 | 80.572295 |
| Accepted concat baseline | 96.482122 | 81.777540 |
| Fresh connected control | 96.770473 | 82.765722 |
| Fresh connected candidate | 97.116494 | 85.151734 |

Candidate minus control: +0.346021 percentage points R@1 and +2.386012 points mAP@R. [Original summary](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-first-selection-score-v2/summary.json), [parent verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-first-selection-score-v2/verification.json). Decision **CONTINUE**; no first-stage confidence interval. Both-seed full selection and product-bootstrap GO remain pending. Sealed validation is ineligible until that GO.

| Engineering measurement | Control / baseline | Candidate / observation | Decision |
|---|---:|---:|---|
| Seed179069 fresh training, whole service (s) | 2188.712 | 2345.673 | Candidate 7.171% slower; admissible cost, no speed win |
| Seed179069 update-core ratio | 1.000000 | 1.059007 | Candidate 5.901% slower |
| Full evaluator CPU-v1, service (s) | Frozen cap 500 | 500.114 | FAIL_TIMEOUT; no qualifying receipt |
| Authority-only observation-v2, wall (s) | No matched speed baseline | 563.699274 | Completed engineering observation; no CPU qualification |
| Repaired authority-only observation-v4, wall (s) | v2: 563.699274 | 409.488395 | 27.357% lower engineering admission time; no product speed or CPU qualification |

Seed179069 training host peak 6103474176 bytes; CUDA allocation peak 2349298688 bytes; zero memory events and swap. [Original candidate verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-candidate-179069-v1/verification.json). Both seed pairs have accepted original training receipts; no retraining or state reuse is needed for the next evaluation gate.

[Original CPU failure](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1/failure.json) remains a failure. [Authority observation](evidence/compact_metric/sop-siglip2-substrate-v1/connected-full-authority-observation-v2/result.json) imported no native roots, peaked at644960256 host bytes, and had zero events/swap. Endpoint-admission callers account for58.3444% of samples; this is not exact input-loop time. The frozen 500-second CPU gate is not rescued by this observation.

The evaluator-owned endpoint admission correction is integrated and pushed: four fresh independent original readers, authenticated transitive callbacks and immutable builtin baselines, complete reader-state preservation, and unchanged terminal predicates. The independent Opus/Astra findings and resolved falsifiers are recorded in the [release decision](evidence/compact_metric/sop-siglip2-substrate-v1/connected-endpoint-reader-review-v1/decision.md). No native speed benefit is claimed yet.

The full affected source-only suite passed under the pinned DGX Python3.13.9: service20.229s, process peak135840KiB, zero swap and memory events. It constructed no model and read no new quality data. [Original receipt and verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-py313-assurance-v3/verification.json). The preceding launcher failed before tests because its runner path was relative to systemd’s working directory; [that failure remains preserved](evidence/compact_metric/sop-siglip2-substrate-v1/connected-py313-assurance-v1/verification.json).

Corrected-source authority observation-v3 failed after188.030s, before native execution, on a registry/origin authentication check; hostpeak589524992B, events0/swap0. Its original receipt is preserved. An executed original-constructor reproduction established that the initializer intentionally omits registry insertion. The finite initializer guard now authenticates that exact original loader and owner chain while preserving the ordinary registry checks. Root falsifiers exposed and repaired two delegate-replacement bypasses before release; the final affected worker suite and pinned-interpreter suite passed. This is not an admission-time speed measurement or CPU qualification. [Failure and causal decision](evidence/compact_metric/sop-siglip2-substrate-v1/connected-full-authority-observation-v3/decision.md).

The repaired exact source-v7 is frozen. Its original authority observation-v4 completed normally: authority409.488395s, service410.152s, hostpeak662966272B, processRSS435156KiB, events0/swap0 and no native imports. Invocation `4b935d2c7b204e63ae1155768f985e77`; original SSH92650 exited0. [Original result and raw evidence](evidence/compact_metric/sop-siglip2-substrate-v1/connected-full-authority-observation-v4/result.json). This is a change across authenticated engineering source versions, not a matched serving benchmark or qualification.

The original first CPU-v5 log measured 41.631175s after admission before exit and76.628351s in exit checks, for two endpoints. Adding those historical remaining costs to the new authority gives a planning estimate527.747921s, above CPU500. The historical admitted marker follows native_start, so that estimate also omits native import/preparation before the marker. Its shared-source/cache preparation intervals alone were25.145242s and2.470563s. The full four-endpoint remaining cost is unmeasured; these planning sums are neither lower bounds nor new runtime results. Hold the unused full CPU-v3 launch while assessing the remaining critical path rather than launching another likely timeout.

The completed [critical-path review](evidence/compact_metric/sop-siglip2-substrate-v1/connected-bootstrap-admission-plan-v1/review.json) found three serial prerequisite scans inside the evaluator’s trainer bootstrap, before its existing batched endpoint reader is created. A bounded source-only implementation will route only those two fixed call sites (CPUv6 plus both mechanics061) through that reader. Original trainer bytes, earlier historical CPU admission, actual-gradient admission and all predicates stay unchanged; savings remain unmeasured.

Ordered gates: falsify that bootstrap dispatch, independently review and freeze actual source, measure its admission path, and qualify it; then one eligible full CPU500 qualification; four independent GPU exports1500 in frozen endpoint order; full paired selection score700 with unchanged bootstrap/floors/cost/parity; only GO admits sealed validation. A quality survivor must then pass matched public B1/B32 decode-to-top-k latency, including10000 interleaved paired calls plus confidence interval for a p99 claim. Current qualification B32 timings are not product latency.

This experiment supplies no new SOP official TEST result, no official In-Shop confirmation, and no new CUB/Cars transfer or matched serving speed win. Historical exploratory results remain historical; the production target is not redefined around this TRAIN panel.
