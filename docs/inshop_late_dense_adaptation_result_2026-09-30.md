# Late dense paired100 terminal result — 2026-09-30

**KILL for this frozen procedure.** Integrity, resources, native packed parity and matched training cost passed; quality failed in both continuation-input seeds. No full application or official read is admitted by this result. The overall production goal remains active, and existing useful Pareto candidates remain preserved.

In-Shop previously observed TRAIN-held: fit13,283 images/2,004 products; held12,599/1,993, query6,354/gallery6,245. Both arms start from the same learned native128 source and reset optimizer/scaler. Candidate opens blocks10–11; control freezes embeddings and blocks0–11 while training blocks12–23, postLN/pool/head and proxies. These are conditional continuation seeds, not independent pretraining replicates. All numbers below are verified packed-score measurements; R1/mAP@R are percent.

| Seed | Control R1 / mAP@R (%) | Candidate R1 / mAP@R (%) | Control / candidate training wall |
|---|---|---|---|
| 179032 | 97.812402 / 85.487008 | 97.623544 / 85.205298 | 116.562 / 125.814 s |
| 179041 | 97.765187 / 85.541657 | 97.544854 / 85.247049 | 116.654 / 126.561 s |

Equal-seed mean candidate-minus-control R1: −0.204596pp, product95% CI [−0.344332, −0.070060]; mAP@R: −0.288159pp, product95% CI [−0.430583, −0.148850]. Query95% CIs are respectively [−0.346239, −0.062952] and [−0.418001, −0.161694]. Frozen5000 shared draws/seed179019. Both positive +0.20pp floors and each-seed signs fail. Training wall ratios1.079369/1.084922 and median-step ratios1.081660/1.084211 pass≤1.50. No public latency measured.

All four CPU qualifications and GPU exports terminated normally; each full12,599-image independent whole/head/packed replay passed. Export service durations232.527/233.299/232.215/232.093s; CUDA allocated peak1,514,226,688bytes; no swap. Final CPU scorer original51145/invocationbd842a94f37c42dabbb886ee97dc7700 terminated0, internal26.345s, peak3,600,244KiB, no GPU/swap. Raw authority, receipts, logs and decision are in [late-dense evidence](evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/late-dense-decision-v4.json).

The earlier native CPU failure came from a full updated-resume mmap remaining live through fit transients. Releasing it after complete strict reload lowered actual peak without changing validation, caps or science. Failed roots/logs remain preserved; the admitted117 closure is immutable.

Next selected intervention: constrain representation drift during the same late adaptation by anchoring fit-image native128 descriptors to the frozen learned source. This directly addresses the demonstrated harm from unconstrained late adaptation; the precise mechanism is a hypothesis, not proved catastrophic forgetting. Superseded control-reuse suggestion: the new same-view source-preservation contract requires four fresh matched arms for current cost and quality; these archived controls remain historical context only. Freeze one source-anchor coefficient and stop rule before execution; no sweep, repeated negative late100, or official selection. First reconcile against the closed-method ledger to prevent rediscovery, then bounded independent planning/dual critique, narrow actual-constructor/mechanics admission, and one paired TRAIN pilot. No new job is authorized by this report alone.

Convergence row: In-Shop TRAIN-held, two seeds, control97.788794%R1/85.514333%mAP@R versus candidate97.584199%/85.226174%; training+7.94%/+8.49% wall; public latency unmeasured; quality−0.204596/−0.288159pp, fixed gateKILL; next decisive test source-anchored late adaptation after frozen admission. Full official In-Shop/SOP quality and matched image-to-top-k speed gaps remain unclosed.
