# SOP true-freeze public serving TRAIN gate, frozen before full-query read

The strict offline-export parity gate failed for both precisions because the public query API uses batches of at most 32 while the archived gallery export used batches of 64. This gate measures the actual public path without changing the six training checkpoints or stored gallery. It is a diagnostic on the same reused TRAIN product holdout, not a new independent quality claim or a relaxation of the failed exact parity gate.

Use the already pinned SigLIP2 snapshot, SOP source archive SHA-256 `1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a`, and native scorer SHA-256 `39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`. For seeds 179024, 179026, 179027 in order, load each same-seed control then freeze via `Siglip2CompactIndex.from_artifacts`, pinned training decision and receipt, with the stored 59,551-row packed TRAIN gallery. Encode every one of the pre-existing 5,851 product-disjoint TRAIN holdout images through the public API in batches of 32. Score each query against the full gallery with self excluded, stable ordinal ties, exact integer-dot and f16-inverse-norm arithmetic; verify native top-10 against that oracle. Record per-query packed Recall@1 and AP@R, image digest, source/checkpoint/gallery/scorer hashes, encode wall and GPU resource use. Run `fp32_autocast` first, then `fp16_native` separately. Never use these results to choose a seed or change a checkpoint.

For each precision and seed, compare freeze to its same-seed control. The exploratory quality rule is the prior paired rule: frozen R@1 and mAP@R point estimates must not be lower, and the paired product-bootstrap 95% lower bound on R@1 must exceed −0.5 percentage points. Stop serial expansion at the first failed pair for that precision; retain failures. Passing this TRAIN diagnostic permits a later, separately frozen full official TEST serving evaluation and matched latency gate. It does not by itself unblock promotion, because strict offline parity already failed and official serving quality/latency remain unmeasured.

## First paired result

Seed 179024 completed on DGX Spark GB10 through the public API. Native top-10 matched the exact packed oracle for all 5,851 queries in each arm and precision; image manifests matched across arms. The receipts, paired decisions, source hashes and service journals are in `docs/evidence/compact_metric/sop-siglip2-substrate-v1/sop-true-freeze-public-train-179024/`.

| Query precision | Control packed R@1 | Freeze packed R@1 | Paired R@1 difference, 95% product CI | Control mAP@R | Freeze mAP@R |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 weights, FP16 autocast | 92.3090% | 92.8901% | +0.5811 pp [0.1215, 1.0530] | 0.767289 | 0.777534 |
| FP16 weights, native | 92.3945% | 92.8559% | +0.4615 pp [0.0170, 0.9229] | 0.767503 | 0.777591 |

Both pass the frozen same-seed TRAIN diagnostic. The public batch-32 path changes some descriptor bytes and ranks relative to the batch-64 offline export, so these numbers cannot replace the official TEST offline result. No serving latency or independent TEST serving quality has been certified yet.
