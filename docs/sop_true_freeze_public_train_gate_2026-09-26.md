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

## Three-seed public TRAIN result

Seeds 179026 and 179027 then completed serially under the same source and exact native scorer. Every one of the six seed/precision pairs passed the predeclared point and product-bootstrap floor. Raw per-query receipts, decisions and journals are in the corresponding `sop-true-freeze-public-train-{seed}/` evidence directories. All means below average three training seeds on the **same reused TRAIN holdout**; they are not across-dataset or across-seed confidence intervals.

| Precision | Seed | Control R@1 | Freeze R@1 | Paired R@1 difference, 95% product CI | Control mAP@R | Freeze mAP@R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 autocast | 179024 | 92.3090% | 92.8901% | +0.5811 pp [0.1215, 1.0530] | 0.767289 | 0.777534 |
| FP32 autocast | 179026 | 91.9843% | 92.5996% | +0.6153 pp [0.1038, 1.1088] | 0.763138 | 0.775434 |
| FP32 autocast | 179027 | 92.5312% | 92.7192% | +0.1880 pp [−0.2741, 0.6397] | 0.769529 | 0.777999 |
| FP32 autocast mean | 3 seeds | 92.2748% | 92.7363% | +0.4615 pp point mean; seed range +0.1880 to +0.6153 pp | 0.766652 | 0.776989 |
| FP16 native | 179024 | 92.3945% | 92.8559% | +0.4615 pp [0.0170, 0.9229] | 0.767503 | 0.777591 |
| FP16 native | 179026 | 92.0185% | 92.5825% | +0.5640 pp [0.0689, 1.0445] | 0.763181 | 0.775640 |
| FP16 native | 179027 | 92.4970% | 92.7363% | +0.2393 pp [−0.2234, 0.6927] | 0.769366 | 0.777672 |
| FP16 native mean | 3 seeds | 92.3033% | 92.7249% | +0.4216 pp point mean; seed range +0.2393 to +0.5640 pp | 0.766683 | 0.776967 |

Native top-10 matched the packed full-gallery oracle for all 5,851 queries in every receipt. The production flag remains SOP opt-in. The public path has verified TRAIN quality under both precision modes, but promotion still needs full official TEST quality for the actual query/gallery batch configuration and matched batch-1/batch-32 latency; the earlier strict offline-export parity failure remains recorded.

## Matched serving latency protocol, frozen before timing

Measure the seed 179024 same-source control and freeze public indexes on the existing DGX Spark GB10, with their own pinned 59,551-row TRAIN galleries and identical 32 query image bytes. Run `fp32_autocast` and `fp16_native` in separate serial jobs. For each precision, load both indexes before timing, verify their training receipts and native scorer, warm each at batch 1 and batch 32, then use 200 four-call ABBA/BAAB balanced blocks at each batch size. Every call includes file read, PIL decode, processor, encoder, packing and exact native top-10; synchronize CUDA at the call boundary. Record raw nanoseconds, p50/p95, cold index load seconds, peak CUDA memory, query IDs and image hashes, and output hashes. A cell needs at least 400 calls, and neither arm may have p95 more than 5% above its matched control. Treat p50/p95 as diagnostics, not p99 certification. A p99 claim requires at least 10,000 calls per arm and the separate paired-block bootstrap protocol. This timing rule is exploratory and does not override the failed strict parity or replace official TEST serving quality.
