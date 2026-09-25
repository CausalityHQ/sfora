# Selected checkpoint at serving batch sizes, 25 September 2026

Seed 179023, selected by the frozen SOP TRAIN-holdout rule, was re-encoded on all 60,502 already-observed official SOP TEST queries at batch sizes **1** and **32**. Both query sets were searched against the **same fixed batch-64 packed gallery** from the authenticated official evaluation, excluding each query's own row. This matches the live serving pattern measured in the paired latency screen. The full-gallery packed-score oracle supplied Recall@1 for every query; native top-10 matched that oracle for the first 32 queries at each batch size.

| Query encoding batch | Packed Recall@1 | Difference from batch 64 | Product-bootstrap 95% interval for difference | Export wall |
| --- | ---: | ---: | ---: | ---: |
| 1 | **91.3292%** | −0.0083 percentage points (5 fewer correct) | [−0.0216, +0.0050] points | 998.715 s |
| 32 | **91.3226%** | −0.0149 points (9 fewer correct) | [−0.0334, +0.0050] points | 513.277 s |
| 64, prior official export | 91.3375% | reference | — | 324.623 s |

All three values are from the same selected seed and the same full TEST gallery; the intervals condition on that seed and the reused TEST protocol. Both deployed batch values are numerically above the rounded published UNICOM ViT-L/14@336 SOP Recall@1 of 91.2%, but the margins are small and there is no paired uncertainty against the published model. The previous three-seed 91.2725% coverage-bank mean is a batch-64 export result. No In-Shop SigLIP2 quality or full p99 latency certification exists yet.

The [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/deployed-batch-quality-v1.json) has SHA-256 `accf890358eb6bb1e37cfaef8f476763c7d9eb16a1684323205d9bf47c57ef45`; the successful DGX Spark unit invocation `922ee1b49a3341c59a4918fdb377bf98` [journal](evidence/compact_metric/sop-siglip2-substrate-v1/deployed-batch-quality-v1.log) has SHA-256 `de9fabf56bd1d19e1f26f980ee576ad5f6edec3728c34237010a12d349afda81`. Remote query embeddings remain at `/home/riomus/runs/sfora-siglip2-deployed-batch-quality-v1/query_batch{1,32}.npy`, SHA-256 `0788371b3d5d4f2ac8804ec8dc7bba0a46fb194005e2f4f95fd6a247733c3577` and `86d74b031475b91a314ad32ab2c06c8f53fb90122f6b42e4e1a814aa0246afee`. The pinned evaluator checkout was clean commit `634b6ef7`; its batch exporter file hash was `effb927f746805f1f0133a4821148b088a3807829d284e3ed9d6c9ecf1f030ee` and the running script hash was `d832bfef83e520857c0f633d4160874aa3da438e8cdbaf7aa65e6f4cda847e67`.

The raw receipt's `export_source_sha256` field hashes the PyTorch inference-mode wrapper (`f22c21a0…`) because the code used the decorated function's `co_filename`. This is a provenance-field bug, not a numeric-path change: the clean pinned checkout and actual exporter file hash above identify the executed source; the code now hashes that file directly for future runs. No quality job was repeated for this metadata correction.

**Decision:** the serving batch shape does not erase the selected checkpoint's observed SOP Recall@1. Continue the frozen In-Shop bank-versus-float transfer, then require official In-Shop query/gallery quality and full paired p99 latency certification before claiming a joint improvement. The current ArcFace, SmoothAP, detached bank, and coverage schedule combination is not established as a novel method.
