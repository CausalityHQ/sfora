# SOP SigLIP2 replication gate stopped, 25 September 2026

The frozen [post-selection protocol](sop_siglip2_member_bank_multiseed_2026-09-25.md)
requires seeds 179020, 179021, and 179022 to complete matched ArcFace,
in-batch float SmoothAP, and full-fit member-bank arms. Seed 179019 selected
the method and is excluded. The first serial DGX unit stopped during the
seed-179020 ArcFace control, after its step-500 log and before step 550, when
`clip_grad_norm_(error_if_nonfinite=True)` found a non-finite gradient norm.
The unit `sfora-siglip2-member-bank-multiseed-v2.service`, invocation
`c0e4924bdcdd403f866c152729aeb273`, exited with status 1. Its
[raw journal](evidence/compact_metric/sop-siglip2-substrate-v1/member-bank-multiseed-v2-failure-journal.log)
has SHA-256 `cb7ac80a796e807f7665c89caf4b712c5c20ab3816cc5aff84c39209b13e06a7`.
The failed output directory has no receipt. Seeds 179021 and 179022 did not
run. This is a failed replication gate under the preregistered rule; the
incomplete ArcFace arm cannot be silently replaced with a diagnostic replay.

## Completed paired evidence, internal selection only

Both completed arms used official SOP **TRAIN** only: 53,700 fit images from
10,186 products, 5,851 holdout queries from 1,132 disjoint products, and the
full 59,551-image TRAIN gallery with self excluded. SigLIP2-L/16@256,
1024-to-128 head, 1,000 updates of 64 images, native exact 130-byte packed
top-10 scorer, seed 179020, model/data/source/initialization/input hashes,
and hardware matched. The paired product-bootstrap draws condition on this
one trained seed and do not quantify training-seed variance.

| Arm | TRAIN holdout Recall@1 | mAP@R | Training wall | Status |
| --- | ---: | ---: | ---: | --- |
| In-batch float SmoothAP | 90.8050% | 0.725659 | 1,152.165 s | Verified receipt |
| Full-fit member bank | 91.5399% | 0.745636 | 1,169.522 s including bank initialization | Verified receipt |
| ArcFace | — | — | — | Non-finite gradient, no receipt |

Bank minus float is **+0.7349 percentage points** Recall@1, product-bootstrap
95% interval **+0.2520 to +1.2074 points**, and **+0.019977** mAP@R,
interval **+0.015380 to +0.024572**. The bank training-wall ratio is
**1.01506×**. The source-bound
[aborted-pair receipt](evidence/compact_metric/sop-siglip2-substrate-v1/aborted-seed020-pair-v1.json)
has SHA-256 `a5c7caef8008ff5f0e2782e2775354dacde84ac15e2dbb468b52bd9cc6a84afc`.
These numbers are neither SOP official TEST nor evidence that the full method
replicated. The earlier published UNICOM 91.2% SOP and 96.7% In-Shop values
refer to official evaluation and are not directly compared with this holdout.

## Decision and precision diagnosis

The bank-versus-float direction repeated once beyond the selection seed, but
the frozen three-arm continuation gate failed. Do not run the prepared
official SOP TEST evaluator or make a SOTA claim from these receipts. Preserve
the original failed unit and completed arm artifacts. A separately labelled
diagnostic replay of seed 179020 ArcFace used the frozen trainer with only
gradient instrumentation; it does not repair the failed gate. Its logged
losses matched the original run through step 500 and it reproduced the
failure at **step 543**, input-batch SHA-256
`070ff753eaca317954f39cd6979b6f565828198c6a1bee25c9a482002f214e9f`.
The loss was finite at **4.480570**, GradScaler was **128**, and 14
parameters had non-finite gradients, all in the first two vision transformer
layers or image/position embeddings. Layer-0 attention output projection had
infinities; earlier attention and embedding gradients contained NaNs. The
head and classifier were absent from the failing set. The
[diagnostic journal](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-seed020-gradient-diagnostic-journal.log)
has SHA-256 `8eadffc6f48829bdce3b7ad739b267c0765f205b27ca20f273963babe79f21d4`.
The [instrumentation receipt](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-gradient-instrumentation.json)
binds the original and modified trainer hashes to the
[exact patch](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-gradient-instrumentation.diff).
This localizes the failure to encoder backward arithmetic but does not yet
prove which operation first overflowed or whether a lower fp16 loss scale
would suffice. A matched precision-stable recipe must be chosen and
preregistered before any new three-seed quality claim.

The live-head source-bank alternative remains unmeasured for quality. Its
guarded, synthetic-geometry **isolated ranking-loss** screen on the same
NVIDIA GB10 measured detached-bank median/p95 **13.693/14.316 ms** versus
live-head **19.560/19.941 ms**, 40 loss-plus-backward calls per arm, exact
forward-loss parity, and maximum incremental GPU allocation **467.4 MB**
versus **522.8 MB**. The [cost receipt](evidence/compact_metric/sop-siglip2-substrate-v1/live-head-isolated-cost-v1.json)
has SHA-256 `1e9d3502050e081edc58d582b42ddd893760165203dce67b53e8a8da7266d56a`.
Its predeclared trainer cost gate passes, but the screen excludes image
encoding, optimizer, export, and actual retrieval. This candidate still
requires a new matched quality and end-to-end performance evaluation before
promotion.
