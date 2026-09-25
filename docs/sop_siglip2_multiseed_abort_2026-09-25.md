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
would suffice. An earlier [seed-179019 gradient replay](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-gradient-diagnostic-v8.log)
found a similar layer-0 failure at step 131 with scale 1024. That was a
different seed and scale, so the two failure steps do not isolate a scale
effect. They do show that a short precision qualification on one seed was
insufficient. A matched precision-stable recipe must be chosen and
preregistered before any new three-seed quality claim.

The live-head source-bank alternative remains unmeasured for quality. A first
[isolated cost screen](evidence/compact_metric/sop-siglip2-substrate-v1/live-head-isolated-cost-v1.json)
omitted the trainer's TF32 setting and persistent bank bytes; it is superseded
for decision purposes. The corrected, guarded, synthetic-geometry
**isolated ranking-loss** screen on the same NVIDIA GB10 set
`matmul.allow_tf32=False` and measured detached-bank median/p95
**13.616/14.074 ms** versus live-head **19.257/19.950 ms**, 40
loss-plus-backward calls per arm with exact forward-loss parity. Maximum
incremental GPU allocation was **467.4 MB** versus **522.8 MB**. The persistent
bank payload itself grows from **27,494,400 bytes** to **219,955,200 bytes**.
The corrected [cost receipt](evidence/compact_metric/sop-siglip2-substrate-v1/live-head-isolated-cost-v2.json)
has SHA-256 `50416fd7e7b0515ae4ced059fcbc17966acd1e1226b117ff1106ae665431a033`.
Its revised trainer cost gate passes, but the screen excludes image encoding,
optimizer, export, and actual retrieval. This candidate still requires a new
matched quality and end-to-end performance evaluation before promotion.

## Same-state precision fork, diagnostic only

A separate instrumentation-only replay captured the model, head, classifier,
exact processed fit batch, labels, and RNG state immediately before the
seed-179020 ArcFace step 543 forward pass. The capture is retained on the DGX
at `/home/riomus/runs/sfora-siglip2-arcface-step543-capture-179020-v1/step543_state.pt`
(SHA-256 `2a639395856bd85bff55f2beaa4c87e77a70667febc1baecd246e85ca66e8c3e`).
The [capture journal](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-step543-capture-journal-v1.log)
(SHA-256 `a68da822be9b158ea6677c384afc9a6321a1f0a2efb791b22183ab3fdda7c0b9`)
records the same capture hash and the original non-finite clipping failure.
This artifact has no optimizer state and is valid for a backward-only precision
comparison, not a training continuation.

The [backward fork receipt](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-step543-backward-fork-v1.json)
(SHA-256 `21ac49997e93b16aa2b3b44fbd677525c97bc90856196bfae5e2daa77addbad3`)
and [raw journal](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-step543-backward-fork-journal-v1.log)
(SHA-256 `0e236557e70622798ff8f7712f2ce61c7aff742f4ed40811ca2bd91e7e9fca11`)
compare the *same* parameters and fit batch without optimizer updates on
NVIDIA GB10, PyTorch 2.12.1+cu130, with TF32 disabled:

| Vision forward precision | Loss scale | Loss | Non-finite parameter gradients | Unclipped parameter-gradient L2 norm |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 128 | 4.480570 | 14, exactly the original failing names | non-finite |
| fp16 | 32 | 4.480570 | 0 | 17,927.47 |
| fp16 | 8 | 4.480570 | 0 | 17,915.21 |
| fp16 | 1 | 4.480570 | 0 | 17,926.64 |
| fp32 | 1 | 4.475663 | 0 | 114.51 |
| bf16 | 1 | 4.486574 | 0 | 121.84 |

The fp16 scale-128 fork reproduced the original loss and all 14 bad gradient
names. Reducing the loss scale made the gradients finite but left their norm
about 156 times the fp32 norm; finiteness alone is therefore an inadequate
qualification. BF16 is the next training-recipe candidate because this
one-state comparison is much closer to fp32. This does not establish where
the first fp16 arithmetic error occurs, guarantee BF16 training stability, or
rescue the frozen three-seed gate. Next, qualify a BF16 ArcFace training run
with a distinct source and receipt, then freeze and execute a new matched
multi-seed protocol before opening official TEST.
