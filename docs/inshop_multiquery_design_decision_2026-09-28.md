# Multi-query pooling: reject this proposed port before implementation

Decision: **KILL this proposed four-query SigLIP2 readout port**. No new
training, token export, official evaluation or production architecture change.
The production single-query pooler and packed 128-D retrieval path remain the
qualified defaults. The joint SOP/In-Shop quality and speed goal remains unmet.

## Evidence checked against the completed review

The single Opus 5.5 consultation `a2d08ddbf9ee4248` completed with exit zero
after 188 seconds, under a $2 maximum cap (not a measured spend). Its raw
[receipt and answer](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-multiquery-design-review-v1/consultation.json)
are preserved. No duplicate review or DGX job was launched.

Independent source inspection found a previously closed four-slot learned
attention readout in the
[24 September decision](joint_quality_performance_decision_2026-09-23.md).
It trained the final UNICOM B/16 transformer block, readout and proxies on
cached tokens for 1,595 updates. On SOP official TRAIN, 5,851 product-disjoint
holdout queries against a 59,551-image TRAIN gallery, slot readout S versus
stronger flatten control A0 measured packed R@1 **75.1325% versus 75.0812%**:
**+0.0513 percentage points**, product-cluster 95% interval
**[-0.5846, +0.6838] points**. Holdout mAP@R was **0.658345 versus 0.661730**,
delta **-0.003385**, interval **[-0.007848, +0.001075]**. This failed the
frozen quality gate; the record explicitly says not to port it to In-Shop.
The arm's online/including-ridge cached-tail training cost was
**64.245/80.260 s** versus control **70.518/70.518 s**. These exclude image
decode and early encoder blocks; no full image-to-top-k latency is implied.

The proposed SigLIP2 port would change the backbone, dataset, initialization
and trainable depth. It is therefore **not an exact replay** of the earlier
experiment. However, no new evidence currently supports reopening that
closed attention mechanism. Its positive comparison against mean pooling
does not establish a gain over SigLIP2's already learned single probe.

Installed Transformers source independently confirms that
`Siglip2MultiheadAttentionPoolingHead.forward` returns `hidden_state[:, 0]`.
Expanding the probe tensor alone would silently discard the extra outputs.
Current training and serving require supported source widths and a linear
compact head; a 4096-to-128 readout requires real loader/training changes.
Those changes were not made for this rejected candidate.

## Limits of the review

Do **not** adopt the review's stronger suggestion that failed fixed spatial
or raw-token matching establishes a necessary impossibility condition for
learned attention. Fixed matching is a different function; training may change
both tokens and their selection. The earlier slot run also had near-duplicate
queries, a source anchor and initialization confounds, explicitly acknowledged
in its terminal decision. It cannot disprove all multi-query architectures.
Likewise, missing token caches do not prove every structural smoke exceeds
120 seconds. A smoke can detect wiring or gradient errors, but cannot prove
retrieval improvement or justify a reopened research arm on its own.

The rejection is a bounded product/research decision based on the prior closed
variant and absence of a new distinguishing signal, not a theorem about model
capacity. There is **no new In-Shop quality, serving latency, QPS, encoder
training cost or VRAM measurement** from this review. No SOTA claim follows.

Next work must address a causally distinct representation/generalization
mechanism with a cheap TRAIN-fit diagnostic and matched controls. The DINOv3
route remains unavailable because neither host has a configured Hugging Face
token; owner grant status is unknown and credential reconciliation was already
reported. No credential workaround or repeated approval request is warranted.
