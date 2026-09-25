# Cached-head speed probe on SOP TRAIN

## Question and scope

Can a live linear 1024-to-128 head learn useful retrieval geometry from one
authenticated cache of pretrained SigLIP2-L/16@256 features, without paying
for 1,000 augmented encoder forward/backward updates? This is a **fit-only
feasibility probe**, not a replacement for full-backbone training or evidence
of a new method. The official SOP TEST and In-Shop protocols remain closed.
The already-used SOP TRAIN product-disjoint holdout is exploratory.

Use the pinned SOP archive (`1ba27b2…7d921a`) and pretrained feature export
(`d15f76e4…8357a`, 59,551 rows). The export receipt reports **508.517 s**
for initial image encoding on NVIDIA GB10; include it in any fresh-system
training-time comparison. Do not treat a pre-existing cache as free. Both
probe arms use the 53,700 fit rows (10,186 products), 5,851 heldout queries
(1,132 disjoint products), the full 59,551-row gallery, self exclusion,
PCA-initialized 128-D linear head, initial fit-only class proxies, the same
class-balanced 64-row schedule, AdamW and exact 130-byte native top-10 scorer.

## Frozen arms and gate

Use seed **179023**, 1,000 cached-head updates, batch 64, ArcFace margin 0.3
and scale 64, head/proxy learning rate `1e-4`, weight decay `0.05`, gradient
clip 1.0. Freeze the encoder and source cache. Arm A optimizes ArcFace only.
Arm B adds coefficient-8 SmoothAP against the full source-feature bank after
projection through the **current** linear head. The source bank is detached;
candidate and anchor projections both update the head. Keep source vectors
unit normalized and compute the loss in FP32. Save the exact schedule hash,
source hashes, initial head/proxy hashes, training wall and peak allocation,
then export and score both trained heads through the same packed scorer.

The speed reference is the existing seed-179023 full-fit detached bank receipt
[`bf16-seed179023-bank-v1.json`](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179023-bank-v1.json),
SHA-256 `20dcf88633c302502d5e7f9cd0a432c9b4186fc1939da420203e52464648ad3d`.
Its `training_wall_including_member_bank_init_seconds` is **1,146.908473 s**.
The speed numerator is **Arm B alone**: the export receipt's 508.516901 s
source-cache encoding plus its measured head initialization and 1,000-step
training walls. Arm A is a scientific control, not part of a deployed Arm B
training run. The baseline's own source-cache acquisition is omitted, so this
is deliberately conservative and is not a complete end-to-end training wall.
Validation, file hashing, optimizer setup, and evaluation are also outside
this defined timing comparison and must be reported separately.

This probe survives only if Arm B gains at least **+0.5 percentage points**
Recall@1 over A on the selected TRAIN holdout, has nonregressing mAP@R, reaches
an above-zero lower 95% product-clustered bootstrap endpoint for that paired
Recall@1 difference, reaches at least **85%** absolute packed Recall@1, and its defined speed numerator is
at most **0.50×** the pinned bank wall (**573.454236 s**). The last threshold
is a speed feasibility gate; it does not count deployment encoding. Report
failures without retuning on this holdout. If it survives, the next
experiment must compare fewer full encoder
updates plus cached-head steps against the 1,000-update bank baseline, with
paired seeds, total training wall and the same scorer. A fit-only pass does
not justify a quality or speed claim on official TEST or In-Shop.

The live-head bank step is measured at roughly 20 ms p95 in the isolated
cost receipt, but the complete cached-head loop and its quality are unmeasured.
The cached-head system has no augmented encoder views and clips head/proxy
gradients separately from the full-backbone recipe. It is therefore a system
feasibility screen, not a same-update optimizer ablation. Peak CUDA allocation
in the probe measures PyTorch's training phase and training-plus-evaluation
phase separately; the export receipt records cache acquisition resources.
Prior art includes memory banks, linear probing, and proxy losses. The only
potential method contribution is keeping gallery candidates current under a
trainable deployment head while amortizing encoder updates; that requires a
later matched ablation and external evaluation.
