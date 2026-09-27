# In-Shop first-16-block freeze gate, 27 September 2026

The 22-block encoder reduced TRAIN wall and memory but failed the frozen
mAP@R floor. This new arm keeps all 24 pretrained SigLIP2 Large/256 vision
blocks and changes only the trainable boundary: freeze embeddings and blocks
0–15 rather than embeddings and blocks 0–11. It targets representation
generalization and training cost. The serving encoder and packed scorer are
unchanged, so this arm cannot establish a serving-speed gain. Partial
fine-tuning is prior art, not a novel similarity-learning method.

Use the official In-Shop **TRAIN** 13,283-fit/12,599-held product-disjoint
split, seed 179024, 1,000 updates × 64 images, BF16, the same pretrained
feature cache, PCA initial head, classifier, member bank, batches,
augmentations, ArcFace plus SmoothAP loss, optimizer, 128-D int8+norm gallery
and exact packed scorer. The archived first-12-block `freeze_emb` control
receipt SHA-256 is
`fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89`:
held-only symmetric packed R@1 98.5475%, mAP@R 0.839099, training wall
747.058 s, peak allocated CUDA 12.939 GB on DGX Spark GB10.

Freeze these gates before seeing treatment quality:

1. One 17-update smoke must verify finite loss/gradients, matching source
   cache, split, schedule and initial head, and frozen pretrained embeddings
   plus blocks 0–15 in the checkpoint. Stop on any mismatch.
2. Run one 1,000-update treatment. On identical held queries and gallery,
   require packed mAP@R at least the archived control point, a paired
   product-cluster bootstrap 95% lower bound for its mAP difference at least
   **−0.003**, packed R@1 no more than **0.2 percentage points** below
   control, and training wall at least **5%** lower. All 1,000 updates and
   checkpoint/source/score receipts must be finite and valid. Otherwise stop
   this arm before another seed or official TEST.
3. A passing seed only authorizes independent seed replication on TRAIN and
   a fresh product-disjoint quality panel. It is not production promotion.
   Official TEST has prior reads; public image-to-top-k latency needs its own
   measured gate, and this architecture cannot make that path faster.

The [trainer](../scripts/train_inshop_siglip2_unseen_gallery.py) SHA-256 is
`0da378d00167e7e6d3a76bdf8ee4dd90fe6f82d05c142c82145767de0da78a6b`.

## Smoke checkpoint

The sole 17-update DGX Spark smoke completed with exit 0, finite loss and
gradients, and a checkpoint hash matching its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-freeze16-179024-v1/smoke.json)
SHA-256 `842749f316938a8c26d7fa51613bb65d741148b7473f6eca1ca948d7297177df`.
Its fit/held rows, schedule, first-ten image inputs, PCA and pretrained
feature cache match the archived same-seed freeze-12 smoke. A direct
checkpoint-to-pretrained tensor comparison found all **259** embedding and
block-0–15 tensors exactly equal, while at least one block-16–23 tensor
changed. The [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-freeze16-179024-v1/smoke.journal.log)
SHA-256 is `dbeffebd6d327c57dcbe36234b29b4cbfbc36965a3f2226b0ec9c11f2d1e7378`.
The subsequent single 1,000-update treatment completed as
`sfora-inshop-freeze16-full-179024-v1` (invocation
`d6a8cdad2b894ac2ba155153fdd85e8b`).

## Full-run decision

The sole DGX Spark GB10 service completed 1,000 finite updates, held-image
export and packed scoring with exit 0. The
[source-bound decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-freeze16-179024-v1/decision.json)
verified source/helper, model, cache, PCA, fit/held rows, batch inputs,
checkpoint and per-query metric authorities. On the same 12,599 official
**TRAIN** product-disjoint held-only symmetric queries/gallery:

| Seed 179024 arm | Packed R@1 | mAP@R | Training wall, 64,000 images | Images/s | Peak allocated CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Embeddings + blocks 0–11 frozen | 98.5475% | 0.839099 | 747.058 s | 85.67 | 12.939 GB |
| Embeddings + blocks 0–15 frozen | 98.4602% | 0.831464 | 617.658 s | 103.62 | 9.745 GB |

Freeze-16 reduced training wall by **17.32%** and peak allocated CUDA by
**24.68%**. Its packed R@1 difference was **−0.0873 percentage points**,
inside the frozen −0.2 pp limit. Its mAP@R difference was **−0.007635**,
with paired product-cluster bootstrap 95% interval
**[−0.009652, −0.005616]**. This fails both the nonnegative mAP point and
−0.003 lower-bound gates. `advance_independent_seeds=false`: stop this arm
before another seed, official TEST or production promotion. The full serving
encoder is unchanged, so no serving-speed gain is implied.

The [treatment receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-freeze16-179024-v1/full.json)
SHA-256 is `63e97cebb8cc0e354e87ed837f2739123e5cdb2078a38827848f5abef0c768cd`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-freeze16-179024-v1/full.journal.log)
SHA-256 is `e320d0bdca02867579b54bc4a07ad223eb0b1bcb8d72b4bbb0ce1d3b7b4867b9`;
and the [decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-freeze16-179024-v1/decision.json)
SHA-256 is `9dc186c9cef6e10821e4c993647239ca39f7f3f8af6559828a64cffaf255533c`.
