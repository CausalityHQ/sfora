# In-Shop 22-block encoder feasibility screen, 27 September 2026

This TRAIN-only screen tests a distinct joint quality and performance lever:
remove the final two blocks from the 24-block pretrained SigLIP2 Large/256
vision encoder while retaining final normalization, pooling, 128-D head,
int8+norm gallery format and exact packed scorer. Depth pruning is prior art;
this is a production architecture experiment, not a novel method or SOTA claim.

Compare fresh seed-179026 `freeze_emb` control and 22-block treatment on the
existing 13,283 fit / 12,599 held official In-Shop TRAIN product-disjoint
split. Both arms use the same 100-update × 64-image schedule, augmentation,
ArcFace plus SmoothAP bank loss, BF16, AdamW, frozen embeddings and first
12 blocks. Recompute the treatment's pretrained 1,024-D source-feature cache,
PCA initial head, classifier and bank through its 22-block encoder; the
control uses the archived full-depth pretrained cache. These initial heads
will differ by design. Holdout remains held-only symmetric packed top-1 and
mAP@R. This repeatedly observed development split is an early feasibility
screen; no official TEST or fresh confirmation quality is claimed.

Freeze the decision before reading treatment quality:

1. Verify authenticated model and partition hashes, exactly 24 pretrained
   blocks before pruning, 22 after, feature-cache receipt and source hashes.
   Run one 17-update treatment smoke and require finite loss/gradients and
   a matching checkpoint hash.
2. Run 100-update baseline and treatment serially on the sole DGX Spark,
   with source, split, schedule, first-ten input and scorer parity. Stop if
   treatment training wall fails to improve by **at least 5%**, or packed
   held R@1 falls more than **0.5 percentage points**, or mAP@R falls more
   than **0.005** versus the 100-update control. These are feasibility limits,
   not a selection claim. Record cache export cost and peak VRAM separately.
3. If those pass, measure synchronized batch-1 encoder latency in an
   interleaved paired benchmark. Require at least **5%** improvement, then
   advance to a 1,000-update replicated training and public image-to-top-k
   p99 gate on a newly reserved product-disjoint panel. No production
   promotion occurs from this 100-update screen alone.

Source SHA-256: [cache exporter](../scripts/export_inshop_siglip2_train_features.py)
`30f46c7536f546a264295da9c22de9cde0738bd02f5ca1bd51dcd1a9b8a659b6`;
[trainer](../scripts/train_inshop_siglip2_unseen_gallery.py)
`92e60e603a88951a071b778ba1f6b2d1c362af9a81fffd78bac117e6c7a823a9`.

## Terminal TRAIN-only result

The 22-block cache exported all 25,882 official TRAIN images in 180.906 s;
its 1,024-D feature-file hash matched its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/cache.json)
SHA-256 `60b277021c858124310b00937e947805fb70739ccaebe8d9142b3bbf6aea2704`.
The archived 24-block source-cache receipt records 216.163 s. These are
separate one-pass exports, not an interleaved public latency measurement.

The first smoke failed **before training**: the service's `PYTHONPATH` omitted
`/home/riomus/runs`, resolving an older trainer helper without
`allow_singletons`. Its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/smoke-v1-failed.journal.log)
SHA-256 is `e293d0ee30ae50328a14de7f733e1a6678bb78dce23d4f32aea460c443aad053`.
The corrected import path pinned helper SHA-256
`e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6`.
The second smoke completed 17 finite updates, verified a checkpoint with
blocks 22–23 absent, and matched its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/smoke.json)
SHA-256 `5e0d73fd2543978a916e02f9ee10ba1b6aae58a80141eadddf7b077663cd3cb6`.
The [cache journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/cache.journal.log)
and [successful smoke journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/smoke-v2.journal.log)
have SHA-256 `61f05c5299b4465d0fbd7a8f608673ce472f12a711ee498c3d7c92f845fd4b0f`
and `f8c03cb63c3dcd03fc14e53e5788cf2439a04959af442b6614c79fc4d79cf545`.

The sole serial DGX Spark GB10 service then completed both 100-update arms,
held-image export and packed scoring with exit 0. The
[source-bound decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/decision.json)
verified identical fit/held rows, schedule, first-ten image inputs, model
files, scorer, source and checkpoint hashes. On the same 12,599 product-disjoint
official **TRAIN** held-only symmetric queries/gallery:

| Arm | Packed R@1 | mAP@R | Training wall, 6,400 images | Images/s | Peak allocated CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24-block control | 97.3411% | 0.775415 | 76.317 s | 83.86 | 12.939 GB |
| 22-block treatment | 97.1903% | 0.766315 | 67.248 s | 95.17 | 11.241 GB |

Training wall improved **11.88%**, peak allocated CUDA fell **13.12%**, and
packed R@1 fell **0.1508 pp**, within its 0.5 pp limit. mAP@R fell
**0.009100**, beyond the frozen 0.005 limit. Therefore
`advance_batch1_latency_screen=false`: stop this depth arm before batch-1
timing, 1,000-update training, official TEST or production integration. No
serving speed or SOTA improvement has been established.

Raw [control](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/control.json)
and [treatment](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/treatment.json)
receipt SHA-256 values are `3da086a15c4ec25b6255d300da13d0220d5311fd6d5babff62d8aaa872ca0e03`
and `6ed19555bd00a226af98aa1caeb261a8c1d0ef057a365ebc413cc97aef5cb160`.
The [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/paired.journal.log)
SHA-256 is `deadc3aa0ab826167a552f4741f518720804cdb8a8c17eb6a4c3352c59e5b30e`;
the [decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-depth22-179026-v1/decision.json)
SHA-256 is `49d3e2bf1caf750e5669004b32857860437ab001f705e905414d973de871fac5`.
