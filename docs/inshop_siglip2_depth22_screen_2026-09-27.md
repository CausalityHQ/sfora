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
