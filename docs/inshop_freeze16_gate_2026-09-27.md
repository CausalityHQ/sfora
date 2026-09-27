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
