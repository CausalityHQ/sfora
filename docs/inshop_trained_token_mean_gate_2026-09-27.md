# In-Shop TRAIN trained-token information gate, 27 September 2026

The three selected true-freeze SigLIP2 Large/256 checkpoints produce a
128-D head on the attention pooler. The final patch-token mean is already
computed inside the same vision forward but is discarded by serving. Test
whether it contains complementary product information after training. For
each image, keep the existing normalized head output `base`, apply the same
trained head to the mean of all final patch tokens, normalize that output,
then normalize `0.75 × base + 0.25 × token`. The coefficient is frozen here;
no labels, projection, checkpoint weights, training schedule or gallery
products are used to select it. Quantize with the existing 128-D int8 plus
fp16 inverse-norm packer and use the exact native packed scorer for the fixed
asymmetric top-1 primary endpoint; use the existing packed reference scorer
for symmetric mAP@R, with exact archived baseline replay. The wire
format remains 130 bytes/image; training cost is unchanged. Serving cost is
unknown until a separate public-path gate.

Use existing full-budget seeds 179026, 179024 and 179027 in that order, their
SHA-256-pinned `freeze_emb` checkpoints, and only the official In-Shop TRAIN
partition: 13,283 fit-product images are outside evaluation; 12,599 held
images from 1,993 disjoint products form the fixed symmetric self-excluded
gallery. The previously fixed asymmetric held roles have 6,354 queries and
6,245 gallery images. Export base and candidate on the same forward for each
seed; do not fit on held labels. Verify each checkpoint and split digest, and
replay the archived base packed symmetric per-query outcomes before accepting
candidate scores. This is an exploratory TRAIN-only screen because this held
panel has been inspected repeatedly. The official query/gallery split is
closed to selection.

Early stop after seed 179026 if candidate-minus-base asymmetric Recall@1 is
nonpositive, or symmetric mAP@R falls by more than 0.002. Otherwise evaluate
the remaining two checkpoints serially. Promote to a public implementation
candidate only if all three seeds have nonnegative asymmetric Recall@1 deltas,
their mean delta is at least **+0.25 percentage points** with a pooled
product-cluster paired-bootstrap 95% lower bound strictly above zero, and
every symmetric mAP@R delta is at least **−0.002**. Record GPU export wall,
peak allocated CUDA, packed gallery bytes and scorer wall. A pass authorizes
an exact public top-10 and image-to-top-10 p50/p95/p99 gate on SOP TRAIN and
In-Shop TRAIN; it does not establish official quality, SOTA, or an improvement
in training speed. Failure retires the fixed token mean fusion without
testing other coefficients on this held panel.
