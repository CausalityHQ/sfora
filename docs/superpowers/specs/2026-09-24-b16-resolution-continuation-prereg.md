# B/16 resolution continuation screen (adaptive preregistration)

This is an exploratory continuation proposed after the frozen B/16@336
screen. It replaces the earlier plan's three 1,000-update runs from scratch;
the change is explicit because the frozen screen showed no robust quality gain.
It cannot establish a new state of the art or replace an independent-seed or
official-protocol result.

## Fixed comparison

Start all arms from the same verified SOP TRAIN B/16@224 ArcFace checkpoint
`a2568c671336b2c5a97587634ed8312c026625e172eadcba0a0df57abf799672`
(seed 179019, 1,000 updates). Restore encoder, 128-D head, classifier,
AdamW optimizer, GradScaler, and RNG state independently for each arm. Run
exactly 200 further ArcFace updates with one frozen identity-balanced batch
schedule on the same 53,700 fit images. A uses 224 input and the source graph;
B uses native 336 input and the resolution adapter; C uses the identical 224
tensor as A, bicubically enlarged to 336, and the same adapter as B. Each
occurrence uses the same seeded source-image crop and flip in all three arms.
Training is sequential on one GPU. Evaluate only terminal checkpoints.

For each arm extract all 59,551 official SOP TRAIN images in row order. The
5,851 heldout images from 1,132 product-disjoint classes are the queries.
Report R@1 on the 59,551-image TRAIN gallery (including fitted identities)
and R@1/mAP@R on the holdout-only gallery, for normalized 128-D float and
130-byte packed vectors. No PCA is fitted or refitted: the learned head is
part of the checkpoint. Exclude self on the full gallery. Use paired
5,000-draw product bootstrap with seed 179019. Record wall-clock training and
encoding time, peak CUDA allocation and host RSS, source hashes, checkpoint
hashes and GPU. This TRAIN screen is biased and not an official TEST claim.

## Decision fixed before the run

Advance B only if B−A full-gallery R@1 is at least +1.0 percentage point
and has a paired product-bootstrap lower 95% endpoint above zero in **both**
float and packed 128-D outputs; B−C float full-gallery R@1 must also have a
lower 95% endpoint above zero. B−A holdout-only float mAP@R must have a
lower 95% endpoint above −0.003. If B−A float full-gallery R@1 is at most
+0.5 point or the holdout mAP bound fails, close this resolution route.
Any ambiguous result remains exploratory and triggers no official claim.
Only after a pass: check CUB classes 101–200 noninferiority, repeat with
independent seeds, then run official SOP and In-Shop protocols and certified
end-to-end p50/p95/p99 latency and training-resource comparisons.

The frozen predecessor is
`docs/evidence/compact_metric/sop-b16-resolution-train-quality-v1.json`;
the paired cost predecessor is
`docs/evidence/compact_metric/sop-b16-resolution-cost-v1.json`.
