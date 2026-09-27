# In-Shop dual-depth compact-head preflight, 27 September 2026

The trained 22-block SigLIP2 screen lost mAP@R, but its **pretrained** block-22
source with an own fit-only PCA-128 head beat the 24-block source by 0.048174
mAP@R on the same held TRAIN products. Test whether both depths together
retain complementary information without another image export or extra encoder
blocks. Concatenate separately unit-normalized 22- and 24-block 1,024-D
pretrained source rows, fit one centered PCA-128 on only the 13,283 fit-product
official In-Shop TRAIN rows, and score the same 12,599 held-product rows with
the existing 130-byte packed scorer and stable ties.

Before reading the result, require the dual-depth packed mAP@R to exceed the
stronger 22-block own-PCA baseline **0.508923** by at least **0.005**, with a
paired product-bootstrap 95% lower bound above zero, and packed R@1 at least
the 22-block baseline **87.1736%**. Validate both cached feature/receipt and
model/row hashes, exact fit/held split, finite features and wire geometry.
Any failure stops this dual-depth treatment before training. A pass permits
one same-architecture 24-block, same-batch/budget/optimizer/scorer paired
100-update training feasibility screen with a 2,048-to-128 head; measured
training, memory and public image-to-top-10 latency must then pass separately.
The intended serving path would read an intermediate activation during the
existing 24-block forward, without a second encoder run, but this cost is
unmeasured. This already-observed TRAIN panel is exploratory and cannot
establish official quality, current SOTA or method novelty.
