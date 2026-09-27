# In-Shop DINOv2-L/224 source preflight, 27 September 2026

This exploratory screen uses the cached native `facebook/dinov2-large` checkpoint
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, its upstream 224-pixel
processor, and only official In-Shop **TRAIN** images. Fit a centered PCA-128
on the same 13,283 fit-product rows as the pinned SigLIP2 Large/256 pretrained
source control; pack the same 12,599 held-product rows into the existing
128-D signed-int8 plus fp16 inverse-norm format. Score all held rows against
the held-only gallery with self excluded and stable ordinal ties. This changes
the encoder and preprocessing together; it tests substrate feasibility, not a
learning-method improvement. The held split has been observed previously, so
even a pass is exploratory.

Frozen advance rule, before reading DINOv2 quality: packed held Recall@1 at
least **83.6652%** and mAP@R at least **0.480749** (both at least +2.0 points
over the authenticated SigLIP2 Large/256 own-PCA pretrained source
**81.6652% / 0.460749**), full 25,882-image encode wall at most **216.163 s**
on the DGX Spark GB10, peak allocated CUDA below **10 GB**, all finite rows,
exact model/processor/partition/source hashes. Any failure stops this substrate
before supervised training, official TEST, or a production API change.
One-pass export wall is a cost screen, not public image-to-top-k latency. A
pass would authorize a separately frozen same-recipe paired training and
public-path latency screen, plus CUB/Cars transfer; it would not authorize a
SOTA or method claim.
