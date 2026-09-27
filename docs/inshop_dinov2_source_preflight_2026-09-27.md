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

## Terminal result and cause check

The sole DGX Spark GB10 export/scoring unit
`sfora-inshop-dinov2-source-v1.service` (invocation
`17673703f7df49419ecded042be24f56`) exited 0. The [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-dinov2-source-v1/receipt.json)
SHA-256 is `64343ad8744ec2784c37c5700fc7daeed37107e37af2e4ba96eff4b8d61ca3ac`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-dinov2-source-v1/service-journal.txt)
SHA-256 is `739af8c862c48a27718fb66789d347ce769df3db8477a5fec76489e3d8e6f9b1`.
Local verification checked all 12,599 per-query outcomes, source hash and
the failed gate. The model, processor, five imported helpers, partition and
feature file are SHA-pinned in the receipt.

| Official TRAIN held-only symmetric panel, 12,599 queries/gallery | Packed R@1 | Packed mAP@R | Full 25,882-TRAIN-image encode wall |
| --- | ---: | ---: | ---: |
| Pinned SigLIP2 Large/256 pretrained own-PCA reference | **81.6652%** | **0.460749** | **216.163 s** |
| DINOv2-L/224 pretrained own-PCA treatment | **43.4717%** | **0.113994** | **191.535 s** |

DINOv2's encode wall passed the cost ceiling and its peak CUDA allocation was
**0.888 GB**, but it missed both frozen quality floors decisively. The two
exports used different model/processor implementations; their sequential
walls are a feasibility screen, not a paired public latency measurement.

A separate source-bound [raw-feature diagnostic](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-dinov2-source-v1/raw-diagnostic.json)
(receipt SHA-256 `bbba015923c1c417af384b22588076cae7fb7554762532aa0063fa9aa8ee5407`,
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-dinov2-source-v1/raw-diagnostic-journal.txt)
SHA-256 `8dfc3acb359e20ae358e0b25d56f8e85201cd66b7d81878d315b63cdd1b6e836`)
reused the exact exported feature SHA and scored normalized 1,024-D float
cosine with the same self exclusion and stable ordinal tie rule. It found
**6,131/12,599 = 48.6626% R@1**, only 5.1909 points above the packed
128-D result. Weak pretrained global source geometry, with additional
compression loss, explains this screen's failure; the result does not bound
what a fully fine-tuned DINOv2 could learn.

**Decision:** stop this native DINOv2-L/224 frozen-source route before
fine-tuning, official TEST, or a production change. Neither its raw nor its
packed source quality supports spending a matched training campaign here.
