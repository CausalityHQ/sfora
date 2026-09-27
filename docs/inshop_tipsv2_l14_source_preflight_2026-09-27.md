# TIPSv2 L/14 compact source preflight, 27 September 2026

The previously tested SigLIP2 depth, resolution, DINOv2 and pose-loss routes
did not close the In-Shop official quality gap. Test a different released
encoder before any new training code. Use Google DeepMind's
[TIPSv2 source and checkpoints](https://github.com/google-deepmind/tips)
at revision `bf10a73765b048edfbfcdf1f7e6ca8292542baae`. Pin its L/14 vision
checkpoint SHA-256 `46e5ac008f02191020b3fe908a60c2d237268b43a864251af2f5354058d3cd77`
and upstream `image_encoder.py` SHA-256
`7e8194f1c93d002cdd96c60bdb3ac71443dc669eb172478a6178992c83828c53`.
The second CLS token is the frozen synthetic-caption-aligned source choice;
the model keeps its native 448-grid positional embeddings and interpolates
to 224-pixel inputs. Use the upstream PIL bilinear resize and `ToTensor`
preprocessing with FP16 model inference.

Encode all 25,882 official In-Shop TRAIN images once on DGX Spark GB10. Fit one
centered PCA-128 on the existing 13,283 fit-product rows only, pack the 12,599
product-disjoint held rows as signed int8 plus f16 inverse norm, and score the
same symmetric self-excluded gallery with stable ordinal ties. Compare against
the stronger existing pretrained 22-block SigLIP2 Large/256 own-PCA baseline:
packed R@1 **87.1736%**, mAP@R **0.508923**. Require TIPSv2 packed mAP@R
at least **+0.005** above that baseline with paired product-bootstrap 95%
lower bound above zero, R@1 no lower, export wall at most the historical
SigLIP2 24-block source export **216.163 s**, and peak CUDA below **10 GB**.
Failure stops this source route before training, official TEST or production.
A pass only permits one 100-update matched training feasibility screen and
paired image-to-top-k latency check; it cannot establish a better system by
itself. The sequential export-wall ceiling is a source cost screen, not a
matched production latency claim. Potential overlap between the model's
pretraining data and the benchmark is not independently excluded, so this
TRAIN-only comparison is exploratory.
