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

## Terminal result

The DGX Spark GB10 service `sfora-inshop-tipsv2-l14-source-v3.service`
completed successfully (invocation `6ebdee0860304d6d8e69f389aa7d6cbd`).
The copied [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-tipsv2-l14-source-v1/receipt.json)
has SHA-256 `1d67b388f86668e9700dafe8011fe0068503e70bb83816b920d4a64fad22f9f5`.
The remote 25,882-image feature file has SHA-256
`bfb9bf7e0c928a697e9dc2bf16dad5e79022d4a2d57bfe0551d30bcc50bc35d4`.
The service [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-tipsv2-l14-source-v1/service.journal.log)
records the terminal exit and result. Source hash on DGX matches the pushed
script, `3d9a8f423487bb5a8ffcbbe1c59c76e9ed0d7822e6899eb39f471a1a82615669`.

| Frozen screen, held TRAIN symmetric gallery | TIPSv2 L/14 | SigLIP2 22-block baseline | Gate |
| --- | ---: | ---: | --- |
| Packed R@1 | 85.3957% | 87.1736% | Failed: candidate lower by 1.7780 pp |
| Packed mAP@R | 0.504019 | 0.508923 | Failed: delta −0.004903 vs required +0.005 |
| Paired product-bootstrap mAP@R delta, 95% | [−0.010692, +0.000824] | reference zero | Failed: lower bound below zero |
| Sequential 25,882-image encoder export | 295.853 s | 216.163 s historical 24-block source export | Failed: +36.9% over ceiling |
| Peak CUDA allocated | 0.892 GB | not comparable here | Passed: below 10 GB |

The 12,599 candidate and baseline per-query AP rows align. Recomputed mean
paired AP delta is −0.004903179582767455; four of five frozen predicates fail.
Stop this source route before training, official query/gallery evaluation, or
production promotion. This is a source feasibility rejection, not evidence
that TIPSv2 cannot improve after a different training or preprocessing recipe.
No public image-to-top-k latency or training cost was measured for TIPSv2.
