# In-Shop rank-matched geometry repair (2026-08-07)

The corrected GPU diagnostic uses actual BN-Inception average-plus-max pooling
and equal extreme-value levels: train uses the second foreign cosine (25,882
rows) and query→gallery uses the first (12,612 rows).

At initialization, foreign cosines were **0.8305031** (train rank 2) and
**0.8304653** (query/gallery rank 1), an excess of **−0.0000378**. Corrected
final embeddings for seeds 0–3 were:

| seed | train rank-2 | query/gallery rank-1 | unseen excess |
|---|---:|---:|---:|
| 0 | 0.6672713 | 0.7094550 | +0.0421836 |
| 1 | 0.6634026 | 0.7062699 | +0.0428674 |
| 2 | 0.6670100 | 0.7085656 | +0.0415556 |
| 3 | 0.6671184 | 0.7087702 | +0.0416517 |

Mean trained excess is **+0.04206** (SD 0.00057), while initialization is
effectively zero. This supports a training-associated change under corrected
pooling and pool size. It is not yet a full Gate-1 closure: initialization is
pre-head 1024-D while trained artifacts are final signed 512-D embeddings.
A final-head-matched control (or trained pre-head export) remains required.

No GPU training candidate was started. GPU use was limited to the diagnostic
cache and rank computation; generated arrays remain on the DGX.
