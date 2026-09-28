# Native update rank-32 diagnostic: approximation premise KILL

The original DGX Spark **CPU-only** unit `sfora-native-delta-rank32-v1`,
invocation `272d5a80895d4761bdd0204de5012b62`, ended with exit0. Script wall
was **4.116 seconds**. It used existing trained and pretrained weights; no
images, features, model forward, CUDA work, training or quality evaluation.

## Measured result

Pinned seed179026 In-Shop TRAIN-fit true-freeze checkpoint versus its exact
SigLIP2 Large/256 pretrained source. Fixed rank32, exact CPU SVD, lexical
order over 72 upper-block attention/MLP weight matrices. The frozen rule
stops at the first matrix retaining less than 80% of its raw update's squared
Frobenius norm.

| First inspected matrix | Measured |
| --- | ---: |
| `encoder.layers.12.mlp.fc1.weight` shape | 4,096 × 1,024 |
| Native dense parameters | 4,194,304 |
| Rank32 factor parameters | 163,840 |
| Captured raw weight-change energy | **40.6174%** |
| Best rank32 residual / raw change Frobenius norm | **77.0601%** |
| Raw change / pretrained weight Frobenius norm | **1.6813%** |
| Remaining matrices unrun under early stop | **71** |

This is a result about approximating a particular trained matrix, not an
encoder quality result. Raw differences include optimizer weight decay and
all learned changes. A small weight residual can still affect retrieval;
Frobenius capture is not a Recall, generalization or hardware latency bound.
It neither tests nor disproves low-rank regularization's ability to learn a
different, better solution. No rank sweep, adapter training or production
parameterization change follows this failed fixed approximation premise.

The native training and deployed image-to-top-k path remain unchanged.
LoRA is a [known method](https://arxiv.org/abs/2106.09685), not a novelty claim.
Any future low-rank learning claim needs matched quality, actual training cost
and merged public-path parity rather than inference from this spectrum.

Verification checked pinned weight hashes on DGX, source against the prior
freeze, tensor geometry, parameter arithmetic, early stop and terminal exit.
Spectrum arithmetic has identity/known-low-rank runnable self-checks; the
measured SVD spectrum was **not independently recomputed**.
Receipt SHA256: `4b7a111bef904be4c9cf657738804415f8ff35b93f70d282295f234090831069`.
[Raw evidence](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-native-delta-rank32-v1/).

## Corrected DINOv3 access blocker

The fresh authenticated attempt on DGX returned `LocalTokenNotFoundError`.
`huggingface_hub.get_token()` also finds no credential locally. Therefore
**owner grant status is unknown**; an earlier anonymous401 did not establish
an authenticated denial. The normal owner grant and credential provisioning
need reconciliation. Operator notified once of this new diagnosis; no secret
search, model weight download or access workaround occurred.
Cross-agent reporting returned `needs a live sender agent`, so the actionable
credential result was sent through the established Telegram route.
