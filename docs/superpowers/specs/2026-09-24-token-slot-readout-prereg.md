# SOP cached-token readout falsifier (exploratory preregistration)

The B/16@336 area-resampled adapter failed its preregistered advancement
conditions. This is a new architecture screen at 224 pixels, using the
authenticated UNICOM B/16 checkpoint and existing SOP TRAIN block-11-input
token cache. It tests whether replacing UNICOM's position-flatten feature
head with content-addressed slots improves the learned 128-D packed vector
while reducing training and serving weight cost. The slot mechanism resembles
published attention-pooling and Set Transformer readouts; novelty is not
assumed. The result is exploratory on a reused TRAIN holdout.

## Data and fixed procedure

The only training source is the official SOP TRAIN inventory: 59,551 images
and 11,318 products. The seed-179019 product split gives 53,700 fit images
and 5,851 holdout queries from 1,132 distinct products. All heads, proxies,
and ridge initialization fit on the fit rows only. No TEST metadata, images,
labels, or descriptors enter this screen. Evaluate holdout-only R@1/mAP@R
and R@1 against the 59,551-row full TRAIN gallery with deterministic self
exclusion. Report normalized 128-D float and 130-byte packed scores.

Use the same authenticated block-11-input cache, one 1,595-update
identity-balanced schedule and seed, last transformer block trainable in
evaluation mode, ArcFace margin 0.3 and scale 64, and 128-D PCA-initialized
head and class proxies as the prior tail control. Reset encoder, head,
proxies, optimizer, and RNG for each arm. Record exact optimizer updates,
source SHA-256s, training/preparation/evaluation wall times, CUDA allocation,
host RSS, model parameters and output checkpoints. Write one per-arm receipt
as soon as it finishes, then a paired terminal receipt. Run GPU arms
sequentially in one durable job.

Fit a single ridge map from the mean of the frozen post-block tokens to the
unit-normalized authenticated 768-D source descriptor, using fit rows only. Fix ridge
regularization to `0.001 × trace(XᵀX)/768` and no tuning. Use that map to
initialize both mean and slot arms. The four slot queries start with
seed-179019 Gaussian scale 0.01; their output projection averages four
identical ridge maps at step zero. The resulting near-mean initial output,
fit-only ridge cost, source-descriptor agreement, and the zero-update M/S
retrieval scores are diagnostics. Record whether the slot attention maps
actually separate on fit rows after training.

## Arms

| Arm | Source readout | Training loss |
| --- | --- | --- |
| A0 | Original position-flatten head, frozen | ArcFace |
| A1 | Original position-flatten head, frozen | ArcFace + source-descriptor cosine anchor |
| M | Mean-pool post-block tokens, ridge-initialized linear map | ArcFace + same anchor |
| S | Four learned content-addressed attention slots, ridge-initialized projection | ArcFace + same anchor |

The anchor target is the archived **pretrained** 768-D source descriptor for
the fit image, with fixed coefficient 1.0. All arms update the same last
block, PCA-initialized 128-D head, and proxies; M and S additionally update
their small readout. A1 is the matched anchor control. A0 reproduces the
strong prior tail ArcFace control; fail preflight if its packed holdout-only
R@1 and full-gallery R@1 differ from 88.5490% and 75.0812% by over 0.2
point. Compare S with the stronger of A0/A1 so any anchor benefit alone
cannot be called a readout benefit. M isolates simple position-free pooling.

## Frozen decision

Advance S only if, relative to the stronger flatten control, packed full
TRAIN gallery R@1 improves by at least +1.0 percentage point with a paired
1,132-product bootstrap lower 95% endpoint above zero; holdout-only packed
mAP@R lower endpoint is above −0.003; and S−M full-gallery packed R@1 lower
endpoint is above zero. Online training time is a diagnostic in this cache
harness, where file gathering and tail work dominate both arms. Report the
one-time ridge fit separately and in S's total training bill. A later live
full-training comparison must beat the matched reference including that
preparation cost. If S's full-gallery R@1 gain is at most +0.5 point, or S
does not beat M, close this slot readout variant. Ambiguous results do not
advance. A pass permits CUB transfer,
matched live image-to-top-10 batch-1/32 profiling, independent seeds, and
only then a port to a stronger trunk and In-Shop. No official or SOTA claim
follows from this screen.

The cached-token run cannot measure the end-to-end training or serving speed
of a full image encoder. The cache timing is a diagnostic, not a product
claim. A slot readout may improve B/16 without reaching the published
UNICOM L/14 quality gate; the latter remains separate.
