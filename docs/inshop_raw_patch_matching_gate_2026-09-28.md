# Raw patch correspondence screen

One hypothesis: global pooling and fixed quadrant averaging discard detailed
garment correspondences whose locations move between photographs. Test the
already trained, unprojected 256 x 1,024 final tokens with **symmetric mean
nearest-token cosine**: mean row maximum and mean column maximum, averaged.
No learned alignment, positional constraint, head projection, coefficient
search or preprocessing change. This is a training-supervision candidate,
not a proposed gallery format or a novelty claim.

The existing shared-head four-quadrant screen failed (50/151 rescues); this
changes representation granularity and matching before the head. The token
mean blend also failed, but did not retain local correspondences. The source
centroid configuration is closed: its useful-fit-direction screen did not
produce sufficiently strong encoder gradients. Do not reopen any of these
fixed configurations.

Run one read-only DGX screen on the archived 151 previously observed official
TRAIN-held miss triples (409 distinct images), pinned seed-179026 checkpoint
and original 256-pixel processor. Original packed triple margins must still
be negative and reproduce within 0.005 before accepting token results. The
baseline is the unchanged deployed global head, not a newly selected model.
Keep the exact archived checkpoint, miss, split, role and model hashes from
the existing spatial-part probe. No official query/gallery labels or new
quality selection panel is read.

Freeze GO at **at least 99/151 positive-over-impostor token wins AND median
token margin at least +0.02**, matching the earlier local-detail screen's
headroom bar. Whole process wall must be at most **60 seconds** and peak CUDA
allocation at most **4 GB**. Any source/replay/numerical/cost/quality failure
kills this exact raw-token matching configuration before training or a full
gallery export. A pass needs matched hit controls and a TRAIN-only bounded
gradient/cost gate before any learned objective. Triple rescue is not full
gallery Recall@1, does not count false-hit losses, and cannot prove SOTA.

Prior art: [MGA](https://arxiv.org/pdf/2302.08902) uses learned multiscale
local aggregation and cross-image attention; Table 1 reports 94.3% In-Shop
R@1, so it does not replace the dated UNICOM 96.7% reference. Its local-detail
motivation supports testing correspondence, not assuming the present score
works. A separate [2025 ESA paper](https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/89458/1/Rethinking%20Metric%20Learning%20Enhancing%20Generalization%20to%20Unseen%20Classes.pdf)
focuses on unnormalized feature-radius shifts and virtual classes. Sfora's
source/head/classifier cosine normalization removes that raw-radius channel;
do not adopt its radius explanation without a distinct angular mechanism.
These primary-source checks do not certify the latest published frontier.
