# In-Shop untrained geometry Gate-1 diagnostic

Pass 73 found a stable corrected Proxy Anchor gap between seen-identity
retrieval (about 0.9925) and unseen retrieval (about 0.9154), plus a nearest-
impostor cosine gap of 0.0383 (unseen higher). Before proposing another loss,
we test whether that crowding is already present in the permitted ImageNet
BN-Inception initialization.

The script `scripts/measure_inshop_untrained_geometry.py` embeds the official
In-Shop train/query/gallery images with the digest-pinned
`bn_inception-52deb4733.pth`, pre-head GAP features, and deterministic center
crop. It trains nothing and fits no parameters. The primary endpoint is the
difference between train nearest-foreign cosine and query-to-gallery nearest-
foreign cosine. If they are approximately equal, the gap is intrinsic and this
lane is closed for train-time losses. If the query/gallery value is materially
higher, the crowding is training-induced and supplies measured provenance for a
new class-separation intervention. This diagnostic is not a method result and
will not be used for tuning.
