# Pass 73 — repaired scalar-loss search: NONE before GPU

The Fable→Claude review first validated its scorer against the repo artifacts:
seed-0 official In-Shop R@1 0.9137009, seed-0 train LOO 0.9950158, and the
four-seed frozen-final mean 0.9153890. It then computed a matched query/gallery
protocol on all four corrected checkpoints. Seen-identity R@1 was 0.99246,
0.99304, 0.99239, and 0.99232 versus unseen 0.91370, 0.91680, 0.91511, and
0.91595. Thus the seen/unseen gap is real (~7.7–7.9 points), but any loss based
only on already-correct training retrieval decisions has at most 0.75% of rows
to affect. Continuous Gram-matrix variants reduce to the already exhausted
reweighting family; non-Gram channels map to the previously closed per-sample,
pair, group, order, transform, or fixed-node families. No candidate survived
Gate 1–2 and no GPU run was authorized.

The review also found a protocol error worth preserving. Pass 60 closed a
class-separation lane using CUB's 1.08:1 between:local ratio, but the same
diagnostic on In-Shop is 3.05, 3.11, 3.22, and 3.15 for the four seeds; after
stratifying by gallery quality it remains above 2:1 (2.18 at the strictest
stratum). This reopens a measured In-Shop premise but does not itself identify
a novel method. In-Shop's 3997 classes and 512-D head also make it structurally
overcomplete, so code-capacity mechanisms are poor one-seed screens.

The next missing Gate-1 measurement is an untrained ImageNet BN-Inception trunk:
compare seen versus unseen nearest-impostor cosine before In-Shop training. If
the +0.0383 impostor gap is already present at initialization, it is not
train-time-removable; if absent, it supplies a causal target for a new method.
This is inference-only and should precede any GPU training. Full consultation
sources include Proxy Anchor (CVPR 2020), CenterPolar/Beyond Seen Bounds,
CouCE, chance-constrained DML, and novel-class DML generation; none supplies a
defensible new candidate for this protocol.
