# In-Shop untrained geometry result

The preregistered diagnostic ran on the DGX with the digest-pinned
`bn_inception-52deb4733.pth`, pre-head GAP features, deterministic center crop,
and no training. It used the verified standard `Img/img` corpus (the first
attempt was correctly rejected because `/home/riomus/datasets/inshop` resolved
to the contaminated `img_highres` tree).

| population | nearest-foreign cosine |
| --- | ---: |
| In-Shop train identities | 0.8353818 |
| query → gallery unseen identities | 0.8244556 |

The initialization does **not** contain an unseen-crowding excess; it is 0.0109
lower on the unseen side. Combined with the four trained corrected checkpoints
(trained unseen nearest-impostor 0.7083 versus seen 0.6700 in the matched
embedding protocol), this supports a measured provenance statement: the
unseen-crowding failure is created during metric training rather than inherited
from the ImageNet BN-Inception representation. This is a diagnostic, not a
method result; it uses a different pre-head representation than the trained
512-D head, so the next candidate must include a matched final-head control.

Artifact: `reports/generated/inshop_untrained_geometry_2026-08-07.json` on the
DGX run workspace (generated reports are intentionally ignored by Git).
