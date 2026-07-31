# Candidate 6: matched-patch positive supervision

Status: gate 2 failed; no implementation or GPU use.

## Gate 1 — provenance: PASS

Three repository measurements motivate this candidate:

1. `region_pa` scored **0.6466 R@1**, **−3.6 pt** against its paired Proxy
   Anchor baseline. Its first implementation used slot-to-slot regional cosine;
   switching to position-tolerant MaxSim recovered **+6.7 pt**, proving that
   corresponding content moves across spatial slots under pose and framing. The
   method still failed because fixed regions were a poor supervision unit.
2. `local_nca` was intended to select a few useful positives, but its measured
   effective-positive count was **31–40 of 40 available**. It supplied almost
   uniform attraction to every same-class image—the same equivalence error as a
   single proxy—and collapsed to 0.5733 mean R@1.
3. Across five independent CUB training packs, global within-class pair ranks
   are stable (Spearman **0.863**) and top-5 positive-neighbour Jaccard is
   **0.411**, 9.06× chance. Selecting compatible image pairs is reproducible;
   the missing step is identifying which content within those pairs corresponds.

## Proposed supervision

For each minibatch, use the existing final convolutional feature grid. Restrict
candidate pairs to reciprocal same-class neighbours in a training-only graph.
Within each eligible image pair, compute detached mutual-nearest patch matches;
only mutually matched patches become positive relations in an auxiliary
contrastive loss. Unmatched patches and distant same-class pairs are neutral,
not forced positive. The ordinary global Proxy Anchor objective remains.

This expands supervision from one binary class relation to observed
instance-and-part correspondences. It derives every target from the benchmark's
training images and activations: no text encoder, diffusion model, external
image, part label, or imported pseudo-annotation. It requires no additional
backbone forward and does not change inference, so expected training cost is
approximately 1× plus token-similarity arithmetic.

It is clearly distinct from SoftTriple and sub-centre ArcFace: neither creates
patch correspondences, there is still one class proxy, and no sample is assigned
to a sub-proxy. Gate 2 must instead search dense contrastive learning,
cross-image correspondence, local-feature DML, part-aware retrieval,
mutual-nearest patch matching, and any method combining those elements on CUB,
Cars, In-Shop, or adjacent retrieval tasks.

## Gate 2 — prior art: FAIL

The relevant prior art is not multi-centre classification but DIML
([Zhao et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.html)).
DIML computes an explicit optimal matching flow between the convolutional
feature maps of two images and substitutes the resulting structural similarity
into existing metric-learning objectives. Its supplement applies that score to
Proxy Anchor as well as pair-based losses, and the paper evaluates CUB200-2011
and Cars196.

Detached mutual-nearest patch pairs are a cheaper correspondence solver than
DIML's optimal transport, and retaining global-only inference differs from its
top-K structural reranking. Neither difference creates a new supervision
mechanism: both methods let cross-image spatial correspondences determine the
positive similarity used to train a DML model. Weakly supervised semantic
alignment also predates DIML and learns dense correspondence from image-pair
category supervision
([Rocco et al., CVPR 2018](https://arxiv.org/abs/1712.06861)).

Candidate 6 therefore stops before preregistration, implementation, or GPU
screening. Its repository motivation remains useful evidence that spatially
moving content is a real failure mode, but the proposed remedy is occupied.
