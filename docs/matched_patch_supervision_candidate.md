# Candidate 6: matched-patch positive supervision

Status: gate 1 passed; prior-art audit required before implementation or GPU use.

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

