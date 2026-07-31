# Candidate 5: local-chord positive expansion

Status: gate 1 passed; prior-art audit required before implementation or GPU use.

## Gate 1 — provenance: PASS

The repository now contains a direct training-only measurement of which
same-class relations survive independent optimization. Five CUB training
embedding packs contain the same 5,864 images and example IDs from independent
runs. Within each class:

- pair-similarity ranks correlate **Spearman 0.863** across run pairs (10th–90th
  percentile 0.808–0.915);
- top-5 positive-neighbour Jaccard is **0.411**, versus **0.045** by chance—a
  **9.06×** enrichment;
- top-1 agreement is **0.376**, versus **0.0088** by chance; and
- 12.6% of images have the exact same nearest class-positive in all five runs.

Thus “which same-class pairs are locally compatible” is reproducible structure,
not merely one trajectory's geometry. A complementary test rejected the broad
pseudo-attribute story: clustering class-centred residuals into 4–32 global
modes gives cross-run adjusted Rand indices of only **0.06–0.07**. The stable
signal is local and class-specific, not a transferable global pose/sex/viewpoint
taxonomy.

These measurements explain two earlier failures. One proxy treats all images in
a class as equivalent; sub-centres discretize the whole class and fragmented it;
Tversky attraction over-collapsed distant positives. The stable local graph
offers a narrower expansion target.

## Proposed supervision

Using only the benchmark's training images, compute frozen-initialization
features once and construct reciprocal top-k graphs separately inside each
class. Only graph edges are eligible for **local chord expansion**: intermediate
backbone features from the two endpoints are convexly mixed and the virtual
point is supervised as belonging to the same class. Non-neighbouring
same-class pairs are never mixed. Ordinary Proxy Anchor supervision remains on
all real images.

This adds virtual positive support along empirically stable local chords. It is
not a multi-centre method: there is still one proxy per class, no assignment of
images to sub-proxies, and no attempt to discretize modes. It is not external
generation: there is no diffusion model, text encoder, imported image, or
additional image synthesis. Mixing existing in-batch features can replace an
equal number of ordinary feature rows, keeping backbone work approximately 1×.

The candidate is allowed to proceed only if gate 2 distinguishes the exact
combination—training-only reciprocal positive graph plus local feature-chord
expansion—from generic MixUp, metric-learning embedding expansion, proxy
synthesis, manifold interpolation, SoftTriple, and sub-centre ArcFace.

