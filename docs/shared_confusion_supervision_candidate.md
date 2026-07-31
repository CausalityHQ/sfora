# Candidate 7: shared-confusion positive supervision

Status: gate 2 failed; no implementation or GPU use.

## Gate 1 — provenance: PASS

`scripts/measure_shared_confusions.py` used five aligned HERD CUB **training**
embedding packs and no test data. For each image it ranked all negative-class
centroids, then compared those rankings between same-class images. The top-5
same-class neighbours had a negative-profile correlation advantage of 0.0838,
0.1028, 0.0824, 0.1000, and 0.0876 across seeds 0–4 (mean **0.0913**).
Within-class embedding similarity and negative-profile similarity correlated
0.7060, 0.7370, 0.7238, 0.6887, and 0.6689 (mean **0.7048**).

The effect is reproducible but not fully independent: embeddings determine both
the positive-neighbour rank and the class-centroid responses. It nevertheless
shows that local same-class compatibility includes a consistent pattern of
which other training classes an image resembles.

## Proposed supervision

For a same-class pair, use a detached temporal target's similarities to
negative-class proxies as a response profile. Reward agreement only in an
automatically selected embedding subspace associated with the pair's shared
hard-negative responses; leave other dimensions neutral. This would replace
uniform same-class equivalence with pair-specific, training-data-only
supervision without external encoders or extra backbone passes.

## Gate 2 — prior art: FAIL

The proposal combines two established mechanisms:

- preserving distances and angles supplied by a teacher in metric learning is
  Relational Knowledge Distillation
  ([Park et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)); and
- selecting and reweighting embedding dimensions for a particular notion of
  similarity is Conditional Similarity Networks
  ([Veit et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Veit_Conditional_Similarity_Networks_CVPR_2017_paper.html)).

A vector of similarities to class proxies is a response/logit profile. Using a
temporal copy rather than a separately pretrained teacher does not change the
supervision mechanism, and deriving the mask from shared hard-negative
responses automates the condition rather than inventing a new kind of target.
Candidate 7 stops before preregistration, implementation, or GPU screening.
