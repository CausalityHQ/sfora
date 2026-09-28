# Source-classifier gradient gate

The global-caption arm is closed. Test one label-only mechanism: all current
identity/rank losses pass through the 128-D affine head, restricting each
pooler's gradient to the head row space plus its normalization direction.
A training-only 1,024-D source classifier could supply useful identity
directions outside that span, without changing the deployed head/scorer.
This is auxiliary supervision, not a novelty claim. Earlier raw-pooler
versus trained-head retrieval did not train such a source classifier.

Before training, run one **<=120-second CPU** diagnostic from the pinned
pretrained cache and original fit-only PCA initializer. Select 512 fit
products with at least two images and at least two products in their clothing
category, ordered by SHA-256 of `inshop-source-classifier-v1\0` plus product
name. Select one image per product by relative-path SHA. No held outcomes.

Compare centered, normalized full-source and PCA-128 class centroids.
Remove the query image from its own positive centroid in **both** spaces;
keep other class centroids fixed. Use all 2,004 fit-class centroids as
candidates, ordinal ties. This is fit prototype classification, not image
retrieval or held Recall@1. Measure ordinary cosine CE at fixed scale 64
for gradients; do not claim it reproduces ArcFace's margin gradient.

Frozen advance rules:

1. Full minus compact prototype classification accuracy at least **+1.0
   percentage point**, with paired product-bootstrap lower95 **above zero**.
2. Median full-source label-gradient norm outside the compact-head/source
   span at least **30%** of its whole gradient.
3. Median true-versus-category-shuffled label-gradient difference in that
   unused span at least **30%** of the true unused gradient.
4. Compact gradient outside the same span at most **1e-4** relative norm;
   finite statistics, source/PCA/partition/fit hashes and positive inventory
   validate; total CPU wall at most 120 seconds.

Any failure stops this fixed source-centroid classifier route before an
encoder run. A pass only authorizes a separately frozen gradient/cost smoke,
then a matched TRAIN held gate. Orthogonality is local at the pooler, not a
guarantee about shared encoder parameter updates or generalization.
