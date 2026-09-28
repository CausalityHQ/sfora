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

## Terminal CPU result

The single original run exited 0 in **4.821838 CPU seconds**, with no
encoder training or held/official evaluation. Receipt:
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-classifier-cached-v1.json`,
SHA-256 `ace3f13e92bc0357c195b307da3fde128780f33a9f7a9a8c12af7b0b8200f29f`.
Protocol and script were pushed at `76e7974b` before metrics were read.

| Gate | Measured result | Decision |
|---|---|---|
| TRAIN fit leave-query-out prototype classification | 452/512 (88.28125%) full vs 435/512 (84.9609375%) PCA-128; +3.3203125 pp, product-bootstrap 95% [+1.7578125, +5.078125] pp | GO |
| Full gradient outside compact/source span | Median 42.3683% | GO |
| True versus category-shuffled unused gradient difference | Median 4452.5822 times true norm | GO; not a strength claim |
| Compact gradient numerical span check | Max relative residual 7.85698e-7 | GO |
| Training cost / serving cost / exact packed retrieval | No encoder run; no serving or packed-scoring change | Not measured by this gate |

Independent receipt replay verified the hit totals, gain, bootstrap interval,
all criteria and wall bound. There were 18 source-only wins and one
compact-only win. Two focused checks cover the projection and actual positive
query removal; Ruff passes. The unusually large shuffled-gradient ratio is
compatible with a tiny true gradient at nearly correct prototypes; it does
not authorize treating this objective as a strong update. A design review
must resolve initializer, loss, coefficient, gradient saturation and shared
clipping before the separately frozen 17-update smoke. The deployed library
is unchanged, and this result cannot establish unseen retrieval quality.

One dual design critique is active, group `781ca331ba644f6b`: Opus
`be0ee1b940794661`, Astra `0ff1b9a4c56b4f65`. No GPU job was started.
