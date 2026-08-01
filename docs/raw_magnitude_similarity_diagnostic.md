# Raw-magnitude similarity diagnostic

Date preregistered: 2026-08-01, after the aggregate and within-identity norm
results but before evaluating any magnitude-aware retrieval score.

## Question

Pre-normalisation magnitude predicts correctness and margin within unseen
identities, but that does not mean it should enter pairwise similarity. It may be
confidence only. The saved normalized directions `z` and raw magnitudes `n`
reconstruct the raw head output `f = n z`, allowing this distinction without
training or another GPU run.

## Frozen comparison

Use the corrected epoch-10 In-Shop query/gallery artifacts and exact official
labels. Report R@1 for exactly three scores, with no coefficient sweep:

1. normalized cosine, `z_q dot z_g` (the reference);
2. raw dot product, `f_q dot f_g`;
3. negative squared raw Euclidean distance, `-||f_q - f_g||^2`.

No test label may choose, scale, centre, or calibrate a score. Raw dot-product
query magnitude is constant within a query ranking, so any difference from
cosine is caused by gallery magnitude. Euclidean uses both endpoint magnitudes.

## Prediction and falsifier

Prediction: at least one raw score improves R@1 by **0.20 point** over cosine.
If neither raw score improves at all, magnitude is predictive confidence but not
useful relational information in either canonical unnormalised metric. Values
between 0 and +0.20 point are descriptive.

This diagnostic cannot establish novelty. Unnormalised DML, norm/quality-aware
similarity and uncertainty-aware comparison are prior art. It only determines
whether a future mechanism must keep magnitude out of pairwise ranking.

## Result

| score | R@1 | delta vs cosine |
| --- | ---: | ---: |
| normalized cosine | **0.84365** | — |
| raw dot product | 0.60051 | **-24.314 points** |
| negative squared raw Euclidean | 0.82163 | **-2.201 points** |

The prediction is decisively falsified: neither canonical raw score improves at
all. Raw dot product is especially destructive because query magnitude is
constant within a ranking while gallery magnitude is not; it promotes high-norm
gallery identities regardless of semantic direction. Euclidean is less extreme
but still loses more than two points.

Magnitude is therefore a confidence/difficulty observable, not pairwise semantic
information that should be multiplied into similarity. Future candidates must
preserve direction-only ranking unless they provide a separately validated
calibration mechanism; simply restoring the discarded radius is closed.
