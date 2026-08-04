# Corrected In-Shop seed-1 training-geometry replication

Date: 2026-08-04. Status: **completed, training-only, no GPU training**.

## Bound artifacts

The analysis used the independently exported final seed-1 training pack,
SHA-256
`5fd172cf7a69bcb1cf6b715793f8a81f12ecf7382a69416f6a3343ca301e3f4d`,
and its matching final checkpoint,
`a25dc22691981e6ad7df899878f448d96d4ac41adbb8e346e10322e93883e580`.
Both scripts reject mismatched digests and neither loads query or gallery data.

## Replicated geometry

| statistic | seed 0 | seed 1 |
| --- | ---: | ---: |
| train leave-one-out R@1 | 0.995477 | **0.995593** |
| eligible leave-one-out errors | 117 | **114** |
| negative nearest-same margin fraction | 0.004523 | **0.004407** |
| negative labelled-proxy margin fraction | 0.001082 | **0.001120** |
| fragmented eligible classes | 1,588 / 3,975 | **1,563 / 3,975** |
| fragmented-class fraction | 0.399497 | **0.393208** |
| median nearest-positive cosine | 0.921561 | **0.921856** |
| median nearest-foreign cosine | 0.676180 | **0.671277** |

The central conclusion replicates: the model nearly saturates observed training
relations, while roughly two-fifths of eligible class 1-NN graphs remain
disconnected. Fragmentation is not evidence of failure to fit.

## Confusion agreement and data hygiene

Nearest-foreign-image / nearest-foreign-proxy class agreement changed from
0.156904 on seed 0 to **0.125918** on seed 1. Its error stratification remained
large: error given agreement was **0.028536** versus **0.001459** given
disagreement, compared with 0.023886 versus 0.001466 on seed 0. The agreement
prevalence is seed-sensitive, but the low disagreement error and elevated
agreement error replicate.

This does not create an inferential residual beyond ordinary image margins.
For all 25,870 non-singleton rows on both seeds, leave-one-out error is exactly
equivalent by definition to a negative nearest-same minus nearest-foreign image
margin. A logistic significance test is therefore invalid through complete
separation.

The near-duplicate audit independently rediscovered the exact same cross-
identity flat-product pair on seed 1: perceptual-hash distance 0, grayscale
correlation 0.999984, and two explained leave-one-out errors. Two errors remain
below the prospectively fixed materiality threshold of 13.

## Method-search consequence

The replicated facts are dataset/model properties, not a live method. Acting on
confusion agreement is hard-negative-class mining or graph-consistency
supervision; repairing fragmentation is contradicted by the project's matched
positive fragmentation association and occupied by multi-centre/topology
methods; duplicate handling is immaterial here and occupied by noise-resistant
DML. The new seed strengthens the reliability boundary but supplies no new
supervision object and authorizes no candidate GPU.
