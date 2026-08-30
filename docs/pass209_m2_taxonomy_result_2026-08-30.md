# Pass209 M2 blinded taxonomy result

Status: **terminal, claim-ineligible evidence**.

The authenticated SigLIP-so400m error manifest contains 103 retrieval errors
from the frozen `1,242 / 1,345` Cars train-band result. Two fresh independent
raters completed the frozen SHA-derived viewing orders and passed the strict
submission validator before comparison.

## Authority

- error manifest SHA-256:
  `64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611`
- rater 1 submission SHA-256:
  `58ca0dbb08477e2955ce9ee2a0e39cb9d78d1a1efcceb8653baff39acf8fef87`
- rater 2 submission SHA-256:
  `25d2cf5181e0dc1cb4b5b2d81a44f5965f4aeae5f94887b9c925a40bed96ad20`
- conservative all-unresolved consensus SHA-256:
  `cffc4c849409e6e370a9becfc573f9d1b740383c6a29b2983f0f40a7e2e5c421`
- canonical taxonomy receipt SHA-256:
  `a63478f0e9e432f8e451f1178f3543572abc6e781006656f33b52e562c9e99eb`
- canonical taxonomy receipt bytes: `268,942`

The canonical receipt has exactly one trailing LF, validates through
`validate_taxonomy_receipt_bytes`, and contains no adjudicated rows or
bootstrap result because the reliability gate failed.

## Raw agreement

- matches: `61 / 103`
- raw agreement: `0.5922330097087378`
- Cohen's kappa: `0.39437211255774873`
- generalized nine-category PABAK: `0.5412621359223301`
- primary disagreements: `42`
- conservative `cannot-judge` or unresolved count: `42`

The preregistered alternatives require raw agreement at least `0.80` or kappa
at least `0.60`, and no more than 15 `cannot-judge` or unresolved pairs. Both
reliability boundaries fail.

Rater 1 prevalence was duplicate 12, suspected-label-integrity 8,
semantic-overlap 43, visually-indistinguishable 13, localized-cue-visible 14,
and global-shape-overridden 13. Rater 2 prevalence was duplicate 12,
suspected-label-integrity 8, semantic-overlap 73,
visually-indistinguishable 1, localized-cue-visible 2,
global-shape-overridden 6, and degraded-observation 1. The disagreement is
therefore primarily the rule boundary between semantic class overlap and
visible fine-grained evidence, not a missing-row or viewing-order defect.

## Decision

The frozen decision is `F-NONE`. M2 does not authorize capacity, invariance,
transfer, data-cleaning, or any other trainable family. Consensus cannot repair
the failed raw reliability gate, so every disagreement was conservatively left
unresolved and no third label was introduced.

The next admissible action is another prospectively frozen, objective,
non-parametric mechanism measurement. The ongoing three-seed pooled control
continues independently to produce M1 and M3 transfer evidence; neither its
partial checkpoints nor this ineligible taxonomy may select a method.
