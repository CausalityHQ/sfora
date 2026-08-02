# In-Shop fragmentation series-confounding preregistration

Recorded after component-versus-series ARI was observed at 0.754--0.761, but
before matching any retrieval outcome on series count or series-size composition.

## Question

The original adjusted fragmented-minus-connected R@1 gaps (+5.875, +5.966,
+5.806 points) controlled exact class size and coarse embedding geometry, but
not how many image series comprise an identity. Since graph components recover
series, the apparent advantage may simply compare multi-series identities with
single-series identities. That would be exposure-composition confounding, not a
benefit of fragmentation.

## Locked analysis

Run independently on the exact seed-0, seed-1 and seed-2 packs. Parse the series
token before the first filename underscore, refusing any malformed path. Reuse
the unchanged symmetrized 1-NN exposure, class leave-one-out R@1 outcome, global
quintiles of mean within-class cosine, and global quintiles of nearest foreign
centroid cosine from the earlier confounding audit.

Compute two exact-stratification estimates:

1. **Primary:** exact sorted series-size signature (for example `(3, 3)`), plus
   the two geometry quintiles. The signature already fixes total class size and
   series count.
2. **Secondary:** exact class size and exact series count, plus the two geometry
   quintiles.

Within each cell containing both exposures, take fragmented-minus-connected
mean class R@1 and weight by the smaller arm count exactly as before. Report
coverage, cells, effective matched weight and the adjusted gap. Do not merge
rare signatures, alter bins, regress, or pool seeds.

## Prediction and falsification

The prospective prediction is that the primary adjusted positive gap collapses
to **<= +1.0 point in every seed**, with at least **20%** class coverage in each.

- The collapse prediction is falsified if the primary gap remains **> +2.0
  points in all three seeds** at >=20% coverage.
- Coverage below 20% in any seed, or a mixture between +1 and +2, is
  inconclusive rather than favourable.
- The secondary estimate is descriptive; it diagnoses whether series count
  alone suffices or the within-series sample allocation matters.

A collapse closes the fragmentation-derived method line: the outcome association
was carried by an observed dataset grouping whose direct and inferred operators
are already occupied. Survival would still be observational and would require a
new causal diagnostic; it would not revive pseudo-label, hierarchy, mining,
topology or diversity candidates.
