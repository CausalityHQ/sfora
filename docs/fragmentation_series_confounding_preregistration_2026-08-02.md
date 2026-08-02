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

## Result (2026-08-02)

The registered collapse prediction **failed in all three seeds**. After exact
series-size-signature matching plus the two geometry quintiles, adjusted
fragmented-minus-connected R@1 remained **+5.476**, **+5.744**, and **+5.587
points**, with **58.21%**, **61.03%**, and **59.45%** coverage. All exceed the
registered +2.0 replicated-survival boundary. Matching only exact class size and
series count gave nearly identical **+5.476**, **+5.749**, and **+5.605**.

The analyzer SHA-256 was
`d708e1f6ae7694743fc569c5763a2a8f579dcb0ed21831a3126cb96aa07ca865`.
Result JSON SHA-256 values for seeds 0--2 were respectively
`d394ef546a78a5e1b6540b6683e9e650766394f4d9d1b334908e216d728194a7`,
`28cb3a701967ba0de92e9049412e703c431c53f68b90f57be2fe1a29c1c5d403`, and
`f53f8413ab406ce01980abd6d779816f805aeb15c1a9bbfb20480a295d58f2f9`.

Series composition therefore explains the component partition but not the
positive outcome association. This is a useful negative against the tempting
“more colourways/classes are simply easier” account. It remains observational:
exact cells balance the registered covariates, not every latent property, and
fragmentation is measured from the same embeddings as leave-one-out retrieval.
No occupied series/hierarchy/clustering operator is revived.
