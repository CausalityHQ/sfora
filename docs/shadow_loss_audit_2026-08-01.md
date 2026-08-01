# Shadow Loss algebra audit — 2026-08-01

Primary source: anonymous withdrawn ICLR 2026 submission, *Shadow Loss:
Memory-Linear Deep Metric Learning with Anchor Projection*, OpenReview
`3fx0Kz6Zfl`.

The paper defines, for anchor `a`, positive `p`, and negative `n`, projection
gaps

`delta+ = | ||a|| - a.p / ||a|| |` and
`delta- = | ||a|| - a.n / ||a|| |`,

then applies `max(delta+ - delta- + margin, 0)`. Its convergence analysis
explicitly assumes `a`, `p`, and `n` are L2-normalised. Under that assumption,
cosines lie in `[-1,1]`, so both absolute-value arguments are nonnegative:

`delta+ = 1 - a.p`, `delta- = 1 - a.n`.

Therefore

`L_shadow = max(a.n - a.p + margin, 0)`,

which is exactly the standard cosine-similarity triplet hinge. The advertised
one-dimensional anchor projection does not define a different objective in the
stated normalised regime. Computing dot products may change an implementation's
temporary storage, but it does not change the similarity supervision or explain
an accuracy gain against a correctly implemented cosine-triplet control.

The submission reports no seed count or uncertainty and is withdrawn. It does
not supply a novel operator or a credible external ceiling for this project.
