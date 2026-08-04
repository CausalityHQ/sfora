# IPSR corrected-corpus Gate-0 result

Date: 2026-08-04. Preregistration:
`ipsr_corrected_corpus_gate0_preregistration_2026-08-04.md`, commit `56655b9`.

## Result

The corrected official-pixel premise passes every frozen condition:

| statistic | result | gate |
| --- | ---: | ---: |
| IPSR preferences | 17,093 | descriptive |
| anchor coverage | 0.660420 | >= 0.50 |
| eligible-class coverage | 0.773836 | >= 0.50 |
| mean initial Bradley--Terry loss | 0.733181 | >= 0.70 |
| closest-quartile pairs rejected | 0.560879 | >= 0.10 |
| farthest-quartile pairs accepted | 0.291475 | >= 0.10 |
| response-graph density | 0.361375 | descriptive |
| multi-component eligible classes | 0.784654 | descriptive |
| valid response signatures | 1.000000 | descriptive |

The response label is therefore not a disguised distance threshold. More than
half of geometrically close same-class pairs disagree in intervention response,
while nearly three tenths of distant pairs agree. The original premise survives
the pixel-corpus repair and is slightly stronger than its quarantined counterpart.

## Independent recomputation

The first run persisted only aggregate JSON, so it was not accepted as a completed
Gate-0 audit. Commit `9e9700e` added a raw pack export and a second NumPy auditor
that imports none of `sfora.arcg`, `sfora.ipsr`, or the training objective. A fresh
six-view forward pass produced:

- `reports/emb/ipsr_corrected_gate0_seed0.raw.npz`, SHA-256
  `76b8312b11dde801cbdbf05d50a6d7b583b037a9843f912b6e95354a2f4ac19c`;
- production aggregate, SHA-256
  `e1cc140bf2df1b87d6e662c453f0fbedd7a8205967b51a36c3d0c1b8a463cea3`;
- production IPSR aggregate, SHA-256
  `749831034fe654e7ad7269c5362ed82fda6be79107d4a38b9f08b1424809a705`;
- independent aggregate, SHA-256
  `4fc906ec1a56da68a9c71ef70877b9d92415ce6cb55675a35b964258130c1727`.

The auditor asserted 25,882 rows, 3,997 identities, 512 dimensions and the fixed
five transformed-view order. It independently reconstructed normalization,
median/MAD signatures, graph edges, stable quartiles, connected components and
the contradicted-preference selection. All ten reported statistics agree exactly
with the production implementation. `tests/test_arcg.py` and `tests/test_ipsr.py`
also pass (4 tests).

## Decision

**Gate 0 and measurement provenance pass; no IPSR training is authorized.** The
old Gate-4 death is retracted because it used `img_highres`, but its small positive
delta is retracted too. The valid local final Proxy Anchor reference is 0.9137009,
whereas the audited comparable In-Shop horizon is 0.939. IPSR would need +2.530
points merely to reach that horizon. The corrected diagnostic establishes the
existence and non-reduction of the relation, but provides no quantitative evidence
that enforcing its ordinal targets repairs unseen-identity errors, much less an
effect of that size.

Consequently the current IPSR operator stops before Gate 3: a forecast above the
horizon would be invented rather than measurement-derived. No corrected screen,
controls, extra seeds or replication are run. The result remains useful input to
future candidate generation: controlled transformation response reveals real
within-class strata on official In-Shop, but a relation's prevalence is not a
causal estimate of retrieval benefit.
