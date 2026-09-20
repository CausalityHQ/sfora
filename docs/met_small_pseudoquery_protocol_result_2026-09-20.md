# MET-small powered pseudo-query protocol result

## Result

The deterministic gallery-derived pseudo-query protocol preserves the coarse
source-float versus PCA64 direction observed on the shifted official validation
queries and passes its frozen validity gate.

| Split | queries | source float768 mMP@5 / R@1 | PCA64 float mMP@5 / R@1 | PCA64 delta |
|---|---:|---:|---:|---:|
| gallery-derived pseudo-query | 3,050 | 0.590541 / 0.523934 | 0.545563 / 0.483607 | -0.044978 / -0.040328 |
| official validation | 129 | 0.695607 / 0.658915 | 0.630362 / 0.581395 | -0.065245 / -0.077519 |

Both PCA64-minus-source deltas are strictly negative on both populations, so
`proxy_direction_valid` is true. The run fit PCA64 only on the 35,257-row
reduced gallery. Exactly one lexicographically selected row from each of 3,050
multi-row train classes formed the pseudo-query population.

The absolute official-validation values differ from prior full-gallery MET
receipts because this protocol removes those 3,050 rows from the gallery. The
direction comparison is internal to this one frozen gallery and must not be
pooled with full-gallery values.

## Interpretation

The 3,050-query protocol is now a higher-power development instrument for
preregistered label-free codec comparisons. It does not reproduce MET's
phone-photo query shift and cannot replace official validation or authorize a
test reveal. A candidate must preserve its direction on official validation;
the proxy alone cannot promote a method.

## Authority

- result:
  `docs/evidence/rank_finished_l14_336_met_small_pseudoquery_protocol_gate_v1.json`;
- result SHA-256:
  `a36bc2cac740ce2c243ebec90246d2ae863669a501cb0a407db4a38ccb679ae1`;
- throwaway driver SHA-256:
  `4cd59e983e135ee62ec3e1760b61ba8385e1ddd5679e2745bd58b8403311a5bd`;
- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- official MET test image remained sealed.
