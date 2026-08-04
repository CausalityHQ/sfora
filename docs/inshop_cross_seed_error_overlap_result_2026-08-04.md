# Corrected In-Shop cross-seed error-overlap result

Date: 2026-08-04. This result was computed after commit `8936e67`
registered the analysis, thresholds, implementation, and unit test.

## Decision

The corrected Proxy Anchor seeds have predominantly stable official-query
errors. Their error-overlap coefficient is **0.7675401522**, above the locked
`0.70` stable-error boundary. This is a descriptive result, not evidence for a
new method.

The secondary same-wrong-identity fraction is **0.6475770925**, below its
`0.70` boundary. Thus the seeds often fail on the same query without reliably
selecting the same wrong identity.

## Locked outputs

| Quantity | Result |
|---|---:|
| Seed-0 final R@1 | 0.9137009425 |
| Seed-1 final R@1 | 0.9167956112 |
| Seed-0 / seed-1 errors | 1,227 / 1,183 |
| Both correct | 12,716 |
| Seed 0 only correct | 275 |
| Seed 1 only correct | 319 |
| Both wrong | 908 |
| Error-overlap coefficient | **0.7675401522** |
| Error-set Jaccard | 0.6045272969 |
| Exact top-1 gallery-row agreement | 0.8084822057 |
| Predicted-identity agreement | 0.9357152905 |
| Same wrong identity given both wrong | 0.6475770925 |
| Oracle either-seed R@1 | 0.9361372908 |

The oracle number is not deployable single-model performance. It uses the test
label to count a query correct whenever either model is correct and is reported
only because it was prospectively locked.

## Artifact and implementation checks

- Seed 0 checkpoint:
  `2b46a68a0364cd204e60068858198f1da699f043897fc93d0c22525b6f635546`.
- Seed 1 checkpoint:
  `a25dc22691981e6ad7df899878f448d96d4ac41adbb8e346e10322e93883e580`.
- The loader required `final_training_state`, paired query/gallery checkpoint
  digests, and identical labels, example IDs, and source paths across seeds.
- The source exporters had already bound both packs to the same independently
  verified official partition and byte-content manifests.
- Float64 cosine retrieval gave byte-identical JSON with chunk sizes 512 and
  127.
- Result/log SHA-256:
  `fc249d020291866c32cf4ba0f87828f470effca8b5be3b7803250f7ce101a639`.

## Consequence

Initialization changes some decisions, but most errors persist at the query
level. The result does not authorize consensus training, cross-seed
distillation, weighting, mining, or an ensemble: those operators are already
occupied, and this audit supplies no new supervision. No GPU follow-up is
authorized.

