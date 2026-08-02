# In-Shop evaluation series-overlap preregistration

Recorded after the underidentified cross-series training retrieval audit, but
before parsing any query/gallery series relationship from the official
`Eval/list_eval_partition.txt`.

## Question

Training leave-one-out retrieval strongly rewards a same-series neighbour. The
official test protocol may or may not offer the same shortcut. If most queries
have at least one gallery positive from their own filename series, a method can
score well while keeping an item split by colourway/acquisition series. If not,
the training diagnostic is irrelevant to benchmark performance.

## Locked metadata-only analysis

Parse the unmodified official In-Shop evaluation partition. For every query,
collect all gallery images with the same item ID and classify availability as:

- both same-series and cross-series positives;
- same-series positives only;
- cross-series positives only; or
- no gallery positive (an invalid benchmark row).

The series token is the basename token before the first underscore, unchanged
from the training audits. Report query counts/fractions in all four categories,
the fraction with any same-series gallery positive, the fraction with any
cross-series gallery positive, and the fraction of all query-positive-gallery
pairs that share series. Do not load images, embeddings, or retrieval outputs.

## Prediction and falsification

The prospective prediction is that **> 75%** of official queries have at least
one same-series gallery positive. This is falsified below **50%**; the interval
between is inconclusive. No threshold is registered for pair fraction.

A pass establishes that the benchmark permits the shortcut, not that a trained
model uses it or that headline R@1 is invalid. A failure disconnects the
fragmentation observation from official evaluation. Either result is a dataset
audit, not a method and not permission to tune the split.

## Result (2026-08-02)

The prediction passed strongly. Of 14,218 official queries, **95.604%** have at
least one same-series gallery positive. **52.884%** have both same- and
cross-series positives, **42.720%** have same-series positives only, and just
**4.396%** are forced to retrieve across series. No query lacks a gallery
positive. Cross-series positives are available to 57.280% of queries. Across all
91,069 query-positive-gallery pairs, 27.422% share series.

The partition SHA-256 was
`cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`, the
analyzer SHA-256 was
`c918602b68668b2169caca14705494ca21fa6c56570384212716a2a07ef0bad5`, and the
result JSON SHA-256 was
`870874ac1f013e26eada1f3fab808c4d01abc84fd6f7d194cf89ffbbf0d1fd1c`.

Official In-Shop R@1 therefore permits—and for 42.72% of queries requires only—
same-series identity retrieval. This does not invalidate the benchmark: the
released task defines item identity, and series/colourway images are legitimate
positives. It does mean the fragmentation association is not evidence for
cross-mode invariance, and a series-aware improvement would be dataset-specific
unless independently replicated where such groups are absent or defined
prospectively.
