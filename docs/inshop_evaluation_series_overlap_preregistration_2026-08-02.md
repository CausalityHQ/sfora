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
