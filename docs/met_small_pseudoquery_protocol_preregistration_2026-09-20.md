# MET-small powered pseudo-query protocol gate

## Question

MET-small's 129 official validation queries are too coarse for selecting among
nearby compact codecs: one Recall@1 outcome is `0.007752`. This diagnostic asks
whether a deterministic gallery-derived pseudo-query protocol provides a
higher-power secondary development signal without opening the official test.

The proxy is permitted to rank codec hypotheses only if it reproduces the
already-observed direction of a coarse source-float versus PCA64 contrast on
the shifted official validation queries.

## Frozen construction

- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- for each train/gallery class with at least two rows, choose the
  lexicographically smallest normalized `train_paths` row as one pseudo-query;
- remove every chosen row from the fit/gallery population;
- retain every singleton row and every nonchosen repeated-class row in the
  fit/gallery population;
- fit centered PCA64 only on that reduced gallery;
- normalize source and PCA64 rows and rank by deterministic cosine top five;
- score both the powered pseudo-queries and all 129 official validation
  queries against the same reduced gallery;
- report mMP@5 and Recall@1 with per-query vectors for source float768 and
  PCA64 float.

No class labels beyond defining relevance and the deterministic one-row
holdout may tune a representation. The pseudo-query population has gallery
positives by construction but lacks the phone-photo query/gallery shift of the
official validation set; it is a development proxy, not a replacement test.

## Frozen interpretation

`proxy_direction_valid` is true only if PCA64-minus-source has the same strict
sign on pseudo-queries and official validation for both mMP@5 and Recall@1.
Zero is treated as disagreement. If false, no codec may be selected using this
proxy. If true, the proxy may power a preregistered label-free codec comparison,
while official validation remains a secondary shift-sanity check and the
official test remains sealed.
