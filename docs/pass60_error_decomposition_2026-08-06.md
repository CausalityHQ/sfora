# Pass 60 cheap diagnostic: CUB error decomposition

Using the existing CUB HERD seed-0 train/test embeddings (read-only export from
the DGX), I computed cosine nearest neighbours and test-class centroids. Test
R@1 is 0.69396 (1813/5924 failures). For each failed query I compared its
similarity to its own class centroid with similarity to the nearest *other*
class centroid.

| failure signature | count | fraction of failures |
|---|---:|---:|
| nearest wrong class image, but own centroid remains closer than the nearest wrong centroid (local within-class dispersion) | 873 | 48.1% |
| own centroid is no closer than the nearest wrong centroid (between-class overlap) | 940 | 51.9% |

The registered Pass-59 reopening condition (between-class errors dominating by
more than 2:1) is **not met**. This does not prove within-class information is
irrelevant; it says neither a pure anti-collapse nor a pure class-separation
intervention is justified by this diagnostic. The result is a useful constraint
for the next blind search: seek a mechanism that addresses the coupled local
and class-level error without being a similarity-matrix reweighting or an
ordinary diversity regularizer.

No GPU run followed this diagnostic.
