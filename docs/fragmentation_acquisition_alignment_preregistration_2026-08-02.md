# In-Shop fragmentation acquisition-alignment preregistration

Recorded after the stable component-partition result (mean pairwise ARI 0.84148)
and after verifying only that all 25,882 exported filenames parse as
`<series>_<pose-code>_<descriptor>.jpg`, but before relating any token to any
learned component.

## Motivation

Repeatable component membership is consistent with semantic within-item modes,
but also with a fixed acquisition artifact. In-Shop paths expose a series token
and explicit descriptors (`front`, `side`, `back`, `full`, `flat`,
`additional`). DeepFashion's released annotations define per-image pose types;
the benchmark is explicitly designed around pose and scale variation. A method
that preserves camera/model/background series would preserve a shortcut, while a
viewpoint result would still enter crowded pose-aware metric learning. The
distinction must be measured before proposing supervision.

Primary dataset source: Liu et al., *DeepFashion: Powering Robust Clothes
Recognition and Retrieval with Rich Annotations* (CVPR 2016) and the released
In-Shop annotation format (`list_bbox_inshop.txt`, pose type per image).

## Locked analysis

Use the same three digest-bound packs and the exact symmetrized k=1 components.
Refuse pack identity/label differences or any filename that does not parse
exactly. Restrict to the 1,439 identities fragmented in all three seeds. For
each seed and each tag independently:

- `series`: the basename token before the first underscore;
- `view`: the final descriptor token before `.jpg` (the numeric pose token is
  redundant in this pack and is reported only as a parse check).

For identities containing at least two values of the tag, compute the
permutation-invariant adjusted Rand index between the learned component
assignment and the tag partition. Report class-macro mean, class-size-weighted
mean, standard error across identities, and eligibility. Also report the paired
per-class series-minus-view ARI difference on identities eligible for both.
Do not merge tags, drop `additional`, tune a graph, or inspect images.

## Prediction and falsification

The prospective prediction is that acquisition series, not nominal view, is the
dominant fixed explanation: every seed's macro series ARI is **> 0.35** and the
three-seed mean paired series-minus-view ARI is **> +0.15**.

- The series-dominance prediction is falsified if mean series ARI is **<= 0.10**
  or the paired difference is **<= 0**.
- A broad filename-metadata explanation is falsified if both series and view
  mean ARI are **<= 0.10**.
- If view ARI is >0.35 and exceeds series by >0.15, the partitions are primarily
  viewpoint-related; if both exceed 0.25 without dominance, the result is mixed.
- Values outside these regions are inconclusive and authorize no method.

Even series dominance cannot prove background/model leakage without inspecting
or independently annotating images; a series may encode a coherent garment
presentation. Conversely, view dominance is not novelty: pose-aware positives,
pose-invariant representation learning and cross-view re-identification are
established. This diagnostic decides whether the fragmentation line supplies an
unobserved factor at all, not whether an occupied factor should become a loss.

## Result (2026-08-02)

The series-dominance prediction passed decisively. Across seeds 0, 1 and 2,
component-versus-series macro ARI was respectively **0.76070**, **0.75801**, and
**0.75411** (class-size-weighted **0.72717**, **0.72309**, **0.71912**) over
1,243 eligible stable-fragmented identities. Component-versus-view macro ARI
was instead **-0.14156**, **-0.14190**, and **-0.14251** over all 1,439. The
paired series-minus-view difference was **+0.92595**, **+0.92366**, and
**+0.92045** (mean **+0.92335**).

Thus the repeatable components overwhelmingly recover the filename series, not
front/side/back/full presentation. Inspection of one prospectively selected
multi-series identity showed the same design in black (`03`) and beige (`05`)
colourways, with multiple views inside each series. That example establishes
that “series” need not be pure background/model leakage; it can encode a real
appearance variant grouped under one benchmark item. It does not establish the
semantics of every series.

The analyzer SHA-256 was
`3f1d94f2bb5835b457a7b8a4281cdbb04f8f9cacdc5dc098c3df21674ccf6287` and the
result JSON SHA-256 was
`ce54879d844f58b5b376ece4cd49ede57dc17c172f75b5ab8903e41c42d194ed`.

The mechanism interpretation changes: fragmentation is not an undiscovered
latent factor extracted by Proxy Anchor. It is largely recovery of a grouping
already present in the training filenames. Using that grouping directly is
hierarchical/subidentity supervision and is dataset-specific; re-deriving it
from pixels is clustering/pseudo-labelling. Both operator families were already
closed at Gate 2. The remaining causal question is whether the positive R@1
association survives matching the number and sizes of series per identity.
