# Fit-only compact-metric selector result

## Result

A three-fold, class-disjoint selector fitted only on authorized training
classes can decide whether to deploy the supervised compact projection or its
PCA initialization.  The frozen decision rule is:

1. assign complete labels to three folds with
   `SHA256("compact-selector-v1:" + decimal_label) mod 3`;
2. in each fold, fit the existing compact metric on the other two folds and
   compare packed int8-128 retrieval with fit-only PCA on the unseen fold;
3. select the learned projection only when pooled mAP@R gain is at least
   `0.003` and pooled Recall@1 gain is nonnegative; otherwise select PCA.

The rule contains no dataset identity, class name, or evaluation-split input.
It was fixed after the Stanford Dogs development result and then evaluated
once on untouched Oxford-IIIT Pet evidence.  No threshold was changed after
the Pet result.

## Development evidence

| Dataset and split | Inner-CV mAP@R delta | Inner-CV R@1 delta | Selected | External learned minus PCA | Direction correct? |
| --- | ---: | ---: | --- | ---: | --- |
| FGVC-Aircraft, SHA-ordered class-disjoint fit classes | `+0.078215` | `+0.033013` | learned | `+0.040785` mAP@R | yes |
| CUB-200-2011, SHA-ordered 100 fit classes | `-0.006808` | `-0.000667` | PCA | `-0.000323` mAP@R | yes |
| Stanford Dogs, class-disjoint fit classes | `+0.007950` | `+0.008167` | learned | `+0.031336` mAP@R | yes |

The original conservative development rule also required a positive
class-clustered bootstrap lower bound.  Dogs exposed that requirement as a
false-negative: its lower bound was `-0.005866`, despite a later external gain
of `+0.031336` mAP@R and `+0.015218` Recall@1.  Dogs was therefore consumed as
development evidence when the simpler rule above was frozen.  It is not an
independent confirmation of that revised rule.

Development receipt authorities:

- Aircraft selector SHA-256
  `bbdac1d173f467775211cadecfced41bffdff27363ab671e6a61a48c72b612f3`;
- CUB selector SHA-256
  `1459b9a2d2a8c7504d84b3a85febfcee7a2faac251aa74fd59ba06d14f814d7f`;
- Dogs selector SHA-256
  `1630445b75d80cfbacd0291c9dc0816479f01b589a37681534c0f1c4a23dec3d`;
- Dogs external result SHA-256
  `c7d80e6a4802582e52ae6410e2071634a9018f99ec0487f66f71e983c3e1f4b8`.

## Untouched Oxford-IIIT Pet confirmation

The confirmation used the official Pet trainval images for 19
SHA-name-ordered fit classes and official test images for the remaining 18
classes.  The frozen UNICOM teacher produced 1,889 fit and 1,776 evaluation
rows.  The feature archive SHA-256 is
`f15e79986efa26031df6e2a9ee297428fa9464e90b178a0a5d0d5b5c32e9a7b1`.

Fit-only three-fold evidence was:

- pooled mAP@R delta: `+0.011230`;
- pooled Recall@1 delta: `+0.001588`;
- class-clustered mAP@R 95% interval:
  `[+0.005856, +0.016646]`.

The frozen rule selected the learned projection.  The one-shot external
class-disjoint result was:

| Representation | mAP@R | Recall@1 |
| --- | ---: | ---: |
| PCA int8-128 | `0.847639` | `0.962275` |
| learned int8-128 | **`0.862759`** | **`0.969595`** |
| learned float-128 | `0.862714` | `0.969595` |
| teacher float-768 | `0.862248` | `0.970158` |

Learned minus PCA was `+0.015121` mAP@R and `+0.007320` Recall@1, so the
untouched confirmation agreed with the fit-only decision and cleared both
frozen gates.  Quantizing the learned 128-dimensional representation to one
signed byte per dimension did not reduce either reported metric at the shown
precision.

The stored selector driver still reports the earlier conservative decision
field. Pet's bootstrap lower bound is positive, so that field and the revised
frozen rule both select the learned projection; the revised result above is
computed directly from the stored aggregate deltas rather than from a rewritten
receipt.

Pet receipt authorities:

- selector SHA-256
  `d835fca611b291d8402f7b55f7a5d85875ffe13c682a896ab7599d7daa3b8d1e`;
- external result SHA-256
  `67ab728083a2a8f3920a0c24872c8ab253fa93966ecf8d5b48473bc1719cf3ce`;
- fitted checkpoint SHA-256
  `09c814443b5efda704f6028810e782afadd410a4d2171e599eaf1c577929c560`;
- selector driver SHA-256
  `0acad37083a66e266dc9f682cf61fbef1857599140c887f77e804ee213247011`;
- Pet exporter SHA-256
  `e14b3f62b4e846a9fb050d1ab3bbae2ac969494a6ef750dda8d3669388a3c3d3`.

## Standard Cars196 panel result

After the Pet confirmation, the unchanged selector and compact learner were
measured on the standard Cars196 metric-learning split.  The pinned
`tanganke/stanford_cars` revision
`9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40` supplied all 16,185 images;
classes 0--97 formed the 8,054-row fit partition and classes 98--195 formed the
8,131-row evaluation partition.  Cars196 had historical exposure elsewhere in
the project, so this is standard-protocol panel evidence rather than a fresh
confirmation.

The fit-only selector measured `+0.033108` mAP@R and `+0.003477` Recall@1.
Its class-clustered mAP@R 95% interval was
`[+0.023754, +0.043083]`, so the frozen rule selected the learned projection.
The one-shot external result was:

| Representation | mAP@R | Recall@1 |
| --- | ---: | ---: |
| PCA int8-128 | `0.767869` | `0.970852` |
| learned int8-128 | **`0.825656`** | **`0.972574`** |
| learned float-128 | `0.825687` | `0.973435` |
| teacher float-768 | `0.826089` | `0.974542` |

Learned int8-128 improved over same-byte PCA by `+0.057787` mAP@R and
`+0.001722` Recall@1.  It retained 99.95% of the float teacher's mAP@R
(`0.825656 / 0.826089`) while using 128 signed bytes per stored item.  The
remaining teacher gap was `0.000434` mAP@R and `0.001968` Recall@1.

Cars196 receipt authorities:

- feature archive SHA-256
  `6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3`;
- selector SHA-256
  `0b2724620bd9494ffdd7c185c78c6143dd485c94abff6a33a43140ffa2023cfc`;
- external result SHA-256
  `def8d3250561af82c9401a8a55674204dcccfea3f583723947550b9c4a849444`;
- fitted checkpoint SHA-256
  `1cd1d633764fd53caefb49f0aa091feb49370455d5e16dfb6c334d7c88a58e9f`;
- exporter SHA-256
  `1b00133936b1c289c4c6d532fb3d7c8ca9937eb25c74cff0424cffc716f8fbe5`.

## Scope

This result validates fit-only model selection for the existing supervised
compact projection.  It does not establish a universal frozen projection:
CUB and the six-domain transfer panel already falsified that stronger claim.
The Pet split is a fresh class-disjoint confirmation of the selector, not a
comparison with published Oxford-IIIT Pet systems that use the standard
same-class train/test protocol.  Cars196 adds a standard zero-shot retrieval
panel result but is not method-specific untouched evidence.  The next
production step is to expose this fit-only choice without changing the compact
encoder or ANN serving path.
