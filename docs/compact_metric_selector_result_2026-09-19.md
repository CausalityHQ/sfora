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

An equal-byte standard-codec screen then fitted Faiss 1.12.0 PQ and OPQ on
exactly the same 8,054 authorized fit rows.  All four arms persist 128 bytes
per item:

| Representation | mAP@R | Recall@1 |
| --- | ---: | ---: |
| PCA int8-128 | `0.767869` | `0.970852` |
| Faiss `PQ128x8` | `0.814752` | **`0.974419`** |
| Faiss `OPQ128_768,PQ128x8` | `0.817630` | `0.973066` |
| Faiss `PQ256x4` | `0.816410` | `0.973681` |
| Faiss `OPQ256_768,PQ256x4` | `0.815397` | `0.973558` |
| learned int8-128 | **`0.825656`** | `0.972574` |

The learned code gains `+0.008026` mAP@R over the strongest OPQ/PQ mAP result
at equal bytes.  Its Recall@1 is `0.001845` lower than the strongest PQ result,
so the evidence supports higher ranking quality per byte, not dominance on
every retrieval metric.  The four-bit controls were added because their 16-way
subquantizers are better matched to this fit-set size; neither displaced the
learned code on mAP@R.  Faiss warned that the 8,054 authorized fit rows are
below its recommended 9,984 rows for internal 256-centroid training;
evaluation rows were not leaked into codec fitting to silence that warning.

Cars196 receipt authorities:

- feature archive SHA-256
  `6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3`;
- selector SHA-256
  `0b2724620bd9494ffdd7c185c78c6143dd485c94abff6a33a43140ffa2023cfc`;
- external result SHA-256
  `def8d3250561af82c9401a8a55674204dcccfea3f583723947550b9c4a849444`;
- equal-byte codec receipt SHA-256
  `80c9d776ad1aab27056c99344d0f2b21498a0ee08d3b6ccc3a886a3027187044`;
- equal-byte codec driver SHA-256
  `c6ebc6807176c312b0790ae9636606f905b3e5b90e75900a32bc9779b7df6491`;
- four-bit codec extension receipt SHA-256
  `17c8b0d166f13f0ba4a10d518818784ada900154bb3749055551af306455854b`;
- four-bit codec extension driver SHA-256
  `3b7021c2339530d114109a49a68284fe81b0f050b4c6e6a1778aa8d55383c742`;
- fitted checkpoint SHA-256
  `1cd1d633764fd53caefb49f0aa091feb49370455d5e16dfb6c334d7c88a58e9f`;
- exporter SHA-256
  `1b00133936b1c289c4c6d532fb3d7c8ca9937eb25c74cff0424cffc716f8fbe5`.

## In-Shop cross-backbone replication

The production selector was then exercised on the official In-Shop Clothes
Retrieval partitions using a different frozen backbone, UNICOM ViT-B/16.  The
authenticated feature archive contains 25,882 training rows from 3,997
identities, 14,218 query rows, and 12,612 gallery rows; training and evaluation
identities are disjoint.  Twelve training identities contain one row each.
Those rows are excluded from inner class-disjoint validation because mAP@R is
undefined for a singleton query class, but remain in the final full-fit PCA and
learned refits as negative-bank evidence.

Every fit-only fold independently favored the learned projection:

| Fold | learned mAP@R | PCA mAP@R | learned Recall@1 | PCA Recall@1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.675906` | `0.522969` | `0.946057` | `0.877603` |
| 1 | `0.670621` | `0.517953` | `0.944451` | `0.869922` |
| 2 | `0.702578` | `0.545016` | `0.955127` | `0.889526` |

The pooled selector deltas were `+0.154314` mAP@R and `+0.069579`
Recall@1, so the unchanged rule selected the learned projection.  On the
official identity-disjoint query/gallery evaluation, the result was:

| Representation | Stored width | mAP@R | Recall@1 |
| --- | ---: | ---: | ---: |
| PCA int8-128 | 128 bytes | `0.449546` | `0.724996` |
| UNICOM teacher float-768 | 3,072 bytes | `0.482066` | `0.758335` |
| learned int8-128 | 128 bytes | **`0.610128`** | **`0.859404`** |

Learned int8-128 improved over same-byte PCA by `+0.160582` mAP@R and
`+0.134407` Recall@1.  It also exceeded this frozen teacher by `+0.128062`
mAP@R and `+0.101069` Recall@1 while using 24 times fewer stored bytes.  This
is a cross-backbone, standard-protocol replication of the selector mechanism,
not a claim against published In-Shop systems: the experiment was not
preregistered as a publication comparison and its receipt is explicitly
claim-ineligible.

A follow-up equal-byte screen fitted four Faiss 1.12.0 codecs on exactly the
same 25,882 authorized training rows.  Each codec stores 128 bytes per item;
its query and gallery codes were decoded and cosine-scored under the same
official protocol:

| Representation | mAP@R | Recall@1 |
| --- | ---: | ---: |
| Faiss `PQ128x8` | `0.462979` | `0.741947` |
| Faiss `OPQ128_768,PQ128x8` | `0.473946` | `0.753060` |
| Faiss `PQ256x4` | `0.463619` | `0.741876` |
| Faiss `OPQ256_768,PQ256x4` | `0.464860` | `0.740399` |
| learned int8-128 | **`0.610128`** | **`0.859404`** |

At equal stored width, the learned code exceeds the strongest standard codec
by `+0.136182` mAP@R and `+0.106344` Recall@1.  This rules out ordinary
same-rate PCA/PQ/OPQ compression as the explanation for the In-Shop gain; it
does not compare against end-to-end image metric-learning systems.

In-Shop receipt authorities:

- feature archive SHA-256
  `730764705dc7dbacefd9c5d0ba1d9f2d65f1b9cbbebd62da84e97fd0d9548a29`;
- result receipt SHA-256
  `7d1f95afd21cdb37d46889db980c52861ad8f16d377a024ff4f72aa0c48d79e6`;
- fitted checkpoint SHA-256
  `28f03a5a17ea10d41032f649f55659f684a190e6e7c52c8c783ddd5be12ddfb8`;
- learned encoder SHA-256
  `3e2a23add6edd6683112a127738c8181d985a06d636402b49e6480525e61f0b5`;
- selector driver SHA-256
  `e17f6a70ac83127bab24ef4736adc078f9f43795d701b7281e6f1b947ddd84ac`;
- candidate production module SHA-256
  `d6a644476ee6b2b8486876e770eb103bdf478197234208da7e00fbf081850654`;
- equal-byte codec receipt SHA-256
  `4500a584964447f8a497e6afc588cc876cff0a39644be2c565a035c1222e0964`;
- equal-byte codec driver SHA-256
  `926f558be3aa39d6ff290921243932baa127e35ec1e333ebc36c79b555150b49`.

## Scope

This result validates fit-only model selection for the existing supervised
compact projection.  It does not establish a universal frozen projection:
CUB and the six-domain transfer panel already falsified that stronger claim.
The Pet split is a fresh class-disjoint confirmation of the selector, not a
comparison with published Oxford-IIIT Pet systems that use the standard
same-class train/test protocol.  Cars196 adds a standard zero-shot retrieval
panel result but is not method-specific untouched evidence.  In-Shop adds a
different-backbone standard identity-disjoint replication, but its exploratory
receipt remains claim-ineligible.  The fit-only
choice is now exposed as `select_compact_metric_projection`; it returns either
the learned projection or a full-fit PCA fallback without consuming evaluation
data or changing the compact encoder and ANN serving path.
