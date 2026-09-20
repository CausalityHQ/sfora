# Fit-only compact-metric selector result

## Result

A three-fold, class-disjoint selector fitted only on authorized training
classes can decide whether to deploy the supervised compact projection or its
PCA initialization.  The frozen decision rule is:

1. assign complete labels to three folds by taking the first eight bytes of
   `SHA256("compact-selector-v1:" + decimal_label)` as an unsigned
   little-endian integer, then reducing it modulo 3;
2. in each fold, fit the existing compact metric on the other two folds and
   compare packed int8-128 retrieval with fit-only PCA on the unseen fold;
3. select the learned projection only when pooled mAP@R gain is at least
   `0.003` and pooled Recall@1 gain is nonnegative; otherwise select PCA.

The rule contains no dataset identity, class name, or evaluation-split input.
It was fixed after the Stanford Dogs development result and then evaluated
once on untouched Oxford-IIIT Pet evidence.  No threshold was changed after
the Pet result.

Integer label identities are part of the deterministic fold authority:
relabeling an otherwise identical corpus can change folds near the selection
threshold. Callers must therefore keep a stable label-to-integer mapping. The
fit path also performs quadratic hard-negative mining and materializes positive
rows up to the largest class; the shipped evidence covers at most 25,882 fit
rows and does not establish memory bounds for classes with thousands of rows.

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

Two additional frozen sources isolate how the selector behaves as upstream
quality improves.  Replaying the identical policy over authenticated UNICOM
ViT-L/14@336 features selected the learned projection in every fold and raised
the official result to `0.654677` mAP@R / `0.875088` Recall@1, versus the
float-768 source at `0.555428` / `0.810100` and same-byte PCA at `0.538349` /
`0.792657`.

The same policy was finally applied to the separately trained rank-finished
UNICOM ViT-L/14@336 source.  This source is the release candidate from model
SHA-256 `ad7e16d28daf32c3ae8d6258444e18c142cdf2e1816448a615be653e9545697b`.
Its authenticated standard readout is `0.776033` mAP@R / `0.944718` Recall@1.
The export used the corrected 52,712-image official pixel corpus, whose
registered image-tree root is `b91f639916fddb1834c2511a7eb5f354d1f301425bee9f7f717e8da48af0ba72`.
Replaying the standard first-512-coordinate metric directly from the exported
archive reproduced `0.77603312` / `0.94471796`; Recall@1 was exact and the
mAP@R difference was `1.8e-8` from batching arithmetic.

All three class-disjoint fit-only folds favored learned int8-128 over PCA.  The
pooled deltas were `+0.015002` mAP@R and `+0.003363` Recall@1, so the unchanged
production rule selected the learned projection.  Official query/gallery
results were:

| Representation | Stored width | mAP@R | Recall@1 |
| --- | ---: | ---: | ---: |
| historical first-512 standard readout | 2,048 bytes | `0.776033` | `0.944718` |
| matched-input float-768 source | 3,072 bytes | `0.779547` | `0.945703` |
| PCA int8-128 | 128 bytes | `0.777572` | `0.946125` |
| learned int8-128 | 128 bytes | **`0.800020`** | **`0.954283`** |

The all-768 source is the matched representation baseline because the compact
learner receives all 768 input coordinates.  Against it, learned int8-128 gains
`+0.020473` mAP@R and `+0.008581` Recall@1 while storing 24 times fewer bytes.
Against same-byte PCA it gains `+0.022448` / `+0.008159`.  The historical
first-512 result is retained only to bind this model to its earlier standard
release evidence, not used as the primary compact-method comparator.  Learned
int8-128 also clears the repository's `0.939` In-Shop Recall@1 frontier, but
this post-hoc composition remains explicitly claim-ineligible.  It is strong
product evidence and motivates a fresh, prospectively frozen replication; it
is not presented as a new publication claim on an untouched test set.

A follow-up linear-attribution panel separates supervised geometry from
quantization conditioning.  Float and int8 scores were nearly identical for
the learned head (`0.800036` versus `0.800020` mAP@R) and differed by only
`0.000384` for ordinary PCA.  PCA whitening fell to `0.745186` int8 mAP@R,
regularized supervised whitening reached only `0.773972`, and the shuffled-label
whitening control fell to `0.669961`.  Thus neither fixed-scale int8 rounding
nor a closed-form second-order whitening control explains the learned head's
`+0.022448` mAP@R advantage over same-byte PCA.

A matched full-rank control then trained the identical objective as a
768-to-768 affine map.  It reached `0.801648` mAP@R / `0.954424` Recall@1 in
float form, only `+0.001612` / `+0.000141` above learned float-128.  The result
localizes the principal gain to supervised metric deformation, not to
compression itself.  Conversely, the 128-dimensional code retains more than
99.7% of the full-rank mAP@R while reducing descriptor payload from 3,072 to
128 bytes.  These controls are post-hoc on the already observed In-Shop test
and therefore explain mechanism without upgrading claim eligibility.

A final mechanism panel tested three narrower explanations under the same
official readout.  Equal-class between-class PCA reached `0.782391` mAP@R /
`0.947742` Recall@1 at int8-128, closing only 21.5% of the learned-versus-PCA
mAP@R gap.  Applying the unchanged learned 128-to-128 recipe inside the fixed
PCA subspace reached `0.790550` / `0.951822`, closing 57.8% of that gap but
remaining `0.009470` mAP@R below the learned 768-to-128 head.  The learned
weight matrix has 45.1% of its Frobenius energy outside the PCA row space, with
a maximum principal angle of 49.4 degrees.  Finally, fitting the ranking head
after row-permuting labels while preserving exact class counts collapsed to
`0.720253` / `0.934027`, `0.057319` mAP@R below PCA.  The evidence therefore
isolates label-dependent supervised geometry that is not reducible to class
means, PCA-space reweighting, or label-independent conditioning.  It remains
post-hoc mechanism evidence, not a prospective quality claim.

The strongest source was a corrected seed-0 BN-Inception ProxyAnchor final
state.  Here the inner learned-minus-PCA evidence was only `+0.001811` mAP@R
and `0.000000` Recall@1, below the frozen `+0.003` mAP gate, so the production
policy correctly returned its PCA fallback.  Official query/gallery results
were:

| Representation | Stored width | mAP@R | Recall@1 |
| --- | ---: | ---: | ---: |
| ProxyAnchor source float-512 | 2,048 bytes | `0.647238` | `0.913701` |
| selected PCA int8-128 | 128 bytes | `0.653438` | `0.914897` |
| exploratory learned int8-128 | 128 bytes | **`0.657014`** | **`0.916796`** |

Thus the conservative deployed choice still improves both metrics while using
16 times fewer stored bytes.  The learned arm is reported to diagnose the
selector margin, not silently substituted for the frozen policy.  Its
additional gain is too small to justify changing a threshold selected before
this result.  These stronger-source runs show that absolute retrieval quality
continues to depend on the upstream representation; the compact method is not
the cause of the remaining gap to higher-capacity end-to-end In-Shop systems.

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
- release module SHA-256 after caller-autograd/autocast boundary repair
  `d93ceccc5fe8cf20ccfe76c6520af61521d1f7197a16242eead3dc11de9f421a`;
- equal-byte codec receipt SHA-256
  `4500a584964447f8a497e6afc588cc876cff0a39644be2c565a035c1222e0964`;
- equal-byte codec driver SHA-256
  `926f558be3aa39d6ff290921243932baa127e35ec1e333ebc36c79b555150b49`;
- ViT-L/14@336 feature archive SHA-256
  `6eae13715e18d7eb99450bade5056538f8f08f1e9b550d0f24ee09e52bb25d0e`;
- ViT-L/14@336 result receipt SHA-256
  `df5b46b6e5d84911b6e9f81553d3ad4b00eb489bc41a4cd88d0c2b98ef7abd2a`;
- ViT-L/14@336 fitted checkpoint SHA-256
  `4e9767b7959749d3a2ea4b60956f4fd8cdd7f192ccaa937c32ec222aedea7555`;
- rank-finished official feature archive SHA-256
  `05cd5901425210c06a3972f5a67acf41c961b2f4b59d6535744f3e3536d036ad`;
- rank-finished compact result SHA-256
  `b9ab3eac27d6f2e158c0d8209ecf79a297ab5f84cc972a8875970134771e915c`;
- rank-finished compact checkpoint SHA-256
  `56d57c92e315be13eb8361f1e72f26f548cece833806fca2142f1d405df752fd`;
- rank-finished mechanism-panel receipt SHA-256
  `8b276c65a2dcb08d18d2344164f162aa98d506ce3420738f2897a43ab8200fe7`;
- rank-finished compact encoder SHA-256
  `748237506aad7df36693630255c88e2175c6c11e00fd6e3df1eefcc7a58ddd96`;
- authenticated rank-finished standard-result SHA-256
  `93ed2130fd1f8e8e03c84f3c9850d04f40f694b6702d8ae5eb86d50b3c23b911`;
- rank-finished linear-attribution receipt SHA-256
  `22fe205422e10ef2e996bcb374ef4a036ea5b3cbaaa1652a4ddde1be22fce808`;
- rank-finished full-rank-control receipt SHA-256
  `adb03d7c0abaf18cbf26720fa418563bdbbf323dd67c08ef99fc57eaea922040`;
- rank-finished full-rank-control checkpoint SHA-256
  `c8b31445b25e1b11cd292d98db8215877aa0cead05469535290b2d20b447a1d8`;
- ProxyAnchor train/query/gallery archive SHA-256 values
  `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`,
  `ef5278fd9aae7a6398a6c74133e6acc0ded05e39647087bdf78459223b9eb761`,
  and `6eb89ff57e7a6002f2ba71f9659e04dabd0cafdb1996be3d85f5211731ba861a`;
- ProxyAnchor result receipt SHA-256
  `69794850a1087c597a2e9f7ecfb70c3ca4bc4f632e966334469b6a55e535e301`;
- ProxyAnchor fitted checkpoint SHA-256
  `346f27ec19425e34cfe42fd90eae6c663e204aa1ed6c985992424a9fda118068`;
- ProxyAnchor selector driver SHA-256
  `5f0a9b09f136a4d9f62e479d4f247fe8945bcb8d710307724560edb1e29eed0f`.

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
