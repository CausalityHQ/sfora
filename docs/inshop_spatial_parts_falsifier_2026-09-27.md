# In-Shop TRAIN spatial-part falsifier

The trained-head global token-mean blend lost 4 net hits on the fixed In-Shop
TRAIN held roles. Test one different mechanism: a product part may agree across
views even when the attention-pooled descriptor prefers an impostor. Use the
existing seed-179026 true-freeze checkpoint and the archived 151 packed misses
only. The official evaluation split is excluded.

For each miss, take its archived query, best positive and best impostor from
the same 12,599-image product-disjoint TRAIN held panel. From the final
16-by-16 vision patch grid, average each nonoverlapping 8-by-8 quadrant,
apply the existing trained 1,024-to-128 head, and unit-normalize each part.
The frozen part score is the mean, over four query parts, of the maximum
cosine to any of four gallery parts. No fit, new weights, sweep, or changed
checkpoint is allowed. Export each distinct image once under the same pinned
model/processor with FP16 autocast as the original held export; verify the
partition and miss receipt digests and all tensor shapes.

Advance only if the part score ranks the archived positive above the archived
impostor for at least **99/151** misses (65.56%) and the median
positive-minus-impostor part-score margin is at least **0.02**. A tie is a
failure. This deliberately demanding screen asks whether part correspondence
has enough headroom to cover the 1.22 percentage-point older official In-Shop
gap before paying to export the whole gallery. It is a causal diagnostic, not
a retrieval result: rescues can be offset by false hits elsewhere. Failure
ends this part lane. Success permits one separately frozen TRAIN full-gallery
paired packed-quality and image-to-top-k cost gate, with unchanged training
recipe and scorer control. Neither outcome licenses an official or SOTA claim.

## Terminal result

The sole DGX Spark GB10 unit `sfora-inshop-spatial-parts-v1` (invocation
`912b00093fa9474590c67f05543f1871`) exited 0. Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-spatial-parts-v1/receipt.json)
has SHA-256 `85812078f2273cdd1f8e555e36fa5b5b75738c51f8212997206df0ce0d68e181`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-spatial-parts-v1/journal.log)
has SHA-256 `618b4e4400149db5284c003fdf1ae226c863fb3ceb8de9e54a61d6d65892d7a4`.
The source SHA-256 is
`71da3788e2dc50c6abb5333318d5702355d90ad23d4c9fc3ab16b79e11ed8577`.
It exported **409 distinct TRAIN-held images** for the fixed 151 triples.
The positive part score beat the archived impostor in **50/151** misses
(33.11%), well below the frozen 99/151 floor. The median part margin was
**−0.006734**, below +0.02. Export and scoring took **3.363 s** after model
load; peak allocated CUDA was **1,886,321,152 bytes**. Local replay checked
the receipt source digest, all 151 margins, win count, median and decision.

**Decision:** reject this fixed spatial-part match and stop before a full
gallery export, new training, official evaluation or production serving edit.
This only falsifies the specified four-quadrant, shared-head score; it says
nothing about other local representations.
