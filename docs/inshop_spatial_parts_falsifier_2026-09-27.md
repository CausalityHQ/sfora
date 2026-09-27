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
