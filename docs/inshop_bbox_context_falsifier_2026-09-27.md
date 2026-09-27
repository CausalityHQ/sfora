# In-Shop TRAIN garment-box context falsifier

The fixed seed-179026 true-freeze checkpoint still misses 151 of 6,354
product-disjoint In-Shop TRAIN held queries. A training-only garment-attention
auxiliary objective is a distinct candidate because product identity loss can
reward model/background context shared within a photo series. The released
`Anno/list_bbox_inshop.txt` file (SHA-256
`b1a67bf2b1bb22ce57b63238dfabfd987f21ef85e93448f01735148f5848f440`)
provides garment boxes. A prior acquisition audit showed series structure,
but series may also reflect real garment colour; background leakage remains
a hypothesis, not an established cause.

First run one read-only, no-fit crop falsifier on the **151 archived TRAIN
miss triples**: query, best positive and best impostor. Verify model,
checkpoint, dataset partition, held roles, archived miss receipt, box file and
every image path by digest. Export each of the at most 453 distinct original
images through the same FP16-autocast trained export and reproduce the
archived packed positive-minus-impostor miss status for every triple and
measure its score drift before any crop result.
Then crop each image to its released garment box (1-based inclusive coordinates),
center on a neutral grey square, process at the pinned 256-pixel resolution,
and encode with the **unchanged** checkpoint/head. Pack with the existing
signed-Int8 plus f16 inverse norm and score the same triple. No crop
coefficient, margin, model or preprocessing variant will be selected.

Advance only if at least **90/151** cropped triples rank the positive above
the archived impostor and the median cropped positive-minus-impostor margin
is at least **+0.02**. A tie fails. This demands roughly 60% triple rescue
because even 90 rescues are only 1.42 points of the 6,354-query panel before
other gallery impostors and false-hit losses; the older official gap is 1.22
points. A fail closes this box-context lane before trainer code. A pass only
permits a frozen full TRAIN held-gallery, paired-seed quality/cost design with
matched cropped/uncropped controls and false-hit accounting. The crop is a
causal stress test, not a deployment proposal. Any trained model would need
fresh exactness and latency qualification; unchanged architecture alone does
not certify the old checkpoint's 15.832 ms p99.

## Source-authority amendment before crop read

The first source-pinned DGX unit failed **before any cropped image was
encoded**. Exporting only the 409 distinct miss-triple images changes their
batch composition from the original 12,599-image export; this likely explains
the maximum original-image packed margin difference of **0.0023614**. The original
`1e-5` score replay rule was too strict for this subset export. Its failed
invocation was `b799db85d79f43c292a1de1fde16cc9f`; the terminal error is
preserved in the [service journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-bbox-context-v1/failed-replay.journal.log)
(SHA-256 `5cbb3a6e437a19ed360fb57c265ba8bdba332cfcb9d8869c5e080aad0d1723ca`).
This is an authority repair, not a quality
result or evidence for the candidate.

Before reading a crop result, freeze one amended guard: all **151 original
triples must still be misses** under the subset export and the maximum
absolute archived-margin drift must be at most **0.005**. This tolerance is
more than twice the observed 0.0023614 and four times smaller than the
candidate's +0.02 median floor. The crop win and margin thresholds stay
exactly as above. If the amended guard fails, stop without a crop result;
do not relax it again.
