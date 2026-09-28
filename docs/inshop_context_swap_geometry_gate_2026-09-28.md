# In-Shop context intervention: CPU geometry gate

The read-only consultation `7b9be1a602d1428e` completed (Fable hit its
budget; automatic Opus fallback exited 0). It proposes replacing training
context using same-product sham and different-product donors. This changes
the association between labels and context; earlier crops kept that context
or removed it without training. Context dependence remains a hypothesis.

The proposed mask outside only the query garment can insert the donor's
garment and corrupt the label. Before any encoder call, freeze a corrected
mask: replace pixels outside the union of query, sham and foreign-donor boxes.
Use that identical mask for both composites. Preserve all query garment
pixels and insert no annotated donor garment pixels. Boxes do not provide
segmentation, so background inside boxes and unannotated garments remain.

CPU-only gate, no held outcomes: choose 512 fit products with at least three
images by SHA-256(label), then one query by SHA-256(relative path). Choose
another same-product image and a different fit product, preferring the same
filename pose, deterministically without augmentation RNG. Verify the pinned
partition, fit and box hashes and 256-square image dimensions; record every
selected path and image hash. Measure both naive donor garment intrusion and
the corrected mask's replaceable fraction. Check the corrected mask against
independent rectangle membership, garment preservation and identical sham
geometry. Any authority or pixel invariant failure stops execution.

Advance only if at least 90% of the 512 triplets can replace at least 20% of
image pixels, and CPU wall is at most 120 seconds. These are intervention
coverage and resource floors, not quality thresholds or proof of context
dependence. Otherwise close this fixed box-safe context-swap intervention.
A pass licenses only a separately frozen TRAIN-fit encoder diagnostic with
matched gallery exclusions and sham; no training or official read yet.

No deployed API or default changes. Annotation supervision must be disclosed
in comparisons. Existing serving measurements do not certify a new trained
checkpoint. The consultation's timing estimate and remembered prior-art
references are unverified and are not adopted as measurements or novelty.

## Terminal: KILL before encoder or training

The sole CPU service `sfora-inshop-context-swap-geometry-v1`, invocation
`b43b589f637d4b0eaf0e2836982da432`, exited 0. Its receipt SHA-256 is
`242443a91088c73060611d27378d7e70ed476d81d33fbc72d1bec640c188a5b8`.
CPU probe wall was **2.848637 seconds**. No GPU, encoder, optimizer, held
outcome, official evaluation or serving timing was used.

| Official TRAIN fit, 512 triplets | Measured | Frozen decision |
| --- | ---: | --- |
| Triplets with at least 20% replaceable pixels | 445/512 = **86.9141%** | Fail: below 90% |
| Median corrected replaceable image fraction | **56.3065%** | Descriptive |
| Naive query-only mask inserts foreign garment pixels | **439/512** | Confound in original proposal |
| Median naive foreign garment intrusion / image area | **5.7030%** | Descriptive |
| CPU wall | **2.848637 s** | Pass: below 120 s |

Independent integer pixel-set enumeration replayed every corrected fraction
and naive intrusion exactly for all 512 triplets. The source hash matched;
the mask unit check passed and Ruff/diff checks were clean. Raw receipt,+journal and terminal verification are in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-context-swap-geometry-v1/`.

Close this fixed box-safe context-swap intervention before encoder F0,
17-update smoke or trainer/API changes. Do not lower the coverage floor or
switch to the contaminated naive mask. This resource/coverage screen does
not prove context independence or rule out segmentation-based interventions.
No new quality or latency number exists. Production defaults remain native.
