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
