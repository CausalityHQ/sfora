# SOP custom image gallery production gate, 27 September 2026

The trained `Siglip2CompactIndex` already loads a pinned TRAIN or official
gallery, but an external user cannot build a new image gallery through that
loader. Add `custom_gallery_image_paths` to the existing artifact loader and
`from_image_paths` for an already loaded encoder. Encode user paths in
32-image batches with the same FP16 serving arithmetic and open the existing
exact native packed scorer. Keep input order as gallery ordinal order. Require
at least ten images because native top-10 returns ten rows. Reject mixed
official/custom gallery arguments before model loading. Keep the authenticated
training checkpoint, model, runtime, and native scorer checks.

Freeze this one-checkpoint integration before the DGX run: use seed-179024
true-freeze SigLIP2 Large/256 FP16, the pinned 59,551-image SOP TRAIN archive,
32 distinct TRAIN images at ordinals 0–31 as an **external user gallery**, and
32 distinct images at ordinals 32–63 as external queries. Bind the source
archive, native library, model snapshot, training receipt/checkpoint and each
image file to SHA-256. Build the gallery through the new public artifact
loader. Independently encode the same 32 gallery images in one batch and
require bitwise equality for every packed code and inverse norm. Submit all
32 query images through `search_images`; require bitwise equality with direct
native search on independently encoded packed queries, and exact top-10
ordinals with score error ≤1e-5 versus the stable packed matrix oracle.
Require finite native scores and no GPU job overlap. Record gallery build
wall, one-call search wall, peak CUDA, source hashes, full result hashes and
the systemd exit status. Any parity failure stops promotion and requires a
root-cause fix. A pass makes custom-gallery serving available as a production
API; it does not establish p99 latency, independent retrieval quality, or
SOTA. Official TEST stays outside this product check.

## Terminal DGX integration

The sole `sfora-sop-custom-gallery-179024-v1.service` exited 0 on DGX Spark
GB10 with no overlapping GPU process. The source-bound
[verifier](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/verify.py)
has SHA-256 `02ff4129b8ff859f2d75524d49b21aab930fad99965949daa116a5183cdb040d`.
Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/receipt.json)
has SHA-256 `b0c9462b6406708b1ea95e7c95755bbae819102158ea664143970f1a70d44d8b`;
the [service journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/service-journal.txt)
has SHA-256 `9b09309c59652fe32e904fdd77126b036e85ab5933caacca04c66e1810af8ee1`.
The loaded custom gallery's 32 packed codes and inverse norms matched an
independent one-batch encode **bitwise**. All 32 external image queries had
public top-10 ordinals and scores identical to direct native search, and
stable packed-matrix top-10 ordinals and scores matched with **0.0** maximum
score difference. Native scores were finite. Gallery storage is **130
bytes/image**; peak allocated CUDA was **1.267 GB**. One gallery build took
**5.550 s** including model loading, encoding and native index creation. One
batch-32 search took **0.267 s**; these single calls are diagnostics, not
latency percentiles or a matched performance result.

**Decision:** promote the additive custom-image-gallery API. It uses the
existing trained encoder and exact packed scorer; the authenticated model,
checkpoint, training receipt, runtime and native-library checks remain. This
gate establishes correctness for one 32-row external gallery and one 32-query
batch, not p99, million-row build cost, official retrieval quality or SOTA.

Applications pass their own ordered `Path` sequence as
`custom_gallery_image_paths` to `Siglip2CompactIndex.from_artifacts(...)`,
with the same required pinned model, training receipt/checkpoint, training
embeddings and native library arguments as the existing loader. The returned
top-10 ordinals index that path sequence. `index.search_images(query_images)`
then serves external PIL images. For an encoder already loaded by the
application, `Siglip2CompactIndex.from_image_paths(encoder=..., native_library=...,
image_paths=...)` builds the same gallery; the resulting index owns its native
gallery and closes it with the existing context-manager API.
