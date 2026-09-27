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

## Frozen realistic-gallery scale check

The first integration used only 32 gallery rows. Keep the same source-bound
seed-179024 FP16 model and native library, then build a new custom gallery
from all **59,551** ordered SOP official TRAIN image paths in the authenticated
archive. Use the first 32 SOP official TEST images only as external product
queries; this is a serving parity check and must not select a model or
estimate retrieval quality. Record an aggregate SHA-256 over the 59,551
gallery image digests and verify the existing pinned TEST image manifest for
the 32 queries. Repeat the first, middle and last gallery encode chunks and
require bitwise packed code/norm equality to the uploaded native gallery.
Require all 32 public top-10 rows to equal direct native results exactly and
the stable full packed-matrix oracle within 1e-5 score error, with finite
scores. Require one gallery build ≤600 s, peak allocated CUDA ≤2.0 GB and
peak parent host RSS ≤6.0 GB. These bounds are frozen before the run; a fail
stops a general scale claim. A pass qualifies this one 59,551-row build and
search configuration, not million-row image ingestion, concurrent service,
quality SOTA or p99 latency.

The first 59,551-row service exited 1 before writing a receipt; its
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/service-scale-v1-journal.txt)
has SHA-256 `758dbf527d4c2a58d75b3a80770de2ce279e4324f8ababf2cca8caf41a0c40ed`.
The diagnostic rerun on the then-current source produced a
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v2/receipt.json)
with SHA-256 `4a486dbf8c2a7fae5a5a7b1488c701170844b0cc859d233c09bf76d89e2dd001`.
All 32 public query top-10 rows matched native and the stable packed oracle,
with 0.0 maximum score error. Build wall was 548.400 s and peak allocated CUDA
1.267 GB. **Peak parent RSS was 6.393 GB, above the frozen 6.0 GB limit**;
this is a failed scale gate. A subsequent 32-row reproduction of the
pre-change full-model loader (source SHA-256 `593e64c8…`, distinct from v2)
peaked at 6.357 GB. The old loader's setup peak alone nearly exhausted the
6.0 GB budget.

The loader had created the complete pretrained image-text model even though
the authenticated training checkpoint contains all 400 vision tensors. It now
constructs the vision model from the pinned config and loads those trained
tensors directly. A meta-device shortcut reduced memory further but left its
nonpersistent position-ID buffer uninitialized and changed packed outputs; it
was rejected. The [diagnostic receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/smoke-v5/receipt.json)
has SHA-256 `ab8ca86c0715f788d3e3d28493f4acaf79335d0bcf8e7915d60b6c0947482835`.
The ordinary vision-model constructor preserves that buffer.
On the corrected source, a new 32-row public integration had gallery and
query packed SHA-256 hashes identical to the original loader, exact public
top-10 and 0.0 score error, with peak RSS **3.530 GB** and build wall **7.594 s**.
Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/smoke-v6/receipt.json)
has SHA-256 `1fa7d5393addb5c1c8df25794bee3da4f7ab588cdb526c9b921306a68b600051`.
The first 32-row build took 5.550 s; these are independent single-call
diagnostics, not a paired latency estimate.

## Terminal corrected scale gate

The sole `sfora-sop-custom-gallery-scale-179024-v6.service` exited 0 on DGX
Spark GB10. Its [source-bound verifier](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v6/verify_scale.py),
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v6/receipt.json),
and [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v6/service-journal.txt)
have SHA-256 `7948f17c3e3e02548d1a27b717d5c643ea44d04adede642fa373cb3339fddf96`,
`7ad9151c4be22901fff663679c759597a6ff3429c0323a19fc6471f1198da287`,
and `761fe57e92c522d1321b98c12e5fe26f58782b41f57425069b52b2f7bb787db2`.
The serving source SHA-256 for this run is
`99fdc2b7f20a01d62faf4f02f63af4e2c65b2391d8c4bf96df81346e8f1135ec`.
All six frozen checks pass: **59,551** ordered gallery images and **32**
external TEST queries, **95** first/middle/last sampled gallery rows bitwise
equal on re-encoding, exact public/native/stable packed top-10 with **0.0**
maximum score error, **547.045 s** gallery build, **1.267 GB** peak allocated
CUDA, and **5.807 GB** peak parent RSS. Gallery storage is **130 bytes/image**.
The one batch-32 image search took **0.269 s**, a single-call diagnostic.
The gallery image-content digest, full packed-gallery digest, and all public
top-10 ordinals and scores are identical to the failed high-memory rerun.
Peak RSS fell by **0.586 GB (9.17%)**; the two builds were not interleaved,
so their wall-time difference is not a speed claim.
The corrected 32-row process peaked at 3.530 GB, while its 59,551-row build
peaked at 5.807 GB. The additional 2.277 GB is not explained by the 7.742 MB
packed gallery; build-phase host allocation remains unattributed, and this run
has only 0.193 GB of headroom below the frozen RSS bound. Repeated builds in
one process and larger user galleries require their own memory checks.

**Decision:** retain the smaller vision-only model loader and the custom
gallery API. This passes the frozen one-gallery scale gate. It does not
establish p99 latency, a million distinct image build, retrieval-quality
gain, or quality/performance SOTA.

## Final dtype-pinned source

Adversarial review found that the vision and head constructors inherited the
host application's default floating dtype. The final loader explicitly moves
both to FP32 before strictly loading the checkpoint, preserving FP32 serving
arithmetic even when the host changes its default. The final serving source
SHA-256 is `73ad606c78ad22e6b377627e4e7c8fe07d7dfaafd07a029c40df2982221a7fd1`.
Its 32-row [smoke receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/smoke-v7/receipt.json)
has SHA-256 `b431de77f6c7800784f50a9fdf157d14ec84b6356c3c9f5655d509ff6181f003`:
the gallery, query, ordinals and scores match the preceding loader bitwise.

The final `sfora-sop-custom-gallery-scale-179024-v7.service` exited 0. Its
[verifier](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v7/verify_scale.py),
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v7/receipt.json),
and [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-custom-gallery-179024/scale-v7/service-journal.txt)
have SHA-256 `4992e64cf6cfaa6de6494e53e4f854d0a8ac941b65aed26da2501cc181a5df2f`,
`a4c689ef5af28583fa24aa07d109f61b205adbabc537a8b45633551a785fe1d0`,
and `67ad1235f0874824441b6bbccc28159b315ac29cde2295c2acbc0908939511d5`.
All six frozen gate checks pass: 59,551 TRAIN gallery rows, 32 external
parity-only TEST queries, 95 re-encoded rows bitwise equal, all public/native
and stable-oracle top-10 exact with 0.0 score error, **546.046 s** gallery
build, **1.267 GB** peak allocated CUDA and **5.776 GB** peak parent RSS.
The one batch-32 image search took **0.265 s** as a single-call diagnostic.
The full packed gallery and all public results match v2 and v6 bitwise.
Against v2, parent RSS fell **0.616 GB (9.64%)**. Build wall times are
unpaired single calls. The final build has **0.224 GB** RSS headroom and
qualifies this one gallery size and process lifecycle; larger or repeated
builds need their own memory checks. Retrieval quality and p99 latency were
not measured in this scale check.
