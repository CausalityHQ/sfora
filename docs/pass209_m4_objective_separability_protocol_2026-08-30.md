# Pass209 M4 objective rescue protocol

Status: **prospectively frozen before any new descriptor or cross-substrate
error-overlap set is computed**.

Correction frozen on 2026-08-31 before any M4 execution: repository review of
the historical launchers established that DINOv2-L used batch 32, whereas both
SigLIP cells used batch 8, and that the historical aggregate/error population
was scored by the CUDA substrate scorer rather than the new CPU reference
scorer. The corrected protocol below preserves those two authorities
separately. No M4 descriptor or result was observed before this correction.

## Purpose and scope

The blinded M2 taxonomy is terminally ineligible: its two strict submissions
matched on only 61 of 103 primary accounts (`0.5922330097087378`, Cohen's
kappa `0.39437211255774873`) and left 42 disagreements. Its frozen decision is
`F-NONE`. M4 replaces that subjective label boundary with a smaller objective
question: do either of the two non-selecting frozen substrates rescue errors
made by the selected SigLIP-so400m substrate, especially the preregistered
dominant pair?

M4 is a corroboration and veto instrument, not an independent estimator of
capacity or transfer. M3 remains the preregistered transfer measurement. M4
may corroborate an M3 branch only when alternative, query-independent frozen
features contain material retrieval evidence. It does not train a model,
choose a layer, tune a threshold, or consume clean Cars classes `49..81` or
official test classes `98..195`.

The dominant object was disclosed before this protocol: the authenticated
SigLIP-so400m error manifest has unordered pair counts `82/83 = 63`, `85/86 =
12`, and `89/90 = 9`; every other pair has at most three errors. Pair `82/83`
is the sole primary pair because it is the unique maximum and contains
`63/103` errors. The other pair tables are descriptive only. No pair may be
added or promoted after descriptor outcomes are visible.

## Immutable authority

- dataset: `tanganke/stanford_cars`, train split, revision
  `9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`
- examples: exactly the 1,345 rows whose labels are `82..97`, in the existing
  authenticated loader order
- legacy order-insensitive full Cars train-split example-set SHA-256 (not the
  1,345-row holdout order):
  `83a7800ee948a816e2fb9a2c9163027d9e90f167abc90052bf220619fa32240f`
- source error manifest SHA-256:
  `64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611`
- DINOv2-L: `facebook/dinov2-large` at
  `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, final CLS readout, expected
  `1,196 / 1,345`; prerequisite receipt SHA-256
  `8d01a2aa7cb122e9db0786e40a397a4dfe64ccec9430f6346a80d3b6a3b973a1`
- SigLIP2-so400m: `google/siglip2-so400m-patch14-384` at
  `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, vision pooler readout,
  expected `1,227 / 1,345`; prerequisite receipt SHA-256
  `55c66314017aac208dd76c542f0b2be5f969b18a4ca422e56a15ef14b15b7f9e`
- SigLIP-so400m: `google/siglip-so400m-patch14-384` at
  `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`, vision pooler readout,
  expected `1,242 / 1,345`; M2 reproduction receipt SHA-256
  `c95088621cdacea5286f1e4634f580ee83d9bed183284f23fc1be9b93bff5089`

These are the original three fp32 substrate-screen cells, fixed before M2.
SigLIP-so400m selected the manifest and is descriptive in M4. Only DINOv2-L
and SigLIP2-so400m are non-selecting rescue devices. The previously observed
SigLIP-base pooled count is excluded because it came from a different fp16
token-screen pipeline and was added only after all aggregate counts were known.

Every cell uses offline, revision-pinned model and dataset loads; DINOv2-L uses
its historical batch size 32 and both SigLIP cells use their historical batch
size 8. Every cell uses
query block 32, full fp32 tower/readout/scoring, deterministic PyTorch
algorithms, TF32 disabled, and the registered readout. A mismatch in model,
receipt, source, example ID/label sequence, row count, descriptor dimension,
finite/unit-norm validation, or processor shape terminates that cell without
analysis. A historical CUDA reproduction mismatch publishes claim-ineligible
diagnostic artifacts and a failed receipt, then terminates without rescue
analysis; it does not discard the completed descriptor measurement.

Descriptor inference is CUDA-only on the registered DGX; retrieval scoring is
CPU-only as specified below. Each cell receipt binds GPU product name, UUID,
compute capability, driver version, CUDA runtime, cuDNN version, and the exact
PyTorch CUDA build. A different or unavailable descriptor device stack fails
authority before publishing a cell.

The SigLIP-so400m v2 prerequisite receipt contains the legacy descriptor
SHA-256 computed as the canonical sorted compact shape/dtype header
`{"dtype":"float32-le","shape":[rows,dimensions]}\n` followed by the exact
row-major little-endian f32 plane. M4 recomputes and requires that digest before
scoring. M4 first reruns the historical CUDA scorer, including its second
normalization, and requires that scorer's complete incorrect-query position set
to equal the frozen manifest's 103 query positions. The separate CPU reference
table is not required to reproduce CUDA counts or positions. Publish every
CPU/CUDA nearest-row divergence
with both row identities, score bit patterns, and same-minus-different margin;
it is descriptive and cannot redefine the fixed source-manifest pair. This
anchors the CPU reference scorer to the CUDA scoring event that defined the
rescue population without falsely requiring backend-identical argmax rows.

The older DINOv2-L and SigLIP2-so400m v1 prerequisite receipts did not publish
descriptor digests or row-level rankings. M4 must not invent them. Those cells
are bound by their exact receipt bytes, source/model/data/readout authority,
new create-once descriptor digest, and exact aggregate-count reproduction by
the historical CUDA scorer.
Their rescue rows are therefore newly generated M4 evidence under the common
CPU scorer, while the SigLIP-so400m population-defining rows are an exact
historical reproduction. This asymmetry is disclosed in the final receipt.

## Exact artifacts and execution

Each of the three cells executes once after the active M1/M3 DGX campaign
releases the GPU. The implementation commit and source-tree digest are sealed
before the first cell. Each cell atomically publishes three create-new files:

1. a canonical JSON authority receipt;
2. a framed binary descriptor file;
3. a canonical JSON ordered query-evidence table.

The descriptor file is exactly:

1. the 16 bytes `SFORA-M4-F32-V1\n`;
2. one unsigned little-endian 64-bit header length;
3. that many bytes of sorted-key compact UTF-8 JSON with exactly one trailing
   LF; and
4. exactly `rows * dimensions * 4` row-major little-endian float32 bytes.

The header binds schema, source revision/tree, dataset/revision, exact example
legacy set digest, a new ordered `(example_id,label)` digest,
split/holdout/dtype, cell/model/revision/readout, `[rows, dimensions]`, payload byte
count, and payload SHA-256. The receipt additionally binds the full descriptor
file SHA-256 and query-table SHA-256. No length, padding, extra byte, NaN,
infinity, zero-norm row, or norm error above `1e-6` is accepted.

For each row, the query table contains exact example/label authority, nearest
row/label, best same-label row, and the float32 bit patterns of nearest score,
best-same score, best-different score, and signed same-minus-different margin.
The scoring authority is one-thread CPU PyTorch matrix multiplication in the
same pinned offline environment as the analyzer: contiguous normalized fp32
rows, query blocks of exactly 32, `torch.set_num_threads(1)`, interop threads
one, deterministic algorithms enabled, autocast disabled, and no CUDA or
reduced-precision backend. Self scores are negative infinity and selection is
the lexicographic maximum `(float32 score, negative row ordinal)`, so the
lowest row wins an exact score tie. The receipt binds the PyTorch version,
build configuration, BLAS backend and version, CPU architecture and ISA flags,
thread settings, `uv.lock` SHA-256, Pillow/libjpeg versions, Transformers
version, and scorer schema.
The analyzer reads only authenticated descriptor bytes, recomputes the table
in a separate invocation of this frozen scorer, and requires byte-identical
canonical evidence. A pre-execution mutation test locks exact score bits,
signed-zero handling, tie behavior, complete 32-row blocks, and the ragged
one-row final block on a fixed synthetic descriptor plane; an environment that
cannot reproduce it fails before any cell runs.

No cell-level overlap, margin, duplicate, or pair result is inspected until all
three cells and their aggregate-count gates are terminal. One analyzer then
authenticates all nine cell files plus the frozen M2 manifest and emits one
canonical `sfora-pass209-m4-objective-rescue-v1` receipt. Partial output is not
scientific evidence.

## P1: gallery-wide exact duplicate audit

Materialize all 1,345 registered band images through the repository's strict
RGB materializer. For each image, hash exactly the 16 ASCII bytes
`SFORA-M4-RGB-V1\n`, unsigned little-endian `u32` width, unsigned
little-endian `u32` height, and exactly `width * height * 3` row-major decoded
RGB8 bytes, with no padding or trailing byte. Width, height, and the byte-count
identity are validated before hashing. For each of the 103 frozen error
queries, report whether any
different-label band row has the same record and list all matching row/example
identities. This is gallery-wide rather than query-to-top-1-only.

No perceptual hash, resize, crop, registration, filename, or human judgment is
used. P1 is descriptive and can trigger a data audit; it never selects or
vetoes a trainable family because the disclosed M2 duplicate prevalence makes
the earlier 0.25 decision threshold unattainable.

## P2: cross-substrate rescue

For each of the 103 frozen SigLIP-so400m errors, recompute correctness from the
DINOv2-L and SigLIP2-so400m query tables. Define `reachable` iff at least one
of those two non-selecting devices is correct. Because every source row is a
SigLIP-so400m error, `universal_three_device_error` is exactly `not reachable`;
derive it rather than treating it as an independent quantity.

Publish both device correctness bits, the reachable bit, the two margins, the
exact overlap table, and the unordered-pair clustered bootstrap for reachable
share. The devices are correlated web-pretrained models. Reachability is
alternative frozen evidence, not model selection, and universal failure is not
Bayes error.

## P3: dominant-pair rescue

The primary panel is the exact 63 source-manifest rows whose unordered pair is
`(82,83)`. This is the complete finite census of the preregistered dominant
error object, not an iid sample from a binomial population. For each
non-selecting device, publish rescued/count and rescue rate.

`dominant_pair_rescuable` is true iff either device rescues at least `0.25` of
the exact 63 rows. The threshold is a descriptive materiality floor on this
census, not a confidence statement and not an analogue of the 97.4 official
test target. Because no population inference or max-over-hypotheses claim is
made, a multiplicity correction is neither applied nor implied.

For pairs `(85,86)` and `(89,90)`, and for the selecting SigLIP-so400m cell,
publish the same raw rates descriptively, but they cannot affect the decision.
The SigLIP-so400m rescue rate on every source-manifest row must be zero by the
exact row-agreement authority above; any nonzero value is an integrity failure.
This prevents winner's-curse reuse of the manifest-producing device and avoids
a twelve-test max-over-device/pair rule.

## Cluster bootstrap

Only the global reachable share is bootstrapped. Partition the 103 source rows
by unordered `(min(query_label, nearest_label),
max(query_label, nearest_label))` pair, sort the `K` nonempty pair blocks
lexicographically, and treat each whole confusion object as one cluster. Use
exactly 10,000 resamples. For each replicate in ascending order, call
`Generator.integers(0, K, size=K)` once from
`numpy.random.Generator(numpy.random.PCG64(seed))`, where `seed =
int.from_bytes(SHA256(b"pass209-m4-objective-bootstrap-v3").digest()[:16],
"big")`. Duplicate an entire pair block whenever its index is sampled. Every
registered block is nonempty, so every replicate denominator is positive.
This keeps the dominant two-way confusion together rather than resampling its
query directions as independent classes.

Publish observed mean, bootstrap mean, NumPy `inverted_cdf`
2.5th/10th/97.5th percentiles, and SHA-256 of all 10,000 consecutive
little-endian float64 values. A reachable-share threshold is met only when its
10th percentile meets it. The bootstrap is not applied to the deterministic
pair-panel decision.

## Decision adapter

M4 records objective evidence. A separate adapter runs only after the three
authenticated M1/M3 seed receipts exist. First matching rule wins:

1. `F4-TRANSFER`: M3 is `T-low` and reachable-share p10 is at least `0.25`.
   Admit only cross-class transfer of the trainable representation.
2. `F4-CAPACITY`: M3 is `T-high` and `dominant_pair_rescuable` is true. Admit
   only trainable input-evidence capacity (resolution, pooling, or
   architecture). Token matching/correspondence remains closed by Pass206/208.
3. `F4-NONE`: every other outcome, including `T-mid`, `T-undefined`, weak
   global rescue, or no material dominant-pair rescue. Gather a
   different objective measurement; do not invent a candidate.

The transfer and capacity branches are mutually exclusive through M3. M4 does
not override M3, infer an absolute representation ceiling, or convert a frozen
device into the later method.

## Leakage, multiplicity, and stopping

- Clean classes `49..81`, official test classes `98..195`, M1 clean metrics,
  and historical Cars test embeddings are unavailable to every M4 process.
- The three-device set is the complete original fp32 substrate ladder; its
  provenance and all three prerequisite receipt digests are disclosed above.
- M4 uses burned classes only and may corroborate a broad family, never a
  layer, model, loss, schedule, crop policy, or checkpoint.
- The later candidate may not copy, initialize from, ensemble, distill from, or
  select the non-selecting rescue device without a new preregistration.
- The complete designer-known evidence before execution is disclosed here:
  the three aggregate counts, the 103-row manifest and pair counts, and M2's
  terminal prevalence/agreement. No cross-device error overlap, descriptor,
  rescue rate, margin, or M3 ratio is known.
- Any output overwrite, extra scientific cell, post-outcome threshold change,
  missing receipt, noncanonical bytes, authority mismatch, or prohibited-class
  load makes M4 `failed`.
- A model/count/descriptor/query-table/manifest authority mismatch is a
  terminal M4 `failed` result and the adapter is not run. It does not authorize
  a rerun, alternate scorer, or replacement cell. The scorer self-test executes
  before any model cell, so a known environment/backend incompatibility fails
  without consuming a scientific attempt.
- Terminal M4 `failed` has the operational consequence of `F4-NONE`: no
  trainable family is admitted from this protocol and the next action is a
  separately preregistered objective measurement. It is not relabeled as a
  scientific `F4-NONE` result and it does not permanently forbid independent
  future evidence.
- Resource limits are conservative operational tripwires, not scientific
  budgets: combined process RSS plus CUDA peak reserved memory at 64 GiB,
  memory PSI full `avg10 >= 0.50`, swap growth at 256 MiB, or 20 minutes
  without an authenticated batch-boundary progress record. No separate CUDA
  cap is used on the unified-memory DGX.
- A resource-stopped cell may resume only from its authenticated batch-boundary
  receipt with the same source/config/output identities and no published final
  file. That continuation is the same cell, not an extra scientific attempt.
  Partial descriptor/query files are deleted after process clearance.

Expected execution after implementation is under one hour of DGX time plus
minutes of CPU analysis. No candidate or clean-validation run is authorized
until M4 and all three M1/M3 seed receipts are terminal and the adapter selects
exactly one family.
