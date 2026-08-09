# Pass 200 RSTA — prospective binding-receipt amendment

## Status and scope

This amendment is frozen **before any RSTA candidate statistic is computed**. It
repairs only the artifact-binding data flow in
`docs/pass200_rsta_candidate_2026-08-09.md`; it changes no candidate formula,
role, batch, support, control, statistic, threshold, prediction, or decision rule.
The existing candidate remains **LIVE-NARROW at Gate 2; Gate 1 unresolved**.

The immutable historical binding receipt is
`docs/pass200_rsta_binding_receipt_d6270a9.json`, with exact SHA-256
`e75944aed5af0fbe53af9febbc9a9a5d30045357eb6b1f086c4ba61e10f82300`.
It was produced by full commit
`d6270a94f14f5e0b4f4a3eeaa23f3f66d9bfaa54` from historical manifest
`docs/pass200_rsta_stage_a_manifest.json` SHA-256
`aafab355a06667a9ca513cddeceb2a0129ea8ee09ce3dec0a19b6839fe15ffb1`.
Its frozen source revision is
`0146f2d1200fec26fcd483005804dbe71ec72786`; its embedded diagnostic SHA-256 is
`78eeb3d0d3f92ad1a0b7e76708851e940a36a1ef260a2618dc58bf7f3fab7f1a`.
The base preregistration remains the byte-exact file SHA-256
`a35cd3469d5561ce59202030dd3c3050e018dbfc537cb0ee0401a1d0340f5857`.

## Measured specification defect

The historical final descriptor packs were reproduced bit-for-bit when the current
source used its historical/default cuDNN arithmetic, in which cuDNN TF32 was
enabled. Enabling deterministic algorithms while retaining cuDNN TF32 also remained
bit-exact. Changing only `torch.backends.cudnn.allow_tf32` to `False` changed the
re-exported descriptors, with maximum absolute differences `0.0005712024867534637`
for train, `0.0008309260010719299` for query, and `0.00044417381286621094` for
gallery. These exceed the frozen `2e-5` binding tolerance.

The original implementation wrongly combined two distinct arithmetic domains in
one process: historical descriptor reproduction and the prospectively frozen
TF32-disabled scientific diagnostic. Widening the tolerance, toggling TF32 inside
the scientific process, or silently skipping current-source binding would each be
post-hoc. The repair is a content-addressed receipt boundary between two processes.

## Frozen two-process boundary

1. The completed historical binding process is represented only by the exact
   immutable receipt named above. It used the historical/default cuDNN-TF32
   arithmetic, exported all train/query/gallery descriptors, matched all frozen
   packs bit-for-bit, recomputed the official retrieval binding, recorded source and
   artifact hashes, computed no candidate value, and terminated.
2. Every integrity-smoke or scientific process is fresh. The controller exports
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process start. Inside the process,
   deterministic configuration is the first torch action: deterministic algorithms
   are fail-closed, cuDNN benchmark is disabled, CUDA-matmul and cuDNN TF32 are both
   disabled, autocast is disabled, and FP32 is retained except for frozen float64
   reductions.
3. The scientific process validates the receipt and both provenance domains before
   loading any artifact semantically. It never re-enables TF32, never exports
   descriptors, never loads query/gallery/prehead arrays, and never calls the old
   full binding loader. It consumes only the digest-bound final training pack,
   checkpoint, report scalars, proxy table, and corrected source pixels needed by
   Stage A.
4. There is one allowed receipt, one exact digest, and four exact seeds. There is no
   fallback receipt, selective seed acceptance, tolerance widening, re-export, or
   arithmetic substitution. Any mismatch is `INVALID`, with no candidate output.

## Historical receipt validation

Before checkpoint or NPZ semantic access, hash the receipt bytes and require the
exact digest above. Parse strict finite JSON with duplicate-key rejection and exact
key sets. Require schema `1`, diagnostic `pass200_rsta_stage_a`, mode
`binding_only`, `candidate_values_computed=false`, verdict `NOT_COMPUTED`, and
`uses_test_data=artifact_binding_only`. Require cross-seed training-row identity and
query/gallery-release flags true, export batch size `128`, descriptor absolute and
relative tolerances both exactly `2e-5`, and seeds `0,1,2,3` exactly once.

Validate the historical producer commit, historical manifest Git blob and SHA,
base preregistration path/hash, artifact schema path/hash, frozen source revision,
diagnostic path/hash, every historical source-file Git blob, and every artifact
path/hash. For every seed and split require the registered row/identity counts,
source-export digest, tolerances, and bit-exact recorded maximum descriptor
difference `0`. Require train example-ID, label, and source-order hashes to agree
across all four seeds. Match receipt retrieval values to digest-bound report and
retrieval JSON scalars without loading query or gallery arrays.

The historical audit validates historical Git blobs; it must not pretend that the
old diagnostic script is the currently executing file. Independently validate the
current scientific commit and the current diagnostic plus every transitive
model/data/loss source file against the amended manifest before Stage A.

## Training-only loader contract

After receipt validation, stream-hash every registered immutable artifact,
including query/gallery/prehead files, without materializing their arrays. Read only
report and retrieval JSON scalars from binding-only artifacts. Validate the trusted
checkpoint's final-state source, full training configuration, student evaluation
source, proxy schema, and exactly one proxy per train identity. Load only the final
training NPZ and require embedded checkpoint/report digests, exact shape, finite
unit rows, labels, unique IDs, literal source paths, and canonical synthesized row
indices. Recompute its train order and source-export hashes and match the receipt.

The returned scientific object contains training-only descriptors/labels/IDs/paths,
the checkpoint model/proxies, receipt provenance, and no query/gallery/prehead
array. Smoke loads seed 0 only after validating the global four-seed receipt;
scientific execution loads all four seeds and preserves the frozen cross-seed
binding.

## CLI and fail-closed execution order

Smoke and scientific require the amended manifest, exact receipt, and output path,
plus exactly one of `--smoke-only` or `--scientific`. The order is:

`fresh process -> deterministic TF32-off configuration -> receipt and manifest
validation -> training-only load -> deterministic tensor cache -> model/derivative
integrity gates -> candidate scoring`.

Receipt failure must occur before checkpoint/NPZ/model access and must create no
output. Smoke and scientific interfaces do not accept a source exporter. Binding
receipt creation, if retained for historical auditability, is a mutually exclusive
command that exits and is never chained to scoring. Both outputs record the receipt
SHA, producer commit, historical manifest SHA, per-seed training-pack/source-export
hashes, and current scientific execution audit.

## Falsification and interpretation

This amendment authorizes only restoration of the already-preregistered integrity
smoke and Stage-A diagnostic. It creates no evidence for RSTA. If any receipt,
artifact, source, deterministic, rotation, adjoint, repeatability, or training-only
boundary check fails, the result is `INVALID`. If integrity passes, the original
Stage-A prediction and all original PASS/FAIL/UNRESOLVED thresholds govern without
change.
