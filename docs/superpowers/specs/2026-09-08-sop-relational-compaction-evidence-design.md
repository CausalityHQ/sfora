# SOP relational compaction evidence design

## Purpose

Produce a second, independently reproducible domain evaluation for Sfora's
generic 64-dimensional relational linear compressor. Stanford Online Products
(SOP) is the fixed domain because the official split, images, pinned UNICOM
source checkout, and authenticated B16/L14 checkpoints are already available
on the DGX. InShop results and labels do not select any SOP parameter.

## Scientific contract

The compressor consumes only paired source/teacher embeddings from the official
SOP training split. It never consumes training class labels, test embeddings,
or test labels. The evaluator uses the official 60,502-image test split as a
symmetric leave-self-out retrieval task and reports MAP@R and Recall@1.

The fixed arms are PCA64 int8, PCA128 int8, and relational-linear64 int8 for
seeds 17, 1729, and 65537. The optimizer, temperature, batch size, epoch count,
packing, and exact packed cosine scorer are identical to the released InShop
recipe. No SOP result changes these values.

Promotion requires every relational seed to exceed PCA64 by at least 0.005 MAP,
with a multiplicity-adjusted class-cluster bootstrap lower bound above zero and
a Recall@1 lower bound above -0.005. The method remains 66 persistent bytes per
item and the paired target-CPU p95 ratio must remain at most 1.10. PCA128 is a
quality reference, not an iso-byte baseline.

This evaluates generalization of the learning procedure after domain-specific,
train-only fitting. It is not zero-shot transfer of the InShop-fitted map and
does not establish universal or cross-modal generality.

## Archive authority

The exporter reads the official `Ebay_train.txt` and `Ebay_test.txt` files,
requires 59,551/11,318 train rows/classes and 60,502/11,316 test rows/classes,
rejects missing, symlinked, or duplicate paths, and binds the ordered records to
a SHA-256 manifest digest. It loads models only from UNICOM revision
`d71992ed969e6c271436ac0a0ee1f3ca61474ac0` and accepts only these checkpoints:

- `UNICOM-ViT-B/16`, SHA-256
  `c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef`;
- `UNICOM-ViT-L/14@336px`, SHA-256
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`.

Each archive has exact keys, float32 finite nonzero embeddings, integer labels,
ordered path digests, per-array SHA-256 values, and canonical metadata. Output
publication is exclusive and atomic; partial output is removed only when owned
by the current process.

## Components

`scripts/export_unicom_sop_embeddings.py` owns official SOP parsing, pinned
UNICOM model loading, deterministic encoding, and archive publication. Its
pure parsing and archive-writing functions accept injected encoders so tests do
not require a GPU or model download.

`scripts/probe_sop_relational_linear.py` owns strict paired-archive loading,
train-only fitting, symmetric leave-self-out retrieval, class-cluster bootstrap,
paired CPU latency, and canonical receipts. It imports the public Sfora method;
no SOP behavior enters `src/sfora/joint_relational_compaction.py`.

## Failure and evidence handling

All CLI paths are absolute and outputs must not exist. Any digest, schema,
cardinality, row-order, checkpoint, device, nonfinite value, self-exclusion, or
positive-count mismatch fails closed before publication. Scientific output is
written to `.partial`, validated, and atomically renamed only after every arm
and gate completes. Interrupted or failed runs are not quality evidence.

DGX execution is a separate bounded operation. One exporter runs at a time;
source and teacher archives are produced in separate processes. The evaluator
runs once after both hashes are frozen. Existing persistent DGX monitoring is
reused; no overlapping evaluator is started.

## Explicit non-goals

- No backbone fine-tuning or serving-time teacher.
- No SOP-specific change to the library method or hyperparameters.
- No class-name semantics in this evidence pass. SOP labels are identifiers,
  not reliable natural-language descriptions.
- No universal or state-of-the-art claim from two image-retrieval domains.
