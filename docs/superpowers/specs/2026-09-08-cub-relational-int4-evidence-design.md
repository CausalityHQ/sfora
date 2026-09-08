# CUB Relational Int4 Evidence Design

## Goal

Test whether Sfora's label-free relational linear compaction transfers to unseen
CUB-200-2011 classes at an equal 66-byte persistent row budget, without tuning
on CUB test classes or weakening deterministic CPU retrieval semantics.

## Frozen evidence

- Dataset: official CUB-200-2011 archive, 1,150,585,339 bytes, SHA-256
  `0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081`.
- Protocol: classes 1-100 train (5,864 images), classes 101-200 test
  (5,924 images); test labels are never used for fitting.
- Source: UNICOM ViT-B/16; teacher: UNICOM ViT-L/14@336px; both use the
  same frozen upstream revision and exact checkpoints.
- Every consumed metadata/image byte must match its member in the authenticated
  archive before and after export. The ordered record manifest and every array
  are independently hashed. Export refuses a modified or untracked upstream
  checkout, and each archive binds the exact CUB and SOP exporter source bytes.

## Arms and gates

The treatment is a 128-dimensional relational projection packed as signed int4
codes plus a two-byte inverse norm: exactly 66 bytes/item. Three fixed seeds are
trained for 1,000 optimizer steps each. Primary equal-byte controls are
PCA128-int4, source-independent rotated PCA128-int4, and train-only
ridge-to-teacher128-int4. Full source/teacher and PCA64-int8 are diagnostic
ceilings/controls, not alternate treatments.

Each treatment seed must exceed every primary control by at least 0.010 MAP@R,
have a multiplicity-adjusted class-bootstrap MAP@R lower bound above zero, and
an R@1 lower bound above -0.005. The deployment seed is fixed at 17. CPU timing
uses one thread, the same packed scorer and score-descending/ordinal-ascending
tie rule as quality, 1,000 warmups and 10,000 measured paired calls. Its p95 may
not exceed 1.10x the folded rotated-PCA control. Results remain
`claim_eligible=false`: CUB is an independent transfer falsifier, not a broad
SOTA claim by itself.

## Publication and provenance

The evaluator accepts local files only and requires explicit archive digests
and the exact committed source revision. Its canonical receipt binds the two
embedding archives, upstream dataset/model/checkpoint/row identities, every
implementation source that affects scoring/statistics, full training recipe,
device/runtime information, all per-query evidence, model bytes, and latency
samples. Before scoring, every executing scientific source must equal its byte
at the registered commit. Outputs are reserved before expensive work and
publish exclusively as one rollback-safe transaction without overwriting or
deleting another run's files.

## Next research boundary

CUB quality stays unopened until the method and controls are committed. Further
method selection occurs only on already-burned SOP evidence. The prioritized
generic hypothesis is quantization-aware output geometry: compare asymmetric
float-query/int4-gallery scoring, fixed rotation, clipped-scale int4, and a
train-only learned orthogonal rotation/straight-through quantization objective.
Fold any accepted rotation into the linear projection and test against matched
PCA/ridge controls. Only the frozen winner may be evaluated once on CUB.
