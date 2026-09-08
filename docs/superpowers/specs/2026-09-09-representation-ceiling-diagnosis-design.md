# Representation-Ceiling Diagnosis Design

## Purpose

Determine whether the remaining compact-retrieval quality gap is caused by
128-dimensional width, the linear source representation, the relational
objective and sampling recipe, or quantization. The diagnosis is train-only
and class-disjoint. It is generic to paired source/teacher embeddings and must
not encode SOP-specific class names, IDs, or thresholds in the library API.

The authenticated SOP comparison establishes 0.4111926207 float and
0.4071995162 symmetric-int4 mAP@R for the 128D relational map, compared with
0.3933490415 and 0.3797123124 for PCA128. The float-to-int4 loss is only
0.0039931046; therefore quantization polish cannot explain the approximately
0.0692 gap to the historical teacher. Asymmetric scoring and rotation/clipping
failed their preregistered gates and are excluded from this diagnosis.

## Frozen train-only protocol

For each split seed `(17, 1729, 65537)`, assign complete training classes to an
80% fit partition or 20% validation partition by a deterministic SHA-256
ordering. The implementation must preserve every class intact, emit ordered
row and class identities, and reject empty partitions or validation classes
without a second validation item. Test rows are not inputs.

Normalize source and teacher rows before fitting. Evaluate every arm on the
same validation rows with self-excluding cosine retrieval, source-ordinal tie
breaks, mAP@R primary, and R@1 secondary. Absolute validation values are not
compared with historical test values because gallery cardinality differs.

The first, closed-form stage contains:

1. `source-full`: normalized source embeddings.
2. `teacher-full`: normalized teacher embeddings.
3. `teacher-pca128`: centered PCA fit only on teacher fit rows, then normalized.
4. `ridge-source-teacher-full`: a ridge affine map from normalized source to
   normalized teacher, fit only on fit rows, then normalized.
5. `ridge-source-teacher-pca128`: the ridge prediction passed through a
   128-dimensional PCA fit only on predicted fit rows, then normalized.
6. `source-pca128`: centered PCA of normalized source fit rows as the compact
   width control.

The ridge solve uses float64 augmented normal equations with an unpenalized
intercept and fixed relative penalties `(1e-6, 1e-4, 1e-2)`. Penalty selection
uses only a deterministic 80/20 class-disjoint sub-split of the outer fit
partition and mean squared error between normalized predictions and normalized
teacher rows. Ties select the smaller penalty. The outer validation partition
is opened once after selection. Rank failure, nonfinite inputs or outputs,
zero-norm outputs, and singular solves fail closed and remain evidence.

## Decision boundary

Across the three outer splits, report per-query outcomes and 10,000 paired
class-cluster bootstrap replicates, seed 17, one-sided 95% lower bounds.
Proposed engineering gates are fixed before execution:

- `ridge-source-teacher-full - source-full >= 0.005` with a positive lower
  bound means the source contains recoverable teacher-aligned structure and
  justifies a neighborhood-sampling experiment.
- `teacher-pca128 - teacher-full >= -0.010` means 128 dimensions are not a
  material width bottleneck. A loss below `-0.030` justifies evaluating
  192/256D compact codes as an explicit bytes/quality product knob.
- A ridge full-width gain below `0.005` stops closed-form head-side claims; it
  does not reject nonlinear heads, but moves the primary quality effort to
  encoder/backbone distillation.

No configuration is selected using SOP test results. Passing this train-only
screen authorizes only the named next experiment, not a release claim.

## Conditional second stage

If and only if the ridge full-width gate passes, compare the existing random
batch relational trainer with a query-independent teacher-neighborhood
sampler. Each update uses 64 uniformly sampled anchors and their 15 nearest
teacher neighbors from fit rows, deduplicated, then uniformly fills to 1,024
rows. Use paired seeds `(17, 29, 43)`, exactly 1,000 updates, identical optimizer
settings, and both float and canonical symmetric-int4 validation. Advance only
for mean int4 gain at least 0.003, positive gain for every seed, float
improvement, and a positive paired class-bootstrap lower bound.

If the ridge gate fails, do not spend DGX time on neighborhood sampling. If a
later full-rank or nonlinear ceiling is desired, it requires a separate frozen
protocol rather than being added adaptively to this receipt.

## Artifacts and safety

The evaluator accepts only authenticated local source and teacher archives,
an exact source commit, and one explicit execution flag. It writes canonical
newline-terminated JSON through reserved partial files with no-clobber final
publication. The receipt binds archive/source hashes, split identities, all
fit parameters, selected ridge penalties, complete per-query evidence,
environment, and every gate calculation. It is always `claim_eligible=false`.

The reusable library module owns deterministic class partitioning, centered
PCA transforms, ridge affine fitting, and normalized affine application. It
must not import experimental scripts or dataset loaders. Dataset authentication,
retrieval scoring, receipt construction, and CLI behavior remain in the script.
