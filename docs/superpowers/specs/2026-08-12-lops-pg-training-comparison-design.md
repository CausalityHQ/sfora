# LOPS-PG Real-Training Comparison

## Goal

Determine whether the train-only LOPS-PG directional PASS converts into a
held-out In-Shop retrieval gain under the existing official Proxy Anchor
recipe. This is an ordinary multi-seed experiment using Git commits and normal
result files. It adds no provenance or authentication framework.

## Arms

Run four objectives under identical data, BN-Inception initialization,
augmentation, balanced 45x4 batches, optimizer, learning-rate schedule, epochs,
checkpoint selection, and cosine inference:

1. `proxy_anchor`: unchanged existing implementation.
2. `proxy_anchor_lops_pg`: unchanged PA scalar loss and proxy gradients; only
   the gradient arriving at each normalized image embedding is projected onto
   the leave-one-out positive-cohesion half-space.
3. `proxy_anchor_compactness`: unchanged PA plus the mean
   `1-cos(z_i, stopgrad(m_i))` over examples with a same-class batch peer,
   weighted `0.1`. This is the conventional positive-pulling explanation.
4. `batch_hard_triplet`: the existing registered implementation and margin.

For LOPS-PG, construct `m_i` from all other same-label embeddings in the live
batch, detach it, normalize it, and compute
`p_i=m_i-z_i<z_i,m_i>`. Given the gradient `g_i` delivered to normalized
embedding `z_i`, replace it by
`g_i-max(<g_i,p_i>,0)*p_i/||p_i||^2` when `||p_i||>=1e-8`. Leave rows without a
same-class peer or with degenerate `p_i` unchanged. The hook must never alter
proxy gradients. It records aggregate conflict/skip counts only.

## Staging

Use seeds `0,1,2`. First run a one-epoch seed-0 smoke for all four arms and
require finite loss, a checkpoint, and a valid retrieval report. Then run the
full official 60-epoch recipe sequentially. If resource cost is material, run
PA and LOPS-PG seeds 0-2 first; run compactness and triplet only after at least
two LOPS seeds complete without structural failure.

GPU arithmetic is not assumed deterministic. Freeze all controllable seeds,
sampling, and configuration, but evaluate paired seed outcomes rather than
byte equality. Preserve each process's exact report and checkpoint.

## Decision

LOPS-PG earns a retrieval GO only if all hold:

1. all three LOPS-PG runs complete without nonfinite loss or structural error;
2. LOPS-PG minus PA final Recall@1 is positive in at least two of three paired
   seeds and has positive mean;
3. mean LOPS-PG minus PA Recall@1 is at least `0.0015` (0.15 percentage points),
   the preregistered practical floor used elsewhere in this research branch;
4. LOPS-PG mean Recall@1 is at least the positive-compactness mean;
5. report LOPS conflict coverage is at least 50% in every seed and skipped-row
   fraction is below 1%;
6. no arm changes inference from normalized embedding cosine.

Batch-hard triplet is a strong benchmark, not a required LOPS win. If it wins,
report that plainly. A GO establishes a useful candidate relative to this
reproduced baseline, not SOTA. A SOTA claim requires matching the published
protocol and uncertainty against current methods separately.

## Implementation and assurance

Add focused Torch helpers for the hook and compactness loss, objective registry
entries, configuration/CLI coverage, and a tiny real training integration test.
Tests must prove: the hook's closed-form gradient, non-conflict identity,
degenerate identity, sibling detachment, unchanged proxy gradient, loss
equivalence to PA before the hook, exact objective dispatch, and successful
checkpoint/retrieval output on the tiny image benchmark.

Before GPU execution, run the focused image-end-to-end and CLI tests, Ruff,
py_compile, and a read-only cross-provider review. Fix concrete source defects;
do not change the CPU confirmation or the six training decision predicates.
