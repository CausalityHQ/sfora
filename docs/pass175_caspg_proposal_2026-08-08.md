# Pass 175 — Conformal Ambiguity-Set Pair Gate (CASPG)

## Status

Gate 2: **LIVE-NARROW; CPU falsifier required.** No implementation or GPU run
is authorized by this proposal alone.

## Gate 1: measured provenance

RSPG measured that target-excluded rival signatures retain 64.49% of within-
class pairs on CUB but only 8.63% on In-Shop. The signal is therefore highly
dataset-dependent: rival identity distributions are nearly class-constant on
CUB, while the 3,997-class In-Shop problem contains substantial within-class
variation. CASPG keeps that measured referent but replaces the uncalibrated
softmax/rival histogram with a split-conformal ambiguity set. The intended
mechanism is to use finite-sample-calibrated uncertainty about rival identities
to decide which same-class pairs remain positive, rather than to weight every
pair. The preregistered CPU test is on the corrected In-Shop training artifact,
not CUB.

## Gate 2: mechanism-level prior-art audit

Conformal prediction produces calibrated label sets (Vovk et al.; Romano et
al., *Uncertainty Sets for Image Classifiers*, NeurIPS 2020). Local-distance
metric learning has been used to improve conformal-predictor efficiency, but
not to gate DML positives. Existing conformal retrieval work calibrates test
candidate sets or uncertainty; it does not use overlap of two different
images' target-excluded sets as a train-time eligibility predicate.

The closest DML calibration antecedent found is Zhang et al., *Threshold-
Consistent Margin Loss for Open-World Deep Metric Learning* (ICLR 2024). It
adds a margin regularizer for threshold consistency and explicitly notes that
ordinary conformal prediction assumes a closed label world; it does not make
conformal sets a cross-image positive gate. This is an adjacent warning, not an
exact match, and keeps CASPG LIVE-NARROW rather than claiming novelty.

The distinction from RSPG/Liao is: CASPG uses calibrated finite prediction
sets with a coverage-derived threshold, and the set-overlap predicate changes
same-class positive eligibility. It is not a raw rival-distribution signature,
instance-neighbourhood contextual loss, uncertainty reweighting, or a
single-image auxiliary objective. This remains LIVE-NARROW pending a focused
search for conformal metric-learning positive mining; any exact antecedent
closes the candidate before CPU work.

## Gate 3: preregistration

Use class-disjoint cross-fitting on the corrected In-Shop training pack. Fit a
linear classifier on the frozen embeddings, reserve a calibration fold, and
form split/Mondrian conformal sets from nonconformity `1 - p(class|x)`. Remove
the target class from each set. Choose the Jaccard threshold `tau` from the
training fold only; do not tune it on the held-out pairs. Report set-size,
coverage, retention, pair-gate AUC, and cross-fitted support-to-query R@1.

The candidate survives Gate 1 only if pair-gate AUC is **>0.70** and selected
support-to-query R@1 improves the all-positive control by **at least 0.5
points**. Either failure kills CASPG without GPU. Empty or near-universal sets
are also reported as a diagnostic failure, not silently repaired by tuning.
