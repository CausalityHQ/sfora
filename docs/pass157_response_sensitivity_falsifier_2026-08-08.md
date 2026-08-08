# Pass 157 — response sensitivity CPU falsifier (NO-GO at Gate 1)

## Test

Using the independently audited corrected In-Shop response pack, each of the
five deterministic views was evaluated against the centered-image gallery,
excluding itself. The outcome was whether the nearest gallery image had the
correct product identity. The control was centered-image leave-one-out margin;
the candidate added image sensitivity

`q_i = max_v (1 - z_i^T z_i^(v))`.

Five identity-hash folds were fit on 129,350 finite crop-view rows. This is the
CPU falsifier preregistered in Pass156; it is diagnostic evidence only and does
not authorize a GPU run.

## Result

Cross-fold AUROC was **0.996462** for the ambiguity-margin control and
**0.996504** after adding response sensitivity: incremental **Delta AUROC
+0.000041**. Crop-view correctness was already **96.16%**. The result is far
below the proposed +0.05 provenance threshold and does not show that response
sensitivity predicts unsafe augmentation beyond ordinary embedding ambiguity.

## Decision

The response-adaptive augmentation near miss is `NO-GO` at Gate 1 in this
operating pack. No augmentation policy, architecture, or GPU screen follows.
This does not invalidate the independently verified response-strata relation;
it limits the causal claim that the scalar sensitivity can identify training
failures.
