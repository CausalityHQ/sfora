# Independent SOP fidelity audit 243

Date: 2026-08-03. Conducted by Claude Opus after the split, source-pixel,
drop-last, warm-up-head, and proxy-initialization repairs, while the corrected
run was live. The auditor was instructed to find mismatches rather than propose
methods and did not edit the repository.

## Verdict

No remaining mismatch capable of moving SOP R@1 was identified. The audit
independently traced the exact registered recipe through optimizer grouping and
weight decay, warm-up freeze/unfreeze, model-mode restoration after evaluation,
trainable BatchNorm, BN-Inception GAP+GMP and internal L2 normalization, BGR
0--255 preprocessing, Kaiming head/proxy initialization, gradient value
clipping, StepLR timing, official product labels/counts, Proxy Anchor algebra,
and diagonal exclusion in full self-retrieval.

## One latent defect, inert for this run

The optimizer's head grouping recognized only ResNet `fc.*`, not BN-Inception
`model.embedding.*`. Unlike the earlier warm-up bug, this has exactly zero
effect on the current SOP and In-Shop reference recipes because
`learning_rate == backbone_learning_rate == 6e-4`. It would silently matter for
a future BN-Inception recipe with split rates. The grouping now shares the same
head-name predicate as warm-up, with an unequal-rate regression test. The live
run need not restart because its two affected group learning rates are equal.

## Metric semantics

The primary report field `recall_at_1` is final-state retrieval. The upstream
published 79.2 is selected best-over-training; SFORA records that separately as
`best_test_recall_at_1`. This is not a training mismatch. Both values must be
reported under their actual semantics, and only final-state retrieval can be
independently reconstructed from the persisted final checkpoint in this run.
