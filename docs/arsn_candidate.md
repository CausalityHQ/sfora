# Candidate ARSN — Amortized Rival-Statistics Normalization (Pass 191)

## Gate 1: provenance

The live CE-BN probe on corrected In-Shop descriptors improved R@1 from
`0.913701` to `0.927275` (+1.357 pt), but applying the label-excluded
normalization directly during training collapsed the end-to-end run to
`0.684344`.  The failure is the train/evaluation mismatch: CE-BN needs labels
and a minibatch at training time, while deployment emits one descriptor from
one image.  ARSN keeps the measured transform but learns a deterministic
per-image approximation to its sufficient statistics.

## Gate 2: prior-art boundary

ARSN is not ordinary batch normalization or a generic teacher loss.  For each
training image, it computes the leave-own-class-out moments `(mu_i, sigma_i)`
from the labelled batch, trains a small predictor `q_phi(h_i)` to regress those
moments with stop-gradient targets, and deploys
`z_i = (h_i - q_mu(h_i)) / q_sigma(h_i)` without labels or a batch.

Adjacent prior art is Batch Renormalization (Ioffe, NeurIPS 2017), EvalNorm
(Singh & Krishnan, arXiv:1904.06031), Learning Using Privileged Information
(Vapnik & Izmailov, JMLR 2015), and teacher-based metric distillation (Yu
et al., CVPR 2019).  These cover batch-statistic correction or generic
privileged/teacher transfer, but the audit found no benchmark-matched DML
method that predicts label-excluded per-instance normalization sufficient
statistics and uses them as the deployed descriptor.  If that distinction is
judged cosmetic, ARSN is DEAD at Gate 2 and receives no GPU.

## Gate 3: preregistration

Before implementation, fit `q_phi` using training folds only and require a
held-out per-coordinate aggregate `R^2 >= 0.10` for both moments (otherwise
the target is not amortizable and the candidate is killed).  The deciding
In-Shop seed prediction is corrected R@1 delta `>= +0.30 pt` versus paired
Proxy Anchor; falsification is `< +0.15 pt` or any non-positive raw delta.
The descriptor path must be deterministic at inference and all results must
report raw best-over-training and the repository's local-trend selection
diagnostic.
