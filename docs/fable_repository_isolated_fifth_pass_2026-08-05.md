# Repository-isolated outcome-only Fable pass

Date: 2026-08-05.

## Execution validity

This pass ran from `/tmp`, outside every Git worktree, with Claude safe mode,
session persistence disabled, and only web search/fetch available. It received
the same neutral outcome brief: problem, legal inputs, deployment and cost
constraints, and audited performance horizons. It received no method hints,
failure catalogue, repository measurements, file paths, or commit metadata.

This is the first pass in this sequence that is both web-enabled and isolated
from repository metadata.

## Result

Fable returned **`NONE`**. After searching thirteen mechanism families, it said
it could not defend a method that was simultaneously novel, legal, and funded
by a measurement strongly enough to cross a matched-capacity horizon on two
datasets. The only nearly open family it named was train-time hubness
penalization, which it rejected itself as density spreading adjacent to
uniformity/anti-collapse work and unsupported by a measured R@1 error budget.

The answer also treated the higher-capacity 0.878 CUB / 0.947 Cars / 0.9448
In-Shop observations as a single matched-capacity target. They are not one
configuration or one method triple: PFML supplies the CUB and Cars observations
with different high-capacity initializations, while CRT supplies In-Shop with
MiT-B2. A future proposal must name its backbone tier and compare like for like.

## Fallback measurement repeats candidate 371

Fable proposed training PFML/ViT on half the CUB training identities, evaluating
on the other half, fitting an oracle linear metric with the held-out labels, and
comparing it with a metric fitted on the training-side identities. It then
proposed repeating across five splits and Cars196 and checking whether the same
gap appears when fitted diagnostically on the true benchmark test classes.

This is candidate 371 again and is rejected for the same reasons:

- a held-out-label metric measures closed-set fitting of that chosen estimator,
  not transferable open-set method headroom;
- decomposition into the training-discriminative span and its complement does
  not identify which information an end-to-end nonlinear backbone can learn;
- five splits on each of two datasets plus metric fits are materially more than
  the claimed 0.6x of one ordinary training; and
- fitting anything with true test-class labels or using its result to choose a
  method contaminates the benchmark and is prohibited.

Nested training-identity splits can lawfully estimate transfer of one specified
metric estimator, but cannot establish an oracle ceiling for all possible
training relations, losses, or architectures.

## Verdict

**`NONE` accepted; fallback DEAD as candidate 371. No diagnostic,
implementation, preregistration, or GPU run.** This is genuine independent
negative evidence, not an impossibility result. The next pass will keep the
problem and frontier neutral but ask directly for a better method, without
inviting a fallback measurement that has now repeated three times.

