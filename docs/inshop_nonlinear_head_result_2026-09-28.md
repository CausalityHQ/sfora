# Cached nonlinear activation filter: stop after seed17

The frozen equal-parameter GELU residual did not beat its half-linear control.
Stop this configuration; do not run seeds23/29, change rank/scaling/steps, or
advance an image-level pair. The joint production quality-and-speed goal remains
unmet. This experiment addresses cached activation placement only.

## Verified quality

DeepFashion In-Shop official TRAIN fit identities only:6645 inner-training
images/1002 products,6638 inner-validation gallery images/1002 products,
6632 eligible self-excluded queries. Six singleton products remain gallery
negatives but supply no queries. Outer TRAIN holdout and official query/gallery
were not read. Previously explored fit identities mean this is an exploratory
inner diagnostic, not independent confirmation or an official benchmark.

| Frozen seed17 arm | Packed R@1 (%) | Packed mAP@R (%) |
|---|---:|---:|
| Inner-training PCA initialization |86.9723|53.8567|
| Half-linear residual control |96.1701|73.5132|
| GELU residual treatment |95.9590|72.8482|
| Treatment minus control (percentage points) |−0.2111|−0.6650|

Each arm ran1000 identical64-image cached FP32 steps. Both applied909 detached
rank updates; remaining whole batches contained singleton anchors and used CE
only. The frozen first-seed gate required strictly greater R@1 than control and
PCA and mAP@R improvement≥0.2 percentage points over control, with no PCA mAP
regression. Treatment failed both control comparisons. The three-seed bootstrap
was not run; no confidence interval or multi-seed conclusion follows.

Treatment residual affine-reconstruction SSE/total-energy fractions were
0.1050008 on inner train and0.1930361 on inner validation, both above frozen0.10.
Thus the tested quality negative meets the declared curvature qualification;
it does not prove every nonlinear head, encoder or training regime impossible.

## Verified resources and execution

DGX Spark GB10 host, CUDA hidden, eight Torch CPU threads. No GPU was used.
Native systemd invocation`16d6a79f940646a1a481040d0719fb64` enforced180s/8GiB
whole-process limits. GNU time reports37.23s wall,2071960KiB maximum process RSS,
exit0. Script's post-import wall was35.0589s; control/treatment optimization
walls16.1869/16.4837s include bank setup. These are CPU cached-training costs,
not encoder training throughput or inference latency. The earlier metadata
preflight separately cost3.28s/1304552KiB; saved-weight replay has its own receipt.
GPU compute apps and running Sfora units were empty after completion.

The synthetic check passed full ArcFace-plus-rank objective/primary/source
initial gradient parity, raw/packed arm equality, live residual learning,
dead-zero detection, duplicate refresh, singleton contracts and product-bootstrap
pass/null examples. It initially exposed an old deployed Sfora rank-loss API;
only the isolated probe source was replaced with repository source before any
optimizer/score. The deployed package was untouched. Exact executable and27
loaded dependency hashes were frozen before outcomes in commit`0d9b9563`.

Independent saved-weight replay uses functional affine/GELU operations rather
than the probe's model class and reproduces all packed per-query R@1/AP values
exactly for both learned arms. It verifies source/cache/weight hashes, reported
means,1000-step inventories and the failed gate. No production package or
protected CuTile candidate changed.

[Raw reports, weights, logs, launch authority and runnable replay](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-nonlinear-head-v1/).
[Reviewed protocol and preflight](inshop_nonlinear_research_scope_2026-09-28.md).

Next decision: retain this negative and choose a materially different source
representation with a credible matched end-to-end cost before another quality
run. Existing failed head/loss/sampling/pixel-transfer configurations stay closed.
A new route must first qualify access/source/processor authority, freeze TRAIN-only
selection and matched resource controls, pass independent critique and cheap
parity/cost gates, then paired quality; official qualification and full-pipeline
latency gates remain required. No new job follows from this result automatically.
