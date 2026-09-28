# Source-classifier gradient gate

The global-caption arm is closed. Test one label-only mechanism: all current
identity/rank losses pass through the 128-D affine head, restricting each
pooler's gradient to the head row space plus its normalization direction.
A training-only 1,024-D source classifier could supply useful identity
directions outside that span, without changing the deployed head/scorer.
This is auxiliary supervision, not a novelty claim. Earlier raw-pooler
versus trained-head retrieval did not train such a source classifier.

Before training, run one **<=120-second CPU** diagnostic from the pinned
pretrained cache and original fit-only PCA initializer. Select 512 fit
products with at least two images and at least two products in their clothing
category, ordered by SHA-256 of `inshop-source-classifier-v1\0` plus product
name. Select one image per product by relative-path SHA. No held outcomes.

Compare centered, normalized full-source and PCA-128 class centroids.
Remove the query image from its own positive centroid in **both** spaces;
keep other class centroids fixed. Use all 2,004 fit-class centroids as
candidates, ordinal ties. This is fit prototype classification, not image
retrieval or held Recall@1. Measure ordinary cosine CE at fixed scale 64
for gradients; do not claim it reproduces ArcFace's margin gradient.

Frozen advance rules:

1. Full minus compact prototype classification accuracy at least **+1.0
   percentage point**, with paired product-bootstrap lower95 **above zero**.
2. Median full-source label-gradient norm outside the compact-head/source
   span at least **30%** of its whole gradient.
3. Median true-versus-category-shuffled label-gradient difference in that
   unused span at least **30%** of the true unused gradient.
4. Compact gradient outside the same span at most **1e-4** relative norm;
   finite statistics, source/PCA/partition/fit hashes and positive inventory
   validate; total CPU wall at most 120 seconds.

Any failure stops this fixed source-centroid classifier route before an
encoder run. A pass only authorizes a separately frozen gradient/cost smoke,
then a matched TRAIN held gate. Orthogonality is local at the pooler, not a
guarantee about shared encoder parameter updates or generalization.

## Terminal CPU result

The single original run exited 0 in **4.821838 CPU seconds**, with no
encoder training or held/official evaluation. Receipt:
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-classifier-cached-v1.json`,
SHA-256 `ace3f13e92bc0357c195b307da3fde128780f33a9f7a9a8c12af7b0b8200f29f`.
Protocol and script were pushed at `76e7974b` before metrics were read.

| Gate | Measured result | Decision |
|---|---|---|
| TRAIN fit leave-query-out prototype classification | 452/512 (88.28125%) full vs 435/512 (84.9609375%) PCA-128; +3.3203125 pp, product-bootstrap 95% [+1.7578125, +5.078125] pp | GO |
| Full gradient outside compact/source span | Median 42.3683% | GO |
| True versus category-shuffled unused gradient difference | Median 4452.5822 times true norm | GO; not a strength claim |
| Compact gradient numerical span check | Max relative residual 7.85698e-7 | GO |
| Training cost / serving cost / exact packed retrieval | No encoder run; no serving or packed-scoring change | Not measured by this gate |

Independent receipt replay verified the hit totals, gain, bootstrap interval,
all criteria and wall bound. There were 18 source-only wins and one
compact-only win. Two focused checks cover the projection and actual positive
query removal; Ruff passes. The unusually large shuffled-gradient ratio is
compatible with a tiny true gradient at nearly correct prototypes; it does
not authorize treating this objective as a strong update. A design review
must resolve initializer, loss, coefficient, gradient saturation and shared
clipping before the separately frozen 17-update smoke. The deployed library
is unchanged, and this result cannot establish unseen retrieval quality.

One dual design critique completed, group `781ca331ba644f6b`: Opus
`be0ee1b940794661`, Astra `0ff1b9a4c56b4f65`. No GPU job was started.

## Review reconciliation and next frozen smoke

Both reviewers support one bounded smoke, with different proposed losses.
Choose Astra's smaller **frozen centroid, margin-zero, scale-64, coefficient
0.1** configuration: it preserves the screened geometry and derivative,
adds no learned classifier and cannot explain a gain merely by relabeling
freely learned classes. Opus proposed a learned ArcFace source classifier;
that changes both derivative and capacity and is deferred. Native ArcFace's
`no_grad` margin replacement preserves a straight-through target gradient;
it is not evidence that target gradients vanish. Keep the main loss intact.
Full independently labeled reviews are archived as
`inshop-source-classifier-design-review-v1.json` beside the CPU receipt.

Use the fit-only normalized source mean and centered, normalized class
centroids, detached throughout. Ordinary training prototypes include the
current image; the diagnostic excluded it. Training loss improvement is
therefore not corroboration of the leave-query-out result.

Only **17 updates each** for matched main-only and auxiliary arms, seed
179024, original batches/initialization, frozen embeddings and first 12
encoder blocks, unchanged main ArcFace and 8x bank SmoothAP. At initial and
terminal probes, record separate main and weighted auxiliary encoder-gradient
norms/cosine, target probabilities and correct routing. Compute the fixed
category-shuffled loss on the same forwards without optimizing it. It is a
wrong-prototype diagnostic, not an identity-breaking generalization control.

Frozen engineering stop rules before any GPU execution:

| Check | KILL threshold |
|---|---|
| Pairing | Any initialization, fit authority, main configuration or any of all 17 input-batch hashes differs |
| Numerical/routing | Any nonfinite loss, gradient or parameter; skipped update; auxiliary reaches head/main classifier; auxiliary fails to reach unfrozen encoder |
| Strength | Weighted auxiliary encoder norm below 1% of main, or majority target probabilities above 1-1e-6, at both initial and terminal probes |
| Dominance/clipping | Weighted auxiliary encoder norm above 25% of main at either probe, or median paired clipping multiplier changes by more than 10% |
| Collapse | Same fixed fit batch compact effective rank or centered variance more than 20% below matched control |
| Cost | Entire paired smoke including setup/diagnostics exceeds 120 seconds; more than 17 updates per arm; memory failure |

Check margin-zero loss/gradient parity, routing and projection against a
nonorthonormal head before launch. QR-orthonormalize current head rows before
projecting after an optimizer step: the initial PCA projector is valid only
while its rows remain orthonormal. Log all step losses, all 17 input hashes,
clipping multipliers, setup/training/diagnostic times and peak memory. Do not
export/read held outcomes during this smoke. Any failure kills this fixed
configuration without coefficient or temperature search. A pass requires a
separately frozen TRAIN-held protocol before more training; it does not
automatically authorize a 100/1000-update run or any official read.

The paired launcher executes control first and applies the same strength,
saturation and dominance rules to its auxiliary diagnostic. If those probes
already reject the fixed configuration, it stops before the auxiliary arm.
The 120-second cap includes subprocess imports, both setups and diagnostics;
on timeout the whole active child process group is killed, and the original
wrapper writes a terminal KILL receipt. The smoke omits checkpoint saving
and all held exports. Raw per-arm logs, initial/terminal probes and receipts
remain preserved even on early rejection.

## Terminal training decision: KILL

The ORIGINAL DGX unit `sfora-inshop-source-centroid-smoke-v1`, invocation
`69ff03ca86a1410087f1cf48cee46937`, exited successfully. Control child PID
761349 completed exactly 17 updates; the wrapper closed normally and
**did not launch the auxiliary arm**. Whole smoke wall was 27.343378 s,
below 120 s. The negative engineering decision is not a crashed run.

| Dataset/split and baseline | Quality and gradient evidence | Cost and decision |
|---|---|---|
| In-Shop official TRAIN fit 13,283 images/2,004 products; unchanged true-freeze main ArcFace+bank | No held R@1/mAP measured. Weighted auxiliary/main encoder norm 0.833670% initially, 0.004986% terminal; both below frozen 1% floor | KILL fixed centroid/CE64/coefficient-0.1 route |
| Same fixed fit batch | Auxiliary loss 0.068080 -> 0.00024954 under **main-only** training; true targets above 1-1e-6: 28.125% -> 57.8125%; head auxiliary gradient absent | Main objective already makes these training prototypes nearly trivial |
| Original control smoke, including diagnostic overhead | 17 stable updates/1,088 training images, no skipped/nonfinite update; 17 input hashes; no checkpoint or held export | Training wall 17.305345 s, 62.870748 images/s; peak allocated CUDA 20,255,933,440 bytes; this instrumented smoke is not a full-training performance benchmark |
| Serving and exactness | No new serving calls or packed-scoring modification; no paired treatment to certify | p50/p95/p99/QPS not measured in this gate; no new quality/speed claim |

Receipt `inshop-source-centroid-smoke-v1/receipt.json` SHA-256
`30ef75e9bcab9cfffb6522c603df894f515582c431b1a426df95aff09750a33c`.
The underlying control receipt SHA, all 17 losses/input hashes, exact frozen
failure replay, normal exit, no second arm and no held export were verified
independently. Raw logs and initial/terminal probes are retained alongside
the receipt. The live-job inspection confirmed no remaining GPU process.

Do not increase coefficient/temperature or swap margins after seeing this
result. Useful directions in a fit prototype screen did not imply a useful
encoder update. The optional loss now lives only in the experimental trainer,
with its routing/parity check; it was removed from the production module.
The current wheel proves the experimental loss is absent and the production
head/serving bytes match their source. The pre-rejection wheel check is
historical evidence, superseded by `source-centroid-closed-wheel-check-v1.json`.
No retraining, 100/1000-update run, p99 gate or official read is authorized by
this rejected configuration. Joint production quality/speed remains unmet.
