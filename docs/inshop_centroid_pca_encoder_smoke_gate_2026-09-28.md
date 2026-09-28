# Centroid-PCA actual encoder smoke: proposed frozen gate

Only the positive CPU F0 licenses this design; no GPU job has started.
Use seed179024, original 13,283-fit/12,599-held partition, native SigLIP2
Large256/freeze_emb, native128 head, ArcFace0.3/64 + bank8, original batch64
schedule, LR/optimizer/BF16, and exact packed/public checkpoint path.

Run native then product-mean initializer serially, 17 updates each, with a
120-second whole-pair watchdog. Refit both initializers using **all fit rows**
for the actual training protocol, declared here before either execution;
the query-excluded CPU F0 remains a different screening instrument. No held
read. Change only the basis fitting input: unit source image rows versus
their equally weighted class means. Preserve Linear1024→128 orthonormal
weights, PCA centering, sign convention, normal classifier initialization
from sums of individually unit-normalized projections, same trainable vision/head/proxies,
same bank construction and all other losses. No whitening/shuffling auxiliary.

Image-PCA centers by the mean of all unit fit-image vectors; product-PCA
centers by the equally weighted mean of product centroids. Both basis and
centering therefore change in this fixed initializer. A positive result
would support the initializer as a whole, not attribute its effect solely
to subspace selection or prove a particular source of nuisance variation.

Prefer extending the existing initializer with one strict explicit basis
choice, defaults unchanged; do not duplicate its head/proxy construction.
The actual mean/components, initializer cost and all source/input hashes
must be recorded in the checkpoint and receipt. Any RNG consumed by new
initializer construction must preserve native loader/augmentation state.
Use the existing mechanics/export helpers, retaining independent stored
training evidence before optional public reload work. No copied trainer.

Require native loss trajectory/17-pixel replay against the original pinned
native17 receipt; identical initial vision and same source/split/schedule;
all17 input hashes equal across arms; all losses/gradients/parameters finite,
no skipped step; frozen embeddings/lower blocks exactly pretrained; initial
head/classifier/bank hashes differ as designed. Record per-group preclip
gradients/clip factor and initializer cost. Candidate median update wall
must be at most1.05× native and peak CUDA at most1.005×; whole pair<=120s.
Record initial/terminal compact variance and effective rank; require variance
>=50% of its own initial and rank>=80% of initial. Public reload on the same
32 fit images must produce exact packed codes/norms compared with in-process
serving at matching dtype and batch. No quality, latency or novelty claim.

Any failed gate closes this fixed initializer before100 updates, seeds or
official evaluation. A pass licenses a separately frozen paired100-update
TRAIN-held gate scored only with deployed128 packed geometry: require R@1
point delta>=0 and mAP@R gain>=+0.5pp with product95% lower>0, together with
the same cost and authority guards. No width/covariance/scale/basis shopping
or initializer refit after a negative. Independent seed confirmation and
full public serving qualification remain necessary before promotion.

## Review reconciliation and exact implementation authority

Opus/Astra group `d8a630a95bce4694` completed: conditional GO after fixing
the contract, not approval of unseen quality. Full separately labelled
answers are archived as `inshop-centroid-pca-design-review-v1.json`.
The normalized-image proxy oracle was corrected; no proxy weighting change
is introduced. Initializer construction uses exactly one Linear per arm,
with equal global RNG state checked on synthetic unequal class counts.
The product branch has a domain-separated digest; native digest bytes remain
unchanged for exact replay. Source defaults retain image PCA.

Pin native17 reference:
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-main-smoke-v1/control/receipt.json`,
SHA `b6c684bcdfb90b3644c5ef66a3de1c1a5d736f46003474775e57c7af2a1c51bf`.
All17 input hashes, PCA/head/bank/vision initialization hashes match exactly;
all17 losses use absolute tolerance1e-5, relative0. Pixel mismatch is KILL.
Matched control/treatment source manifests must match; old source manifest
necessarily differs and is not treated as paired-source evidence.

Initial PCA components/mean, class counts (including singleton count),
feature/fit authority, basis name, initializer cost, center distance and
projected-center distance versus median raw projection norm are stored in
the raw checkpoint. Every expected fit class must have a count>0; proxy and
bank rows must be finite/nonzero. PCA rank/gap failure is KILL, not relaxation.
Update cost window is steps2–17, with initializer cost reported separately.
Stored preclip group norms and global norms provide the clipping factor
min(1,1/(norm+1e-6)); no gradient-size floor is selected from the output.

Public exactness uses **64 fit images in two matched32 batches**, captured
from the live terminal FP32 vision/head with FP16 encoder autocast and
serving-compatible processor before public reload. Compare exact codes and
f16 inverse norms, alongside exact saved/live tensors and existing parent
state checks. The common parity helper now accepts this independent live
reference; it does not use reloaded vision as the sole reference for this gate.

## Native authority repair before treatment execution

Original unit `sfora-inshop-centroid-pca-smoke-v1`, invocation
`b3173d16f99648d3b9d25abec6b5c775`, exited0 after **50.641871 s** with the
native arm only. It stopped because the new head hash concatenated raw
weight/bias bytes while the reference used the existing named-parameter
digest. All17 native pixels/losses, finite state, geometry and exact live
packed/public parity passed. No treatment optimizer update or quality read
occurred. This is a verifier defect, not a candidate quality rejection.

Independent reconstruction from the saved initial PCA tensors reproduces
the reference canonical head hash
`398ea349f25a5c9deeaa21adc5679356022277417a2f6f4e49121d1949817695`
exactly. PCA, bank and initial vision hashes also match. Correct the new
record to call the existing `parameter_digest` helper. Preserve the original
receipt without rewriting its KILL label. A new source-pinned, uniquely named
v2 pair is allowed once; all original acceptance thresholds, inputs,
architecture and initialization remain fixed. No gate relaxation.

## Terminal v2: GO to separately frozen100 quality gate

Sole repaired unit `sfora-inshop-centroid-pca-smoke-v2`, invocation
`ebc87cfb982e4b7fbe6d6be94ec451b7`, exited0; whole pair **92.782331 s**.
Receipt SHA-256:
`9ae62ed93a7546e4d71b11b12a145bdca23908ec781e60731340d44fd9744e1f`.
All16 criteria pass and independent terminal replay verifies the pixel,
geometry, live-public parity, finite-state and cost rules. Both arms have
17 stable actual encoder updates; no held/official query or quality read.

| Official TRAIN fit17, DGX Spark GB10 | Native image-PCA | Product-mean PCA |
| --- | ---: | ---: |
| Training wall excluding bank/initializer | 15.862303 s | 15.974447 s |
| Training wall including bank initialization | 17.238077 s | 17.265319 s |
| Sampled images/s including bank initialization | 63.1161 | 63.0165 |
| Peak allocated CUDA | 12,246,201,856 bytes | 12,246,201,856 bytes |
| Median update wall, steps2–17 | 0.833352 s | 0.835514 s |
| Head/proxy initializer wall, reported separately | 1.238468 s | 0.487583 s |
| Initial→terminal compact variance | 0.851326→0.902825 | 0.863861→0.913763 |
| Initial→terminal effective rank | 24.230318→23.549534 | 23.432932→23.315868 |
| Live terminal→public reload codes/norms, 64fit/two32 batches | Exact | Exact |

The 17 input batches, source/model/fit/schedule hashes and initial vision
match exactly. The native PCA/head/bank authorities and all17 losses replay.
Head/proxy/bank initialization changes only as declared. Cost observations
are a short mechanical screen, not hardware latency or convergence evidence;
neither arm has image-to-top-k p50/p95/p99/QPS or unseen quality from this run.

Raw receipts, logs and terminal replay are in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-centroid-pca-smoke-v2/`;
full initializers/checkpoints remain at the corresponding remote result root.
Next: a fresh paired100 source-pinned run on fixed6,354 query/6,245 gallery
TRAIN roles, native128 packed scorer, R@1 nonnegative point and mAP@R>=+0.5pp
with conditional lower>0, plus cost guards. Declare its exact protocol before
execution. No seeds, official read or production default until it qualifies.
