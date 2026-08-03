# Official Proxy Anchor SOP benchmark repair 239

Date: 2026-08-03. Written and committed before the deciding run.

## Why this run exists

The corrected official SOP partition has not yet been evaluated with the
registered Proxy Anchor SOP recipe. The first attempted correction run used a
ResNet-50 project recipe and was aborted at about 9% when the mismatch was
found. It produced no report. This run repairs the benchmark only; it is not a
method test and cannot support a novelty claim.

## Locked recipe

Use exact recipe ID `proxy_anchor.sop.official-51db570`, sourced from official
Proxy Anchor repository revision
`51db57031e38f75c03f69bbdfad1a3233afd9787` (README SOP command plus
`code/train.py` defaults). The material settings are BN-Inception with pinned
`bn_inception_52deb4733` weights, 512 dimensions, avg-max pooling, Kaiming head,
Proxy Anchor alpha 32 and delta 0.1, one proxy per class, AdamW, weight decay
`1e-4`, batch 180 with shuffled sampling, learning rates `6e-4` for backbone and
embedding head and a 100x proxy multiplier, one warm-up epoch, 60 training
epochs, step decay by 0.25 every 20 epochs, trainable BatchNorm, gradient clip
10, and test evaluation every epoch. No recipe field may be overridden except
seed 0, worker count, output paths, and final-checkpoint persistence.

An execution audit after this document's first commit found that the harness
kept the final incomplete training batch while upstream sets `drop_last=True`.
That first launch was stopped after about three minutes and produced no report.
The recipe now explicitly sets `drop_last_train_batch=True`: 59,551 images at
batch 180 give 330 updates per epoch and exactly **19,800** updates over 60
epochs. A trace showing 19,860 updates falsifies the repaired invocation.

The repaired-batch launch was itself stopped at step 200 when a warm-up audit
found a second harness mismatch: the generic code recognized only ResNet's
`fc.*` head, while BN-Inception names its new head `model.embedding.*`. It
therefore froze the embedding head during warm-up, contrary to upstream. The
head selector is now covered by a regression test. This launch also produced no
report. The admissible run must train `model.embedding.*` and metric proxies
during epoch 1 while freezing only the pretrained backbone.

The loader must independently verify the digest-pinned official membership:
11,318 training classes / 59,551 images and 11,316 test classes / 60,502 images,
with disjoint product identities.

## Prediction and falsification

The official Proxy Anchor repository reports about **0.792 R@1** on SOP for its
improved official settings. Predict seed-0 best-over-training R@1 in
**[0.777, 0.807]**. A value outside that interval falsifies faithful local
reproduction and blocks method comparisons until the discrepancy is diagnosed;
it is not evidence for or against any proposed method.

Report two distinct quantities:

1. the report's code-derived best-over-training test R@1, explicitly labeled as
   test-selected and not independently reconstructable without a best checkpoint;
2. final-training-state test R@1, independently recomputed from a saved final
   checkpoint and official-test embeddings.

Export final-state official train and test embeddings only after training.
Their metadata, counts, split hashes, normalization, labels, and recomputed
leave-one-out retrieval must pass Gate 0 before they can motivate a candidate.
One seed repairs provenance but does not establish a method effect.
