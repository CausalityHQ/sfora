# SOP official-split correction 234

Date: 2026-08-03. Written before any corrected-split training run.

## Defect

The SOP loader used the `JamieSJS/stanford-online-products` corpus, extracted
eBay product IDs from its `id` field, sorted them lexicographically, and assigned
the first and second halves to training and test. That is not Stanford Online
Products' official split, which is defined by `class_id` in `Ebay_train.txt` and
`Ebay_test.txt`.

Against the official metadata, the old 11,317-class training set contains 11,093
official training products and **224 official test products**. Its test set
contains 11,092 official test products and **225 official training products**.
The old image counts were 59,355/60,698 instead of the official 59,551/60,502.
Consequently all historical SOP training claims in this repository are
noncanonical and cannot supply protocol-grade evidence.

The correction resolves product membership using the two metadata files from
the digest-pinned `pawlo2013/StanfordOnlineProducts` revision
`11cf4cdcd89ac209f05277fcabebf10f47d3467d`, while retaining the complete image
corpus. The pinned SHA-256 digests are:

- train: `77abb1e82af49f2f1f272dc6bd0b8480904f58742091764f9511b152cad1824e`;
- test: `63d559dc1ca1e03671224c58a26b18985bbff22ae20344d41dabcfe3873cc276`.

## Preregistered correction check

Re-run the previous strongest plain-PA SOP recipe exactly: ResNet-50/512,
Proxy Anchor, seed 0, batch 256, two samples per class, learning rate `2e-4`, one
proxy per class, 60 epochs, evaluation every 10 epochs. The invalid-split result
was best-over-training R@1 **0.72147** at epoch 50.

Prediction: because 98.0% of each class set is unchanged, the corrected result
will differ by no more than **0.5 point**. A change of at least **2.0 points**
would falsify that prediction and identify split leakage/membership as a major
cause of the reproduction gap. An intermediate change is recorded without a
causal claim. This run repairs the benchmark and may provide new training-split
measurements; it is not itself a novel-method experiment.

The corrected run must save the model and training embeddings so subsequent
hypotheses use official training identities only. No method screen is authorized
until those measurements produce an operator that survives prior-art review.
