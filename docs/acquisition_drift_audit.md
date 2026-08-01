# In-Shop acquisition-gap drift audit

**CPU-only measurement recorded 2026-07-31.** Identical within-identity pairs
were compared in the existing one-step and exact epoch-10 In-Shop exports.

| checkpoint | same-group cosine | cross-group cosine | acquisition gap |
|---|---:|---:|---:|
| after one optimization step | 0.6947 | 0.6696 | **0.0251** |
| epoch 10 | 0.8199 | 0.6396 | **0.1804** |
| change | +0.1252 | -0.0300 | **+0.1553** |

The acquisition gap grows **7.18×** in the first ten epochs. Overall same-class
cosine rises only from `0.6764` to `0.6882`: the small aggregate improvement hides
a large redistribution in which same-session pairs contract and cross-session
pairs separate.

The one-step export is essentially the pretrained BN-Inception backbone plus a
barely trained random embedding head, so it is not a pristine ImageNet-feature
measurement. It is nevertheless the correct early-training comparator available
in this repository. The result shows that most of the epoch-10 shortcut is created
during Proxy Anchor training rather than already present at the first update.

This motivates monitoring acquisition-gap *change*, not merely its final size.
`docs/ncdd_candidate.md` records why the obvious temporal constraint still fails
novelty.

## Exact Proxy Anchor gradient decomposition

At epoch 10, the full-dataset Proxy Anchor embedding gradient was separated into
its own-proxy positive and foreign-proxy negative terms. For every same-class pair,
the first-order gradient-descent change in cosine was computed after tangent-plane
projection.

| pair type | positive-term similarity rate | negative-term similarity rate | total |
|---|---:|---:|---:|
| same acquisition group | `+7.77e-5` | `+1.72e-4` | `+2.49e-4` |
| cross acquisition group | `+4.16e-5` | `+1.94e-4` | `+2.35e-4` |

The positive term predicts acquisition-gap growth of `+3.61e-5` per unit step.
The negative term predicts `-2.20e-5` and therefore partially *corrects* the gap;
net predicted gap growth is `+1.41e-5`. Both terms increase almost every
same-class pair's similarity, so the issue is not repulsion between sessions.
Single-proxy attraction contracts the already-local session clusters faster than
it joins them.

This is a local, full-dataset gradient calculation rather than a measured SGD
trajectory decomposition, so its scale should not be extrapolated over epochs.
Its directional attribution is nevertheless exact for the audited objective and
checkpoint.
