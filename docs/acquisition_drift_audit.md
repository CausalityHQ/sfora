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
