# Candidate 41: acquisition intraclass-correlation suppression (AICS)

**Gate-2 death recorded 2026-07-31; no implementation or GPU run.**

## Gate 1: PASS

In-Shop acquisition groups are nested within garment identities and explain a
large geometric component: mean within-group cosine is `0.8199`, versus `0.6396`
across groups, and cross-group-only training R@1 is `0.5542`. Group sizes are
balanced, so this is a variance-component defect rather than pseudoreplication.

Inspired by random-effects ANOVA, AICS would minimize the intraclass correlation
ratio

`between-acquisition-group scatter / total within-identity scatter`

while retaining ordinary Proxy Anchor. Unlike a global camera discriminator, it
would allow the group labels to be nested and non-comparable across identities.

## Gate 2: FAIL

The normalization is new notation, but the supervised relation is established:

- [Learning View-Specific Deep Networks (Feng et al.,
  2018)](https://arxiv.org/abs/1803.11333) introduces a cross-view Euclidean
  constraint and cross-view center loss to reduce same-identity center differences
  across views.
- [Hetero-Center Loss (Zhu et al.,
  2020)](https://arxiv.org/abs/1910.09830) explicitly pulls modality-specific
  centers of the same identity together.
- [Camera-Aware Style Separation and Contrastive Learning (2021)](https://arxiv.org/abs/2112.10089)
  combines camera-style separation with a camera-aware contrastive center loss.

All three use the same mechanism: partition each identity by a nuisance condition
and reduce distances among the partition-specific centers. Replacing a sum of
center distances by an ANOVA/ICC ratio changes scale normalization, not what
supervision exists. Nested rather than globally shared group names do not alter the
center operator.

**Verdict: DEAD at Gate 2.** The acquisition variance remains a strong measured
benchmark defect, but AICS is cross-view center alignment in statistical language.
