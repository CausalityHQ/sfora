# Candidate 273: compression-distance positive eligibility

**Verdict: DEAD at Gate 2. No diagnostic, implementation, or GPU.**

## Origin and measured motivation

A bounded Claude Sonnet critique proposed borrowing alignment-free phylogenetic
similarity from algorithmic information theory. RSPG's target-excluded contextual gate
retained 0.6449 of CUB same-class pairs but 0.0863 on In-Shop and then lost 5.72 R@1
points; ARCG's model-derived relation was selective (0.3631) but also failed as
eligibility. Those measurements motivate asking whether a fixed, model-free relation
could supply within-class structure without self-erasure or rival identities.

For canonical pixel byte strings and a fixed lossless compressor `C`, the proposed
operator was:

```text
NCD(i,j) = [C(x_i || x_j) - min(C(x_i), C(x_j))]
           / max(C(x_i), C(x_j))
tau_y = within-class p-th percentile of NCD for class y
positive(i,j) = 1[label_i = label_j and NCD(i,j) <= tau_y]
```

Pairs above the threshold would become unknown. This is genuinely distinct from a
learned embedding distance at execution time.

## Gate 2: the source of supervision is occupied

Compression distance is already an image-retrieval similarity, not an untried transfer
from genomics:

- Cerra and Datcu,
  [A fast compression-based similarity measure with applications to content-based
  image retrieval](https://arxiv.org/abs/1210.0758), define Fast Compression Distance
  and an explicit color-image retrieval system.
- Nikvand et al.,
  [Perceptually Inspired Normalized Conditional Compression
  Distance](https://arxiv.org/abs/1810.00059), develop compression similarity for image
  processing, texture classification, and face recognition.
- Guha et al., *Image similarity using sparse representation and compression distance*
  (2018), apply conditional compressibility to image clustering, retrieval, and
  classification.
- Cilibrasi and Vitanyi's normalized compression distance and later generalized
  compression dictionary distances already establish its use as a generic supervised
  and unsupervised similarity/kernel signal.

Using a known image-similarity metric to threshold labeled same-class edges does not
create a defensibly novel supervision source. The positive-to-unknown policy is the
same narrow distinction rejected for other established pair descriptors.

## Mechanistic risk (secondary)

Raw-pixel compression primarily captures repeated low-level statistics, file encoding,
background, and texture. Canonical decoding removes file-container artifacts but not
the studio-background versus natural-background dataset dependence already exposed by
RSPG/ARCG. A perceptual transform would move directly toward Nikvand et al.; a learned
compressor would add cost and make the signal another representation distance.

Candidate 273 therefore stops at prior art. No CPU diagnostic is warranted.
