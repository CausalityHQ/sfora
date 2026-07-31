# Candidate 4: graded within-class relation supervision

Status: **failed gate 2 on prior art; no implementation or GPU run**.

## Gate 1 — provenance: PASS

Two failures in this repository point to the same defect in ordinary class-label
supervision:

1. Sub-center Proxy Anchor reached about **0.675 R@1** on CUB, roughly **1.7
   pt below** the corrected Proxy Anchor reference. Treating within-class
   variation as several discrete modes fragmented the class and hurt unseen-class
   transfer.
2. Tversky similarity reached **0.6758 R@1** on CUB, roughly **1.6 pt below**
   Proxy Anchor, and was **−4.63 pt** on the low-noise In-Shop screen. Its
   defining intervention was to give distant true positives more overlap-based
   attraction. The large negative indicates that many same-label images are
   distant for useful reasons; forcing every class-positive relation toward the
   same geometry destroys information.

The common supervision problem is that a binary class label says only
“same class,” while the retrieval embedding must preserve graded visual
relations within that class. Discrete sub-centers over-separate those relations;
uniform positive attraction over-collapses them.

Candidate 4 would add **pair-specific ordinal supervision** inside each training
class. A frozen, non-label feature source would identify which same-class pairs
share more local attributes, yielding constraints such as “within class c,
image a is more similar to b than to d.” Proxy Anchor would retain its ordinary
identity supervision, while the ordinal constraints preserve partial similarity
instead of declaring new identities or merely changing the similarity score.
This changes what supervision exists.

The proposal is measurement-derived: it predicts that preserving graded
within-class structure can avoid both measured failure modes. It is not yet a
method claim. No effect size, recipe, or GPU work is allowed unless a primary
literature audit finds that the supervision mechanism itself is unoccupied.

## Gate 2 — prior art: FAIL

The audit searched continuous and ordinal DML labels, multiple notions of
similarity, automatically generated graded pair labels, within-class variance
preservation, and latent fine-grained supervision.

The mechanism is occupied:

- *Deep Metric Learning Beyond Binary Supervision* starts from the same
  criticism of binary class equivalence, uses continuous/structured labels, and
  preserves both the order and ratios of label distances in embedding space
  ([Kim et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Metric_Learning_Beyond_Binary_Supervision_CVPR_2019_paper.pdf)).
- *HIER: Metric Learning Beyond Class Labels via Hierarchical Regularization*
  discovers a latent semantic hierarchy without hierarchy annotations and uses
  it as richer, fine-grained supervision on standard DML benchmarks
  ([Kim et al., 2022](https://arxiv.org/abs/2212.14258)).
- *Deep Metric Learning by Online Soft Mining and Class-Aware Attention*
  explicitly preserves useful intra-class variance by emphasizing more-similar
  positives instead of treating every positive relation uniformly
  ([Wang et al., 2018](https://arxiv.org/abs/1811.01459)).
- *Data-Efficient Large Scale Place Recognition With Graded Similarity
  Supervision* automatically re-annotates binary pair relations with continuous
  similarity labels and trains retrieval descriptors from them
  ([Leyva-Vallina et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Leyva-Vallina_Data-Efficient_Large_Scale_Place_Recognition_With_Graded_Similarity_Supervision_CVPR_2023_paper.pdf)).

Using a frozen visual or language model to manufacture the ordinal labels does
not create a new supervision mechanism; it substitutes a label source in an
established continuous/structured-label framework. Language-derived DML
supervision is itself already occupied by *Integrating Language Guidance into
Vision-Based Deep Metric Learning*.

Candidate 4 therefore stops at gate 2. It receives no preregistration,
implementation, or GPU screen.
