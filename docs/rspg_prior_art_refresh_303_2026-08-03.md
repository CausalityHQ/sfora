# RSPG adversarial prior-art refresh before corrected-data work

Date: 2026-08-03. Completed before the corrected operating-point pack and before
any corrected RSPG candidate training.

## Claim under attack

RSPG uses each image's distribution over **other class identities**, excluding
its own class, as a contextual signature. Agreement between two different
same-class images is a discrete eligibility test: a rejected labelled positive
becomes unknown. The novelty claim is this complete operator, not contextual
similarity, class signatures, positive mining, or graph supervision separately.

## Primary-source neighbours

- Suh et al., *Stochastic Class-Based Hard Example Mining for Deep Metric
  Learning* (CVPR 2019), maintain online class signatures and use them to choose
  hard **negative classes** efficiently. They neither compare two same-class
  samples' distributions over rival identities nor remove their positive
  relation. [Primary source](https://openaccess.thecvf.com/content_CVPR_2019/html/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.html)
- Wang et al., *Deep Metric Learning by Online Soft Mining and Class-Aware
  Attention* (AAAI 2019), continuously reweights positives by direct embedding
  similarity. It keeps them positive and has no target-excluded class-level
  contextual descriptor. [Primary source](https://arxiv.org/abs/1811.01459)
- Xuan et al., *Improved Embeddings with Easy Positive Triplet Mining* (WACV
  2020), chooses the closest positive and permits multiple modes. That is direct
  proximity mining, not rival-signature agreement.
  [Primary source](https://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf)
- Wu et al., *Contextual Similarity Distillation for Asymmetric Image Retrieval*
  (CVPR 2022), use contextual descriptors as a training signal but compare two
  models on the same input; they do not adjudicate a relation between different
  same-class samples. [Primary source](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html)
- Liao et al., *Supervised Metric Learning to Rank for Retrieval via Contextual
  Similarity Optimization* (arXiv:2210.01908), optimize the gap between binary
  labels and instance-neighbourhood contextual similarity. Every labelled
  same-class pair retains target label one; context changes the loss value, not
  whether that positive exists. [Primary source](https://arxiv.org/abs/2210.01908)
- Zhuang et al., *Deep Semi-Supervised Metric Learning With Mixed Label
  Propagation* (CVPR 2023), add/remove affinity-graph edges to propagate pseudo
  labels for unlabeled data. It does not revise known same-class supervision from
  target-excluded rival distributions. [Primary source](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_Deep_Semi-Supervised_Metric_Learning_With_Mixed_Label_Propagation_CVPR_2023_paper.pdf)
- Chen et al., *Confusion-Based Metric Learning for Regularizing Zero-Shot Image
  Retrieval and Clustering* (TNNLS 2024), add adversarial energy- and
  diversity-confusion regularizers. It changes representation regularization,
  not pair eligibility. [Primary record](https://doi.org/10.1109/TNNLS.2022.3185668)

Targeted searches through current CVF, NeurIPS, arXiv, and OpenReview results for
class-confusion distributions, contextual positive mining, same-class graph
gates, and positive-to-unknown supervision found no later paper implementing the
complete operator. Search failure is not proof of novelty, so the claim remains
**LIVE NARROWLY**, with the earlier operational warning intact: on the invalid
high-resolution corpus, 70.5% of accepted edges overlapped an equal-cardinality
distance gate. A corrected gain must strictly beat distance, soft-context, and
instance-neighbourhood controls before the signature can receive causal credit.
