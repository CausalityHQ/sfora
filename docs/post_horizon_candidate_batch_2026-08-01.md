# Post-horizon candidate batch: candidates 140--144

Date: 2026-08-01. Generated with Claude from the repository's saved In-Shop
embedding and local-feature packs after candidates 136--139 and the
VAPNet/AdvRF horizon correction. No diagnostic, implementation, or GPU run was
started.

The generation prompt required each proposed diagnostic to imply a new
supervision relation and excluded pair mining, multi-centre/proxy methods,
uncertainty weighting, covariance augmentation, distillation, generic
regularisation, and similarity substitution. Claude proposed five diagnostics;
an immediate adversarial pass returned **NONE defensible** under those rules.

## Shortlist and gate decisions

1. **Density-gradient alignment.** Correlate the direction of a within-class
   density gradient with retrieval success, then pull samples toward the dense
   region. This is not a new supervision object. Ghosh, Singh, and Vatsa, *On
   Learning Density Aware Embeddings* (CVPR 2019), already iteratively moves
   class centres toward dense regions and attracts embeddings there:
   https://openaccess.thecvf.com/content_CVPR_2019/html/Ghosh_On_Learning_Density_Aware_Embeddings_CVPR_2019_paper.html.
2. **Class-size-conditioned margins.** Correlate rival-class cardinality with
   retrieval outcome, then scale margins by cardinality. This is scalar
   reweighting/margin adaptation, not new supervision. Class-frequency
   reweighting is explicit in Cui et al., *Class-Balanced Loss Based on
   Effective Number of Samples* (CVPR 2019), and data-dependent margins/mining
   are explicit in hierarchical and adaptive metric losses:
   https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html.
3. **Local rank stability under perturbation.** Measure top-k instability after
   small embedding perturbations, then penalise rank changes. Zhou et al.,
   *Adversarial Attack and Defense in Deep Ranking* (ECCV 2020), already
   formulates ranking perturbations and trains a retrieval defense on CUB,
   Cars196, and SOP: https://arxiv.org/abs/2106.03614.
4. **Persistent-homology defects.** Measure within-class first Betti numbers,
   then remove loops. Persistent-homology penalties that simplify learned
   decision/representation topology are established by Chen et al., *A
   Topological Regularizer for Classifiers via Persistent Homology* (AISTATS
   2019), and later representation-learning work:
   https://proceedings.mlr.press/v89/chen19g.html.
5. **Local curvature.** Estimate Hessian curvature of the neighbourhood distance
   field, then flatten high-curvature regions. This is manifold regularisation,
   directly represented by Pei et al., *Curvature Regularization to Prevent
   Distortion in Graph Embedding* (NeurIPS 2020):
   https://proceedings.neurips.cc/paper/2020/hash/eeb29740e8e9bcf14dc26c2fff8cca81-Abstract.html.

Strictly, all five fail Gate 1 because the proposed diagnostic is not yet a
repository measurement and an unobserved result cannot be provenance. The
primary-source Gate-2 check was still performed because it is cheaper than
building the diagnostics and establishes that even a favourable number would
only motivate an occupied regularizer or weighting rule. Running the CPU
diagnostics would therefore be HARKing with no live downstream operator.

## Verdict

There is no shortlist survivor. Candidates 140--144 are recorded as cheap
pre-GPU deaths. This batch strengthens rather than weakens the stopping case:
the remaining easily measurable geometric properties map to density,
cardinality, robustness, topology, and curvature regularisation, all already
covered operator families. The GPU remains idle.
