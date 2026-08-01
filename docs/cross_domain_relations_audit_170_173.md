# Cross-domain relation audit: candidates 170--173

Date: 2026-08-01. This batch examined higher-order networks, open-set speaker
verification, molecular conformational ensembles, and counterfactual image
composition. No implementation or GPU run followed.

## 170. Simplicial identity supervision

Higher-order contrastive learning on simplicial complexes is established (for
example HONC/Topo-MLP). More importantly, these image datasets do not observe
irreducible higher-order incidence: a class is supplied only as an equivalence
relation. Building simplices from embedding neighbourhoods makes the additional
label circular and reduces to contextual/hypergraph contrastive learning.
**DEAD AT GATE 1/2.**

## 171. Speaker-style variability subspaces

Open-set speaker verification already models within-speaker nuisance variation
and between-speaker identity with metric losses, PLDA, and variability subspaces.
Transferring that mechanism to images is class-conditional subspace or nuisance
invariance learning, already closed in candidates 150, 152, 159, and 162.
**DEAD AT GATE 2.**

## 172. Conformational-ensemble identity

Protein ensemble comparison assumes multiple experimentally or physically
generated conformations of the same molecule. CUB/Cars/In-Shop supply no
correspondence asserting that two images are conformations along the same latent
coordinate. Estimating ensembles from class embeddings is density/multi-center
learning; generating them violates the no-external-generator constraint.
**DEAD AT GATE 1/2.**

## 173. Counterfactual local-evidence transplantation

Transplanting or erasing regions and assigning donor/recipient mixtures is
multi-image augmentation with mixed supervision. Metrix explicitly generalizes
mixup over inputs, intermediate representations, embeddings, and metric-learning
targets. Embedding Expansion synthesizes feature points, while Intra-Class Part
Swapping covers fine-grained images. Without independently observed causal part
labels, calling the mixture an intervention does not change its semantics.
**DEAD AT GATE 2.**

Primary sources:

- Topo-MLP / HONC: <https://arxiv.org/abs/2312.11862>
- Open-set speaker metric learning: <https://arxiv.org/abs/2003.11982>
- Metrix: <https://arxiv.org/abs/2106.04990>
- Embedding Expansion: <https://arxiv.org/abs/2003.02546>
- Intra-Class Part Swapping: <https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Intra-Class_Part_Swapping_for_Fine-Grained_Image_Classification_WACV_2021_paper.pdf>
