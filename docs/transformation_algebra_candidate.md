# Candidate 152: cross-instance transformation algebra

Date: 2026-08-01. Status: **DEAD AT GATE 2**. No diagnostic,
implementation, or GPU run.

## Gate 1 provenance

After exact class-size matching, disconnected same-class In-Shop 1-NN graphs
retrieve 3.534 R@1 points better, and the local-feature audit found a 6.67-point
MaxSim recovery. These measurements suggest that useful identity evidence may
live in transformations between appearances rather than in class contraction.

The candidate would fit low-rank local-feature operators `T_ij` between
same-class images, enforce `T_ij T_jk ≈ T_ik` across triples, share a dictionary
of transformation atoms across training classes, and train the final embedding
difference `z_j - z_i` to predict the atom. The operator module would be
training-only.

## Gate 2 prior art

The mechanism is occupied by transformation-equivariant/group-structured
representation learning:

- Sadeghi, Zitnick, and Farhadi, *VISALOGY: Answering Visual Analogy Questions*
  (NeurIPS 2015), learns an embedding in which image pairs exhibiting analogous
  transformations are close using a quadruple Siamese architecture:
  https://arxiv.org/abs/1510.08973.
- Winter et al., *Structuring Representations Using Group Invariants* (NeurIPS
  2022), learns embeddings whose unknown input transformations have simple
  latent group actions, explicitly grounding the method in composable group
  structure and invariant polynomials:
  https://papers.neurips.cc/paper_files/paper/2022/file/dcd297696d0bb304ba426b3c5a679c37-Paper-Conference.pdf.
- Hadji, Derpanis, and Jepson, *Representation Learning via Global Temporal
  Alignment and Cycle-Consistency* (CVPR 2021), uses cycle-consistent latent
  correspondence as representation supervision:
  https://openaccess.thecvf.com/content/CVPR2021/html/Hadji_Representation_Learning_via_Global_Temporal_Alignment_and_Cycle-Consistency_CVPR_2021_paper.html.

Learning a dictionary rather than fixing a group, using still-image class pairs
rather than temporal pairs, or discarding the auxiliary module at test changes
the transformation estimator and application. The supervisory object remains a
composable latent action/cycle-consistent cross-instance transformation.

Candidate 152 is **DEAD AT GATE 2**.
