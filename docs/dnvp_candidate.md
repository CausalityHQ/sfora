# Candidate 47: determinantal niche-volume preservation (DNVP)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

Exact epoch-10 Proxy Anchor gradients show that the positive term increases
same-acquisition similarity at `7.77e-5` per unit step versus `4.16e-5` across
acquisitions. This is the measured cause of the training-induced acquisition
collapse. Inspired by ecological niche-volume measures and determinantal point
processes, DNVP would maximize a regularized log-determinant of centered
same-class embeddings while Proxy Anchor preserves identity. Unlike pair
reweighting, the determinant couples the whole class set and penalizes loss of
independent within-class directions.

## Gate 2: prior art

The aggregate is different, but the supervision operation is occupied:

- Du et al., *An Adaptive Deep Metric Learning Loss Function for Class-Imbalance
  Learning via Intraclass Diversity and Interclass Distillation*, TNNLS 2024,
  explicitly generates diverse within-class features in a DML loss:
  <https://pubmed.ncbi.nlm.nih.gov/37379193/>
- Wang et al., *Ranked List Loss for Deep Metric Learning*, CVPR 2019, learns a
  class hypersphere specifically to preserve intra-class structure rather than
  compress all positives:
  <https://arxiv.org/abs/1903.03238>
- Duboudin et al., *Encouraging Intra-Class Diversity Through a Reverse
  Contrastive Loss*, ICCV workshop 2021, directly reverses same-class attraction
  to preserve intra-class diversity:
  <https://openaccess.thecvf.com/content/ICCV2021W/AROW/html/Duboudin_Encouraging_Intra-Class_Diversity_Through_a_Reverse_Contrastive_Loss_for_Single-Source_ICCVW_2021_paper.html>

A DPP log-determinant is a set-level diversity aggregate, but it still supplies
the established instruction to spread same-class embeddings while retaining a
discriminative loss. Candidate 47 is **DEAD at Gate 2**.

