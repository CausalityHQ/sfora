# Candidate 49: compositional decoupled relational distillation (CDRD)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

Across five final CUB HERD packs, only 7.81--11.10% of the implemented τ=0.1
neighbourhood-distribution mass falls on same-class samples even though the
nearest neighbour is same-class 89.23--95.07% of the time. CDRD, inspired by
subcompositional coherence in Aitchison geometry, would separately match:

1. total mass assigned to same- versus different-class relations; and
2. the conditional ranking within each subcomposition.

This prevents batch cardinality from coupling relation-type mass to the detailed
dark ordering.

## Gate 2: prior art

The mechanism is occupied. Decoupled Knowledge Distillation factorizes ordinary
KD into target-class versus non-target mass and the conditional distribution
among non-target classes—the same probabilistic decomposition:

- Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html>

Relational KD establishes transfer of inter-sample geometry in metric learning,
and multi-stage decoupled relational KD explicitly combines relational transfer
with decoupling:

- Park et al., *Relational Knowledge Distillation*, CVPR 2019:
  <https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html>
- Wang et al., *Multi-stage Decoupled Relational Knowledge Distillation with
  Adaptive Stage Selection*, ICLR 2024 submission:
  <https://openreview.net/forum?id=4QtywskEyY>

Changing the partition from output target/non-target classes to inter-sample
same/different labels adapts the established decoupled-relational operator to
DML. The compositional-data interpretation does not make the supervision new.
Candidate 49 is **DEAD at Gate 2**.

