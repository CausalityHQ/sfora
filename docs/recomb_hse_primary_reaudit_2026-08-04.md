# RECOMB primary-source re-audit: HSE closes the mixed-hybrid lane

Date: 2026-08-04.

## Why this correction is necessary

The original candidate-370 audit named Metrix, CutMix, Proxy Synthesis, and
Memory-Based Virtual Classes, but missed the closest benchmark-matched paper:
Yang et al., **HSE: Hybrid Species Embedding for Deep Metric Learning** (ICCV
2023). HSE is not a generic classification-mixup citation. It constructs
pixel-space hybrids with CutMix, MixUp, and a modified GridMask, adds an
auxiliary metric-learning loss for those hybrids, and evaluates on CUB-200,
Cars196, and SOP.

Primary sources:

- Yang et al., *HSE: Hybrid Species Embedding for Deep Metric Learning*, ICCV
  2023: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_HSE_Hybrid_Species_Embedding_for_Deep_Metric_Learning_ICCV_2023_paper.html
- Gu, Ko, and Kim, *Proxy Synthesis: Learning with Synthetic Classes for Deep
  Metric Learning*, AAAI 2021: https://arxiv.org/abs/2103.15454
- Ren et al., *A Simple Data Mixing Prior for Improving Self-Supervised
  Learning*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_A_Simple_Data_Mixing_Prior_for_Improving_Self-Supervised_Learning_CVPR_2022_paper.pdf
- Venkataramanan et al., *It Takes Two to Tango: Mixup for Deep Metric
  Learning*, ICLR 2022: https://openreview.net/forum?id=ZKy2X3dgPA

## Mechanism comparison

HSE does not assign a hybrid a persistent hard recipe identity. Its hybrid
contains content from parent classes A and B; HSE treats those parents as weak
positives, chooses a nearby parent example and a nearby unrelated-class hard
negative, and optimizes an auxiliary NCA-like hybrid loss. It therefore
occupies the important proposition that spatially mixed cross-class images can
provide additional benchmark-matched DML supervision.

Proxy Synthesis independently constructs an interpolated embedding and proxy
from one pair of original embedding/proxy pairs and computes the proxy loss as
if that pair were a synthetic class. The paper explicitly describes the
synthetic proxy as the representative and the synthetic embedding as a data
point of the synthetic class. Metrix occupies mixed input/feature/embedding
targets in DML. SDMP makes mixtures from the same source instances positive in
self-supervised learning.

The only literal residue in RECOMB is:

> Two composites made from four disjoint source images are positive solely
> because they share an ordered pair of parent class labels and a spatial mask,
> while their real parents and reversed/shared-part recipes are negatives.

That sentence is narrower than HSE, SDMP, or one-shot Proxy Synthesis. It is
not enough to reverse the Gate-2 verdict. The virtual identity is still fully
specified by an occupied cross-class mixing recipe; repeating the recipe with
disjoint source samples supplies ordinary within-virtual-class positives, and
the reversed/shared-part cases supply hard-negative sampling. The novelty
claim rests on the conjunction of generator persistence, sampling, and
negative construction rather than on a new observable source of supervision.
HSE makes this conclusion stronger because it directly occupies mixed spatial
hybrids as additional DML supervision on the same benchmarks.

## Gate 1 and effect-size ruling

No verified repository measurement identifies hard recipe identities as the
missing cause of retrieval error. The closest evidence is adverse or
non-identifying: position-tolerant MaxSim recovered 6.67 points relative to a
broken fixed-slot regional comparison, but trained `region_pa` remained about
3.6 CUB points below paired Proxy Anchor, and the frozen Cars MaxSim probe lost
1.47 points. These measurements show that alignment affects a weak regional
readout; they do not show that impossible cross-class composites encode unseen
identity structure.

Fable's frozen CUB prediction was 0.778, but it assigned only a 35--40% chance
of clearing the 0.766 horizon with confidence. There is no repository-derived
effect decomposition supporting the forecast, and this project has already
falsified additive component predictions.

## Verdict

**Candidate 370 remains DEAD at Gates 1 and 2.** HSE is a material omitted
near-neighbour and is recorded here as a correction. The narrow
disjoint-source recipe-equivalence residue is a sampling-level conjunction
without measured provenance, not sufficient novelty under the registered
protocol. No diagnostic, implementation, preregistration, or GPU run is
authorized by this re-audit.

