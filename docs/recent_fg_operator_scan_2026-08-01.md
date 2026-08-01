# Recent fine-grained operator scan: candidates 145--146

Date: 2026-08-01. This scan followed the 144-candidate stopping audit and
searched 2025--2026 fine-grained retrieval and adjacent local/global retrieval
work for a supervision family missing from the catalogue. Claude was separately
asked for a counterexample under the same exclusions: Opus and Sonnet timed out
without verdicts; Haiku returned `NONE`. Silence from timed-out reviews is not
evidence and is not counted.

## 145. Learnable visual-concept decomposition

**Provenance.** The repository's local-feature audit found a 6.67-point recovery
from position-tolerant MaxSim relative to fixed-coordinate comparison. A
plausible response is to learn a small set of concept slots from regional
features and aggregate them into a single test descriptor.

**Gate 2.** This is occupied. Wang et al., *VCE: Visual Concept Embedding for
Open-set Fine-grained Image Retrieval* (Knowledge-Based Systems 2025), uses
learnable concept vectors, cross-attention to regional features, orthogonality,
and graph constraints to produce independent composable visual concepts for
open-set FGIR:
https://doi.org/10.1016/j.knosys.2025.114311. VAPNet (NeurIPS 2023) is the
earlier training-only learned-attribute neighbour. Renaming slots as local
evidence atoms or changing their aggregation does not create a new operator.

## 146. Local-similarity-to-global distillation

**Provenance.** The same MaxSim recovery suggests using expensive local matching
only during training: compute a local MaxSim relation matrix and force the
single 512-D global descriptor to reproduce it.

**Gate 2.** This is similarity distillation, not a new supervision relation.
Roth et al., *Simultaneous Similarity-based Self-Distillation for Deep Metric
Learning* (ICML 2021), explicitly transfers similarities from auxiliary
high-dimensional embedding and feature spaces into the deployed DML embedding
at negligible inference cost:
https://proceedings.mlr.press/v139/roth21a.html. Lebailly et al., *Global-Local
Self-Distillation for Visual Representation Learning* (WACV 2023), explicitly
studies local-to-global self-distillation and geometric matching:
https://openaccess.thecvf.com/content/WACV2023/papers/Lebailly_Global-Local_Self-Distillation_for_Visual_Representation_Learning_WACV_2023_paper.pdf.
Using MaxSim as the teacher kernel changes the teacher statistic, not the
distillation operator.

## Result

Both candidates are **DEAD AT GATE 2**. No implementation, diagnostic export,
or GPU run follows. The recent scan also found VCE, noise injection, mixture-of-
experts ReID, and local/global descriptor fusion; these reinforce already
covered attribute, regularisation, conditional-feature, and local-retrieval
families rather than opening a new supervision object.
