# Pass 123 — post-ECTR candidate batch (2026-08-07)

This batch is generated from repository evidence, not from an armchair loss
swap.  The corrected Pass-67 decomposition attributed 51.9% of CUB failures to
the between-class component, while RSPG retained 64.49% of CUB same-class pairs
but only 8.63% on In-Shop.  ECTR then passed its CPU feasibility test but its
full In-Shop arm is currently far below the paired baseline.  The implication
is that a useful next mechanism should model *within-class factors* without
relying on cross-class identities, and must change the supervised object rather
than only reweighting pairs.

The Fable/Claude blind-review service is unavailable until its quota reset; this
is recorded as a process limitation, not silently substituted with a claimed
independent review.

## Gate-2 triage

### GCPE — Gradient-Conflict Positive Eligibility: DEAD at Gate 2

**Proposed mechanism.** For two labelled same-class images, compute detached
per-pair gradients of their positive and hard-negative objectives.  A pair is
eligible as positive-to-unknown only when its update direction is compatible
with the class's running gradient cone; incompatible same-class pairs are not
treated as positives.  This would make the supervised object depend on whether
the pair helps the shared representation, rather than on embedding distance.

**Prior art.** The unified DML gradient analysis already shows that common DML
losses reduce to pair-weight assignment (Wang et al., ICLR 2019), and
distributionally robust pair weighting explicitly optimizes the pair-weight
distribution (Qi et al., ICLR 2020).  PCGrad/CAGrad-style gradient-conflict
methods occupy the compatibility test and projection itself.  Applying that
test to positive eligibility is an application-level combination, not a
defensible new mechanism.  No implementation or GPU run is authorized.

### DPA — Dynamic Prototype Assignment: DEAD at Gate 2

**Proposed mechanism.** Reassign class labels to a fixed set of hyperspherical
prototypes online by bipartite matching, so a class can rotate between factor
components during training while the deployed descriptor remains single-vector.

**Prior art.** Saadabadi et al., *Hyperspherical Classification with Dynamic
Label-to-Prototype Assignment* (CVPR 2024), already formalizes the sequential
gradient-descent plus bipartite-matching assignment.  Discrepant and
multi-instance proxies (Zou et al., ICCV 2023) occupy the class-factor proxy
variant.  DPA is therefore dead before GPU.

### ERG — Exchangeability-Residual Gate: DEAD at Gate 2

**Proposed mechanism.** Maintain a detached, class-conditional residual
coordinate system from *within-class* embeddings.  For a same-class pair,
convert each residual vector to rank/sign coordinates under the running
class-conditioned covariance (a copula-like signature), and retain the pair as
positive only when the two signatures are exchangeable at a preregistered
level.  This is a hard positive-to-unknown gate based on within-class factor
compatibility; it is not a multi-center proxy, a distance weight, or a
single-image auxiliary loss.

**Adversarial reduction.** The earlier PEBH measurement and RPEX audit already
showed the relevant mechanism family: derive a within-class representation
from a peer, use it to decide or improve a same-class relation, and retain a
self-only deployment branch.  Wang et al. (CVPR 2016), X-ReID (Shen et al.,
2023), Cross-GAN, and feature-exchange/transfer blocks occupy that family;
residualizing the signal and changing the decision to an exchangeability test
does not change the supervision object.  Deep Relational Metric Learning
(Zheng et al., ICCV 2021), covariance embedding, and variational metric
learning add further adjacent coverage.  ERG is therefore **DEAD at Gate 2**;
no diagnostic, implementation, or GPU run is justified.

## Next gate

After the Pass-119 controller, fixed random rerun, and selection analysis finish,
continue with SRC's implementation-level distinction audit and controls.  No
ERG CPU diagnostic is authorized because Gate 2 already failed.

## Primary sources

- Wang et al., *A Unified View of Deep Metric Learning via Gradient Analysis*, ICLR 2019: https://openreview.net/forum?id=Skf5qiC5KQ
- Qi et al., *A Simple and Effective Framework for Pairwise Deep Metric Learning*, ICLR 2020: https://openreview.net/forum?id=SJl3CANKvB
- Saadabadi et al., *Hyperspherical Classification with Dynamic Label-to-Prototype Assignment*, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Saadabadi_Hyperspherical_Classification_with_Dynamic_Label-to-Prototype_Assignment_CVPR_2024_paper.html
- Zheng et al., *Deep Relational Metric Learning*, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Deep_Relational_Metric_Learning_ICCV_2021_paper.html
