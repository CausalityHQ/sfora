# Post-OAPF candidate batch: 177--182

Date: 2026-08-02. This batch was generated after OAPF failed its blinded
reliability gate and candidates 175--176 died at prior art. It was reviewed
with Claude before implementation. No candidate in this batch authorizes GPU
work.

## Measured constraints

- OAPF's independent crop-tail radius was not a repeatable endpoint property:
  global pack Spearman **0.317593** and within-class-residual Spearman
  **0.184057**.
- ARCG's deterministic response relation was real, but using it as pair
  eligibility erased useful class attraction; IPSR's stronger unsatisfied
  cross-instance response ordering gained only **+0.060 pt** after selection
  correction.
- After matching class size, disconnected In-Shop within-class 1-NN graphs
  were associated with **+3.534 R@1 points**. Fragmentation is therefore the
  other measured opening, but it is not permission to relabel an old diversity
  penalty.

## Gate-2 outcomes

### 177. Consensus intervention ordering — DEAD

Two same-class images would vote on which controlled augmentation damages
identity evidence more, creating an ordinal label only when their votes agree.
This is a data-dependent descriptor inside the augmentation-ranking operator of
Fu et al., *Deep Metric Learning with Self-Supervised Ranking* (AAAI 2021,
<https://doi.org/10.1609/aaai.v35i2.16226>). More decisively, candidate 20
(IPSR) already tested a cross-instance response-derived ordinal target and
failed at **+0.060 pt** corrected. Changing the ranked objects from real peers
to augmented views does not repair that mechanism and needs extra image passes.

### 178. Determinantal class volume — DEAD

A class-batch log-determinant would preserve higher-order volume while Proxy
Anchor preserves identity attraction. This is intra-class diversity
regularisation, not a new label. Joint Representation Diversification already
balances metric discrimination with representation diversity (Chu et al., ICML
2020, <https://proceedings.mlr.press/v119/chu20a.html>); reverse contrastive
intra-class diversity, DVML, and *Determinant regularization for Deep Metric
Learning* (<https://openreview.net/forum?id=NUBuJsAq1U>) occupy the same
objective class. A DPP changes the diversity functional, not the supervision.

### 179. Response-code parity — DEAD

Quantising deterministic response order into a code and asking same-class pairs
to predict code agreement/XOR is ordinary augmentation-aware auxiliary
prediction (AugSelf/ECOC). The code is a derived nuisance descriptor; it does
not add a retrieval relation and candidate 20 already showed that response
agreement is not relevance.

### 180. Complementary evidence union — DEAD

Complementarily mask two same-class images and require their pooled evidence to
identify the class. This is set/multiple-instance classification combined with
part erasing. Attention ensembles, part-based DML, compositional DML, and
same-class mixing already create complementary regional evidence. The new mask
schedule changes the view generator, not the class/set supervision.

### 181. Nuisance-delta transplantation — DEAD

Transfer a within-class feature difference to an image of another class and
retain the recipient label. This is feature synthesis occupied by DVML,
Embedding Expansion, Metrix (<https://arxiv.org/abs/2106.04990>), and the
trajectory-transfer family already audited as candidate 45.

### 182. Counterfactual negative abstention — DEAD

Mark different-class pairs unknown when controlled transformations make their
evidence collide. This is false-negative cancellation/hardness-conditioned
negative mining. It also routes through cross-class relations, which the CUB
RSPG diagnostic showed to be nearly non-differentiating (64.49% gate density).

## Adversarial generation result

Claude was separately asked for mechanisms imported from coding theory, causal
inference, ecology, control, and experimental design. Its proposed class codes
were ECOC without semantic information; causal instruments were not observed;
class niche width was variance regularisation; stability regions were
adversarial robustness; and semantic blocks required external annotations or a
teacher. None passed the source-and-operator constraint.

This is an exhausted batch, not a proof of impossibility. The useful update is
that neither deterministic response nor observed fragmentation can be routed
through another ranking/diversity functional and called a new supervision
mechanism.

## Immediate topology follow-up: 183 — DEAD

Candidate 183 replaced a point proxy by an admissible class shell/level set so
same-class samples would be supervised as members without collapsing to one
point. PFML already gives same-class attraction a zero-force interior;
chance-constrained DML explicitly characterises class-proxy covering radii
(Gürbüz et al., WACV 2024,
<https://openaccess.thecvf.com/content/WACV2024/html/Gurbuz_Deep_Metric_Learning_With_Chance_Constraints_WACV_2024_paper.html>),
and non-isotropic probabilistic proxy DML represents class-specific spread
(Kirchhof et al., <https://arxiv.org/abs/2207.03784>). A shell changes the
feasible-set geometry, not what label relation exists. It is **dead at Gate 2**.
