# Fable HDE collision audit (candidate 373)

Date: 2026-08-05.

## Execution caveat and frozen proposal

An outcome-only Fable retry ran from `/tmp` in safe mode with web-only tools and
no repository access. It was given the legal similarity-learning task but, in
the quota-probe retry that unexpectedly began running, the prompt abbreviated
the audited frontier instead of supplying its numeric values. The output is
therefore a valid independent mechanism proposal but **not** a valid completion
of the user's requested problem-plus-current-SOTA experiment. A full numeric-
frontier retry remains required.

Fable proposed **Held-out-class Density Equalization (HDE)**:

1. rotate an 80/20 split of training classes each epoch;
2. apply the ordinary discriminative DML loss only to the 80% subset;
3. use an embedding queue to estimate each sample's mean cosine similarity to
   its `k` nearest foreign-class neighbours;
4. penalize variance of that per-sample density; and
5. add a Sinkhorn-derived uniform-column-marginal loss so every gallery sample
   is retrieved equally often.

The intended test-time result is an ordinary normalized descriptor and cosine
ranking. Fable argued that if gallery local-density terms were constant, CSLS
would be rank-equivalent to cosine for each query. It forecast **+0.5 to +2.0
R@1 on CUB and In-Shop**, plausibly zero on Cars, and proposed a new class-split
CSLS/NNN diagnostic requiring at least +1.0 point before implementation.

## Frontier audit fails

The output replaced the supplied current frontier with older IBC/PFML rows and
claimed that large recent in-scope gains came only from external models. That is
false. The audited comparable-CNN targets are **AdvRF 0.766 CUB / 0.949 Cars**
and **VAPNet 0.939 In-Shop**; both operate from benchmark pixels and labels with
ordinary ImageNet initialization and single-model, single-view descriptors.

HDE's own +0.5--2.0-point range does not supply a quantitative argument for
crossing those targets. Starting from the repository's corrected BN-Inception
In-Shop final reference 0.9137 would leave a much larger gap, and Fable did not
name a stronger matched HDE base whose frozen score plus the expected effect
crosses 0.939. It therefore fails the standing outcome independently of novelty.

Fable also cited arXiv 2503.10526 as CSLS. That source is **NeighborRetr**;
classical CSLS is Conneau et al., *Word Translation Without Parallel Data*
(ICLR 2018). The algebraic observation remains true only under its premise: for
a fixed query, CSLS preserves cosine order if the gallery correction is exactly
constant. HDE's foreign-class training-queue statistic is not the unseen
gallery's CSLS term, and a Sinkhorn plan is globally coupled rather than a
separable local correction, so the proposed loss does not guarantee the claimed
rank equivalence.

## Gate 1: no positive repository provenance

No verified repository measurement identifies hubness reduction as a causal
opportunity. The proposal inferred it from the deployment constraint and
external cross-modal gains.

The historical CUB hubness audit is directionally adverse: across 17 saved
models, CSLS changed R@1 by **-0.65 points** and Sinkhorn by **-3.16 points** on
average while sharply reducing skew. Those old CUB artifacts are not promoted
to the post-audit-321 verified tier, so they cannot alone falsify every modern
HDE implementation. They do, however, refute Fable's description of positive
headroom as already known in this repository.

The corrected evidence boundary supplies no replacement positive result. The
prospectively verified corrected In-Shop reference establishes scores and
stable query errors, not gallery k-occurrence skew or benefit from CSLS/NNN.
Candidate 366 had already rejected generic hubness as an identified error cause.

HDE is therefore an externally motivated regularizer awaiting a new diagnostic,
not a method derived from a repository measurement. It fails Gate 1 before the
proposed three-dataset diagnostic or GPU use.

## Gate 2: every executable component is occupied

### Density-aware supervised DML

Li et al., *Deep Metric Learning with Density Adaptivity* (2019), explicitly
integrate embedding concentration into end-to-end supervised DML as a plug-in
density regularizer for contrastive, N-pair, and triplet objectives, motivated
by open-set generalization. Roth et al., *Revisiting Training Strategies and
Generalization Performance in Deep Metric Learning* (ICML 2020), independently
connect embedding density/compression to transfer and train a regularizer on
CUB, Cars, and SOP.

- https://arxiv.org/abs/1909.03909
- https://proceedings.mlr.press/v119/roth20a.html

Choosing a foreign-neighbour `k`-density rather than class concentration or a
spectral statistic changes the density estimator, not the method class.

### Training-time hub and anti-hub balancing

NeighborRetr (2025) computes training-sample centrality from a queue, adjusts
training relations using that centrality, and adds a Sinkhorn uniform-marginal
regularizer explicitly so anti-hubs receive balanced retrieval probability. It
then deploys ordinary similarity without a test-time memory bank. This exactly
occupies HDE's queue-centrality and uniform-column-marginal operations. Its
cross-modal task changes available labels, but not those mathematical
operators.

Primary source: https://arxiv.org/abs/2503.10526

Trosten et al., *Hubs and Hyperspheres* (CVPR 2023), additionally establish
training embeddings to reduce hubness for unseen episodes, albeit in
transductive few-shot classification rather than open-gallery DML.

Primary source:
https://openaccess.thecvf.com/content/CVPR2023/html/Trosten_Hubs_and_Hyperspheres_Reducing_Hubness_and_Improving_Transductive_Few-Shot_Learning_With_Hyperspherical_Embeddings_CVPR_2023_paper.html

### Rotating class holdout is not a novel unseen-class attachment

With HDE's epochwise rotation, a nominally held-out class has already shaped the
backbone through the discriminative loss in earlier epochs and will do so again
later. The density term therefore does not observe a class unseen by the
discriminative learner; the load-bearing novelty claim is false as specified.

Making the split genuinely disjoint requires fold-specific models/updates or a
fixed data sacrifice. Class-disjoint episodic DML is occupied exactly by Zheng,
Lu, and Zhou's *Deep Metric Learning With Adaptively Composite Dynamic
Constraints* (TPAMI 2023), which meta-optimizes a constraint generator on a
disjoint-label subset. Candidates 228, 336, and 353 already close this
attachment. Applying an occupied density loss on the outer fold is a component
conjunction, not a new supervision referent.

Primary source: https://ieeexplore.ieee.org/document/10008092

## Verdict

**DEAD at Gates 1 and 2; no diagnostic, implementation, preregistration, or
GPU.** HDE has no positive repository provenance, composes established density
regularization, queue-centrality and Sinkhorn anti-hub balancing with occupied
class-disjoint episodic training, and does not cross the actual supplied
frontier under its own forecast. Its useful process residue is that abbreviated
outcome prompts let Fable silently replace the current SOTA; future runs must
include the numeric frontier verbatim.

