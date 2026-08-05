# Pass-15 non-loss and cross-field prior-art guardrails

Date: 2026-08-05. Scope: critic-side preparation during the capacity wait.
This file is not input to the blind proposer or the independent frozen-proposal
reviewer, and it is not a verdict on a candidate that does not yet exist.

The point is to stop a vocabulary change from masquerading as a mechanism
change when the blind proposer ranges over activations, architectures,
optimizers, training algorithms, and imports from other sciences. Each item
below was checked against a primary paper or proceedings page.

## Activation and representation support

**Expanding Hyperspherical Space (EHS), Deng and Xiang, WACV 2024.** EHS
attributes an incomplete/crowded embedding hypersphere to ReLU, uses an
odd-symmetric activation to fill the sphere, reserves a region for future
classes, and adds pseudo-instances. A proposal whose contribution is merely an
odd/symmetric activation or reserved angular volume for unseen identities is
therefore occupied. A genuinely different activation proposal must identify a
different measured failure and an executable property beyond sign symmetry or
empty-space reservation.

Primary source:
https://openaccess.thecvf.com/content/WACV2024/html/Deng_Expanding_Hyperspherical_Space_for_Few-Shot_Class-Incremental_Learning_WACV_2024_paper.html

## Recurrent/predictive architecture

**Deep Predictive Coding Network, Wen et al., ICML 2018.** PCN uses recurrent
bottom-up prediction errors and top-down predictions to iteratively refine a
visual representation before classification. “Biology-inspired predictive
coding,” recurrent refinement, or feedback-error cycles are not novel by
themselves. Under this project's fixed one-view global-descriptor deployment,
a survivor must also count every inference cycle and distinguish its learned
operator from ordinary recurrent feature refinement.

Primary source: https://proceedings.mlr.press/v80/wen18a.html

**Hybrid-Attention Decoupled Metric Learning, Chen and Deng, CVPR 2019.** DeML
already attacks selective/partial visual learning in zero-shot image retrieval
with multiple attention-specific learners, object attention via random-walk
graph propagation, and adversarial channel attention. A part/expert/attention
architecture needs a mechanism-level distinction from this direct DML
precedent, not just a different router or number of heads.

Primary source:
https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid-Attention_Based_Decoupled_Metric_Learning_for_Zero-Shot_Image_Retrieval_CVPR_2019_paper.pdf

## Optimizer and training-controller mechanisms

**Deep Metric Learning via Adaptive Learnable Assessment (DML-ALA), Zheng,
Lu, and Zhou, CVPR 2020.** DML-ALA learns a sequence-aware sample assessor by
meta-learning. Each episode uses disjoint-label train/validation subsets; a
one-gradient-updated metric's held-out-label performance trains the assessor.
Thus learned sample weighting, bilevel controllers, and simulated
train/test-identity optimization are occupied unless the proposed controller
changes the observed object or update field in a substantive way.

Primary source:
https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html

General sharpness-aware optimization is also heavily occupied; a DML-specific
proposal must do more than apply SAM or class-conditioned perturbation. The
local critic should demand a measured DML failure, a matched compute control,
and a distinction from ordinary flat-minimum or stochastic-gradient-noise
regularization.

## Statistical-physics imports

**Mean Field Theory in Deep Metric Learning, Furusawa, 2023.** This work
explicitly imports ferromagnetic mean-field theory into DML, replacing
pairwise interactions by learned mean fields and deriving classification-form
MeanFieldContrastive and MeanFieldClassWiseMultiSimilarity objectives. A
proposal is not novel because it calls proxies order parameters, energies a
Hamiltonian, or pair aggregation a mean-field approximation. It must produce a
different executable interaction and causal prediction.

Primary source: https://arxiv.org/abs/2306.15368

## Synthetic unseen-class anticipation

**Proxy Synthesis, Gu, Ko, and Kim, 2021.** Proxy Synthesis interpolates
embeddings and proxies into synthetic classes explicitly intended to mimic
unseen classes and smooth decision boundaries, on the standard retrieval
benchmarks. Any chemical recombination, interpolation, reaction-mixture, or
virtual-species analogy that compiles to synthetic embeddings/proxies/classes
collides unless its supervision referent and gradient field differ.

Primary source: https://arxiv.org/abs/2103.15454

## Optimal transport, graph diffusion, and structural dynamics

**Batch-Wise Optimal Transport Loss, Xu, Sun, and Liu, CVPR 2019.** This work
solves an optimal-transport program over each batch to learn an
importance-driven distance metric, emphasize hard examples, and train the
representation end to end. An earth-mover, mass-flow, chemical-transport, or
balanced-reaction story that compiles to batch assignment/reweighting is
occupied at mechanism level.

Primary source:
https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Learning_With_Batch-Wise_Optimal_Transport_Loss_for_3D_Shape_Recognition_CVPR_2019_paper.html

**Learning Intra-Batch Connections, Seidenschwarz, Elezi, and Leal-Taixé,
ICML 2021.** This method refines embeddings by learned message exchange among
all samples in a batch and evaluates on CUB, Cars196, SOP, and In-Shop. A
reaction-diffusion, interacting-particle, or graph-neural update is not new if
its executable step is batch-neighbour message passing followed by an ordinary
DML objective.

Primary source: https://proceedings.mlr.press/v139/seidenschwarz21a.html

**Contextual Similarity Optimization, Liao, Tsiligkaridis, and Kulis, ICML
2023.** Contextual loss optimizes neighbourhood-set structure together with
cosine similarity for supervised retrieval. Diffusing labels/similarities on a
kNN graph, matching neighbourhood overlap, or using graph context as a
training signal must be distinguished from this direct standard-benchmark
precedent.

Primary source: https://proceedings.mlr.press/v202/liao23b.html

**Structural Matching for interpretable DML, Zhao et al., ICCV 2021.** This
line uses optimal transport for learned structural image matching. A proposal
that transports spatial/token parts rather than batch samples must confront
this part-level retrieval precedent as well as the batch-wise OT loss above.

Primary source:
https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.pdf

## Directional statistics and topology

**Directional Statistics-based DML, Zhe, Chen, and Yan, 2018.** This method
models normalized embeddings with a von Mises--Fisher distribution and derives
a hyperspherical retrieval objective with a global view of the embedding
space. A statistical-mechanics or molecular-orientation analogy that ends in a
vMF energy is occupied.

Primary source: https://arxiv.org/abs/1802.09662

**Differentiable persistent-homology precedents.** Persistent-homology losses
are already used to train visual models, including topology-preserving
contrastive learning and persistent-topology alignment of real and synthetic
feature graphs. These are adjacent rather than exact standard-DML precedents,
so topology is not blanket-closed. But a future proposal must specify which
Betti/persistence quantity is measured, prove it cannot be minimized by an
irrelevant nuisance topology, and distinguish its gradient from ordinary
neighbour-graph or spectral regularization. Merely adding a persistence-image
distance is not a novelty case.

Primary sources:
https://openaccess.thecvf.com/content/CVPR2026/html/Li_Fixed_Anchors_Are_Not_Enough_Dynamic_Retrieval_and_Persistent_Homology_CVPR_2026_paper.html
and
https://openaccess.thecvf.com/content/CVPR2026/papers/Meng_TopoCL_Topological_Contrastive_Learning_for_Medical_Imaging_CVPR_2026_paper.pdf

## Boosting, direct-gradient, and adaptive-training algorithms

**BIER: Boosting Independent Embeddings Robustly, Opitz et al., 2018.** BIER
splits the final descriptor into an embedding ensemble and trains its learners
as an online gradient-boosting sequence, reweighting examples from preceding
learners and encouraging diversity. It evaluates on all four target benchmark
families without extra deployed parameters. Boosted descriptor blocks,
sequential residual experts, or “ecological niche” sub-embeddings therefore
need a distinction stronger than a new diversity penalty.

Primary source: https://arxiv.org/abs/1801.04815

**GOAL, Xuan and Chen, WACV 2023.** GOAL analyzes DML objectives through their
embedding gradients and directly implements desired gradient combinations even
when no scalar loss integrates to that field. The paper is image--text
retrieval rather than the present image-only lane, so it is adjacent rather
than an exact benchmark collision. It nevertheless occupies “optimize the
gradient law instead of inventing a loss” as a general contribution. A future
proposal must supply a distinct field, causal premise, and image-only controls.

Primary source:
https://openaccess.thecvf.com/content/WACV2023/html/Xuan_Dissecting_Deep_Metric_Learning_Losses_for_Image-Text_Retrieval_WACV_2023_paper.html

**ADroDML, Huai et al., IJCAI 2019.** ADroDML derives a DML generalization
bound and adaptively learns dropout retention rates instead of fixing them.
Learned stochastic-depth/dropout schedules or a bound-driven retention
controller are occupied unless their learned object and bound differ
materially.

Primary source: https://www.ijcai.org/proceedings/2019/352

Natural-gradient, Fisher--Rao, differential-game, and optimal-control
optimizers are well-established outside DML. They are not blanket closures for
a DML-specific update, but the scientific analogy itself supplies no novelty.
The local critic must reduce the actual parameter update and compare it with
natural gradient, second-order preconditioning, SAM, gradient surgery, and the
meta-controller precedents above. Without a verified causal premise and a
matched-compute optimizer control, a geometry/control label is insufficient.

## Pooling, parts, uncertainty, and normalization architectures

**Generalized Sum Pooling for Metric Learning, Gürbüz et al., ICCV 2023.** GSP
is a trainable generalization of global-average pooling that selects and
reweights feature subsets, adds cross-batch regularization for zero-shot
transfer, and evaluates on SOP, In-Shop, CUB, and Cars. A learned pooling
operator, spatial/channel selection rule, or transport-inspired aggregation is
occupied unless its executable invariant differs materially from GSP and the
part-level structural-matching precedent.

Primary source:
https://openaccess.thecvf.com/content/ICCV2023/papers/Gurbuz_Generalized_Sum_Pooling_for_Metric_Learning_ICCV_2023_paper.pdf

**Attention-based Ensemble for DML, Kim et al., ECCV 2018.** This method gives
multiple embedding learners different attention masks and adds a divergence
loss to promote part diversity on standard DML retrieval benchmarks. Together
with DeML and BIER, it occupies the basic architecture of diverse
part-specialized heads concatenated or ensembled into a descriptor.

Primary source:
https://openaccess.thecvf.com/content_ECCV_2018/html/Wonsik_Kim_Attention-based_Ensemble_for_ECCV_2018_paper.html

**Multi-Head DML Using Global and Local Representations, Ebrahimpour et al.,
WACV 2022.** Global/local multi-head representation learning is likewise a
direct retrieval precedent. A capsule, expert, or anatomical-part proposal
must show more than multiple heads receiving different spatial evidence.

Primary source:
https://openaccess.thecvf.com/content/WACV2022/papers/Ebrahimpour_Multi-Head_Deep_Metric_Learning_Using_Global_and_Local_Representations_WACV_2022_paper.pdf

**Introspective DML, Zheng et al., 2022.** IDML learns a semantic embedding and
an accompanying uncertainty embedding, then makes uncertainty-aware pairwise
similarity judgments on CUB, Cars, and SOP. A confidence, evidential, variance,
or ambiguity head is occupied unless it changes supervision or training action
beyond uncertainty-conditioned comparison.

Primary source: https://arxiv.org/abs/2205.04449

**MDProp, Singh, Kakizaki, and Araki, ACML 2024.** MDProp generates several
adversarial/unadversarial feature-space distributions and assigns them
separate batch-normalization paths, improving clean DML retrieval as well as
robustness. Learned or routed normalization remains possible, but a proposal
needs a measured distribution split and must separate the normalization effect
from augmented-example training.

Primary source: https://proceedings.mlr.press/v222/singh24a.html

## Adjudication rule

These are guardrails, not blanket closures. Similar language does not kill a
future proposal; an equivalent learned object, observation channel, and
gradient action does. Conversely, a renamed activation, recurrent block,
meta-weighting loop, mean field, or synthetic-class construction is dead at
Gate 2 even if its motivating analogy comes from a different discipline.
