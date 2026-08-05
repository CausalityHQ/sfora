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

## Adjudication rule

These are guardrails, not blanket closures. Similar language does not kill a
future proposal; an equivalent learned object, observation channel, and
gradient action does. Conversely, a renamed activation, recurrent block,
meta-weighting loop, mean field, or synthetic-class construction is dead at
Gate 2 even if its motivating analogy comes from a different discipline.
