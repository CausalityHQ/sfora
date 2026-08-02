# Candidate 214: fixed-probe learning-dynamics supervision

Date: 2026-08-02. Status: **DEAD at Gate 2**. No diagnostic,
implementation, or GPU run.

## Gate 1 premise

Nominally identical fixed-seed runs in this repository differ by **1.08 R@1
points**, while candidate 203 could not attribute within-epoch scalar loss
variation because batch composition, random transforms, parameter movement and
CUDA nondeterminism were confounded. A cleaner measurement would evaluate a
fixed unaugmented In-Shop training panel at fixed checkpoints and record, per
image, labelled-proxy ownership, nearest-positive identity, first acquisition,
and forgetting transitions.

This diagnostic would be valid as description: fixed inputs remove transform
and batch-composition noise from the measurement path, and parameter movement
is the intended signal. It still cannot attribute training loss variance to
batch composition, which would require an intervention rather than a probe.

## Gate 2 reduction

No method action survives. Every proposed statistic is a deterministic function
of the current parameter state and training labels, not an independently
observed relation. It can enter later training only as:

- a per-example coefficient, schedule, or inclusion decision (curriculum,
  dynamic sampling, or self-paced learning);
- a per-interaction mask (mining, co-teaching, or T-SINT interaction
  selection);
- a stored historical target (temporal ensembling or self-distillation);
- a trajectory-derived relabeling (clustering, subcentres, or multi-proxy
  assignment); or
- a gradient modifier (regularization).

The class-disjoint setting makes the import weaker than closed-set dataset
cartography: the statistic is indexed by training examples and cannot be
computed for unseen test identities without becoming an ordinary test-time
uncertainty or similarity estimate. Moreover, one trajectory is poorly
identified against the repository's measured 1.08-point run divergence;
stabilizing it with multiple runs violates the roughly-1x budget and becomes
cross-model interaction selection already killed as candidate 134.

Primary neighbours:

- Toneva et al., *An Empirical Study of Example Forgetting during Deep Neural
  Network Learning*, ICLR 2019, <https://arxiv.org/abs/1812.05159>.
- Swayamdipta et al., *Dataset Cartography: Mapping and Diagnosing Datasets
  with Training Dynamics*, EMNLP 2020,
  <https://arxiv.org/abs/2009.10795>.
- Liang, Zhao, and Chen, *Dynamic Sampling for Deep Metric Learning*,
  <https://arxiv.org/abs/2004.11624>.
- Ibrahimi et al., *Learning with Label Noise for Image Retrieval by Selecting
  Interactions*, WACV 2022,
  <https://openaccess.thecvf.com/content/WACV2022/html/Ibrahimi_Learning_With_Label_Noise_for_Image_Retrieval_by_Selecting_Interactions_WACV_2022_paper.html>.

## Verdict

Candidate 214 is **DEAD at Gate 2**. The proposed panel remains a valid
descriptive instrument, but it cannot create a new supervision object under
class-disjoint retrieval and therefore does not justify its diagnostic cost.
