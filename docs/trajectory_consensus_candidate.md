# Candidate 2: cross-trajectory consensus supervision

Status: **failed gate 2 on prior art; no implementation or GPU run**.

## Gate 1 — provenance: PASS

The proposal follows from two measurements in this repository:

1. Nominally identical fixed-seed CUB runs differ by as much as **1.08 pt**
   Recall@1. GPU nondeterminism therefore sends training down materially
   different trajectories; the variation is not merely uncertainty in reading
   one curve.
2. Replacing best-over-training with top-5 checkpoint averaging leaves the
   six-seed paired standard deviation of `pa_distill − proxy_anchor` essentially
   unchanged (**0.367 pt** for the maximum versus **0.363 pt** for top 5).
   Temporal smoothing within one trajectory cannot recover supervision that is
   unstable between trajectories.

The earlier checkpoint-variance idea failed this causal test: observations from
one path do not reveal which relations would survive another path. Candidate 2
addresses that exact defect by training two independently perturbed replicas and
allowing a new relation to become a target only when their retrieval
neighbourhoods agree.

Concretely, each replica supplies its nearest cross-instance neighbours under
independent augmentation, minibatch order, dropout, and optimization noise.
The intersection of the two neighbourhoods creates consensus pseudo-positive
relations; high-confidence neighbours proposed by only one replica are withheld
rather than forced into the existing label-only objective. Ground-truth
class-positive and class-negative supervision remains unchanged. The candidate
therefore changes **what supervision exists** by adding cross-trajectory
consensus relations, rather than changing the similarity function, mining
weight, loss regularization, or checkpoint readout.

The measured prediction behind the mechanism is directional: relations stable
across two genuinely diverging trajectories should transfer better to unseen
classes than relations selected from either trajectory alone. This gate does
not yet assign an effect size. A numeric prediction and falsification threshold
are permitted only if the mechanism survives gate 2.

## Gate 2 — prior art: FAIL

The audit searched dual-network agreement, mutual metric learning,
consensus-neighbour pseudo-labels, cross-view graph intersection, and
relationship-disagreement selection in DML and adjacent retrieval.

The mechanism is occupied:

- *Unsupervised Domain Adaptation with Noise Resistible Mutual-Training for
  Person Re-identification* maintains two networks for collaborative clustering
  and mutual instance selection. Its selection explicitly uses peer confidence
  and **relationship disagreement** between the networks
  ([Zhao et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1391_ECCV_2020_paper.php)).
- *Mutual Mean-Teaching* collaboratively trains two identically structured
  networks with different initializations, uses their temporally averaged
  models to generate online soft pseudo-labels, and applies those labels through
  a soft triplet objective for person retrieval
  ([Ge et al., ICLR 2020](https://arxiv.org/abs/2001.01526)).
- *Graph Consistency Based Mean-Teaching* constructs similarity graphs from
  multiple teacher networks and imposes graph-consistency supervision between
  teacher and student retrieval models
  ([Yang et al., IJCAI 2021](https://www.ijcai.org/proceedings/2021/121)).
- *Part-based Pseudo Label Refinement* computes cross-agreement from k-nearest
  neighbours in two feature spaces and uses it to refine retrieval
  pseudo-labels
  ([Cho et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Cho_Part-Based_Pseudo_Label_Refinement_for_Unsupervised_Person_Re-Identification_CVPR_2022_paper.pdf)).

These are not merely papers that train two models. They instantiate the
candidate's defining causal move: use agreement or disagreement between
independently varying retrieval representations to decide which inter-instance
relations become supervision. Replacing their unsupervised target-domain
setting with labelled CUB/Cars/In-Shop training and taking a hard neighbourhood
intersection would be a benchmark adaptation and selection-rule variant.

The repository's between-trajectory instability remains a valid motivation for
testing such established machinery, but it cannot support a genuinely novel
method claim. Candidate 2 stops at gate 2 and consumes no GPU.
