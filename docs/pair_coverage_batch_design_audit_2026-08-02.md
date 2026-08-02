# Candidate 204: cross-epoch positive-pair coverage audit

Date: 2026-08-02. Gate-2 audit performed before any diagnostic, implementation,
or GPU work.

## Proposed mechanism

Replace memoryless random within-class grouping with a near-resolvable balanced
block schedule so that, under the same fixed batch and epoch budget, each
same-class image pair has nearly equal realized co-occurrence multiplicity.
The intended distinction was label-only cross-epoch state: no embedding
hardness, external model, loss reweighting, or memory bank.

## Verdict

**DEAD at Gate 2.** A random M-per-class sampler already gives every same-class
pair the same inclusion probability. A balanced block schedule leaves that
first moment unchanged and reduces only the finite-run variance of realized
pair counts. The mechanism is therefore global batch assignment plus balanced
experimental design, both occupied.

The closest primary sources are:

- Sachidananda, Yang, and Zhu, *Global Selection of Contrastive Batches via
  Optimization on Sample Permutations* (ICML 2023). GCBS explicitly formulates
  contrastive batch construction as a global assignment/permutation problem
  and changes which samples coexist without changing the model. Uniform
  positive-pair coverage is a different assignment objective inside that
  established mechanism.
- Chisaki, Fuji-Hara, and Miyamoto, *Combinatorial Designs for Deep Learning*
  (Journal of Combinatorial Designs 2020; arXiv:1809.08404). It replaces random
  dropout selection with a combinatorial design specifically to balance the
  realized frequency of edges. Moving the balanced-design object from dropout
  edges to batch-pair edges is an application, not a new mechanism.
- Clémençon, Colin, and Bellet, *Scaling-up Empirical Risk Minimization:
  Optimization of Incomplete U-statistics* (JMLR 2016), and Papa, Clémençon,
  and Bellet, *SGD Algorithms based on Incomplete U-statistics* (NeurIPS 2015),
  treat the selection of observed pairs for pairwise ERM, including metric
  learning, as a sampling-design problem.
- Wang et al., *Cross-Batch Memory for Embedding Learning* (CVPR 2020), starts
  from the same finite-batch pair-availability limitation and makes relations
  available across batches through memory.

The narrow residue—carrying exact label-only pair counts across epochs—is not
enough. It is the pair-level version of reshuffling to remove sampling-count
variance, not a new supervision relation. No exact paper using BIBD schedules
for within-class DML groups was found, but novelty by application is below the
project's mechanism-level bar.

## Consequence

Do not calculate pair-coverage imbalance as candidate provenance and do not
implement the sampler. Even a large finite-run imbalance would only quantify
the variance that this occupied assignment/design mechanism controls. The GPU
remains idle for a candidate that survives both provenance and prior art.

Primary sources:

- https://proceedings.mlr.press/v202/sachidananda23a.html
- https://arxiv.org/abs/1809.08404
- https://www.jmlr.org/papers/v17/15-012.html
- https://proceedings.neurips.cc/paper_files/paper/2015/hash/67e103b0761e60683e83c559be18d40c-Abstract.html
- https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html
