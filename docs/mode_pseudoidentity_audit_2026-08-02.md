# Candidate 185: mode pseudo-identities — Gate-2 death

Checked before the independent fragmentation replication produced either seed.
No candidate implementation or GPU screen was run.

## Provenance and proposed causal edit

At the seed-0 In-Shop epoch-10 operating point, disconnected within-class 1-NN
graphs were associated with **+3.534 class-balanced leave-one-out R@1 points**
after exact class-size matching. Candidate 185 proposed clustering each training
identity's early embedding graph, assigning each component a separate
pseudo-identity, and training Proxy Anchor on those component labels. Unlike a
soft diversity penalty, this would explicitly change the labels: different
modes of one annotated identity would repel one another as different classes.

## Gate 2

**DEAD.** The operator is supervised class splitting by clustering, followed by
ordinary proxy classification on pseudo-labels. Clustering-based pseudo-label
metric learning is established, and two closer supervised DML neighbours occupy
the intended mechanism:

- Levi et al., *Reducing Class Collapse in Metric Learning with Easy Positive
  Sampling* (ICLR 2021, <https://openreview.net/forum?id=QQzomPbSV7q>) preserve
  multiple within-class subclusters by connecting each anchor only to its
  nearest same-class positive;
- Sanakoyeu et al., *Divide and Conquer the Embedding Space for Metric Learning*
  (CVPR 2019,
  <https://openaccess.thecvf.com/content_CVPR_2019/html/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.html>)
  jointly partition data and embedding dimensions into smaller metric-learning
  subproblems.

Subcentres and pseudo-label refinement occupy the remaining variants. The only
literal distinction—repelling components that share a ground-truth identity—is
not new information; it knowingly injects false-negative labels determined by
the current embedding. A parent-identity loss added back would turn the method
into hierarchical/subcentre DML. Therefore even a clean replication of the
fragmentation marker cannot authorize this operator.
