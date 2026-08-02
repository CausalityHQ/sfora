# DAMLRRM primary-source re-audit

Date: 2026-08-02. Purpose: challenge a load-bearing prior-art closure after 210
candidate failures, not confirm it by assumption.

## Verdict

**Closure confirmed.** Xu, Yang, Deng, and Zheng, *Deep Asymmetric Metric
Learning via Rich Relationship Mining* (CVPR 2019), explicitly rejects
attracting every possible labelled positive pair. Within each category it
constructs a minimum-cost spanning tree so arbitrary positive samples have at
least one direct or indirect path, while only selected tree edges receive the
direct relation. It evaluates the mechanism on CUB-200-2011, Cars196, and SOP.

This is the exact supervision primitive for which the repository cites it:

- selecting sparse within-class positive edges while preserving connectivity;
- replacing all-pairs adjacency by existential transitive connectivity;
- deriving the selected graph from visual distance.

Therefore candidates 32, 68, 189, and 194 remain closed. Changing the edge
estimator from learned embedding distance to a model-free pixel or regional
descriptor can change robustness or cost, but it does not create a new
supervision mechanism. A soft relaxation becomes pair weighting or continuous
similarity supervision instead.

The audit also checked that the paper is not merely test-time reranking: the
minimum-spanning-tree relations are used during metric training, alongside its
asymmetric data streams. This removes the only distinction that could have
reopened the route.

Primary source:
https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.html
