# Pass 110: current-literature escape-route audit (no GPU)

This batch was checked after Pass109 collapsed. It is a prior-art boundary
record, not a claim that the papers reproduce our corrected baseline.

| Proposed escape route | Primary source | Gate-2 disposition |
| --- | --- | --- |
| Projected positive-curvature/hypersphere embedding | *Deep metric learning in projected-hypersphere space*, 2025, DOI/ScienceDirect record; evaluates CUB, Cars, SOP, In-Shop | **DEAD.** The exact geometry change and benchmark family are occupied. |
| Incremental margin plus feature-map standard-deviation objective | Winston & Kang, *IMSDO*, Neurocomputing 655 (2025), DOI 10.1016/j.neucom.2025.131376; evaluates all four retrieval datasets | **DEAD.** Dynamic margin/curriculum and variance regulation are explicit prior art. |
| Embedding Space Augmentation with hard-sample confidence and principal-direction updates | Park et al., *Rethinking Metric Learning: Enhancing Generalization to Unseen Classes*, IEEE Access 13 (2025), DOI 10.1109/ACCESS.2025.3637551 | **DEAD.** It directly occupies the measured unseen-class generalization/variance route. |
| Shadow scalar-projection loss | Khan et al., *Shadow Loss: Memory-linear Deep Metric Learning with Anchor Projection*, withdrawn ICLR 2026 submission, OpenReview 3fx0Kz6Zfl | **DEAD for novelty.** It is a public prior claim on CUB, Cars, SOP, and In-Shop; a replication could be a baseline, not our new method. |
| Hyperbolic hierarchical ranking | *Hierarchical ranking in hyperbolic space*, 2026, PubMed record | **DEAD.** Hyperbolic/hierarchical DML is an occupied similarity geometry and the method is already benchmark-matched. |
| Training-free retrieval/refinement pipelines | WISER, CVPR 2026 | **OUT OF SCOPE.** It uses multi-stage retrieval/refinement rather than one train-time fixed descriptor. |

The repeated pattern is now explicit: changing metric curvature, margin
schedule, feature variance, projection geometry, or ranking aggregation does not
create an unoccupied supervision primitive. No implementation or GPU run was
authorized from this batch. The next candidate must introduce a measured
data-derived relation not already present in the repository's failure catalogue;
otherwise the defensible result is a mechanism-level negative, not another
underpowered screen.
