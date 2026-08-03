# Candidate 314: transition-state exclusion

RSPG's failure suggested that attracting sparse positives can erase the base
signal. The proposed repair was to supervise same-class interpolation arcs only
by excluding foreign-class intrusions, without adding endpoint attraction.

**Gate 2: dead.** LoOp (Vasudeva et al., ICCV 2021) searches for hard-negative
embeddings in precisely the region between positive endpoints, and interpolation
based metric-learning methods use the same transition-region constraints.
Changing the exclusion from a point to an interpolation arc is a mining/negative
constraint variant, not a new supervision primitive. No implementation or GPU.

Primary source: https://openaccess.thecvf.com/content/ICCV2021/papers/Vasudeva_LoOp_Looking_for_Optimal_Hard_Negative_Embeddings_for_Deep_Metric_ICCV_2021_paper.pdf
