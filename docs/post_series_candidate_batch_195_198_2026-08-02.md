# Post-series candidate batch 195--198: Gate-2 audit

Date: 2026-08-02. This shortlist was generated from the completed In-Shop
measurements, not from analogy: stable components align with filename series at
ARI 0.754--0.761, and fragmented identities retain +5.48--+5.74 R@1 points after
exact series-size and geometry matching. No candidate below was implemented or
run on GPU.

## 195. Cross-series bridge obligation — DEAD

**Proposed relation.** Within an identity, make pairs from different image
series the privileged positive obligations while same-series pairs receive only
ordinary identity supervision. This tries to bridge colourway/acquisition modes
without collapsing every local neighbour.

**Why it is occupied.** The identity label already makes every cross-series pair
positive; privileging a subset changes mining or weighting, not what supervision
exists. Camera-aware person Re-ID has the same group-labelled operator at scale:
Wu et al., [*Unsupervised Person Re-Identification by Camera-Aware Similarity
Consistency Learning*](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Person_Re-Identification_by_Camera-Aware_Similarity_Consistency_Learning_ICCV_2019_paper.html)
explicitly aligns intra- and cross-camera similarity, while Lee et al.,
[*Camera-Driven Representation Learning for Unsupervised Domain Adaptive Person
Re-identification*](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Camera-Driven_Representation_Learning_for_Unsupervised_Domain_Adaptive_Person_Re-identification_ICCV_2023_paper.html)
encourages cross-camera diversity within an identity. Replacing camera with
filename series is a new group estimator, not a new supervision relation.

## 196. Series child / identity parent equivalence — DEAD

**Proposed relation.** Treat each series as a child equivalence class while
retaining the item identity as a parent, so local series structure and global
membership coexist.

**Why it is occupied.** This is hierarchical/subcentre metric learning exactly.
SoftTriple explicitly assigns multiple centres per class to represent local
clusters ([Qian et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.html)).
Hierarchical Proxy-Based Loss is evaluated on In-Shop and represents subordinate
and shared proxy structure ([Yang et al., WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Yang_Hierarchical_Proxy-Based_Loss_for_Deep_Metric_Learning_WACV_2022_paper.html)).
HIER learns latent hierarchy beyond class labels ([Kim et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.html)).
Observed series labels make estimation easier but do not alter the operator.

## 197. Series-nuisance quotient — DEAD

**Proposed relation.** Make item identity predictable while making series
unpredictable, or explicitly align the per-series conditional distributions, so
the embedding retains design and removes acquisition/colourway variation.

**Why it is occupied.** This is group/domain invariance. Camera-aware adversarial
domain adaptation already removes camera subdomains ([Qi et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.html));
CamStyle treats camera style as the same nuisance through augmentation
([Zhong et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhong_Camera_Style_Adaptation_CVPR_2018_paper.html)).
Moreover, the repo's observation points in the opposite causal direction:
preserved series separation correlates positively with retrieval, so quotienting
it is not measurement-motivated beyond a control.

## 198. Leave-one-series-out retrieval meta-objective — DEAD

**Proposed relation.** Within each identity, use some series as support and a
held-out series as query, optimising retrieval across that split so success
cannot rely on within-series neighbours.

**Why it is occupied.** The relation remains identity equivalence; only the
episode/sampling and outer objective change. Deep Meta Metric Learning already
learns set-based support/query metrics across sampled subsets
([Chen et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.html)).
Worst-series variants are group distributionally robust optimisation, whose
objective is explicitly minimax risk over prespecified groups (for a primary
modern formulation, [Awasthi et al., ALT 2024](https://proceedings.mlr.press/v237/awasthi24a.html)).
Neither changes the supervision relation, and both exceed a simple novel DML
claim.

## Verdict

**NONE survives Gate 2.** The measurements expose a real and unusually stable
dataset structure, but every direct edit lands in positive mining/weighting,
hierarchical multicentre DML, group-invariant representation learning, or
episodic/group-robust optimisation. The series token is also In-Shop-specific,
so none supplies the required second-dataset mechanism. The faithful Cars RS@k
run remains the next independent measurement source.
