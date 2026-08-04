# Fable two-seed gap candidate batch 348--352

Date: 2026-08-04. Model: `claude-fable-5`, maximum effort, read-only tools,
web disabled, with no model fallback. Primary-source checks below were then
performed independently.

## Verified input packet

The pass received only corrected evidence: two independently verified final
In-Shop Proxy Anchor models, final R@1 **0.9137009425** and **0.9167956112**;
training leave-one-out R@1 about **0.9955** for each; replicated fragmentation
near **39%**; and replicated concentration of the sparse training errors in the
nearest-image/nearest-proxy confusion-agreement stratum. It was instructed that
two seeds do not estimate variance and that the seen/training versus
unseen/test gap is descriptive, not causal.

## Decision

**ZERO SURVIVORS; no implementation, diagnostic, or candidate GPU.** Five
mechanism directions were generated and rejected before proposal. The result
is stronger than “no idea occurred”: each direct response to the verified
premise reduces to an excluded operator or already established method family.

### 348. Unseen-identity relation supervision

Held-out-identity episodes, fold teachers on disjoint classes, or a set encoder
that emits representatives for novel classes all expose class-holdout tasks
during training. That is episodic/meta metric learning, which the search has
already tested or excluded. Deep Meta Metric Learning explicitly samples
support/query subtasks and learns set-based distances, so the mechanism is
occupied ([Chen et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.html)).

### 349. Cross-seed confusion-structure supervision

The prevalence change from 0.1569 to 0.1259 has only two observations and
cannot support a stability mechanism at Gate 1. More fundamentally, a single
model cannot observe cross-seed agreement unless training runs multiple models;
using that information becomes consensus/ensemble relational distillation or
disagreement weighting. Knowledge distillation with pairwise and higher-order
relations is already explicit in JRD ([Chu et al., ICML
2020](https://proceedings.mlr.press/v119/chu20a.html)).

### 350. Confusion-agreement-localized supervision

Selecting, scaling, or suppressing rows where nearest-foreign-image and
nearest-foreign-proxy identities agree is hard-negative mining or weighting.
Adding an equality term between their structures is graph/relational
consistency. The underlying training-error statistic is also outcome-defining:
negative nearest-same minus nearest-foreign margin exactly defines a finite
leave-one-out error. No residual supervision survives the reduction.

### 351. Fragmentation-preserving supervision

Preserving or generating within-class modes is precisely intra-class variation
modelling; assigning those modes representatives is multi-centre learning.
DVML explicitly models class-independent intra-class variation and generates
samples from it ([Lin et al., ECCV
2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html));
DRML explicitly learns intra- and inter-class relational structure ([Zheng et
al., ICCV
2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Deep_Relational_Metric_Learning_ICCV_2021_paper.html)).
The observed fragmentation/R@1 association is weak and causally unidentified,
so repair also lacks Gate-1 support.

### 352. Deployment-population repair

Using the unlabeled query/gallery population at inference is transductive
retrieval, reranking, or test-time adaptation, not label-only inductive
similarity learning. Updating on test inputs is an established operator
([Sun et al., ICML
2020](https://proceedings.mlr.press/v119/sun20b.html)); aggregation or
abstention changes the deployment claim. It therefore cannot fill the standing
single-vector inductive objective.

## Closure and reopening condition

The two verified seeds improve the reliability boundary and now show that most
official-query errors are stable across initialization. They still provide no
new training observation channel. A new candidate requires either (a) a
validated observation/annotation channel shared by at least two target
benchmarks whose gradient does not reduce to weighting, mining, regularization,
metric substitution, multi-centre learning, meta-learning, or distillation; or
(b) an explicitly disclosed change of task. No method is preregisterable from
this batch.

