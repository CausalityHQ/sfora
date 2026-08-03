# Verified-evidence Fable candidate batch 336--340

Date: 2026-08-03. Status: **SHORTLIST REPORTED; ALL FIVE DEAD BEFORE
IMPLEMENTATION OR CANDIDATE GPU.**

## Model and evidence control

The first attempted Claude run was rejected because its transcript emitted a
`model_refusal_fallback` event when it requested web tools. It was stopped and
none of its content was used. The deciding pass was launched in safe mode with
model and fallback both pinned to `claude-fable-5`, only `Read`, `Grep`, and
`Glob` available, and no web or subagent tool. Every assistant-message record
identified `claude-fable-5`; no fallback event occurred. The main agent then
performed the primary-source search independently.

Fable was restricted to the digest-bound corrected In-Shop artifacts, the
byte-bound CUB audit, the pinned Cars corpus, corrected RSPG measurements, and
the reliability boundary in audits 321--335. It returned no live mechanism.

## 336. Cross-fold unseen-structure target

**Provenance.** Corrected In-Shop train leave-one-out error is 129 / 25,882
(0.4984%), while the final unseen-identity test error is 8.630%. The gap says
that fit diagnostics on seen identities do not measure the deployed problem.

**Proposal.** Train fold-specific teachers on disjoint class subsets and make a
student reproduce each teacher's relations on classes unseen by that teacher.

**Verdict: DEAD at Gate 2.** The target is a Gram matrix from an auxiliary
model; the executable operation is relational distillation under class-disjoint
episodes. Candidates 49--51, 61, and 228 already close those mechanisms.
Independent primary-source checks also find explicit class-disjoint episodic
metric learning in [Jung et al., ACCV
2022](https://openaccess.thecvf.com/content/ACCV2022/papers/Jung_Few-Shot_Metric_Learning_Online_Adaptation_of_Embedding_for_Retrieval_ACCV_2022_paper.pdf)
and set-based meta metric learning in [Chen et al., ICCV
2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.html).
The 17.3x error ratio is descriptive, not an effect estimate: train and test
identities have different exposure by construction.

## 337. Jackknife self-influence-free positive force

**Provenance.** All 117 corrected In-Shop leave-one-out errors with a defined
same-class margin have negative nearest-same minus nearest-foreign margin.

**Proposal.** Attract each image to a fold proxy or influence-corrected proxy
whose estimate excludes that image's own training contribution.

**Verdict: DEAD at Gates 1 and 2.** The error/margin statement is definitional,
not causal evidence for self-influence. The edit changes the estimator of the
same image-to-proxy relation and reduces to replica proxies, influence
estimation, or a control variate, already closed by candidates 58, 135, 224,
281, and 327. No new supervision relation enters.

## 338. Proxy radial-gauge optimizer quotient

**Provenance.** Normalized proxies have loss-invisible radial coordinates while
ordinary Adam and coupled decay can retain radial optimizer state.

**Proposal.** Constrain proxies to the unit sphere and transport Adam moments in
the tangent space.

**Verdict: DEAD at Gate 1 and occupied at Gate 2.** No verified measurement
shows material proxy-norm or effective-learning-rate dispersion. More
decisively, tangent/manifold adaptive optimization is established by
[Becigneul and Ganea, ICLR
2019](https://openreview.net/forum?id=r1eiqi09K7); gauge fixing was already
closed by Hoffer and Pernici, and candidates 229 and 310 close the surrounding
space-tuning family. This is optimizer geometry, not new similarity
supervision.

## 339. Official CUB attribute/part channel

**Provenance.** CUB's official distribution contains human attribute and part
annotations in addition to the image, label, and bounding-box fields bound by
the corpus audit. Those extra annotation files have not themselves been
independently byte/semantic-verified in this repository.

**Proposal.** Use train-class attributes or part visibility to define graded
pairs or part-conditioned positives, retaining label-only inference.

**Verdict: DEAD at Gate 2 and outside the benchmark-matched claim.** This is a
real new annotation channel, but the operations are established attribute-
specific similarity, graded supervision, and part-structured matching. [Zhao et
al., ICCV
2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.html)
already evaluates structural matching on CUB, Cars, and SOP; [Kim et al., CVPR
2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Deep_Metric_Learning_Beyond_Binary_Supervision_CVPR_2019_paper.html)
establish continuous/structured supervision for DML. It is also not comparable
to label-only baselines and supplies no matched second-dataset channel under
Gate 7.

## 340. Pair-margin co-fluctuation graph

**Provenance.** The only eligible temporal measurement reports 64.99% nearest-
same persistence and final errors of 0.4518% versus 0.5849% for persistent and
changing rows, a 0.133-point single-trajectory difference.

**Proposal.** Prospectively log fixed-pair margins and use correlated SGD
fluctuations as a pair-interaction graph.

**Verdict: DEAD at Gates 1 and 2.** The measured association is small and from
one trajectory. Consuming the graph is still graph consistency, mining, or
weighting; estimating it from training dynamics is influence/forgetting-event
tracking. Candidates 128, 214, 317, and 319 already bracket these variants. No
logging rerun is justified.

## Batch verdict and reopening boundary

No candidate survives to preregistration. The batch adds one useful distinction:
CUB attributes/parts are a genuinely new information channel, but using them is
an annotation-assisted task change with occupied operators, not a label-only
DML novelty result. The other four proposals remain functions of model
geometry, optimizer state, or trajectory and reduce to already catalogued
families.

The evidence therefore supports infrastructure work and a possible
**evidence-quality contribution** (correct corpus, artifact-preserved final
states, multi-seed uncertainty), but not a novel mechanism claim. Reopening the
mechanism search requires a validated new observation/annotation channel shared
by at least two benchmarks, or an explicit deployment/task change. Neither is
currently present.
