# Fable deployment and quality candidate batch 341--346

Date: 2026-08-03. Status: **ALL SIX DEAD BEFORE IMPLEMENTATION OR CANDIDATE
GPU.**

## Model and evidence control

The deciding Claude process was restarted with both model and fallback pinned to
`claude-fable-5`, maximum effort, safe mode, and only read-only repository tools.
Web, shell, write, edit, notebook, and subagent tools were disabled. Every
substantive assistant record identified `claude-fable-5`; no fallback event
occurred. A small Haiku bookkeeping entry in the CLI usage record was not an
assistant answer and contributed no candidate content. The main agent checked
the resulting mechanism verdicts independently against primary sources.

Only the corrected evidence packet in audit 321 and subsequently verified
artifacts were supplied. In particular, legacy benchmark numbers, wrong-corpus
In-Shop runs, selected-only checkpoints, and model-ambiguous results were not
accepted as provenance.

## 341. Identity-set gallery deployment

**Proposal.** Aggregate all gallery images of a known In-Shop identity at
deployment and compare a query with the identity set rather than individual
images.

**Verdict: DEAD at Gates 1 and 2.** Taking the maximum query--set similarity and
then selecting the best identity is algebraically identical to ordinary
nearest-image top-1. Nontrivial learned or attention aggregation changes the
deployment object to a multi-image identity template and is established by set
and template aggregation, including Neural Aggregation Networks. It cannot be
claimed as a single-image DML method.

## 342. Train-calibrated selective retrieval

**Proposal.** Abstain on uncertain queries using a threshold calibrated from
corrected train leave-one-out margins.

**Verdict: DEAD at Gates 1 and 2.** The corrected train leave-one-out error is
0.4984%, while the unseen-identity test error is 8.630%; the seen-identity
calibration population does not estimate deployed error. Selective retrieval,
risk--coverage control, and uncertainty-aware retrieval are established. Risk
Controlled Image Retrieval moreover evaluates Cars196 and CUB explicitly. This
is a deployment guarantee problem rather than a better embedding mechanism.

## 343. Duplicate-conflict robust supervision

**Proposal.** Detect content-equivalent samples with inconsistent identities
and suppress or relabel their training relations.

**Verdict: DEAD at Gates 1 and 2.** The verified near-duplicate audit found only
one cross-identity pair responsible for 2 / 129 corrected In-Shop train
leave-one-out errors, below the locked materiality threshold. Noise-resistant
DML and ranking-based instance selection are occupied by PRISM and related
methods. Editing evaluation labels would repair the benchmark rather than
improve similarity learning.

## 344. Paired-resolution intervention

**Proposal.** Use each image's response to a paired resolution degradation as
new supervision for resolution-stable retrieval.

**Verdict: DEAD at Gates 1 and 2.** No verified packet measurement attributes
error to resolution. The operator is already augmentation-response/self-
distillation or resolution-asymmetry training, covered by S2SD, AugSelf, and
Large-to-Small Resolution Asymmetry in Deep Metric Learning. A new transform
does not create a new supervision primitive.

## 345. Acquisition-randomized supervision

**Proposal.** Treat acquisition variables as randomized interventions and learn
an identity representation invariant to them.

**Verdict: DEAD at Gates 1 and 2.** The verified packet contains no identified
acquisition--error association, and no matched acquisition variable exists
across the accepted datasets. Once specified, the mechanism is domain/nuisance
invariance, augmentation consistency, or controlled ablation, all previously
occupied. The causal language does not supply randomization that the data lack.

## 346. Declared ensemble or artifact pack

**Proposal.** Combine several independently trained embeddings or expose the
artifact pack as the deliverable rather than proposing another training loss.

**Verdict: DEAD as a novelty candidate.** Ensembling is established, changes
deployment cost, and does not define a novel similarity-learning mechanism.
Several historical artifacts are also quarantined, so a heterogeneous pack
would mix incomparable corpora and selection states. It may improve quality as
an explicitly declared system, but cannot satisfy this project's novelty goal.

## Correction: shared official annotation channels do exist

Candidate 339 said CUB attributes/parts lacked a matched second-dataset channel.
That factual premise was too strong. The official CUB distribution includes
bounding boxes, part locations, and attributes. The official DeepFashion
distribution includes bounding boxes, clothing landmarks, and attributes, and
the corrected In-Shop tree already contains per-image boxes and pose types.
Thus CUB and In-Shop share broad **attribute** and **landmark/part** channel
types, although their vocabularies and semantics are not aligned.

This correction does not revive the obvious annotation-assisted candidate.
[DeepFashion](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html)
already jointly predicts clothing attributes and landmarks with a triplet
retrieval loss and uses estimated landmarks to pool or gate features. Attribute-
specific similarity, structured/continuous DML supervision, part-aware
retrieval, and learning using privileged information are independently
established. Using the files also changes the ordinary class-label-only
benchmark supervision budget and the channel meanings do not transfer directly
between birds and clothes.

The corrected census therefore refutes a stopping-premise detail but produces
no live mechanism. A future annotation-assisted proposal must specify a new
operator beyond auxiliary prediction, feature gating, pair grading, or
distillation; bind the exact official files and train-only use; disclose the
extra supervision; and replicate across at least two datasets despite their
non-aligned annotation ontologies. No such proposal survived this batch.

## Batch verdict

No candidate reaches preregistration. The useful outputs are negative: gallery
max aggregation is an identity, train-to-test calibration is population-
mismatched, duplicate conflicts are too rare, resolution and acquisition ideas
lack measured provenance and collide with occupied operators, and ensembles do
not satisfy novelty. The annotation census is corrected, while its most direct
use is already present in the primary literature.
