# Research agenda beyond the closed operator search

Date: 2026-08-01

## Why broaden the assumptions

The 173-candidate audit closes the combination of training-data-only supervision,
roughly Proxy-Anchor training cost, a single 512-dimensional cosine descriptor,
and standard class labels. Continuing inside that box repeatedly maps new names
to pair weighting, mining, regularisation, distillation, reconstruction, or an
established similarity operator.

This document asks a different question: what knowledge from recent retrieval
and adjacent fields changes the assumptions enough to create real information?

## What recent work establishes

Patel, Tolias, and Matas, *Three Things to Know about Deep Metric Learning*
(2024), identify a strong modern recipe: a large-batch differentiable surrogate
close to Recall@k, similarity-level mixup, and foundation-model initialization.
Their combined recipe nearly saturates popular benchmarks. This means a new
method must be compared with substantially more than ResNet-50 Proxy Anchor.
It also means an apparent gain from a modern encoder is not evidence for a new
supervision mechanism.

LoDisc (Shi et al., 2024) establishes training-data-only global/local
self-supervised pretraining for fine-grained visual recognition. VAPNet and VCE
then occupy learned local attributes/concepts for open-set fine-grained
retrieval. DIVA (ICLR 2025) establishes diffusion feedback as a way to enrich
fine visual detail in a pretrained CLIP encoder. BLENDER uses multiple powerful
external models for fine-grained retrieval. These results jointly close broad
claims based merely on local SSL, visual concepts, or generative feedback.

## Track A: establish the honest empirical frontier

This track is not a novelty claim. It is necessary experimental infrastructure.

1. Reproduce the large-batch differentiable Recall surrogate under the standard
   splits and report memory, wall time, and seeds.
2. Add similarity-level Metrix under the same initialization and evaluation.
3. Compare ImageNet, DINO-family, and CLIP-family initializations, explicitly
   labelling their unknown benchmark-contamination status.
4. Evaluate 512-D and 2048-D heads separately. VAPNet and AdvRF show that the
   general published horizon uses 2048-D/200-epoch recipes; a 512-D result is a
   controlled ablation, not the overall frontier.
5. Preserve raw and selection-corrected reporting and use paired seeds.

This track tells us whether any later mechanism beats a current baseline rather
than a historically convenient one.

## Track B: assumption changes that can add information

### B1. Auditable visual pretraining

Train an object/part-aware visual teacher only on the benchmark training split or
on a corpus whose complete digest and deduplication against benchmark images are
available. External SAM/DINO/CLIP features may be useful, but a region-quality
score is not a contamination audit. A publishable claim needs source provenance,
near-duplicate search, and a matched randomly initialized or ImageNet control.

The research question is not whether a larger teacher helps. It is whether a
specific visual primitive learned without test-image exposure supplies open-set
retrieval information beyond LoDisc/VAPNet/VCE.

### B2. New observations rather than new losses

Collect or expose a relation unavailable in category labels: verified viewpoint,
part correspondence, temporal adjacency, product-instance linkage, or human
fine-grained similarity judgements. Once observed, these can define supervision
that is not derivable from the class-equivalence relation. The price is a new
benchmark claim and annotation budget; this must not be presented as a drop-in
method on the old protocol.

### B3. Transductive retrieval as a separate task

Gallery adaptation, diffusion, query expansion, and test-time memory can exploit
unlabelled test-set structure. They may improve deployed retrieval, but are
transductive and must be reported separately from single-image inductive DML.
A new method here must beat strong diffusion/query-expansion baselines and state
whether gallery updates couple queries.

### B4. Multi-vector deployment

The repository's untrained local MaxSim recovery shows local evidence exists,
while trained regional Proxy Anchor shows naive local supervision destroys it.
Relaxing single-vector deployment permits learned token sets, but ColBERT-style
late interaction, DIML, VCE, local alignment, and multi-vector retrieval are
already adjacent. A credible new contribution would need a new compression or
training mechanism plus latency/storage comparisons, not MaxSim alone.

## Ideas rejected during this broader pass

- Filtering external-teacher regions by transferability does not audit teacher
  contamination and is quality weighting plus distillation.
- Two-stage local SSL followed by identity aggregation is a pipeline combination
  adjacent to LoDisc, VAPNet, and ordinary frozen-pretrainer transfer.
- Gallery memory is query expansion/diffusion/test-time adaptation unless its
  update rule creates a substantively new operation.
- Multiple counterfactual heads are attention diversity or embedding ensembles.
- Same-class cross-image masked prediction is cross-instance masked modelling or
  JEPA-style latent prediction; its semantics also differ between In-Shop
  product instances and CUB/Cars category labels.

## Ranked next actions

1. **Strong-baseline reproduction:** highest information value and required for
   every future performance claim; not novel.
2. **Auditable train-split-only local teacher:** scientifically clean but likely
   expensive; novelty depends on a mechanism beyond existing local SSL and
   concept learning.
3. **New-relation dataset/annotation:** strongest route to truly new supervision,
   but changes the benchmark and requires data work.
4. **Transductive or multi-vector task:** potentially useful engineering result,
   explicitly outside the original inductive single-vector claim.

No GPU candidate is preregistered here. The document prevents a useful recipe or
assumption change from being mislabeled as a novel similarity-learning method.

## Primary sources

- Patel, Tolias, and Matas, *Three Things to Know about Deep Metric Learning*:
  <https://arxiv.org/abs/2412.12432>
- Shi et al., *LoDisc*: <https://arxiv.org/abs/2403.04066>
- Venkataramanan et al., *Metrix*: <https://arxiv.org/abs/2106.04990>
- Wang et al., *DIVA*: <https://proceedings.iclr.cc/paper_files/paper/2025/file/b8d32f60f69cc1fe461be9712af1528d-Paper-Conference.pdf>
