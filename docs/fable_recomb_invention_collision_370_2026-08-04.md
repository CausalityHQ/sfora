# Candidate 370: RECOMB invention and exact repeat collision audit

Date: 2026-08-04.

## Frozen proposal

A sixth clean, catalogue-blind `claude-fable-5` pass proposed **RECOMB**. It
would splice stage-3 feature-map regions from two different training classes,
assign the ordered parent-class/mask recipe a hard virtual identity, create two
instances of that virtual identity from four disjoint source images, and use a
contrastive objective with reversed and shared-part recipes as constructed hard
negatives. Auxiliary BatchNorm would isolate composite statistics and all
composite machinery would be discarded at inference.

The frozen prediction was CUB R@1 **0.778** at five seeds with a
ResNet-50/GAP/2048-D/200-epoch recipe. Fable separately estimated only a
35--40% probability of clearing 0.766 with confidence-interval separation.

## Gate 1

The proposal does not have the required positive repository provenance. It
infers "class-boundary scarcity" from cross-dataset class counts and from a
cross-lane CUB comparison that jointly changes architecture, pretraining data,
method, dimension, and recipe. Neither comparison identifies spatial composite
identities as the missing intervention. The claim that `C` labels limit the
encoder to about `log2(C)` useful bits is not an information bound on training:
pretrained features, image-dependent gradients, augmentation, and continuous
within-class inputs remain available.

The repository's closest measured evidence points against the regional arm:
position-tolerant MaxSim recovered 6.67 points relative to a broken
fixed-coordinate regional score, but the trained `region_pa` arm still lost
about **3.6 CUB points** against paired Proxy Anchor. That recovery says spatial
alignment matters inside a poor regional representation; it does not say hard
virtual identities from impossible cross-class composites improve retrieval.

## Gate 2: exact repeat of candidates 210 and 367

RECOMB is the second catalogue-blind Fable pass to rediscover candidate 210's
**binding-error composite supervision**, after candidate 367's Composite Class
Expansion:

- spatially mix content from two training identities;
- treat the composite as neither parent and therefore as a hard synthetic
  identity; and
- train multiple synthetic examples of that identity against other real and
  synthetic identities.

Metrix/CutMix occupy mixed input or feature construction and its supervision.
Proxy Synthesis generates both synthetic embeddings and synthetic proxies that
operate as synthetic classes specifically to mimic unseen classes in
class-disjoint DML. Memory-Based Virtual Classes independently occupies adding
virtual identities to reduce over-focus on seen classes.

RECOMB's "recipe consistency" is the ordinary requirement that two examples of
the same synthetic class be positives. Disjoint source images change how those
examples are sampled. Reversed and shared-part recipes are hard-negative
construction/mining among synthetic classes. A spatial mask, auxiliary BN, and
feature-map rather than embedding interpolation change the generator and
training hygiene. None changes the supervision primitive identified in the
primary-source re-audit: synthesize support from known identities, create a
virtual identity, and optimize it in a metric loss.

The proposal's own literature section found Proxy Synthesis, L2A-NC, MIRAGE,
Memory-Based Virtual Classes, Metrix, CutMix, PatchUp, and negative data
augmentation, but declared survival by requiring the exact conjunction of
sampling and consistency details. Under the protocol's mechanism-level novelty
standard, this is a conjunction fallacy: composing occupied synthetic-class,
mixing, contrastive-positive, and hard-negative operations is not a new source
of supervision.

Primary sources and records:

- Gu, Ko, and Kim, *Proxy Synthesis: Learning with Synthetic Classes for Deep
  Metric Learning*, AAAI 2021: https://arxiv.org/abs/2103.15454
- Venkataramanan et al., *It Takes Two to Tango: Mixup for Deep Metric
  Learning*, ICLR 2022: https://openreview.net/forum?id=1fD8rW-5Et
- `docs/cross_field_candidate_batch_205_210_2026-08-02.md`, candidate 210;
- `docs/proxy_synthesis_primary_reaudit_2026-08-02.md`;
- `docs/fable_cce_invention_collision_367_2026-08-04.md`.

Fable again said VAPNet's mechanism could not be verified. The official
NeurIPS 2023 primary paper is available and already recorded in candidate 367:
https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf

## Verdict

**DEAD at Gates 1 and 2. No diagnostic, implementation, preregistration, or
GPU.** RECOMB is not merely adjacent to candidate 367; it repeats its exact
hard-composite-virtual-class mechanism with a contrastive rather than proxy
parameterization. The independent recurrence is useful evidence that synthetic
compositional classes are an attractor of outcome-only invention, not evidence
that the occupied mechanism becomes novel on repetition.
