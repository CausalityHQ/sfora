# Candidates 216--217: topology and metric compatibility

Date: 2026-08-02. No diagnostic, implementation, or GPU was used.

## 216. Raw-input persistent topology alignment

**DEAD at Gate 1 and Gate 2.** The proposed provenance was In-Shop's stable
within-class embedding partition (mean cross-seed ARI **0.84148**). That
measurement does not observe raw-pixel topology. The same components align with
filename acquisition series at ARI **0.754--0.761** and with simple view
descriptors at about **-0.142**. Preserving their input topology would therefore
risk preserving acquisition/colorway structure; no repository measurement
identifies the intervention's direction as beneficial.

The exact operator is established:

- Moor et al., *Topological Autoencoders* (ICML 2020),
  <https://arxiv.org/abs/1906.00722>, aligns input- and latent-space persistent
  topology as representation supervision.
- Dan et al., *TopoFR: A Closer Look at Topology Alignment on Face Recognition*
  (NeurIPS 2024), <https://arxiv.org/abs/2410.10587>, uses perturbation-guided
  persistent-homology alignment specifically to improve unseen-benchmark face
  generalization, plus uncertainty-based hard-sample weighting.
- Li et al., *Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent
  Homology for Dataset Distillation* (arXiv:2602.24144, 2026),
  <https://arxiv.org/abs/2602.24144>, aligns persistent images of real and
  synthetic feature graphs. No conference venue was verified; the earlier
  catalogue attribution to CVPR 2026 was corrected.

Candidates 117 and 143 already close persistence-diagram pair eligibility and
class-cloud topology regularization. Changing the source space to raw pixels is
an estimator/application change, not a new supervision object. Fixed raw-input
topology would also require an extra unaugmented pass under the repository's
random-crop recipe, violating the roughly-1x constraint before filtration cost.

## 217. Cross-trajectory metric compatibility

**DEAD at Gate 2 with a causal-target defect.** The motivating fixed-seed
**1.08 R@1-point** spread establishes nonlinear trajectory divergence from GPU
nondeterminism. It does not identify distance-scale mismatch. R@1 is invariant
to monotone rescaling, so scale interoperability cannot directly stabilize the
measured outcome. Matching ranks instead is relational distillation.

The complete compatibility action is established:

- Shen et al., *Towards Backward-Compatible Representation Learning* (CVPR
  2020), <https://arxiv.org/abs/2003.11942>;
- Meng et al., *Learning Compatible Embeddings* (ICCV 2021),
  <https://arxiv.org/abs/2108.01958>, including direct, forward, backward, and
  cross-model compatibility through class-center alignment and compactness;
- Ramanujan et al., *Forward Compatible Training for Large-Scale Embedding
  Retrieval Systems* (CVPR 2022), <https://arxiv.org/abs/2112.02805>;
- Zhou et al., *BT2: Backward-Compatible Training with Basis Transformation*
  (ICCV 2023), <https://arxiv.org/abs/2211.03989>;
- Seo et al., *Metric Compatible Training for Online Backfilling in Large-Scale
  Retrieval* (WACV 2025), <https://arxiv.org/abs/2301.03767>.

Changing the purpose from deployment compatibility to generalization leaves
the observed variables and alignment loss unchanged. A second head is roughly
2x and repeats cross-model consistency/co-teaching; an earlier checkpoint is
temporal self-distillation or EMA. Candidate 134 already rejected two-head
consensus from the same 1.08-point measurement. Candidate 217 has no distinct
supervision object and no identified causal route to R@1.
