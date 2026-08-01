# Evidence-bounded stopping audit

Date: 2026-08-01

## Scope of the claim

This audit asks whether the current iterative search can name a method that
simultaneously has (1) provenance in a measurement from this repository, (2) a
defensible new supervision operator after primary-source review, (3) no
external pretrained generator or text model, and (4) roughly baseline training
cost. It does **not** claim that no future similarity-learning method can
exist. It claims that the present evidence does not justify another GPU arm
without weakening one of the registered gates.

## Empirical closure

The search produced several real measurements, not a collection of arbitrary
ideas:

- Weight averaging improved CUB by 0.414 point but failed to replicate on
  In-Shop over three paired seeds (mean +0.070 point, sd 0.166). Dual-timescale
  EMA then failed its own additivity prediction.
- RSPG found that target-excluded rival signatures retained 64.49% of
  same-class pairs on CUB but only 8.63% on In-Shop. The signal is
  dataset-dependent, and its positive-to-unknown operator erased the trained
  geometry.
- ARCG found an augmentation-response graph with 36.31% density and substantial
  independence from ordinary distance. Its selected edges were already
  satisfied, while replacing ordinary positive attraction was destructive.
- IPSR found stable intervention strata, but their order did not predict a
  useful retrieval objective.
- Five independent CUB teachers agreed strongly on same-class geometry
  (Pearson 0.9098) and less strongly across classes (0.8127). The closed
  image-by-image cross-class interaction was reproducible (Pearson 0.5710) but
  only 4.75% of cross-class variance.
- TIRD isolated that interaction and still failed catastrophically on In-Shop:
  raw R@1 0.8301 and selection-corrected 0.8267 versus paired Proxy Anchor
  0.9024. Normalizing a small residual promoted it to unit-scale pressure.

These results close the tempting post-hoc repairs. Reweighting RSPG/ARCG edges
is established pair weighting; complement attraction is hard-positive mining;
rescaling TIRD is Gram-loss weighting; selecting stable tetrads is ordinal
relation mining.

## Operator coverage

The numbered verdict catalogue now reaches candidate 68. Its mechanisms cover
the executable ways found to alter supervision from training data alone:

| Operator family | Repository evidence | Nearest established operator |
| --- | --- | --- |
| Split a labelled class into modes | sub-centre arm failed | SoftTriple and sub-centre classifiers |
| Grade, select, or abstain on labelled pairs | RSPG/ARCG failed; ACPC descriptor adds no new relation | Beyond Binary Supervision, General Pair Weighting, AdaSP, HAP2S |
| Synthesize or interpolate same-class support | no measured provenance beyond ordinary augmentation | Metrix, Embedding Expansion, DVML, PartMix |
| Add local/part correspondence | no unresolved local-frame identification | DIML, weakly supervised alignment, part-aware re-ID |
| Condition on cross-class context | RSPG is highly dataset-dependent | contextual-similarity optimization/distillation, hierarchical proxies |
| Transfer teacher geometry | TIRD failed; residual energy only 4.75% | RKD, SPKD, differential/ranking KD, Gram transfer |
| Add higher-order or global set structure | HIST is already the benchmark target | HIST hypergraphs, facility-location DML, batch optimal transport |
| Model density, uncertainty, or distributions | no repository defect outside weighting/multi-centre geometry | density-aware DML, chance-constrained DML, DVML |
| Use temporal, multi-view, or intervention stability | consensus and response candidates failed or were occupied | multi-teacher agreement, cross-view consistency, AugSelf/InstaAug |
| Import semantic evidence | potentially useful but outside the clean-data constraint | BLenDeR and language/generative guidance |

## Final adversarial literature pass

The last pass deliberately searched outside pair/triplet losses for operator
families that the catalogue might have missed:

- Song et al., *Deep Metric Learning via Facility Location* (CVPR 2017),
  https://openaccess.thecvf.com/content_cvpr_2017/html/Song_Deep_Metric_Learning_CVPR_2017_paper.html,
  already optimizes global clustering structure on CUB, Cars196, and SOP.
- Lim et al., *Hypergraph-Induced Semantic Tuplet Loss* (CVPR 2022),
  https://openaccess.thecvf.com/content/CVPR2022/html/Lim_Hypergraph-Induced_Semantic_Tuplet_Loss_for_Deep_Metric_Learning_CVPR_2022_paper.html,
  already makes multilateral sample-to-class hyperedges the supervision object;
  it is the HIST baseline this project is trying to beat.
- Xu et al., *Learning With Batch-Wise Optimal Transport Loss for 3D Shape
  Recognition* (CVPR 2019),
  https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Learning_With_Batch-Wise_Optimal_Transport_Loss_for_3D_Shape_Recognition_CVPR_2019_paper.html,
  already uses batch-level optimal transport to allocate metric pressure.
- Ghosh et al., *On Learning Density Aware Embeddings* (CVPR 2019),
  https://openaccess.thecvf.com/content_CVPR_2019/html/Ghosh_On_Learning_Density_Aware_Embeddings_CVPR_2019_paper.html,
  and Gurbuz et al., *Deep Metric Learning With Chance Constraints* (WACV
  2024),
  https://openaccess.thecvf.com/content/WACV2024/html/Gurbuz_Deep_Metric_Learning_With_Chance_Constraints_WACV_2024_paper.html,
  occupy density, coverage-radius, and feasibility formulations.
- Lin et al., *Deep Variational Metric Learning* (ECCV 2018),
  https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html,
  already models class-independent intra-class variation and generates support,
  evaluated on CUB, Cars196, and SOP.

None supplies a new repo-motivated candidate; instead they close the remaining
global, hypergraph, transport, probabilistic, and distributional branches.

## Post-audit horizon correction and independent challenge

A 2024--2026 primary-source scan corrected one premise without reopening a
method: PFML (Bhatnagar and Ahuja, CVPR 2025) reports five-run, single-view
ResNet-50 results of 0.734 on CUB and 0.927 on Cars196. The earlier claim that
the CUB ceiling above roughly 0.715 was unoccupied is withdrawn. This makes the
novelty/performance target harder; it does not supply a new operator. This
repository already contains a faithful, unit-tested PFML implementation whose
reproduction collapsed to R@1 0.0155, so an unregistered PFML In-Shop run is
not a novelty experiment.

Two additional read-only Claude reviews were given the complete failure
catalogue and instructed to attack the stopping conclusion. The successful
review returned `NONE`. Its three attempted escapes—magnitude-conditioned
margin, proxy-level gradient surgery, and proxy-quadruple Gram transfer—were,
respectively, occupied by MagFace/AdaFace, PCGrad-family optimization, and
similarity-preserving relational KD. Its only genuinely absent raw observable
was pre-normalisation embedding magnitude, but no DML checkpoints were retained
and its obvious use is precisely occupied quality/hardness weighting. A second
Opus review timed out without a verdict and is not counted as evidence.

## Decision

There is no defensible candidate left under the current constraints. Every
candidate that survived provenance either failed a registered experiment or
reduced, before GPU use, to an established operator with a new descriptor,
mask, normalization, or application. Starting another arm now would violate
Gate 1 or Gate 2 and would make the search less rigorous, not more persistent.

The search should reopen only on new evidence that identifies an operation not
represented above—for example, a new annotation-free observable that creates
a new supervision object rather than selecting or weighting known labels—or if the project
explicitly relaxes the no-external-knowledge or roughly-1x-cost constraint.
Until then, the negative catalogue and its cross-dataset measurements are the
research result.
