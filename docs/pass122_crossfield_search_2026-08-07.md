# Pass 122 — cross-field search while ECT-R runs (2026-08-07)

This is an offline Gate-2 search only.  The Pass-119 ECT-R deciding run is
still the sole active GPU job, so this note does not authorize another run.

## Measurement anchor and candidate

The retained CUB decomposition says that 48.1% of failed queries have a
nearer own-class centroid but the wrong nearest image.  That is evidence for a
local-evidence failure, but it does **not** by itself show that an attribution
map is a useful pairwise signal.  A candidate motivated by that observation is
**counterfactual-evidence agreement (CEA)**: for two different images of the
same class, compute the class-score drop caused by masking each spatial cell;
make the pair positive-to-unknown unless their top evidence cells agree.  The
claim is a discrete cross-instance supervision gate, not a saliency auxiliary
loss or an inference-time explanation.

CEA is **Gate 1 unresolved**.  Before any implementation, a frozen operating-
point checkpoint must provide pair-level evidence: close same-class pairs with
agreement, close pairs with disagreement, and distant pairs with agreement.
The preregistered CPU test is a held-out pairwise association test against the
nearest-neighbour correctness label; failure to beat a distance-only gate by
0.05 AUC (or fewer than 5% usable pairs) kills the candidate.  No checkpoint
currently exists that can answer this honestly, so no number is invented.

## Adversarial Gate-2 triage

* **Adaptive assessor / meta-sampling** is dead at Gate 2.  DML-ALA (Zheng,
  Lu, and Zhou, CVPR 2020) already learns a sequence-aware assessor by a
  train/validation meta-objective.  A CEA-weighted sampler would change the
  assessor's input, not the supervision object.
* **Environment-game metric learning** is dead at Gate 2.  IRM and IRM games
  (Arjovsky et al.; Ahuja et al., ICML 2020) already use environment-wise
  invariance/game objectives.  Splitting augmentations into environments and
  applying the same risk game is an optimizer/objective transplant.
* **CEA remains LIVE-NARROW, not novel.**  Counterfactual Attention Learning
  (Rao et al., ICCV 2021) uses intervention-derived attention as a training
  signal, but the checked primary description does not gate supervision
  between two different same-class images.  That distinction may survive, but
  it must be tested at Gate 1 and then re-audited against any pairwise
  attribution or saliency metric-learning paper before GPU use.

Primary sources checked:

* Zheng et al., *Deep Metric Learning via Adaptive Learnable Assessment*, CVPR
  2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html
* Ahuja et al., *Invariant Risk Minimization Games*, ICML 2020:
  https://proceedings.mlr.press/v119/ahuja20a.html
* Rao et al., *Counterfactual Attention Learning for Fine-Grained Visual
  Categorization and Re-Identification*, ICCV 2021:
  https://openaccess.thecvf.com/content/ICCV2021/html/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.html

## Decision

Do not implement CEA or launch a screen from this note.  The queued CIS/SRC
path remains the next executable candidate after ECT-R's corrected random
control and selection analysis.  If CIS/SRC close, CEA's only defensible next
step is a cheap operating-point diagnostic using a trained checkpoint; a
one-step random-head export is explicitly invalid because it measures a
different representation.
