# Verdict on the method search (2026-07-30)

Fifteen candidates, fifteen failures. Before starting a sixteenth, the whole search was
put to an independent adversarial review (Codex, read-only, literature-grounded, asked
explicitly to argue the strongest case for **stopping**). This records what came back,
including the parts that correct claims made elsewhere in this repository.

---

## 1. Two corrections to our own claims

**H3 is not a novel technique.** Cai et al., *Exponential Moving Average Normalization
for Self-Supervised and Semi-Supervised Learning* (CVPR 2021), identify the same
teacher/student BatchNorm mismatch and propose EMA-ing the normalisation statistics —
which is exactly `ema_teacher_ema_buffers`. We rediscovered EMAN independently. The
result survives as an **audit** finding (the defect is live in DML momentum-teacher
code; here is its measured cost across two bases and six paired seeds), not as a method.
`docs/HANDOFF.md` §3 and `docs/results.md` now say so at the point of claim.

**The GPA fusion number is not a win.** 0.7490 at 512-d is *transductive* — the
Procrustes rotations are fit on test-set geometry. The honest inductive figure, frozen
on the disjoint train split, is +1.4 pt at 3× training cost. That is an ensemble result,
not algorithmic headroom.

## 2. My three candidates, all dead

| candidate | verdict | why |
| --- | --- | --- |
| **Gauge-fixing the embedding frame** | already done, *and* wrong on mechanism | Fixed reference frames exist as fixed classifiers (Hoffer et al., *Fix Your Classifier*, ICLR 2018; Pernici et al., *Regular Polytope Networks*, TNNLS 2022). More importantly the single-model claim is wrong on first principles: gauge fixing removes an **exactly flat** symmetry, not a harmful curvature direction, so it cannot improve a cosine-retrieval solution. A covariance eigenbasis is also fragile — near-equal eigenvalues swap, eigenvector signs stay ambiguous. |
| **Snapshot-Procrustes fusion** | wrong on mechanism | My own suspicion confirmed with a clean argument: every minibatch loss is invariant to a joint rotation, so its gradient has **no first-order component along the rotation orbit**, and shared initialisation pins the frame. The gauge problem does not arise within one run. Snapshot differences are genuine geometry changes, so GPA would *remove* diversity rather than align it. Also 1× training but not 1× inference. Prior art anyway: Huang et al. ICLR 2017; Izmailov et al. UAI 2018. |
| **Sinkhorn-annealed multi-center assignment** | already done, motivation wrong | SoftTriple (Qian et al. ICCV 2019) is fixed-K soft multi-center; Sinkhorn-balanced prototype occupancy is the core of SwAV (Caron et al. NeurIPS 2020). And the motivation fails: the 1.08 pt fixed-seed spread does **not** identify discrete center assignment as its source. Balanced occupancy can manufacture fictitious modes. |

## 3. The one direction with a credible prior is occupied

Changing *what supervision exists* — the only untried class — points at expansion of
intra-class support. BLenDeR is positive evidence for one implementation, not closure
of that direction
([Kolf et al., arXiv v1 2026](https://arxiv.org/abs/2601.20246)). Its +3.7 CUB and
+1.8 Cars claims are single-run evidence: the paper reports no seed count, error bars,
confidence intervals, or paired multi-seed test. It imports Stable Diffusion 1.5
alongside LLaVA-Next, CLIP ViT-L/14, and Segment Anything without a contamination
audit for CUB/Cars, does not isolate gains from imported knowledge, and discloses
neither end-to-end GPU cost nor a cheap recipe (20,000 LoRA steps per category and
150 samples per class per attribute are reported).

The baselines are **not** the objection. BLENDER's own Proxy Anchor baseline is
72.7 CUB versus the published 69.7, and its controlled BLENDER-PA result is 74.6
(+1.9); PF is 73.3 → 77.0 (+3.7) against published PF 73.4. Evaluation is
like-for-like ResNet-50, single-view, one 512-dimensional descriptor. The weak-
baseline/capacity defect that invalidates IDEAL as ceiling evidence does not apply
to BLENDER. `docs/blender_verification.md` records the primary-source audit.

The forecast for a faithful digest-pinned six-seed reproduction is roughly +1 to +2
CUB points, not +3.7. More importantly, BLenDeR leaves open simpler, approximately
1x, contamination-controlled **non-generative** supervision expansion derived only
from the training images. Language guidance is occupied, but it is excluded from this
open class for the same imported-knowledge reason.

## 4. Why the search should stop

The argument, as put:

- Our baselines reproduce their papers to within **0.51–0.58 pt**, so pipeline failure
  is no longer a credible source of headroom.
- Fifteen candidates spanning relational, hypergraph, local-neighbourhood, regional,
  hubness, kernel, asymmetric and capacity mechanisms produced no repeatable
  single-model gain.
- The one positive shrinks from +0.890 in-sample to **+0.427** out-of-sample, and its
  two proposed causal stories were both directly refuted.
- **The fixed-seed spread (1.08 pt) is larger than the entire claimed gain**, and larger
  than HIST's 0.58 pt gap to its published number. Detecting +0.5 pt on the HIST leg
  needs 17 seeds.
- Best-over-training compounds this: it is a **maximum over ~50 noisy correlated
  evaluations**, so it is upward-biased, and the bias grows with a method's noise. A
  noisier method is flattered relative to a stable baseline.
- In-Shop makes the ceiling worse, not better: PA and HIST differ by 0.03 pt against an
  SD of 0.12 pt.
- Every apparently large gain evaporates under the single-model constraint.

**Conclusion: the identifiable single-model headroom is smaller than the measurement
system's resolution.** On these benchmarks, at this architecture, a sixteenth candidate
is not distinguishable from noise even if it works.

## 5. What to do instead

Pivot to the measurement paper. It is stronger than a sixteenth method claim and it is
genuinely ours: paired-seed protocol, fixed-seed GPU nondeterminism, power calculations
that are per-*comparison* rather than per-dataset, transductive/test-fit leakage,
gauge-aligned ensemble accounting, a fifteen-entry negative corpus with mechanisms, and
the EMAN-class defect with its measured cost. This extends Musgrave, Belongie & Lim,
*A Metric Learning Reality Check* (ECCV 2020), with unusually strong controlled
evidence — including something that paper did not have: interventions that were
**pre-registered and then refuted**.

## 5b. Second review: the stability class, which the first review never covered

The first review only judged the three candidates named to it. Today's finding — that
the limiting noise is run-to-run *trajectory divergence*, not evaluation noise — points
at a class that was never asked about. It was put to a second independent review.
**The stop recommendation held.**

**Flat-minimum optimisers (SAM/ASAM/SWA/SWAD) on these benchmarks are genuinely
untested** — no benchmark-matched evaluation exists as of 2026-07-30. But my motivation
for them is wrong, and the refutation is clean:

> The fixed-seed result shows *global trajectory bifurcation*, not evidence that the
> final solution sits in a locally sharp basin. SAM controls the latter.

Chaos in *which solution you reach* and curvature *around the solution you reached* are
different properties. My "chaos ⇒ sharpness ⇒ SAM" chain does not connect. Add that
parameter-space flatness is not a reliable OOD predictor (Andriushchenko et al., ICML
2023, found sharper solutions sometimes generalise *better* OOD), that the zero-shot gap
is across **classes** rather than samples, and that SAM doubles gradient cost — and it is
a bounded ablation at best, not a direction.

**The variance framing is genuinely unoccupied**, and this is the one thing that
survived. Musgrave, Belongie & Lim (ECCV 2020) ran ten seeds and reported confidence
intervals *"to be less subject to random seed noise"*, and proposed MAP@R partly for
checkpoint stability — but they never proposed a method to reduce run-to-run variance,
never decomposed seed vs GPU-nondeterministic variance, and never argued a
same-mean/lower-variance method should be preferred. BIER (Opitz et al. ICCV 2017) is
the closest, reporting CUB R@1 54.41±0.43 → 55.33±0.05, but variance was a by-product of
a boosting claim, not the endpoint.

It is still not a paper on its own. A negative mean plus a variance ratio from six seeds
would need ~20–30 runs per cell, both backbones, all three datasets, class-disjoint
validation checkpointing, a pre-declared non-inferiority margin, and tail-risk metrics
(failure probability below a target R@1, expected retries) — and a *mechanism* for why
trajectory bifurcation shrinks. The deeper objection is sharper than the statistics:
best-test checkpointing is not merely a noisy estimator, it is **test-set feedback**, and
the right response is to replace it with a class-disjoint validation rule rather than
debias it.

**Cheap cross-run diversity is crowded.** DiVA (ECCV 2020), IDEAL (*Pattern Recognition*
2026), DFML (CVPR 2023), DREML, DIABLO. My K-heads-on-a-shared-backbone idea is
explicitly the weak baseline form: DFML identifies shared-backbone *mixed gradients* as
the limitation of embedding ensembles, and Havasi et al. (ICLR 2021) found shared-trunk
multi-head diversity "quite lacking" — MIMO exists precisely because of it. Heads can
learn rotations and proxy arrangements; they cannot learn the distinct low- and mid-level
filters that nine independently optimised backbones produce.

**One number worth noting.** IDEAL reports HIST on CUB **69.7 → 72.3** and Cars
87.4 → 90.3. If that reproduces, the headroom above HIST is not zero — it is *already
taken*. That would make reproduction, not invention, the only route to "outperforms".

## 5c. ⚠️ That conclusion is withdrawn — IDEAL does not foreclose the search

The IDEAL citation was load-bearing for §5b's claim that the headroom is occupied, so it
was verified against primary sources rather than cited. **It does not support the
claim.**

The paper is real: Zelin Yang, Lin Xu, Shiyang Yan, Haixia Bi & Fan Li, *IDEAL:
Independent domain embedding augmentation learning*, **Pattern Recognition** 170,
article 112024 (Feb 2026), [doi:10.1016/j.patcog.2025.112024](https://doi.org/10.1016/j.patcog.2025.112024).
The numbers were quoted correctly. But:

**Its HIST baseline is anomalously weak.** IDEAL reports HIST at 69.7 on CUB. HIST's own
authors publish **71.4**. Our digest-pinned six-seed reproduction is **70.82**. So the
advertised +2.6 is measured from a floor 1.7 pt below the original paper and 1.12 pt
below ours. Against our baseline the gap is **+1.48, not +2.6**. No explanation for the
69.7 appears in the accessible text.

**Its inference is not like-for-like.** It appears to use a fixed **four-view rotation
ensemble** at test time. Comparing that to single-view HIST is a compute and capacity
difference, not purely an algorithmic one — the exact question this project asks of its
own ensemble results (§ GPA, transductive).

**Nobody has reproduced it.** No independent reproduction, critique, or failed
replication was found. Meanwhile HIST *itself* has a
[ReScience C reproduction study (2023)](https://openreview.net/forum?id=JJQbk2hIQ5) that
spent 1,108 GPU-hours, fell ~1.5 R@1 short on CUB with the authors' own configurations,
and received no clarification from the authors. HIST is known to be
configuration-sensitive, which makes an unexplained 69.7 baseline more suspect, not less.

**Forecast for a digest-pinned, paired, six-seed reproduction:** +0.8 to +1.0 under
IDEAL's four-view inference; **0 to +0.8, centred on a few tenths, under a single-view
budget matched to ordinary HIST inference.** The arithmetic: replacing 69.7 with our
70.82 takes +2.6 → +1.48; a differential checkpoint-selection bonus of the size we
measured (0.42 pt) takes it to ~+1.06; seed effects and augmentation parity can consume
several more tenths.

**Consequence: the headroom above HIST is not credibly occupied.** The strongest external
reason to stop searching does not survive contact with its own sources. That does not
manufacture a method — the fifteen failures and both reviews' other findings stand — but
it removes a false ceiling, and it is exactly why the claim was worth checking before
being written into a strategic conclusion.

**A quiet validation, worth recording.** Our HIST reproduction is 70.82 with sd 0.67 over
six seeds — a 95% interval of roughly 70.1–71.5, which **contains HIST's published
71.4**. The pipeline reproduces the paper; IDEAL's baseline is the outlier, not ours.

## 6. That last claim, now measured (`scripts/measure_selection_bias.py`)

Best-over-training reports `max` over ~60 test evaluations. A maximum over noisy
observations of a curve overshoots the curve, and the overshoot grows with the noise.
Estimated per run by taking the trend at the selected epoch from its **neighbours only**
— excluding the selected point, whose own noise is what selection exploited.

**Every reported number in this project is inflated**, and the inflation is not small:

| dataset | arm | reported | trend | selection bonus |
| --- | --- | ---: | ---: | ---: |
| CUB | `proxy_anchor` | 0.6919 | 0.6842 | **+0.77 pt** |
| CUB | `hist` | 0.7082 | 0.7047 | **+0.35 pt** |
| CUB | `pa_distill` | 0.6985 | 0.6901 | +0.84 pt |
| CUB | `narrow64` | 0.6293 | 0.6224 | +0.69 pt |
| In-Shop | `hist` | 0.9038 | 0.9025 | +0.14 pt |
| In-Shop | `proxy_anchor` | 0.9035 | 0.9015 | +0.20 pt |

Three things follow.

1. **The bonus tracks noise, as predicted.** In-Shop (σ = 0.12 pt) gets 0.14–0.37 pt;
   CUB (σ = 0.57 pt) gets 0.35–0.84 pt. On a *flat* simulated plateau with 0.5 pt
   evaluation noise the estimator recovers **+1.16 pt** of pure selection — larger than
   most published DML gains, from a truth with no improvement in it at all.
2. **It differs between arms, so it contaminates comparisons.** CUB `proxy_anchor` gets
   +0.77 and `hist` only +0.35. The reported PA−HIST gap is therefore flattered toward
   PA by ~0.42 pt; corrected, HIST's advantage is *larger* than the leaderboard shows.
   Any leaderboard mixing a stable method with an unstable one is partly ranking
   stability.
3. **Our own effect survives.** `pa_distill − proxy_anchor` goes +0.658 → **+0.592**
   corrected; `herd − hist` +0.298 → +0.187. The distillation gain is not a selection
   artifact, which is the first thing this tool was pointed at.

It also reproduces a known failure as a sanity check: `local_nca` collapsed and peaked
in its first epochs, and best-over-training reported **0.5733 against a 0.3394 trend** —
a 23.4 pt selection bonus on a run that never worked.

### A better summary statistic does not buy power — the noise is in the training

The obvious follow-up: if best-over-training is a noisy max, a less noisy summary should
detect effects with fewer seeds, which would reopen the search on practical grounds. It
does not. Paired `pa_distill − proxy_anchor` on CUB, 6 seeds:

| summary statistic | mean Δ | paired sd | seeds for 80% power on +0.5 pt |
| --- | ---: | ---: | ---: |
| max (reported) | +0.658 | 0.367 | 4.2 |
| mean of top 5 epochs | +0.554 | 0.363 | 4.1 |
| mean of top 10 epochs | +0.608 | 0.414 | 5.4 |
| mean of last 15 epochs | +0.842 | 0.554 | 9.6 |

Averaging over epochs debiases the *mean* (+0.658 → +0.554) but leaves the variance
untouched. So the limiting noise is **not** evaluation noise on a shared trajectory —
if it were, top-5 averaging would collapse it. It is genuine run-to-run divergence:
different runs follow different trajectories, and no readout statistic can recover what
the training did not do consistently.

This sharpens the fixed-seed result too. The 1.08 pt spread at an identical seed is not
jitter in measuring one trajectory; GPU nondeterminism sends training down *materially
different paths*. It also closes the last practical escape route from the stopping
argument: you cannot out-measure this with a cleverer metric.

**A caught bug, worth recording.** The first version of this script keyed artifacts on
arm name and immediately produced `inshop/pa_distill − proxy_anchor = +3.401 pt` — the
exact spurious number that superseded pre-`22f7dd6` artifacts fabricate. It now keys on
recipe digest and resolves the current digest before pairing, which is the rule
`analyze_reference_matrix.py` already enforced. If the current recipe cannot be resolved,
it refuses the comparison. The same trap, in a new script, within an hour of writing it.

## 7. Third audit: can the measured trajectory instability motivate candidate sixteen?

**Audit written 2026-07-30 before spending GPU on a new method.** The standing objective
remains a genuinely novel similarity-learning method, not a new evaluation of an old
optimizer. Weight averaging is useful evidence but is Polyak/SWA, and therefore cannot
satisfy that objective.

The project's strongest new measurement is that training trajectories diverge materially:
top-5 epoch averaging does not reduce the paired variance, and nominally identical
fixed-seed runs spread by 1.08 pt. Four method ideas follow naturally. None survives both
the mechanism check and primary literature.

### 7a. Weight pairs by their stability across checkpoints — rejected

The proposal is to keep temporal means and variances of pair similarities, then down-weight
relations whose sign or rank changes during training. It sounds tailored to the measured
instability, but it makes the same category error as the SAM proposal in §5b. The damaging
variance is **between trajectories**: GPU nondeterminism sends nominally identical runs to
different solutions. A variance estimate collected along one trajectory does not identify
which of its relations would survive another trajectory. Our readout experiment already
shows that smoothing observations on one path leaves the across-run paired sd unchanged.

The nearby method space is occupied as well. General Pair Weighting and Multi-Similarity
already cast DML as informative-pair selection and weighting
([Wang et al., CVPR 2019](https://arxiv.org/abs/1904.06627)); distributionally robust
pair weighting explicitly optimises over an uncertainty set
([Qi et al., ECCV 2020](https://arxiv.org/abs/1912.11194)); and uncertainty-guided metric
learning directly weights pairs by predictive confidence and uncertainty
([Devalraju et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Devalraju_Uncertainty-Guided_Metric_Learning_without_Labels_WACV_2025_paper.pdf)).
Changing the uncertainty estimator to checkpoint variance would be a variant, not a new
supervision mechanism, and its causal premise is unsupported by our measurement.

### 7b. Learn pair weights on disjoint validation classes — already done

This is the statistically correct response to zero-shot class generalisation: split the
training classes, take a metric-learning step on one subset, and learn a constraint or
weighting rule from retrieval on disjoint labels. It would also replace test-set
best-checkpoint feedback with an inductive signal.

But the method exists. *Deep Metric Learning with Adaptively Composite Dynamic
Constraints* constructs disjoint-label episodes, performs a one-gradient update on one
subset, and meta-learns its constraint generator from performance on the held-out subset
([Chen et al., TNNLS 2023](https://pubmed.ncbi.nlm.nih.gov/37018614/)). This is not merely
generic meta-learning near our proposal; it is the same class-disjoint DML mechanism.
Reimplementing it under cleaner recipes could be a reproduction, not a novel method.

### 7c. Momentum-stabilise proxy geometry rather than network weights — already done

The proposal is to average class proxies or local proxy neighbourhoods so the supervision
geometry moves more slowly than the network. It is attractive because Proxy Anchor's
proxies are otherwise updated by the same noisy optimiser as the embedding.

Again the benchmark-matched method space is occupied. *Piecewise-Linear Manifolds for Deep
Metric Learning* uses a momentum encoder to construct data manifolds and combines
point–point, proxy–point, and proxy-neighbourhood objectives
([Bhatnagar et al., ACML 2023](https://proceedings.mlr.press/v234/bhatnagar24a/bhatnagar24a.pdf)).
ProxyGML already builds adaptive proxy similarity subgraphs and label-propagated
neighbourhoods on CUB, Cars, and SOP
([Zhu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html)).
A moving-average proxy update is at most a simpler optimiser for an occupied supervision
structure.

### 7d. Environment-invariant similarity — occupied and unsupported here

One could read zero-shot classes as environments and penalise class-specific or
background-specific distances. *Deep Causal Metric Learning* already learns
environment-invariant attention and task-invariant embeddings for exactly this
generalisation argument
([Deng & Zhang, ICML 2022](https://proceedings.mlr.press/v162/deng22c.html)).
More importantly, this project has measured no environment annotation or spurious-feature
mechanism that predicts a gain. It would be literature-driven speculation, not an
experiment implied by our evidence.

### Verdict of the third audit

No genuinely novel mechanism survives. The remaining unoccupied observations are
**evaluations and measurements**:

1. EMA/SWA weight evaluation has not been benchmarked cleanly on Proxy Anchor or HIST
   zero-shot retrieval, but the technique is old.
2. Best-over-training systematically under-credits stable methods because they collect a
   smaller selection bonus.
3. Seeds-required is comparison-specific, and fixed-seed GPU nondeterminism can exceed
   the claimed method gain.
4. Published teacher-normalisation machinery (EMAN) has not been adopted consistently in
   DML, and the measured cost here is 0.3–1.4 pt.

Those can support a measurement paper. They do not become a novel similarity-learning
method by being combined. At the time of this third audit, candidate sixteen remained
unregistered and unqueued; §§8–11 record the subsequent protocol loop and stopping
verdict.

## 8. Candidate 1 under the iterative protocol: dual-timescale EMA

**Gate 4 failure recorded 2026-07-31.** Candidate 1 was motivated by the CUB
factorial: the relational-distillation target preferred EMA momentum 0.999
(+0.91 pt versus +0.30 pt at 0.99), while the evaluated average preferred 0.99
(+0.46 pt versus +0.07 pt at 0.999). `pa_dual_ema_bnfix` therefore used a slow
0.999 teacher for relational supervision and a separate BN-correct 0.99 average
for evaluation.

The prior-art audit found role-specific and dual-timescale EMA teachers in
adjacent fields, but no benchmark-matched implementation of this exact
single-student DML mechanism. Novelty was explicitly qualified: without a
material gain over averaging alone, the combination would not be distinct
enough from its established components to defend.

The preregistered In-Shop seed-0 screen compared:

| arm | recipe digest | raw best R@1 | corrected R@1 |
| --- | --- | ---: | ---: |
| Proxy Anchor | `16a3bc844c81` | 0.9024 | 0.9015 |
| BN-correct fast average | `80f57f183966` | 0.9043 | 0.9033 |
| BN-correct dual EMA | `79f9d35c4eea` | 0.9044 | 0.9040 |

Raw dual-minus-average was **+0.014 pt**; selection-corrected it was
**+0.077 pt**. The candidate required at least +0.24 pt raw, a positive
corrected delta, and absolute raw R@1 of at least 0.9048. It met only the
corrected-sign condition and therefore **failed gate 4**.

The mechanism is informative, not merely the negative score: once both arms
evaluate the same BN-correct fast average, the slow relational teacher adds
effectively nothing on In-Shop. That is consistent with the prior three-seed
BN-correct distillation result (−0.04 pt versus Proxy Anchor). The opposing EMA
timescale preferences measured on one CUB seed did not expose a transferable
supervision conflict. **Decoupling target and evaluation timescales is not the
constraint: a single EMA was not losing anything recoverable.** Candidate 1
receives no confirmation run and no further dual-EMA GPU time.

The subsequent three-seed averaging confirmation produced raw deltas of
**+0.18 / −0.13 / +0.15 pt**, mean **+0.068 pt** (sd 0.169, paired t
p = 0.5589, exact sign p = 0.500). Selection correction increased the estimate
to **+0.203 pt** (sd 0.157). Thus the raw CUB effect (+0.41 pt) did not replicate
on In-Shop. The correction again shows that the standard protocol under-credits
the stabler arm, but neither estimate supports a method claim at n=3. Weight
averaging does not advance to Cars or a momentum sweep.

## 9. Candidate 2: cross-trajectory consensus supervision

**Gate 2 failure recorded 2026-07-31; no GPU used.** This candidate addressed a
specific weakness in the earlier checkpoint-stability proposal. Fixed-seed CUB
runs spread by 1.08 pt, while top-5 checkpoint averaging left the six-seed
paired sd almost unchanged (0.367 pt for the maximum versus 0.363 pt for top
5). A single trajectory cannot reveal what survives another trajectory.
Candidate 2 therefore proposed two independently perturbed replicas and allowed
new neighbourhood relations to become supervision only when the replicas
agreed.

That supervision mechanism is prior art in retrieval:

- NRMT uses two networks, collaborative clustering, and mutual instance
  selection based on peer confidence and relationship disagreement
  ([Zhao et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1391_ECCV_2020_paper.php)).
- Mutual Mean-Teaching uses independently initialized collaborative networks
  and their temporal averages to generate soft triplet supervision
  ([Ge et al., ICLR 2020](https://arxiv.org/abs/2001.01526)).
- GCMT constructs teacher similarity graphs and supplies graph-consistency
  supervision
  ([Yang et al., IJCAI 2021](https://www.ijcai.org/proceedings/2021/121)).
- PPLR refines person-retrieval pseudo-labels using cross-agreement between
  k-nearest-neighbour sets from two feature spaces
  ([Cho et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Cho_Part-Based_Pseudo_Label_Refinement_for_Unsupervised_Person_Re-Identification_CVPR_2022_paper.pdf)).

The exact hard intersection and supervised benchmark setting are details, not a
new source of supervision. The defining move—letting agreement between
independently varying retrieval representations determine which relations
supervise training—is occupied. Candidate 2 stops before preregistration,
implementation, or GPU screening.

## 10. Candidate 3: controlled synthetic intra-class support

**Gate 1 failure recorded 2026-07-31; no GPU used.** Controlled generation of
new within-class views is the remaining obvious way to change what supervision
exists. It is not, however, motivated by this repository's measurements.
Sub-center Proxy Anchor lost about 1.7 pt, Tversky's distant-positive
intervention lost about 1.6 pt, and same-resolution multi-crop was neutral.
Those observations do not diagnose missing intra-class support. Variable-size
multi-crop collapsed because of frozen BatchNorm and therefore supplies no
positive evidence for the coverage story.

BLenDeR implements expensive pretrained-generative expansion and reports +3.7
R@1 on CUB and +1.8 on Cars, but its unreviewed single-run evidence and imported
Stable Diffusion knowledge do **not** occupy cheap, data-only, non-generative
expansion
([Kolf et al., arXiv v1 2026](https://arxiv.org/abs/2601.20246)). Candidate 3's
specific proposal still failed provenance at the time; its literature rationale
for closing the entire supervision-expansion class is withdrawn.

## 11. Iterative-search stopping verdict

The protocol's stopping condition is met. This is not a claim that no future
similarity-learning method can exist; it is the narrower, evidence-backed
conclusion that **no genuinely novel mechanism is defensible from the
measurements in this repository**:

1. Candidate 1 was the only mechanism with both numeric provenance and a
   qualified novelty case. It failed the preregistered In-Shop screen:
   dual-timescale EMA added +0.014 pt raw over averaging, versus +0.24 required.
2. The remaining between-trajectory measurement motivated candidate 2, but
   two-network agreement/disagreement as retrieval supervision is established
   prior art.
3. Expensive pretrained-generative support expansion is demonstrated by BLenDeR,
   but cheap, data-only, non-generative expansion remains open. The stopping
   verdict is therefore superseded for that class; candidates 4 onward record
   the reopened loop.
4. Every other measured defect already maps to a failed mechanism or established
   method: teacher normalization to EMAN, trajectory averaging to Polyak/SWA,
   class-disjoint meta-supervision to adaptive dynamic constraints, proxy
   neighbourhood supervision to ProxyGML/piecewise-linear manifolds, and
   environment invariance to Deep Causal Metric Learning.
5. The remaining novel observations concern **measurement**, not a new training
   signal: comparison-specific power, fixed-seed GPU nondeterminism,
   best-over-training winner's-curse reversal, digest/provenance failures, and
   the benchmark-specific cost of missing EMAN behavior.

Generating another arm without a new repository measurement would violate gate
1. Renaming or slightly changing an occupied mechanism would violate gate 2.
The honest outcome of this loop is therefore a negative method search and a
positive measurement contribution. No further method GPU run is warranted on
the current evidence.

## 12. Candidate 4: graded within-class relation supervision

**Gate 2 failure recorded 2026-07-31; no GPU used.** Sub-center Proxy Anchor
lost roughly 1.7 pt on CUB, while Tversky's attempt to rescue distant positives
lost roughly 1.6 pt on CUB and 4.63 pt on In-Shop. Candidate 4 interpreted those
failures as evidence that binary class labels overstate within-class
equivalence: discrete modes fragment the class, but uniform attraction
over-collapses useful variation. It proposed pair-specific ordinal constraints
from a frozen feature source.

The causal mechanism is already established:

- continuous/structured DML labels and distance-ratio preservation in
  [Kim et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Metric_Learning_Beyond_Binary_Supervision_CVPR_2019_paper.pdf);
- annotation-free latent fine-grained hierarchy supervision in
  [HIER](https://arxiv.org/abs/2212.14258);
- explicit preservation of useful intra-class variance by soft positive mining
  in [Wang et al., 2018](https://arxiv.org/abs/1811.01459); and
- automatic continuous re-annotation of binary retrieval pairs in
  [Leyva-Vallina et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Leyva-Vallina_Data-Efficient_Large_Scale_Place_Recognition_With_Graded_Similarity_Supervision_CVPR_2023_paper.pdf).

A frozen encoder changes where the graded labels come from, not what supervision
the method uses. Candidate 4 fails novelty before preregistration or
implementation.

## 13. Candidate 5: local-chord positive expansion

**Gate 2 failure recorded 2026-07-31; no GPU used.** Five aligned CUB training
embedding packs established that within-class relation structure is real:
pair-rank Spearman is 0.863 across independent runs and top-5 positive-neighbour
Jaccard is 0.411, 9.06× chance. Global class-centred residual modes were not
stable (cross-run ARI 0.06–0.07), so the candidate correctly targeted local
class-specific structure.

It proposed reciprocal same-class kNN edges and feature interpolation only
along those edges, creating virtual positives without a generator or additional
backbone work. That exact supervision class is occupied. Metrix mixes inputs,
intermediate features, or embeddings for DML
([Venkataramanan et al., ICLR 2022](https://openreview.net/pdf?id=ZKy2X3dgPA));
Embedding Expansion combines feature points into synthetic DML samples without
extra model-size or training-speed cost
([Ko & Gu, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)).
Nearest-neighbour restriction is a locality policy for that existing mechanism,
not a new supervision source. Candidate 5 stops at gate 2.

## 14. Candidate 6: matched-patch positive supervision

**Gate 2 failure recorded 2026-07-31; no GPU used.** `region_pa` lost 3.6 pt,
but replacing fixed slot cosine with position-tolerant MaxSim recovered 6.7 pt;
`local_nca` meanwhile treated 31–40 of 40 same-class images as effective
positives and collapsed. Together with stable global positive-neighbour ranks,
this motivated selecting compatible image pairs and making only their matched
local patches positive.

The mechanism is occupied by DIML, which computes optimal-transport matching
between cross-image feature maps and uses the structural similarity inside
Proxy Anchor and other DML objectives on CUB and Cars196
([Zhao et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.html)).
Mutual-nearest patches rather than optimal transport and global-only rather
than structural-reranked inference are implementation choices, not a new source
of supervision. Earlier weakly supervised alignment work also derives dense
correspondence from same-category image pairs
([Rocco et al., CVPR 2018](https://arxiv.org/abs/1712.06861)). Candidate 6
stops before preregistration or implementation.

## 15. Candidate 7: shared-confusion positive supervision

**Gate 2 failure recorded 2026-07-31; no GPU used.** Across five independent
CUB training packs, top-5 same-class neighbours had a mean **+0.0913** advantage
in the correlation of their rankings over negative-class centroids. Pairwise
embedding similarity and negative-profile similarity correlated **0.7048**.
The proposed method would use shared hard-negative response profiles to select
which embedding dimensions make a same-class pair positive.

The mechanism is not novel. A proxy-similarity vector is a response/logit
profile, and preserving teacher-supplied metric relations is Relational
Knowledge Distillation
([Park et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)).
Learning masks that select embedding dimensions for conditional notions of
similarity is Conditional Similarity Networks
([Veit et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Veit_Conditional_Similarity_Networks_CVPR_2017_paper.html)).
Self-derived conditions and a temporal teacher alter target acquisition, not
the underlying supervision primitive. Candidate 7 stops before preregistration.

## 16. Weight averaging: dataset-specific CUB effect

**Replication failure recorded 2026-07-31; line closed.** BN-correct weight
averaging (`pa_ema_avg_bnfix`) changed In-Shop R@1 by **+0.19, −0.12, and
+0.14 pt** against seed-paired Proxy Anchor: mean **+0.070 pt**, sd **0.166**,
paired t = **0.73**, with only two of three signs positive. Seed 1 was negative.
The selection-corrected estimate was +0.203 pt, but the registered raw retrieval
effect is the deciding quantity and is neither practically nor statistically
established.

This is a failure to replicate the CUB +0.414 pt effect, not evidence for a
smaller universal benefit. The mechanism—Polyak/SWA-style temporal weight
averaging—reduced trajectory noise on CUB but did not transfer to In-Shop under
the required BatchNorm-correct evaluation. No Cars run and no momentum sweep
are warranted.

The measurement lesson is stronger than the method result. The CUB effect
looked solid after two seeds and had a tight sd of **0.060 pt**, yet it was
dataset-specific. **Tight variance on one dataset is evidence about repeatability
on that dataset, not evidence that an intervention is real across datasets.**
This joins the winner's-curse result: checkpoint-selection correction can alter
arm rankings, while low within-dataset seed variance can still leave an entire
method claim non-replicating out of dataset.

## 17. Dual-timescale EMA: no recoverable timescale conflict

**Registered falsification recorded 2026-07-31; line closed.** On In-Shop,
`pa_dual_ema_bnfix` beat the matched BN-correct averaging arm by only **+0.014
pt raw** and **+0.077 pt selection-corrected**, versus the preregistered minimum
raw gain of +0.24 pt and absolute threshold of 0.9048. It therefore failed its
own Gate-4 condition.

The factorial suggested that relational targets preferred momentum 0.999 while
evaluated weights preferred 0.99. Decoupling those roles did not recover their
putative sum: once both methods evaluated the same BN-correct fast average, the
slow relational teacher contributed effectively nothing. The apparent CUB
timescale conflict was not a transferable constraint. The accidental CUB
seed-0 continuation was terminated before producing a result; no further
dual-EMA or momentum-sweep GPU time is permitted.

## 18. Candidate 8: rival-signature positive graph

**Gate-4 failure recorded 2026-07-31; line closed.** Candidate 7's response
distillation and conditional-mask mechanism remains dead. RSPG uses the same
measured rival-profile structure for a different operation: agreement over
negative-class identities decides whether a same-class pair becomes a positive
edge, while failed pairs are unknown.

The closest prior art includes Contextual Similarity Distillation, which makes
contextual descriptors a training signal
([Wu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html)),
and supervised Contextual Similarity Optimization
([Liao et al., arXiv:2210.01908](https://arxiv.org/abs/2210.01908)). RSPG's only
possible distinction is a target-class-excluded descriptor over rival class
identities that gates same-class pairs from positive to unknown, rather than an
instance-kNN contextual loss. The adversarial audit verdict is **LIVE,
narrowly**, conditional on ablations showing that this operator—not generic
contextual training—causes any gain.

The mandatory CUDA-disabled diagnostic on an existing CUB training-only pack
retained **109,375 / 169,596 edges**, density **0.6449**, above the fixed 0.60
maximum. Its multi-component fraction was **0.2500**, exactly the minimum. The
graph is therefore too close to indiscriminate same-class supervision and the
candidate dies before a valid GPU screen. Thresholds were not tuned.

The explicit mechanism test does reject a geometrically close pair with
disagreeing rival signatures and accept a distant pair with agreeing signatures,
so the implementation-level distinction from Easy Positive/OSM is real. The
data-level gate is what fails. An In-Shop process mistakenly started before the
CPU-first task specification was read was terminated after epoch 10; its partial
curve and in-training graph are procedurally invalid and excluded.

### Process finding: a dataset-unspecified go/no-go is not reproducible

The CPU gate specified thresholds and representation timing but did **not** name
the dataset. It was therefore reasonably run on the available CUB training pack
while gating an In-Shop experiment. That answers whether RSPG partitions CUB,
not whether the signal exists in the data about to be trained. The observed
split—CUB density 0.6449 versus partial-run In-Shop density 0.0863—is itself
evidence that the gate is dataset-dependent.

The adjudicated correction is option (b): rerun the unchanged diagnostic on
In-Shop training embeddings, CPU-only. This choice is contaminated because both
dataset outcomes were already seen. Any resulting number must state that fact
in the same paragraph, and a passing main screen requires **two seeds**, not one.
Future go/no-go registrations must name dataset, split, representation source,
checkpoint/epoch, and whether computing that representation counts as GPU work.

Option (b) was first executed with CUDA disabled using the official In-Shop Proxy
Anchor initialization after one CPU optimizer step, the only independently
materializable fixed source available without new GPU work. **The choice was
contaminated by already knowing the favourable partial-run graph.** The corrected
In-Shop graph retained **606 / 153,115 edges** (density **0.0040**, below the
0.05 minimum) and had multi-component fraction 0.9997. That result is invalid as
the candidate decision: after one step the embedding head is essentially random,
whereas the method constructs its graph from a trained epoch-10 state. Together,
CUB/one-step In-Shop/partial epoch-10 In-Shop densities of
0.6449/0.0040/0.0863 prove that the gate depends on both dataset and
representation stage; omitting either makes a go/no-go underspecified.

The final adjudication pays the real cost: train plain official Proxy Anchor for
10 In-Shop epochs, export final training embeddings without periodic test
selection, then run the frozen diagnostic on CPU. A "free CPU diagnostic" that
requires a trained representation is not free unless the exact checkpoint
already exists. That dependency must be budgeted and registered up front.

The exact operating-point rerun then passed: **13,253 / 153,115 edges**, density
**0.0866**, multi-component fraction **0.8735**, closely reproducing the already
seen partial-run 0.0863/0.8703. This paragraph is intentionally marked as a
**contaminated confirmation** because the favourable answer was known first.
Under the adjudicated rule it warrants two full In-Shop RSPG seeds, not one;
both must clear raw R@1 0.9085 before any novelty ablation is activated.

### Deciding In-Shop screen: catastrophic self-erasure

**The decision path was contaminated by the dataset-unspecified diagnostic defect
described above.** On the resulting full In-Shop seed 0, RSPG reached raw
best-over-training R@1 **0.8452 at epoch 10**, versus the paired Proxy Anchor
seed-0 **0.9024** (a **-5.72 pt** result) and the preregistered 0.9085 minimum.
`measure_selection_bias.py` reports **0.7262 selection-corrected** for RSPG versus
**0.9015** for the paired baseline (a **-17.53 pt** corrected delta). That
corrected value must not be interpreted as an ordinary winners-curse estimate:
the estimator assumes a locally smooth curve, but the selected epoch is exactly
the structural discontinuity where RSPG activates. It is reported because the
protocol requires both numbers; the raw threshold alone already kills the arm.

The failure mechanism is stronger than the headline number. Before activation,
epoch-10 R@1 was 0.8452. Replacing Proxy Anchor's own-class positive-proxy term
with attraction to detached graph neighbours immediately drove the logged loss
from ordinary scale to approximately **0.0017**, then **0.0001**, while R@1 fell
to 0.6415 at epoch 11 and **0.4251 at epoch 60**. At the epoch-40 refresh the
graph retained only **1,451 / 153,115 edges**, density **0.0095**, down from
13,615 edges and density 0.0889 at activation and below the registered 0.05
minimum. Multi-component fraction rose to 0.9972. The model therefore erased the
rival-agreement structure that supplied its positives, leaving a nearly empty
graph whose detached-edge objective was trivially satisfied. Positive-to-unknown
gating did not merely select useful supervision; without the own-class proxy
anchor it created a self-reinforcing route to no positive supervision.

The registered rule required both contaminated-path seeds to clear 0.9085 before
ablations. Once completed seed 0 landed at 0.8452, seed 1 could not change the
conjunction and was terminated after one minute; the soft-JS, distance-gate and
instance-context controls were never run. This is not an ablation-stage novelty
failure: RSPG died earlier on absolute performance. Candidate 18 is **DEAD**.

### Mechanistic finding: rival identity carries dataset-dependent structure

The density split is more informative than either gate decision alone. On CUB,
retaining **64.49%** of same-class pairs means that the question “which other
classes does this image resemble?” has nearly the same answer for most images
of a bird class: they confuse with the same small set of rival species. The
target-excluded rival signature therefore supplies little additional
within-class information and RSPG degenerates toward the original class label.
On In-Shop, retaining only **8.66%** of pairs at the trained operating point
shows that rival identities genuinely distinguish samples within a class. With
3,997 training identities, the cross-class reference set is rich enough for the
same construction to become selective.

This is a measured property of these datasets, not merely an implementation
outcome. It constrains the next search batch: supervision routed through
cross-class identities has weak empirical support as a source of CUB
within-class structure. A CUB candidate should instead derive structure from
within-class appearance factors, viewpoint, or instance-level relations that do
not route through rival classes. Cross-class relational candidates should target
many-class datasets such as In-Shop or SOP and must not be presented as a
dataset-general solution without replication.

## 19. Augmentation-response compatibility graph: real structure, no useful objective

**Gate-4 mechanistic failure recorded 2026-07-31; line closed.** ARCG was the
post-RSPG candidate derived from measurements rather than analogy: rival-class
signatures were nearly vacuous on CUB (density 0.6449) but selective on In-Shop
(0.0866), while position-tolerant region matching beat fixed coordinates by
6.67 points. It therefore used each image's response to deterministic flip and
spatial-crop interventions as a within-class factor signature, retaining a
same-class edge only when two different images' signatures agreed.

The adversarial Gate-2 audit found direct prior art for every ingredient but not
the operator-level conjunction. AugSelf and EquiMod represent augmentation
response within one source image; NNCLR/Easy Positive select other instances by
embedding proximity; ScoreCL/CLVS/CoCor continuously weight augmented views;
view-aware re-identification uses known or predicted nuisance factors. None
retrieved work used agreement between two different images' controlled-response
vectors as a binary positive-to-unknown eligibility gate. The novelty survived
narrowly and required later soft-weight and distance-gate controls if the main
arm won.

The exact epoch-10 In-Shop diagnostic passed strongly: 55,594 / 153,115 edges,
density 0.3631, 80.93% multi-component classes, 53.07% closest-quartile
rejection, and 28.02% farthest-quartile acceptance. The full run reproduced it
almost exactly at activation: **55,729 edges, density 0.3640, 81.84%
multi-component, 53.37% close rejection, and 28.00% far acceptance**. The
signature is real and is not ordinary distance mining.

The supervision operator failed catastrophically. At epoch 10, before ARCG
activation, raw R@1 was **0.8463** and loss 2.3593. Replacing Proxy Anchor's
own-class positive-proxy term with attraction to eligible detached graph
neighbours dropped loss to **0.0017** within 100 steps and R@1 to **0.7005** at
epoch 11; by epoch 15 loss was **0.0005** and R@1 **0.6637**. The run was stopped
because the nearly zero objective contained no recovery force. The epoch-10
best is pre-activation, far below the registered 0.9059 minimum, and is not a
completed ARCG benchmark. No final artifact or defensible selection-corrected
estimate exists; Gate 6 was not reached, and no controls or replication ran.

Mechanistically, the eligible same-class neighbours were already beyond Proxy
Anchor's positive margin when frozen. Their pair loss was therefore satisfied
at construction. Removing the unsatisfied own-class proxy term removed useful
attraction, leaving a rapidly satisfied negative-only objective. Retaining the
proxy term or choosing a new harder pair margin after seeing the curve would
change ARCG into an added regularizer and contaminate the preregistration. This
is the second independent graph to show the same constraint: finding genuine
intra-class structure is not enough; a positive-to-unknown graph must also
define an unsatisfied, non-self-erasing training target.

## 20. Interventional principal-stratum ranking: stable signal, irrelevant order

**Gate-4 failure recorded 2026-07-31; line closed.** IPSR was motivated by
ARCG's measured response/distance inversions and its objective collapse. It kept
Proxy Anchor intact and added only ordinal preferences that were provably
unsatisfied: for an anchor, a response-compatible but farther same-class peer
should outrank the closest response-incompatible peer. The construction was
inspired by principal stratification and Bradley–Terry paired comparisons.

The Gate-2 audit found close prior art in Deep Metric Learning with
Self-Supervised Ranking (Fu et al., AAAI 2021), its TCSVT synthesis-ranking
extension, intra-class ranking losses, hard-positive mining, and adversarial
same-source ordinal constraints. IPSR survived narrowly because those methods
rank transformed/synthetic variants around one source or derive order from
embedding distance; IPSR ranked three distinct real images using agreement of
their empirical intervention-response profiles.

The preregistered diagnostic passed with 16,455 preferences, **63.58% anchor
coverage**, **73.36% class coverage**, and initial Bradley–Terry loss **0.7359**.
The full run reproduced it (16,303 preferences; 62.99% / 73.31%; loss 0.7352),
kept total loss at an ordinary scale, and retained a nonzero 0.71–0.73 ranking
loss. The epoch-40 refresh also remained stable at 16,401 preferences and
63.37% / 73.81% coverage. Thus this is not ARCG/RSPG self-erasure.

The complete In-Shop screen reached raw best R@1 **0.9034**, below the registered
0.9059 gate and existing HIST 0.9038. Against paired Proxy Anchor seed 0 the
raw delta was only **+0.091 pt**. Selection correction gives IPSR **0.9013** and
the shared-seed baseline **0.9007**, delta **+0.060 pt**. Both are below the
measured 0.12-point In-Shop sigma; no controls, extra seeds, or replication are
warranted.

The mechanism is a clean negative: augmentation response encodes a stable
nuisance stratum, but agreement of nuisance sensitivity does not establish
which same-identity image should rank closer for unseen-identity retrieval. The
ordinal target remained substantially unresolved and its small training effect
did not transfer to retrieval. A measured intra-class relation is not
automatically a useful relevance order.

## 21. Response-adaptive augmentation dosing: occupied by InstaAug

**Gate-2 prior-art death recorded 2026-07-31; no GPU spent.** RAAD used the
ARCG/IPSR result as augmentation-safety evidence rather than a relevance order:
crop-sensitive images would receive milder crop distributions, analogous to
per-unit dose titration. The mechanism is directly occupied by InstaAug (Miao
et al., ICML 2023), which learns input-specific augmentation distributions to
capture local crop invariances. AdaAug learns class/instance-adaptive policies,
iMAS weakens augmentation per hard instance, and Soft Augmentation changes
targets with crop severity. Replacing their learned policy or hardness with a
frozen response score is not method novelty. Candidate 21 is **DEAD at Gate 2**.

## 22. Reaction-norm transport and homeostatic factor competition

**Shortlist exhausted before GPU.** RNT proposed cross-image/cross-view
supervision derived from augmentation reaction norms. Fu et al.'s Deep Metric
Learning with Self-Supervised Ranking (AAAI 2021) already uses crop,
perspective, and colour transformations to create intra-class ranking
supervision, and its TCSVT 2022 synthesis extension plus later intra-variance
ranking work generalize that mechanism. Matching transformed views across
images by a response descriptor is an adjacent sampler, not a defensible new
supervision class, and would cost roughly 2x training.

HFC proposed reallocating positive loss toward under-covered response strata.
That is continuous sample/loss reweighting—occupied by general pair weighting,
hardness-adaptive objectives, and instance-adaptive supervision—and its Gate-1
premise was absent: the repository measured response heterogeneity, not
per-stratum gradient starvation. Both candidates are **DEAD before
implementation**.

## 23. Interventional differencing nuisance residualization: occupied by NAP

**Gate-2 prior-art death recorded 2026-07-31; no GPU spent.** IDNR treated
controlled augmentations like fixed-effects interventions: within-image
embedding differences would cancel identity, estimate an augmentation nuisance
subspace, and define a fixed orthogonal quotient for Proxy Anchor training and
retrieval. This followed directly from ARCG finding stable heterogeneous
augmentation response and IPSR showing that response agreement is not retrieval
relevance.

The mechanism is already nuisance attribute projection (Solomonoff, Campbell,
and Quillen, 2007 and earlier 2004--2006 work): estimate within-identity nuisance
directions by an eigenproblem, orthogonally remove them, and compare identity in
the retained space. Weighted and nonlinear NAP variants cover covariance
weighting and learned feature spaces. Simard, LeCun, and Denker's tangent
distance is additional image-transformation precedent. Controlled augmentation
pairs are a cleaner estimator, not a new method class. Candidate 23 is **DEAD at
Gate 2**.

## 24. Counterfactual identity-label invalidation: occupied by PNDA

**Gate-2 prior-art death recorded 2026-07-31; no GPU spent.** ARCG's crop
response and IPSR's negative result motivated a narrower intervention: if a
controlled crop destroys the frozen model's own-class evidence, change that
augmented observation from positive to unknown instead of using response to
relate two images. Miyai et al.'s PNDA (WACV 2023) already makes the same
per-image semantic-eligibility decision and assigns a transformed view as
positive or negative accordingly. A supervised proxy-response detector and an
unknown rather than negative target are implementation choices, not a novel
mechanism. Candidate 24 is **DEAD at Gate 2**.

## 25. Occupancy-bag Proxy Anchor: occupied by Bag Exponential Loss

**Gate-2 prior-art death recorded 2026-07-31; no GPU spent.** Inspired by
repeated-survey occupancy models, OBPA would assign an identity label to a bag
of augmented views under an at-least-one-positive assumption, allowing an
occluded crop to be uninformative without selecting or relabeling it directly.
Martinez-Cortes et al.'s Bag Exponential Loss (Pattern Recognition 2021) already
trains deep image retrieval with bags of matching images under exactly this
latent-positive MIL assumption and dynamically weights relevance inside each
bag. Augmented views and class proxies are an application choice. Candidate 25
is **DEAD at Gate 2**.

## 26. Conformal acceptance-set similarity: insufficient calibration support

**Gate-1 feasibility death recorded 2026-07-31; no GPU spent.** CAS proposed
replacing point attraction with leave-one-out, class-conditional conformal
acceptance regions. The official In-Shop training split makes that operator
undefined or too coarsely resolved for most identities: among 3,997 classes, 12
have one image, 10 have two, 416 have three, 1,575 have four, and 671 have five.
Most classes therefore offer only three or four leave-one-out calibration
scores. Pooling across identities would discard class-conditional calibration
and leave an occupied kNN/radius loss. Candidate 26 is **DEAD at Gate 1**.

## 27. Class-transmitted negative immunity: algebraically redundant

**Gate-2 death recorded 2026-07-31; no GPU spent.** In-Shop's diverse rival
profiles and few examples per identity motivated sharing a rival discovered by
one class member as negative supervision for all peers. Proxy Anchor already
labels every image negative against every nonmatching proxy, however, so the
proposal adds no relation: it only reweights or enlarges the margin of an
existing one. AdaptiveFace's hard-class mining/class-adaptive margins,
confusion-based metric learning, and ordinary pair weighting occupy that
remainder. Candidate 27 is **DEAD at Gate 2**.

## 28. Reciprocal-risk distillation: occupied by contextual similarity optimization

**Gate-2 death recorded 2026-07-31; no GPU spent.** Exact epoch-10 In-Shop
neighbors expose a strong training-set signal: leave-one-out R@1 is 0.9382;
mutual top-1 cases are 0.9691 correct versus 0.8977 otherwise, and the 1.03% of
queries lacking top-10 reciprocity are only 0.4060 correct. Direct use is
transductive reranking. The apparent single-image alternative—optimize
training-set reciprocal context so ordinary cosine retrieval internalizes
it—is already Liao et al.'s supervised contextual similarity optimization,
whose definition explicitly uses reciprocal-neighbor sets and query expansion.
Wu et al. CVPR 2022 adds contextual distillation precedent. Candidate 28 is
**DEAD at Gate 2**.

## 29. Matched-control negative supervision: occupied by pose-matched contrast

**Gate-2 death recorded 2026-07-31; no GPU spent.** A new CPU audit found that
21.19% of the 1,600 epoch-10 In-Shop top-1 training errors repeat within an
ordered identity-pair cell, motivating causal-style nuisance matching within
confused negative identity pairs. MCNS would match different-identity images on
their controlled-augmentation response and contrast them with nuisance held
approximately fixed. *Unmasking Puppeteers* already introduces a pose-matched
contrastive loss whose different-identity negatives share pose/expression so
the gradient isolates identity. Deep View-Aware Metric Learning is older
adjacent precedent. A response signature instead of pose is an estimator
substitution, not method novelty. Candidate 29 is **DEAD at Gate 2**.

## 30. Reopened-loop stopping argument: data-only supervision expansion

The strategic opening remains real: BLenDeR does not establish a clean,
reproducible ceiling for cheap data-only intra-class expansion. But repository
measurements plus pre-GPU prior-art audits now cover the plausible mechanism
classes rather than merely a list of loss variants:

1. **split a class into latent modes:** sub-centre Proxy Anchor failed here, and
   SoftTriple/sub-centre classifiers occupy the mechanism;
2. **grade or select same-class relations:** candidate 4 is occupied by Beyond
   Binary Supervision, HIER, and soft-positive mining;
3. **synthesize support between selected positives:** candidate 5 is occupied
   by Metrix and Embedding Expansion;
4. **add spatial/part correspondence:** candidate 6 is occupied directly by
   DIML and weakly supervised semantic alignment;
5. **condition similarity on shared negative relations:** candidate 7 reduces
   to relational distillation plus conditional feature masks;
6. **import or generate additional evidence:** BLenDeR is positive but expensive
   external-generative evidence, explicitly outside the contamination-controlled
   constraint; standard augmentation/multi-view supervision was already tested
   and was neutral or failed under corrected recipes.

ARCG adds the previously untried data-only appearance-factor branch. Its
augmentation-response graph was selective and demonstrably independent of base
distance, so Gate 1's motivating structure exists; it still failed because the
binary graph edges were already satisfied at the operating checkpoint and
replacing proxy positives erased the attractive objective. RSPG failed through
the same positive-to-unknown interface with a different, cross-class signature.
The common failure is now the supervision operator, not the descriptor.

The new diagnostics support the premise—same-class relations are stable and
local, while global pseudo-modes are not—but every defensible way found to turn
that structure into supervision is either measured negative or established
prior art. A further arm would currently require relaxing gate 1, disguising an
occupied mechanism, or adding external knowledge. Under the registered search
protocol, none warrants preregistration or GPU use. This is an evidence-bounded
stopping claim, not a claim that no method can ever exist; the loop should
reopen only when a new repository measurement identifies a supervision
operation outside these six covered classes.
