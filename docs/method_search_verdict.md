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

That selectivity is not semantic evidence. Reconstructing the exact registered
graph from the epoch-10 pack shows that it retained **11,806 / 41,312 (28.58%)**
same-acquisition pairs but only **1,447 / 111,803 (1.29%)** cross-acquisition
pairs. Consequently **89.08%** of all accepted edges shared the filename
acquisition token. RSPG mainly rediscovered the session/model/background
shortcut that ordinary In-Shop R@1 already rewards. This sharpens its death:
the target-excluded rival signature was dataset-selective, but on the dataset
where it was selective its positive-to-unknown decisions largely encoded a
known nuisance variable, then the replacement objective self-erased.

The formally non-distance gate also largely collapsed to distance mining in the
operating distribution. Its 13,253 edges overlap **9,343 / 13,253 (70.50%)**
with the equal-cardinality closest-pair gate (Jaccard **0.5444**). Among
cross-acquisition pairs, accepted edges have mean cosine **0.8469** for
different named views and **0.8794** for matching views, versus **0.6286** and
**0.6660** when rejected. The preregistered synthetic asymmetry test correctly
proved that the implementation *can* accept a distant agreement and reject a
close disagreement; it did not establish that those cases are common in real
data. Future novelty diagnostics must measure operational overlap with the
nearest occupied control, not only exhibit a separating counterexample.

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

## 30. Difference-in-differences equivariance: occupied vector geometry

**Gate-2 death recorded 2026-07-31; no GPU spent.** DiDE proposed preserving
identity contrasts under a shared augmentation by equalizing clean-to-augmented
embedding displacements across images, allowing nonzero equivariant motion
instead of enforcing invariance. TraVeLGAN (CVPR 2019) already preserves equal
transformation vectors across images, and Difference Vector Equalization (AAAI
2026) explicitly equalizes embedding difference vectors across samples to
preserve geometry. Established equivariant/AugSelf work supplies the
augmentation setting. Clean-to-augmented endpoints are an application change,
not method novelty. Candidate 30 is **DEAD at Gate 2**.

## 31. Mutual-ownership proxy calibration: occupied by Calibrate Proxy

**Gate-2 death recorded 2026-07-31; no GPU spent.** The exact In-Shop epoch-10
audit found a striking directional proxy defect: 99.975% of proxies choose their
own empirical centroid, but only 70.303% of centroids choose their own proxy;
only 65.308% of images score their own proxy highest. MOPC would impose a
centroid-side reciprocal ownership margin without pulling individual samples to
the centre. Robust Calibrate Proxy Loss already uses real sample information
and a calibration loss to move proxies toward class feature centres, and
Proxy-AN covers the sample-centric/proxy-centric distinction. Reciprocal
Voronoi phrasing is not a new mechanism. Candidate 31 is **DEAD at Gate 2**.

## 32. Rooted factor spanning supervision: occupied by DAMLRRM

**Gate-2 death recorded 2026-07-31; no GPU spent.** ARCG's 81.84%
multi-component class graph and self-erasure motivated using exactly `n-1`
within-class edges plus one persistent proxy root. DAMLRRM (Xu et al., CVPR
2019) already builds a minimum-cost spanning tree within each category so every
positive pair has a direct or indirect path while avoiding all-pairs collapse,
and evaluates it on CUB, Cars196, and SOP. Adding ordinary Proxy Anchor as a
root solves an implementation failure but only composes two established
mechanisms. Candidate 32 is **DEAD at Gate 2**.

## 33. Local-response transport supervision: underidentified local frames

**Identifiability/Gate-2 death recorded 2026-07-31; no GPU spent.** LRTS tried
to replace scalar response agreement with cycle-consistent transport between
per-image augmentation-response frames. Five displacement vectors in 512
dimensions do not identify an ambient transport; a pairwise 5x5 Procrustes map
estimated from the same frames self-fits the residual. A differentiable frame
requires centre plus five augmented forwards or a stale memory, violating the
roughly-1x constraint, while a one-view version cannot define holonomy.
Cycle-consistent correspondence, transformation synchronization, and
gauge/equivariant local-frame learning occupy the remaining abstraction.
Candidate 33 is **DEAD before diagnostic**.

## 34. Exposure-gated proxy updates: premise falsified by loss normalization

**Gate-1 death recorded 2026-07-31; no GPU spent.** Uniform In-Shop batching
means a typical proxy appears positively in only about one of 23 batches, which
initially appeared to expose it to chronic negative-only drift. The exact Proxy
Anchor arithmetic compensates this: positive loss is averaged over roughly 176
present proxies while negative loss is averaged over all 3,997. For an
average-frequency class, `0.044/176` and `1/3997` are both about 0.00025.
Masking absent proxy gradients would instead suppress negative proxy motion by
roughly 23x. Update frequency was mistaken for update weight; the measured
proxy asymmetry does not support EGPU's mechanism. Candidate 34 is **DEAD at
Gate 1**.

The corrected follow-up does find an ordinary frequency effect: proxy-centroid
cosine correlates 0.276 with identity count, averaging 0.0876 for four-image
classes versus 0.1906 for classes with at least 21 images. This belongs to
class-frequency correction, balanced losses, and adaptive-margin prior art; it
does not rescue EGPU or justify a renamed arm.

An image-level follow-up makes the negative conclusion sharper rather than weaker.
At the exact epoch-10 operating point, images owned by their labelled proxy have
leave-one-out R@1 `0.9656`, versus `0.8865` when a foreign proxy scores highest: a
large **7.91-point conditional gap**. But Proxy Anchor already optimizes exactly
the labelled-versus-foreign proxy inequalities defining that event. Using the
event changes sample weights, curriculum, or margin, not what supervision exists.
The defect predicts errors but does not identify a novel intervention.

## 35. Proxy-neighbour disagreement curriculum: occupied hard mining

**Gate-2 death recorded 2026-07-31; no GPU spent.** At the exact epoch-10
In-Shop checkpoint, labelled-proxy-owned images have leave-one-out R@1 `0.9656`
versus `0.8865` for foreign-proxy-owned images, a **7.91-point gap**. PNDC would
combine this global proxy certificate with local-neighbour correctness to gate or
schedule uncertain samples. Proxy Anchor already weights gradients by relative
sample hardness; Suh et al.'s stochastic class-based hard mining already uses
online class signatures followed by instance-level refinement; hard-aware
point-to-set losses cover soft weighting. PNDC changes the hardness estimator,
not the supervised relations. Candidate 35 is **DEAD at Gate 2**.

## 36. Detailed-balance confusion flow: premise reversed by sparse null

**Gate-1 death recorded 2026-07-31; no GPU spent.** Exact epoch-10 In-Shop
errors initially looked like irreversible class flow: 82.16% of connected class
pairs were one-way and mass-weighted imbalance was 67.25%. A thermodynamics-
inspired arm would symmetrize aggregate class-pair error flux. But destination
permutations preserving the complete source and receiver marginals yield only
0.43% reciprocal directed cells versus **30.28% observed**, and 99.43% imbalance
versus **67.25% observed**. Relative to the correct sparse null, the learned error
graph is already exceptionally reciprocal and balanced. Candidate 36 is **DEAD
at Gate 1**; the raw one-way fraction was a sparsity artefact.

## 37. Gradient-coalition supervision: real conflicts, occupied operators

**Gate-2 death recorded 2026-07-31; no GPU spent.** Exact full-dataset Proxy
Anchor tangent gradients at epoch 10 disagree for **17.94%** of In-Shop
same-class pairs; 46.08% have cosine below 0.5, and gradient agreement correlates
only 0.212 with embedding similarity. The signal is real. But Proxy Anchor has no
image-to-image positive relation to gate. Sample or batch selection is GRAD-MATCH;
one-step generalization weighting in DML is DML-ALA; projecting conflicts is
PCGrad; adding compatible pair attraction returns to the already-falsified
RSPG/ARCG interface. Candidate 37 is **DEAD at Gate 2** because every executable
operator is occupied, not because the motivating measurement vanished.

The estimator itself supplies a process correction: per-image proxy-softmax
gradients falsely showed zero conflicts and mean cosine 0.891. Only the exact
Proxy Anchor log-sum-exp aggregation revealed the 17.94% conflict rate. A
convenient surrogate cannot adjudicate a mechanism whose premise depends on the
operating loss.

Acquisition metadata does not rescue an operator: same-session pairs have a
higher outright gradient-conflict rate than cross-session pairs (21.02% versus
16.80%), even though their mean gradient cosine is higher. Session similarity and
loss-gradient conflict are distinct effects, not a causal chain.

## 38. Cross-session privileged supervision: occupied camera-aware retrieval

**Gate-2 death recorded 2026-07-31; no GPU spent.** In-Shop filenames expose
acquisition groups and named views. At the exact epoch-10 checkpoint, same-item
same-group pairs have cosine `0.8199`, versus `0.6396` across groups, and 90.90%
of nearest neighbours share the acquisition token. The model clearly exploits a
photoshoot/model/background shortcut. Privileging cross-group same-item positives
is nevertheless the established camera-aware retrieval mechanism: Wu et al.
align intra- and cross-camera similarities, Lee et al. explicitly increases the
role of same-identity cross-camera images, and Qi et al. aligns camera subdomains.
Using a DeepFashion acquisition token instead of camera ID is a benchmark
adaptation. Candidate 38 is **DEAD at Gate 2**.

The corrected diagnostic strengthens the dataset finding: cross-acquisition-only
training R@1 is `0.5542` on identities where such a positive exists, versus
ordinary `0.9382`. But the official partition gives 95.60% of test queries at
least one same-token gallery positive and only 57.28% any cross-token positive.
Headline In-Shop R@1 therefore partly rewards the acquisition shortcut. Fixing
it may be scientifically desirable while moving against this project's registered
screen; exploiting it would not constitute a retrieval-method contribution.

## 39. Acquisition-cluster robust PA: pseudoreplication premise absent

**Gate-1 death recorded 2026-07-31; no GPU spent.** Equal-weighting In-Shop
acquisition groups was motivated by the 0.8199 within-group versus 0.6396
cross-group cosine gap. But multi-group identities have mean group-size CV only
0.056; image-weighted and group-balanced centroids have cosine 0.99985. Balancing
changes nearest-centroid accuracy by only +0.0039 point and slightly reduces mean
proxy alignment. Candidate 39 is **DEAD at Gate 1**: the session shortcut is
geometric, not unequal-sample pseudoreplication.

## 40. Legacy synthesis variants: stale novelty claim withdrawn

**Gate-2/protocol death recorded 2026-07-31; no new GPU spent.** The repository
still called group-mean and confusion-guided Proxy Synthesis “novel.” Existing
paired CUB final-epoch deltas over an older ResNet-50 baseline are: vanilla
`+0.523 ± 0.556` point, group mean `+0.422 ± 0.726`, and confusion-guided
`+0.264 ± 0.469` (mean ± paired sd, n=3). Each is seed-0-heavy; group and
confusion-guided each have a negative seed. The artifacts lack curves for
selection correction and were never screened on In-Shop. More importantly,
mixing virtual classes is Proxy Synthesis/Embedding Expansion/Metrix, while
group aggregation and confusable-pair sampling are estimator and mining choices.
Candidate 40 is **DEAD at Gate 2** and the source comments' novelty wording is
retracted.

## 41. Acquisition ICC suppression: occupied cross-view center alignment

**Gate-2 death recorded 2026-07-31; no GPU spent.** The 0.8199 within-session
versus 0.6396 cross-session cosine gap motivated a random-effects/ANOVA loss that
minimizes between-session center scatter relative to total within-identity scatter.
Cross-view center loss, Hetero-Center loss, and camera-aware contrastive center
loss already pull condition-specific centers of one identity together. An ICC
denominator and nested group IDs change normalization and indexing, not the
supervision mechanism. Candidate 41 is **DEAD at Gate 2**.

## 42. Negative-control differential distillation: occupied pair-difference KD

**Gate-2 death recorded 2026-07-31; no GPU spent.** The acquisition gap is
mostly training-induced: same- versus cross-group cosine differs by only `0.0251`
after one step but by `0.1804` at epoch 10, a **7.18× amplification**. NCDD would
use the early snapshot as a negative control and apply a one-sided hinge to growth
in `sim(same-session) - sim(cross-session)`. Pairwise Difference Relational
Distillation already transfers differences between pairwise similarities in
object re-identification; RKD and similarity-preserving KD cover the relational
teacher operator. Selecting acquisition-labelled entries and penalizing only one
direction is a mask/hinge variant. Candidate 42 is **DEAD at Gate 2**, while the
training-induced shortcut remains a strong measurement result.

## 43. Rate-balanced proxy attraction: measured cause, occupied weighting

**Gate-2 death recorded 2026-07-31; no GPU spent.** Exact full-dataset
epoch-10 gradients show that Proxy Anchor's positive term increases same-session
similarity at `7.77e-5` versus `4.16e-5` across sessions, while its negative term
partially corrects the gap (`1.72e-4` versus `1.94e-4`). Positive attraction, not
foreign-proxy repulsion, causes the local acquisition-gap drift. A homeostatic
controller could equalize those rates, but it only reweights existing positive
relations; General Pair Weighting, DML-ALA, and camera-diversity losses occupy the
operator. Candidate 43 is **DEAD at Gate 2** despite a successful causal
decomposition.

## 44. Residual-agreement positive preservation: occupied contextual pseudo-labeling

**Gate-2 death recorded 2026-07-31; no implementation or GPU.** RSPG's exact
In-Shop graph overlaps the equal-cardinality distance gate by 70.50%, leaving
3,910 context-agreeing edges outside ordinary proximity selection. RAPP would
retain Proxy Anchor and add positives only on this disagreement residual, thus
avoiding RSPG's self-erasure. STML (Kim et al., CVPR 2022) already combines
pairwise and reciprocal-neighbour contextual similarity as cross-instance
relational pseudo-labels and explicitly analyzes the high-context/low-distance
case. Liao et al. (arXiv:2210.01908) brings contextual-similarity optimization
to supervised metric learning. Restricting that established relation to a
binary residual and composing it with Proxy Anchor is a mask/loss-composition
variant. Candidate 44 is **DEAD at Gate 2**.

## 45. Transformation-response transplantation: occupied feature trajectory transfer

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** ARCG's 36.31%
edge density and distance-disagreement rates motivate using real controlled-
transformation responses without treating response agreement as relevance.
TRRT would transplant an observed same-class donor's augmentation displacement
onto a recipient as a counterfactual positive. FATTEN (Liu et al., AAAI 2018)
already models and transfers pose-induced feature trajectories to synthesize
target-pose features, and Embedding Expansion establishes synthetic feature
support in DML. Directly observing the displacement and response-matching its
donor alter estimation and matching, not the supervision mechanism. Candidate
45 is **DEAD at Gate 2**.

## 46. Repeated-measure set completion: occupied cross-camera generation

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** The 7.18x
training amplification of In-Shop's acquisition gap and RSPG's 28.58% versus
1.29% same-/cross-group gate rates motivate predicting a held-out acquisition
group from the other repeated observations of an identity. Camera-Conditioned
Stable Feature Generation (Wu et al., CVPR 2022) already synthesizes missing
cross-camera identity features for ReID, while set-to-set cross-view metric
learning directly supervises identity sets across cameras. Filename tokens and
deterministic centroid regression alter group estimation and generator capacity,
not the supervision mechanism. Candidate 46 is **DEAD at Gate 2**.

## 47. Determinantal niche-volume preservation: occupied intra-class spreading

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** Exact gradients
show Proxy Anchor's positive term contracts same-acquisition pairs nearly twice
as quickly as cross-acquisition pairs. An ecology/DPP-inspired log-determinant
could preserve the volume of centered within-class embeddings while the proxy
maintains identity. IDID already explicitly generates intra-class diversity in
DML, Ranked List Loss preserves class hyperspheres instead of compressing all
positives, and reverse contrastive loss directly spreads same-class examples.
The determinant changes the set aggregate, not the supervision instruction.
Candidate 47 is **DEAD at Gate 2**.

## 48. Same-class patch recombination: occupied within-identity PartMix

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** The roughly
6.7-point CUB MaxSim recovery motivates recombining authentic local evidence
from two same-class training images into a new positive. CutMix supplies the
generic image operation, and PartMix (Kim et al., CVPR 2023) directly synthesizes
within-identity positive part mixtures for contrastive person retrieval. Pixel
versus descriptor mixing and donor sampling are implementation choices.
Candidate 48 is **DEAD at Gate 2**.

## 49. Compositional decoupled relational distillation: occupied DKD + RKD

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** On five saved
final CUB HERD packs, same-class samples receive only 7.81--11.10% of the τ=0.1
neighbourhood mass although the nearest neighbour is same-class 89.23--95.07%
of the time. The live distillation target is therefore dominated by graded
cross-class relations. An Aitchison-inspired decomposition would separately
match same/different total mass and conditional rankings. Decoupled KD already
factorizes target/non-target mass from the non-target conditional distribution;
RKD transfers inter-sample geometry, and decoupled relational KD combines these
operators. Relabelling the partition for DML is not novel. Candidate 49 is
**DEAD at Gate 2**. The diagnostic is retained because it narrows the mechanism
behind the only intervention that replicated on CUB and Cars.

## 50. Consensus-stable relational distillation: occupied multi-teacher agreement

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** Across five
aligned final CUB HERD packs, different-class pair similarities are strongly
reproducible (mean Pearson 0.8127, Spearman 0.7936), so the dark geometry is not
seed noise. Filtering relational targets by replica agreement is nevertheless
the mechanism of multi-teacher agreement KD; its relational component already
weights distance and angular knowledge by teacher agreement. Ensemble KD and
adaptive disagreement weighting further occupy the class, and multiple complete
teachers violate the roughly-1x cost constraint. Candidate 50 is **DEAD at Gate
2** while the stability measurement remains useful.

## 51. Residualized dark-relation distillation: occupied pairwise differences

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** Class-pair means
explain 52.57--58.90% of cross-class similarity variance across five final HERD
packs and reproduce at Pearson 0.9037, while image-pair residuals remain stable
at Pearson 0.6980. RDRD would subtract class-pair fixed effects and distil only
the residual that Proxy Anchor does not encode. Pairwise Difference Relational
Distillation and D3still already transfer differential pairwise similarity
matrices for object/image retrieval. Fixed-effect centering chooses a contrast
inside that operator rather than defining new supervision. Candidate 51 is
**DEAD at Gate 2**.

## 52. Tetrad interaction relational distillation: failed at Gate 4

**Pre-registered 2026-08-01 before implementation.** Exact cross-class ANOVA on
five final HERD packs isolates an image-by-image interaction with 4.75% variance
share but reproducible cross-seed structure (Pearson 0.5710). TIRD transfers the
EMA teacher's closed 2×2 similarity contrast over two images from each of two
classes, cancelling class-pair and both single-image main effects. RKD transfers
absolute pair/angle geometry; PDRD and D3still transfer first-order pairwise
similarity differences for ranking; standard quadruplet losses impose labelled
margins. No audited source transfers this two-class tetrad interaction. In-Shop
seed 0 was predicted at raw R@1 0.9090 and falsified below 0.9085.

**Gate-4 death recorded 2026-08-01.** The complete official In-Shop seed-0
screen reached raw best-over-training R@1 **0.8301** at epoch 59, versus paired
Proxy Anchor **0.9024**: **-7.237 points**. Selection correction moved TIRD to
**0.8267** and the paired delta to **-7.405 points**, so volatility made the raw
best slightly optimistic rather than rescuing it. It missed its absolute 0.9085
falsifier by **7.84 points**. No additional seed or dataset is warranted.

The mechanism failure is stronger than “distillation did not help.” The
measured interaction was real but small: only 4.75% of cross-class similarity
variance. Cosine-normalizing all cross-class interaction entries promoted that
small, sign-changing residual to a unit-scale global target. Training lagged
Proxy Anchor drastically, oscillated, and only recovered to an approximately
0.83 ceiling after 60 epochs. Isolating information not already encoded by the
base loss does not establish that it is compatible with the dominant geometry;
normalization can erase the variance share that made the component minor.
Candidate 52 is **DEAD at Gate 4**.

## 53. Reopened-loop stopping argument: data-only supervision expansion

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

## 54. Effect-size-calibrated tetrad distillation: occupied Gram matching

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** TIRD's In-Shop
screen failed by -7.237 raw and -7.405 selection-corrected points after cosine
normalization promoted a 4.75%-variance interaction residual to unit-scale
pressure. ECTD would preserve that measured effect size by squared-error
matching class-centred interaction Gram entries normalized by total teacher
Gram energy. Similarity-Preserving KD already matches teacher/student pairwise
similarity matrices; Full Kernel Matrix Transfer and geometry-aware KD use
matrix-difference/Frobenius Gram objectives; DRDKD adds centred Gram matching.
The closed tetrad is a mask/contrast inside that established operator, while
total-energy calibration is scalar loss normalization. Candidate 54 is **DEAD
AT GATE 2**. Running it would be a post-result loss-weight sweep, not a new
supervision method.

## 55. Cross-fitted tetrad eligibility: occupied ordinal relation mining

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** CFTE would
compute TIRD's tetrad under two augmentation views, retain sign-agreeing
high-magnitude relations, and transfer their ordinal quartet constraints.
Pairwise Ranking Distillation already defines ranking transfer for an arbitrary
relational function over input n-tuples; D3still transfers strict similarity
differentials; Cross-View Consistency KD combines view agreement with
confidence-based teacher-signal mining. The tetrad selects the relational
descriptor and cross-fitting selects a reliability mask inside those established
operators. Candidate 55 is **DEAD AT GATE 2**.

## 56. Augmentation-complement positive completion: occupied positive mining

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** ARCG measured
an augmentation-response graph independent of base distance but failed after
replacing ordinary positive attraction. ACPC would retain Proxy Anchor and add
attraction specifically to same-class pairs with disagreeing response
signatures, interpreting disagreement as missing appearance-factor coverage.
AugSelf already establishes augmentation-response information as a learned
descriptor; AdaSP and HAP2S already adaptively select or emphasize difficult,
visually diverse same-class positives; proxy-based diversity expansion is also
established by SEE. No audited paper used ACPC's exact response score, but the
score merely chooses which already-labelled positive receives more pressure.
It is a new hardness descriptor inside an occupied positive-mining operator,
not new supervision. Candidate 56 is **DEAD AT GATE 2**.

## 57. Coverage audit: evidence-bounded stop

**Stopping audit recorded 2026-08-01; no implementation or GPU.** After the
post-TIRD shortlist was exhausted, an adversarial taxonomy pass checked the
remaining apparent operator classes. Global clustering is occupied by
facility-location DML; multilateral hypergraph supervision by HIST; batch-level
assignment by optimal-transport DML; density and feasibility by density-aware
and chance-constrained DML; distributional intra-class modelling and generated
support by DVML. Together with candidates 1--56, this covers splitting,
selecting/weighting, synthesizing, local alignment, contextual conditioning,
teacher transfer, global/setwise structure, probabilistic coverage, and
multi-view/temporal stability. The full argument and primary citations are in
`docs/search_stopping_audit_2026-08-01.md`.

The bounded verdict is that no candidate currently satisfies both repository
provenance and mechanism-level novelty under the data-only, roughly-1x-cost
constraints. Another GPU arm would require relaxing Gate 1, relaxing Gate 2,
or importing external knowledge. This is not a universal impossibility claim;
it is the registered stopping condition supported by the present evidence.

## 58. Teacher-gradient control variate: occupied variance reduction

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** Reopening the
search beyond fixed supervision operators suggested using the EMA teacher only
as a zero-mean control variate for stochastic Proxy Anchor gradients, preserving
the base objective in expectation while reducing the measured trajectory noise.
The provenance was the 0.35--0.84-point best-over-training inflation and the
selection-corrected distillation gain. Safaryan, Peste, and Alistarh,
*Knowledge Distillation Performs Partial Variance Reduction* (NeurIPS 2023),
already establish KD as stochastic-gradient variance reduction. Subtracting a
running expectation to make the auxiliary gradient unbiased is classical
control-variate/SVRG machinery. Applying that occupied optimizer to Proxy Anchor
does not define new similarity supervision. Candidate 58 is **DEAD AT GATE 2**.

## 59. Reproducible ternary class betweenness: occupied ordinal geometry

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The five CUB teachers reproduce class-pair geometry strongly (class-pair variance
component Pearson 0.9037), motivating a test for stable ternary statements of
the form “class B lies between A and C.” A positive diagnostic would supervise
class-centroid collinearity/barycentric order rather than another pair weight.
But ordinal embedding and triplet networks already learn relative comparison
relations; Piecewise-Linear Manifolds for DML explicitly models local linear
submanifolds with proxies; HIST already makes multilateral class relations the
supervision object. Betweenness is a particular tuple descriptor inside
established ordinal/manifold/hypergraph geometry, not a new operator. Candidate
59 is **DEAD AT GATE 2**.

## 60. Directed augmentation-transition supervision: occupied equivariance

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
ARCG's In-Shop response graph was selective (density 0.3631), stable through
activation (0.3640), and substantially independent of base distance, motivating
a directed intervention object: for a controlled augmentation, record whether
an image's nearest rival class changes from A to B and supervise that
action-conditioned transition. This would be a directed relation rather than an
agreement gate. But AugSelf already supervises augmentation information;
orthogonally equivariant contrastive learning (CARE) learns augmentation action
on representations; and FATTEN explicitly models and transfers pose-induced
feature trajectories. Replacing the trajectory endpoint with a rival-class
identity is a descriptor change inside established equivariant/trajectory
supervision. Candidate 60 is **DEAD AT GATE 2**.

## 61. Episodic unseen-class supervision: no distinct provenance and occupied

**Gate-1/2 death recorded 2026-08-01; no implementation or GPU.** A possible
escape from fixed train-class supervision was to partition training identities
into pseudo-seen and pseudo-unseen episodes and optimize retrieval on the held-
out identities, importing bi-level meta-learning. The repository establishes
ordinary zero-shot retrieval performance but contains no numeric defect showing
that an episodic objective corrects a failure distinct from generalization
itself, so the proposal lacks Gate-1 provenance. Independently, few-shot metric
learning already uses episodic training on non-target classes (for example
Channel-Rectifier Meta-Learning), while meta/factorized DML explicitly targets
unseen-class generalization. Candidate 61 is **DEAD before preregistration**.

## 62. Augmentation-commutator supervision: occupied transformation composition

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
ARCG's stable 0.3631--0.3640 response-graph density suggested measuring whether
two controlled transformations have a reproducible order effect,
`f(T1(T2(x))) - f(T2(T1(x)))`, and supervising that four-view commutator. This
would expose factor interaction unavailable in static packs. But composable
augmentation encoding (CATE) already represents sequences of augmentation
parameterizations; class--pose decomposition and steerable equivariant learning
supervise transformation group action; equivariant reconstruction explicitly
handles combined augmentations. The commutator is a particular algebraic
summary of composition inside that established operator. Candidate 62 is
**DEAD AT GATE 2**.

## 63. Single-model effective-rank compression: diagnostic falsified

**Gate-1 diagnostic failure recorded 2026-08-01; CPU only, no GPU.** Existing
narrow-head failures did not establish whether a normally trained 512-D model
actually uses all dimensions, so a train-fit/test-apply PCA sweep was registered
before reading the result. Unprojected CUB HERD seed-0 Recall@1 was 0.6940.
Centered fitted rank 512 fell to 0.6818 (-1.215 points), already violating the
fixed 0.10-point geometry tolerance; rank 128 reached only 0.6661 (-2.785
points) despite retaining 88.55% of train variance. The mean/origin is part of
cosine retrieval geometry, and variance compression does not reveal unused
capacity. Candidate 63 is **DEAD AT GATE 1**.
