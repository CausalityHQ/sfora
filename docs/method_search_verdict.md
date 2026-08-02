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
ranking work generalize that mechanism. A primary re-audit of the latter (SGAR,
Liu et al., EAAI 2024) found that it generates isotropic Gaussian radial latent
points and ranks them by construction intensity, nearly verbatim SR/SSR. Its
single-run +0.3--0.5 point claims have no uncertainty, use six test-tuned
hyperparameters, and contain baseline transcription errors in the accessible
arXiv version. It strengthens synthesis-ranking prior art but supplies no
real-image relation. Matching transformed views across
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
Proxy-AN covers the sample-centric/proxy-centric distinction. DADA (Ren et al.,
AAAI 2024) is the more direct distribution-level occupant: it adversarially
aligns sample and proxy populations and enforces category-posterior agreement
between them, using feature mixtures as an intermediate domain. Reciprocal
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

## 64. Mean-direction factorization: occupied and unsupported improvement

**Gate-2 death recorded 2026-08-01; no further diagnostic or GPU.** Candidate
63 established that subtracting the train mean at full rank costs 1.215 CUB
points, so the cosine embedding origin is useful. Explicitly separating a
mean-direction scalar from the centered residual is adjusted-cosine/whitening
geometry; a query-conditioned origin is local scaling/CSLS; a class-conditioned
origin is proxy or density modelling; teacher mean/covariance transfer is
MMD/CORAL/Gram distillation; homogeneous coordinates are the standard bias
construction. More fundamentally, raw uncentered cosine already retains the
useful mean at zero cost. The measurement supplies evidence against removing it,
not evidence that a more complex reintroduction can improve it. Candidate 64 is
**DEAD AT GATE 2**.

## 65--67. Recent-paper gap proposals: occupied implementation variants

**Gate-2 deaths recorded 2026-08-01; no implementation or GPU.** A 2024--2026
horizon scan found PFML, Anti-Collapse, Realigned Softmax Warping, DDML, and
CouCE. Three apparent gaps were audited: (65) learning rather than hand-setting
the field/warping curve is established meta-loss/loss-shape learning; (66)
streaming whole-dataset coding rate is a memory/EMA implementation of an
existing coding-rate/covariance regularizer; and (67) automatic nuisance-factor
discovery is established disentangled representation learning. None changes the
supervision or similarity operator. Candidates 65--67 are **DEAD AT GATE 2**.

The scan does correct the external ceiling: PFML (CVPR 2025) credibly reports
five-run ResNet-50 single-view results of 0.734 CUB and 0.927 Cars196. The old
statement that roughly 0.715 CUB was unoccupied is withdrawn. PFML reports no
In-Shop result, leaving In-Shop as the appropriate first screen for a novel arm.

## 68. Minimum-spanning same-class bridges: occupied graph-positive mining

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The RSPG post-mortem showed that rival-class signatures are nearly vacuous on
CUB (0.6449 retained density) but selective on In-Shop (0.0863 at the trained
operating point). One attempted escape was to ignore rival identities and make
only the minimum set of within-class appearance edges needed to connect each
class eligible positives, thereby supervising component bridges instead of all
same-class pairs. This is graph construction from embedding proximity followed
by hard positive selection: graph-based positive mining, local-manifold DML,
self-supervised ranking, and graph-consistency objectives already contain that
operator. ProxyGML and STML are especially close instances of propagating or
selecting supervision on an embedding graph. Calling the selected edges a
minimum spanning forest changes the sparsifier, not what supervision exists.
Candidate 68 is **DEAD AT GATE 2**.

## 69--71. Claude-proposed subspace, isotropy, and gradient objectives: occupied regularizers

**Gate-2 deaths recorded 2026-08-01; no diagnostic, implementation, or GPU.**
After the evidence-bounded stopping audit, a fresh Claude proposer was forced
to use raw observables absent from the saved normalised packs. It proposed:
(69) class-private/shared PCA leakage followed by gradient orthogonalisation;
(70) a cross-class margin-isotropy tensor followed by directional-margin
equalisation; and (71) sample-to-class gradient-direction entropy followed by
gradient-alignment minimisation. None satisfies the requested mechanism class.

Candidate 69 is explicitly a shared/private disentanglement and orthogonality
regularizer. Deep Disentangled Metric Learning (Park et al., AAAI 2025,
https://doi.org/10.1609/aaai.v39i19.34184) already separates class-specific and
class-agnostic features for DML, and fine-grained DML with an orthogonalisation
constraint is also published (Xiao et al., *Knowledge and Information Systems*,
2026, https://doi.org/10.1007/s10115-026-02716-2). Candidate 70 is exactly a
hyperspherical/margin geometry regularizer; Learning with Hyperspherical
Uniformity (Liu et al., AISTATS 2021,
https://proceedings.mlr.press/v130/liu21d.html), Deep Metric Learning with
Spherical Embedding (Zhang et al., NeurIPS 2020), and Non-isotropy
Regularization for Proxy-based DML (Roth et al., CVPR 2022 workshop/arXiv
2203.08547) occupy the operator. Candidate 71 directly constrains gradients;
GradML already derives DML losses from per-sample gradient behaviour, while
PCGrad-family and class-adaptive gradient optimisers occupy alignment and
projection. Measuring a new tensor does not make its downstream regularizer a
new supervision object. Candidates 69--71 are **DEAD AT GATE 2**.

## 72. Cross-instance masked reconstruction: no identified pairing variable

**Gate-1/2 death recorded 2026-08-01; no implementation or GPU.** A Claude
round proposed taking different same-class images A and B, conditioning a
masked-token decoder for B on A's global code, and using the reconstruction as
an auxiliary target beside Proxy Anchor. The intended new supervision object
was information shared across different instances rather than a weight on their
positive edge. The proposal fails at the level of the data-generating relation:
under uniform same-class sampling, A is exchangeable with every other
same-class image. There is no fact saying *which* A should predict B.

If a particular A predicts B better than a shuffled same-class A, that pairing
must have been induced from visual proximity, part correspondence, pose, or a
neighbour graph—the already occupied selection/local-matching operator. If it
does not, the optimal decoder ignores A-specific information and learns a
class-conditional reconstruction prior, an established supervised-autoencoding
or conditional-generation auxiliary objective. Same-image masked modelling,
positive-pair prediction, cross-view re-identification reconstruction, and
class-conditioned autoencoders occupy the adjacent space. Claude's suggested
diagnostic required the chosen A to beat a random same-class A by 15% MSE, but
no registered variable defines that chosen pairing; the threshold therefore
tests an unavailable relation. Candidate 72 is **DEAD BEFORE PREREGISTRATION**.

## 73--77. Claude-proposed comparison algebras: degenerate or reparameterized

**Gate-1 algebraic deaths recorded 2026-08-01; no implementation or GPU.** A
comparison-operator round required a symmetric, subquadratic score on one
512-D vector per image that was not a monotone transform of cosine. All five
proposals failed their own specification before prior-art search.

- **73, Clifford geometric-product score:**
  `||x wedge y||^2 = ||x||^2 ||y||^2 - (x dot y)^2`; for normalized vectors the
  proposed score is affine in squared cosine. It discards the sign and is not a
  new geometric observable.
- **74, pairwise Procrustes contraction:** an independently optimized rotation
  can map any one equal-norm vector exactly onto any other. On normalized
  embeddings every pair has zero residual, so the comparison is constant.
- **75, rough-path signature over coordinates:** embedding coordinates have no
  canonical sequence order. The score changes under an arbitrary basis
  permutation/rotation and its truncated form is a polynomial feature kernel,
  not a justified retrieval relation.
- **76, “directional Fisher--Rao”:** the supplied formula was a learned
  coordinate-wise mixed norm, not a Fisher--Rao metric. It is occupied metric
  learning and basis-dependent scaling.
- **77, anisotropic cosine:** diagonal scaling followed by cosine is exactly
  absorbable into the preceding linear embedding head; labeling the matrix a
  metric tensor does not change the model class.

Candidates 73--77 are **DEAD BEFORE GATE 2**. The mechanism lesson is that a
single-vector comparison must either respect the embedding's arbitrary basis,
in which case orthogonal-invariant scalar information collapses to norms and
inner products, or introduce learned structure equivalent to another embedding
head or established metric tensor.

## 78--82. Claude-proposed structured outputs: separable feature-map equivalence

**Gate-1 algebraic deaths recorded 2026-08-01; no implementation or GPU.** A
second operator round allowed structured per-image outputs but required each
image to be encoded independently. The proposed conditional transformation
`P(x)v(x)` (78) is simply another vector-valued embedding head. Frobenius inner
products between predicted moment matrices (79) are dot products after
vectorisation. Weighted orthogonal subspaces (80) are a concatenated/rescaled
embedding. A per-image rotation and scale used as a Mahalanobis tensor (81) is
established local/probabilistic metric learning. RBF coefficients plus an RBF
of the base vector (82) are a direct-sum kernel feature map. None creates a
non-separable comparison.

Candidates 78--82 are **DEAD BEFORE GATE 2**. More generally, any positive
semidefinite comparison between independently encoded finite objects has a
feature-map representation; changing vectors into matrices or named moments
does not evade that equivalence. A live comparison candidate must justify a
non-separable or non-kernel interaction and show why it is meaningful rather
than merely nonlinear.

## 83. Covariance-commutator similarity: occupied matrix comparison

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The failure of first-order multi-vector Region Proxy Anchor (0.6466 mean CUB,
3.6 points below its paired baseline) suggested testing a genuinely
non-separable comparison of per-image second-order feature operators. The
candidate would encode an intermediate feature-map covariance `C(x)` and rank
with the Frobenius norm of the commutator
`C(x)C(y) - C(y)C(x)`, which measures failure of shared eigendirections.

Both halves are occupied before DML implementation. Bilinear/global covariance
pooling is established for fine-grained recognition and retrieval (including
MPN-COV and SOLAR). More decisively, Glashoff and Bronstein, *Matrix
commutators: their asymptotic metric properties and relation to approximate
joint diagonalization* (Linear Algebra and its Applications, 2014,
https://doi.org/10.1016/j.laa.2013.09.020), explicitly study commutator norm as
a matrix dissimilarity and note joint diagonalizability as a similarity
criterion for 3D shapes. Combining an established covariance descriptor with
its established matrix comparison does not define a novel similarity operator.
Candidate 83 is **DEAD AT GATE 2**.

## 84--88. Reusing pretrained-backbone internals: pseudo-pairing or feature transfer

**Gate-2 deaths recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Because ImageNet initialization is already part of the benchmark recipe, a
Claude round was asked whether otherwise unused classifier logits,
intermediate activations, BN deviations, Jacobians, or residual-path patterns
could create a new supervision object without adding another model. Every
proposal mapped those descriptors into pseudo-positive/pseudo-negative pairs
or asked the metric embedding to preserve descriptor similarity:

- top-k ImageNet-logit overlap (84),
- layerwise channel-rank agreement (85),
- BN-deviation cluster agreement (86),
- layerwise Jacobian-profile agreement (87), and
- residual-path sparsity-code agreement (88).

These are descriptor substitutions inside pair mining/gating or
similarity/feature distillation. Dark knowledge, activation transfer, neural
tangent/gradient kernels, and general pair weighting occupy the mechanisms.
Moreover, ImageNet logits explicitly expose imported semantic categories;
having the classifier weights present in the initialization does not make
their semantic targets contamination-free. Candidates 84--88 are **DEAD AT
GATE 2**.

## 89--93. Internalising the ensemble residual: excluded mechanisms in disguise

**Gate-2/algebraic deaths recorded 2026-08-01; no implementation or GPU.** The
largest measured headroom remains the seed ensemble: about 0.705 single-model
CUB R@1 versus 0.7468 for five models, with a transductive 512-D GPA fold
retaining 99.4% of the nine-model pack. A Claude round was required to capture
that residual in one deterministic backbone/vector while excluding
distillation, multiple heads, snapshots, weight averaging, Procrustes, MC
inference, and extra backbone capacity. Its proposals were input routing (89),
an input-dependent transform trained on seed disagreement (90), Hessian/stable-
rank shaping (91), auxiliary seed-mode prediction (92), and disagreement-
conditioned isotropy annealing (93).

These are, respectively, mixture-of-experts, local metric learning plus
ensemble distillation, loss-geometry regularization, auxiliary ensemble
distillation, and covariance regularization plus ensemble weighting. Renaming
the disagreement target a mode does not remove the teacher. More generally, a
single deterministic model cannot acquire information about cross-seed
disagreement unless training observes multiple seeds/outputs, maintains
multiple hypotheses, averages trajectories, or adds capacity; those are
exactly the excluded and occupied mechanisms. Candidates 89--93 are **DEAD AT
GATE 2**.

## 94. Single-seed consensus prediction: train-fit map worsens retrieval

**Gate-1 diagnostic failure recorded 2026-08-01; CPU only, no GPU.** The large
ensemble gap and the 98.9% inductive fold raised a distinct question: is the
five-seed consensus mostly a deterministic calibration of information already
inside one seed? Before measurement, a fixed uncentred ridge map from seed-0
train embeddings to the five-seed train-fit GPA consensus was registered, with
all fitting on train classes and frozen application to disjoint test classes.

Seed 0 scored 0.6940 CUB R@1; the five-seed concat scored 0.7350 and train-fit
GPA consensus 0.7205. The ridge prediction scored **0.6813**, a **-1.266-point**
change. An orthogonal control remained exactly 0.6940. This decisively violates
the registered +0.5-point minimum and shows that the linear map overfits
train-class cross-seed relations rather than recovering test-class ensemble
information. Candidate 94 is **DEAD AT GATE 1**. The remaining ensemble gain is
not available through a simple single-embedding calibration.

## 95. All-miss ensemble complementarity: real but too rare

**Gate-1 diagnostic recorded 2026-08-01; CPU only, no GPU.** A preregistered
five-pack analysis asked whether concatenation succeeds when every individual
seed misses. It rescued 15 of 5,924 CUB queries (**0.253%**), below the predicted
0.5% but above the 0.1% falsifier. Among those 15, the median worst-seed rank of
the correct class was 7 and 73.3% had a correct-class image in every seed's top
10, clearing the registered consistency prediction. Thus score averaging can
occasionally combine a consistently near correct class against different
seed-specific distractors, but the event is too rare to explain the main pack
gain or justify a method arm. Rank aggregation, consensus, and ensemble
distillation are established in any case. Candidate 95 is **INCONCLUSIVE AT
GATE 1 AND DOES NOT ADVANCE**.

## 96. Shadow-style anchor projection: exactly cosine triplet when normalised

**Gate-2/algebraic death recorded 2026-08-01; no implementation or GPU.** A
fresh 2026 horizon result claimed a scalar anchor-projection loss that improves
CUB, Cars, SOP, and In-Shop. In the withdrawn Shadow Loss manuscript's own
L2-normalised setting, however, its gaps reduce to `1 - a.p` and `1 - a.n`.
The proposed hinge is therefore exactly
`max(a.n - a.p + margin, 0)`, the ordinary cosine triplet hinge. A different
buffer implementation does not create a new similarity operator. The paper
also gives no seed count or uncertainty. Candidate 96 is **DEAD AT GATE 2**.

## 97. Norm-aligned principal-direction augmentation: occupied by ESA

**Gate-2 death recorded 2026-08-01; no implementation or candidate GPU.** A
pre-result audit of raw embedding magnitude considered whether norm could drive
something beyond quality weighting. ESA (Park et al., IEEE Access 2025,
https://doi.org/10.1109/ACCESS.2025.3637551) already links norm to confidence,
identifies train--test scale drift, lowers confidence for hard samples, and
augments embeddings along class principal eigendirections. It reports matched
three-seed Proxy Anchor improvements of +0.50 CUB and +0.73 Cars196. Norm-ranked
mining, norm consistency, and manifold alignment additionally reduce to hard
mining or geometry regularization. Candidate 97 is **DEAD AT GATE 2** regardless
of the pending magnitude diagnostic's sign.

## 98--101. Claude source audit: rank symmetry, graded positives, frequency, and manifold

**Gate-2/algebraic deaths recorded 2026-08-01; no implementation or candidate
GPU.** An adversarial generation round was constrained by the measured RSPG/ARCG,
ensemble-complementarity, ridge-consensus, regional, and TIRD failures. Four
apparent directions did not survive mechanism-level reduction.

- **98, reciprocal ranking-inversion supervision:** penalising disagreement
  between the rank of `x` from `y` and `y` from `x` is reciprocal-neighbour or
  listwise ranking supervision. With a symmetric metric it supplies no new
  relation beyond the same pair/set rankings already optimised by established
  listwise DML.
- **99, within-class appearance ranking:** tiering same-class peers using frozen
  appearance similarity and assigning tiered margins is graded positive mining
  or margin weighting. *Deep Metric Learning Beyond Binary Supervision* (Kim et
  al., CVPR 2019) already occupies learned continuous similarity beyond binary
  labels.
- **100, class-frequency supervision:** class cardinality is a distinct measured
  property but can enter this model only through sampling, class/example weights,
  adjusted margins, or added capacity. It is imbalance handling, not a new
  supervision operator.
- **101, unlabeled-manifold co-similarity:** neighbourhood and second-order
  co-similarity become clustering, contextual similarity, graph propagation, or
  pseudo-pair eligibility. HIER (Kim et al., CVPR 2023) and contextual similarity
  optimisation (Liao et al., arXiv:2210.01908) occupy the closest forms, while
  this repository has directly falsified multiple graph interfaces.

Candidates 98--101 are **DEAD BEFORE PREREGISTRATION**. The source taxonomy and
its limitation are recorded in `docs/search_space_source_audit_2026-08-01.md`:
future candidates must state both a source of information and an operator outside
the occupied reduction for that source.

## 102. Sparse-class covariance support: occupied by IAA

**Gate-2 death recorded 2026-08-01; no implementation or candidate GPU.** The
RSPG split (64.49% CUB versus 8.66% In-Shop pair retention) and In-Shop's many
small identities suggest estimating within-class appearance variation and
borrowing variation statistics across related classes. This is not open.
Chen et al., *Intra-class Adaptive Augmentation with Neighbor Correction for
Deep Metric Learning* (arXiv:2211.16264), estimate class-wise embedding
covariances, use neighbouring classes to correct sparse estimates, and sample
adaptive virtual embeddings for DML losses. They report CUB, Cars196, SOP,
In-Shop, and VehicleID experiments at roughly 2% runtime overhead. Embedding
Expansion, Symmetrical Synthesis, Proxy Synthesis, and variational/adversarial
sample generation are additional adjacent priors enumerated by that paper.

Changing the covariance estimator, conditioning it on this repository's
response signatures, or using a stricter uncertainty threshold would be a
variant of the same distribution-estimation-and-synthetic-support operator.
Candidate 102 is **DEAD AT GATE 2**.

## 103. Origin-anchored semantic radius: incompatible with the measured representation

**Algebraic death recorded 2026-08-01; no implementation or GPU.** The fitted
PCA diagnostic found that subtracting the train mean lowers disjoint-test CUB
R@1 by 1.215 points. Claude proposed interpreting the mean direction as an
origin-anchored semantic radius. That explanation is incompatible with the
measurement: the saved retrieval embeddings are L2-normalised, so every sample
has radius one. Mean subtraction changes pairwise angles; it does not reveal
class-conditional distance from the origin.

Explicitly privileging the mean direction would be an affine or anisotropic
metric/head transformation, while supervising magnitude would be norm/quality
modelling. Both are occupied and were excluded from this generation round. The
1.215-point observation remains evidence that centring is unsafe for cosine
retrieval, not provenance for new supervision. Candidate 103 is **DEAD BEFORE
GATE 2**.

## 104. Confusable negative-to-unknown editing: hierarchical relation weighting

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** The failures of
positive-to-unknown RSPG/ARCG leave a superficially different edit: preserve all
same-class positives, but turn a foreign-class negative into unknown when the
two training classes are empirically confusable. This could avoid erasing Proxy
Anchor's attractive term while preserving features shared by nearby classes.

The operation is occupied. HIER (Kim et al., CVPR 2023) discovers inter-class
hierarchy and supplies graded relations beyond binary class labels. Kim et al.,
*Deep Metric Learning Beyond Binary Supervision* (CVPR 2019), learn continuous
semantic similarity rather than treating every different-class relation as
equally negative. False-negative cancellation supplies the exact remove-from-
the-negative-set operation in contrastive learning. A hard unknown threshold is
therefore a binary relation weight/mask, not a new supervision operator; deriving
it from proxies, neighbours, response signatures, or early confusion changes
only its estimator. Candidate 104 is **DEAD AT GATE 2**.

## 105. Pre-normalisation magnitude is predictive, but every direct action is occupied

**Gate-1 diagnostic passed 2026-08-01; no candidate GPU.** At the exact
epoch-10 In-Shop Proxy Anchor operating point, correctly hooked raw head-output
magnitude correlated **0.18675** with query R@1 correctness and **0.32574**
(Spearman) with retrieval margin. Both clear the preregistered 0.15/0.20
prediction, and train identity ICC was **0.57754**. The first export that returned
unit norms was rejected as an instrumentation failure before adjudication; the
corrected hook and its regression test were committed before replay.

The measurement establishes a missing observable, not a novel operator.
MagFace, AdaFace, IDML, SEC, ESA, and IAA occupy magnitude/quality-aware margins,
similarities, confidence, augmentation, and synthetic support. Norm-ranked
mining, norm-conditioned gates, norm consistency, and dynamic margins reduce to
the same quality weighting or geometry regularisation. Candidate 105 therefore
**PASSES GATE 1 BUT CANNOT ENTER GATE 3** without a separately stated and audited
operator. No method run follows from the positive number.

## 106--108. Claude norm-conditioned operators: endogenous targets or excluded scaling

**Algebraic deaths recorded 2026-08-01; no implementation or GPU.** A
result-conditioned Claude round was given the positive magnitude correlations
and explicitly excluded known quality weighting, margins, mining, gates,
uncertainty similarity, regularisation, augmentation, synthetic support, and
architecture changes.

- **106, norm-stratified ordinal supervision:** bin identities by the model's
  current raw norm and require the high-norm bin to have a larger cosine margin.
  This initially appeared to survive, but failed adversarial follow-up. Raw norm
  is endogenous and PA removes it before similarity, so the model can change
  radial scale without adding angular retrieval information. The rule rewards a
  partition the model itself creates, has discontinuous moving bins, reverses
  the causal interpretation of a correlation, and unjustifiably promotes a
  query-level association to an identity-level ordering. Claude retracted it as
  self-reinforcing regularisation rather than supervision.
- **107, norm as latent margin predictor:** multiplying similarity by a learned
  function of norm is exactly quality scaling/dynamic margin modelling.
- **108, norm-inversive dual comparison:** requiring rankings to agree under
  ordinary and norm-rescaled paths is a norm-conditioned auxiliary branch and
  sample-dependent scaling, not a non-separable relation.

Candidates 106--108 are **DEAD BEFORE GATE 2**. The positive diagnostic remains
descriptive unless an exogenous target and a non-occupied action are both found.

## 109. Within-identity norm signal survives; action remains occupied

**Gate-1 refinement passed 2026-08-01; CPU only.** Because train identity ICC was
0.57754, the aggregate norm result could have been entirely identity difficulty.
A separately preregistered decomposition rejected that explanation. Across
14,205 query images in 3,972 repeated identities, within-identity centred norm
correlated **0.14170** with centred correctness and **0.20972** (Spearman) with
centred retrieval margin, clearing the registered 0.10/0.15 prediction. Between-
identity correlations were still larger at **0.25870** and **0.45322**.

Raw magnitude therefore encodes both image-level quality and identity-level
difficulty. This is stronger provenance for quality-aware DML, but MagFace,
AdaFace, IDML, SEC and ESA remain prior art for every direct action found. The
measurement is **LIVE; THE METHOD SEARCH DOES NOT ADVANCE** without a novel
operator.

## 110. Restoring raw magnitude to similarity catastrophically hurts

**Gate-1 diagnostic failed 2026-08-01; CPU only.** The positive within-identity
quality signal motivated a preregistered test of whether raw magnitude also
contains relational information. It does not. On the frozen In-Shop artifact,
normalized cosine scored **0.84365** R@1, raw Euclidean **0.82163** (-2.201
points), and raw dot product **0.60051** (-24.314 points). The prediction that a
canonical raw score would gain at least 0.20 point was falsified.

Raw dot ranks by gallery magnitude times cosine because query magnitude is a
query-wise constant. Its collapse shows that identity/quality scale is not
semantic compatibility. Magnitude should remain confidence, not similarity;
unnormalised retrieval and norm-product comparisons are **DEAD AT GATE 1**.

## 111. Degradation-ordered norm supervision: occupied quality auxiliary task

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** The surviving
within-identity norm signal and the failure of raw-magnitude ranking suggest an
exogenous target: apply controlled degradations to one image and require raw
head norm to decrease monotonically with registered severity, while cosine
direction retains identity.

This is not an open supervision operator. MagFace and AdaFace establish
recognition-utility/quality control through embedding magnitude; AugSelf
predicts augmentation parameters as an auxiliary representation task; and
synthetic-degradation supervision is a standard mechanism in face/image quality
assessment (including degradation-representation learning such as DSL-FIQA,
CVPR 2024). The proposed ordering is an auxiliary quality regularizer, and the
trained scalar is discarded by the benchmark's direction-only retrieval—the
raw-similarity diagnostic shows why. Candidate 111 is **DEAD AT GATE 2**.

## 112. Relative similarity smoothness: constant bound or Jacobian regularisation

**Algebraic death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Claude proposed perturbing an embedding and penalising the ratio of cosine-score
change to perturbation size. For a unit gallery vector `y`, however,
`s(x + e, y) - s(x, y) = e dot y`, whose absolute value is already bounded by
`||e||`. Noise injected after the encoder makes the proposed penalty independent
of the learned model (apart from a normalization projection); there is no new
supervision to optimize.

Moving the perturbation to pixels or intermediate activations makes the term an
ordinary Jacobian/Lipschitz/consistency regularizer, with spectral normalization,
gradient penalties, virtual adversarial training, and augmentation consistency
as prior art. The proposal also explicitly violated the generation round's
no-regularizer constraint. Candidate 112 is **DEAD BEFORE GATE 2**.

## 113--114. Density routing and conformal class sets: weighting and pseudo-labels

**Algebraic deaths recorded 2026-08-01; no implementation or GPU.** A Claude
round was asked to import a genuinely different operator from topology, causal
inference, cooperative games, coding theory, or statistical decision theory.
Its density-stratified supervisor (113) routes high-, middle-, and low-density
examples to different standard objectives. Claude's own reduction wrote it as a
weighted sum of those objectives; changing weights discontinuously by density
does not escape weighting/curriculum.

Its fallback (114) constructs an online conformal set of plausible class proxies
and applies a multi-positive margin against proxies outside the set. Static or
online, this is a self-generated multi-label/pseudo-label mask plus an ordinary
set margin. Recomputing the target from the current embedding makes it dynamic,
not algebraically new; HIER, contextual-similarity optimisation, clustering,
and confidence-set pseudo-labelling occupy the mechanism. It also replaces a
known training label with an endogenous class set without a new information
source. Candidates 113--114 are **DEAD BEFORE GATE 2**.

## 115. Cross-seed prototype-offset drift: tautological diagnostic, occupied action

**Gate-1 failure recorded 2026-08-01; no diagnostic or GPU.** Claude proposed
measuring each training image's offset from its class mean across independent
seeds. The registered-looking prediction that 10--25% of images would lie in the
top quartile is tautological: exactly 25% lie there by definition. Raw offset
vectors are also incomparable until the independently rotated embedding gauges
are aligned, after which the statistic is another form of the ensemble/trajectory
disagreement already audited in this repository.

Most decisively, the proposed interventions were to reweight high-drift samples
or mine their hard negatives. Thus even a non-tautological drift statistic would
predict an occupied weighting/mining estimator, not new supervision. Candidate
115 is **DEAD AT GATE 1**.

## 116. Raw norm as directional concentration: exactly NIR prior art

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** The paired
findings that norm predicts within-identity correctness while raw norm-product
similarity collapses suggest a coherent probabilistic interpretation: direction
is semantic mean and raw magnitude is concentration/uncertainty on the sphere.

Kirchhof et al., *A Non-isotropic Probabilistic Take on Proxy-based Deep Metric
Learning* (ECCV 2022, arXiv:2207.03784), make exactly this argument. They model
image embeddings as vMF natural parameters (direction plus norm concentration),
derive non-isotropic vMF class proxies, and study point/distribution and
distribution/distribution metrics on standard DML benchmarks. IDML and Bayesian
metric learning are adjacent uncertainty-aware comparisons. Projected-normal
instead of vMF, or Proxy Anchor instead of ProxyNCA++, changes the distribution
family/base loss rather than the mechanism. Candidate 116 is **DEAD AT GATE 2**.

## 117. Persistence-diagram positive gate: descriptor metric plus pair mining

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The motivating measurement was the In-Shop region diagnostic: replacing fixed
coordinate matching (R@1 **0.5775**) with position-tolerant MaxSim (**0.6442**)
recovered 6.67 points, although both remained below the global descriptor. This
suggested testing whether topology of an intermediate feature map could preserve
arrangement while tolerating displacement. The candidate would compute a
persistence diagram per image and change a labelled same-class pair from positive
to unknown when the two diagrams disagree.

The operator does not survive a mechanism-level decomposition. Persistence images
and sliced-Wasserstein and learned kernels already turn diagram agreement into a
similarity (Adams et al., 2017; Carriere, Cuturi, and Oudot, ICML 2017; Zhao and
Wang, 2019). Differentiable topology layers already place persistence-derived
functionals inside learned objectives (Gabrielsson et al., AISTATS 2020). RETA
(Li et al., CVPR 2026) additionally aligns persistence images of mutual-kNN
feature graphs during vision training. Thresholding any established similarity
to accept or reject a training pair is ordinary pair mining; substituting a
diagram distance for cosine distance does not create a new supervision mechanism.
The claimed hard positive-to-unknown gate is therefore an obvious composition of
two occupied operators, not a defensible novelty. Candidate 117 is **DEAD AT GATE
2**.

## 118. Counterfactual patch-swap supervision: spatial Metric Mix

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
This branch asked whether donor/recipient labels from controlled cross-image
patch swaps could add information that the current embedding cannot erase: paste
a region from identity `b` into identity `a` and supervise the composite by the
known contribution of both source identities. It is exogenous and data-only, but
the supervision operator is occupied.

Venkataramanan et al., *It Takes Two to Tango: Mixup for Deep Metric Learning*
(ICLR 2022), define Metric Mix (Metrix) precisely to mix examples and their
metric-learning targets. Their formulation covers input, intermediate-feature,
and embedding mixing and explicitly studies anchor/positive/negative source
pairs. CutMix changes the interpolation mask from dense to spatial; donor and
recipient labels remain the same mixed target. Patch/part erasing, swapping, and
identity-preserving augmentation are additionally established in person ReID.
Recasting Metrix's mixture coefficient as the pasted area or a measured feature
effect does not create new supervision; it is mixup target assignment or
reweighting. Candidate 118 is **DEAD AT GATE 2**.

## 119. Cross-instance transformation parallelograms: equivariance supervision

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The post-stopping loop searched for exact relational labels created by controlled
interventions rather than extracted from the current embedding. The first proposal
applies the same known transform `T` to two images and requires
`f(Tx)-f(x) = f(Ty)-f(y)`, so a transformation has a shared displacement across
identities. ARCG's reproducible, non-distance augmentation-response structure is
the repository provenance.

This is an equivariance constraint, not a new similarity-learning supervision
object. Feige, *Invariant-Equivariant Representation Learning for Multi-Class
Data* (ICML 2019), already decomposes class-invariant and transformation-equivariant
latents. Marchetti et al., *Equivariant Representation Learning via Class-Pose
Decomposition* (AISTATS 2023), train explicitly from relative symmetry information,
and symmetric embedding networks learn a known feature-space action for input
transformations (Park et al., ICML 2022). Matching displacement vectors is a
linear parameterization of the same group-action consistency; using it beside
Proxy Anchor makes it an auxiliary equivariance regularizer. Candidate 119 is
**DEAD AT GATE 2**.

## 120. Norm-stratified metric partitions: conditional metric plus uncertainty

**Gate-2/algebraic death recorded 2026-08-01; no diagnostic, implementation, or
GPU.** Asked for an information-producing operation outside the 119-candidate
taxonomy, Claude proposed partitioning examples by raw-norm regime and selecting
a different comparison metric per regime, with a fallback suggestion to train
separate direction and scale heads.

The first operation is a conditional similarity network or mixture of metrics:
the norm routes a pair to an existing score, and any continuous version is
adaptive weighting. The second is the established semantic-direction plus
quality/concentration decomposition occupied by NIR, MagFace/AdaFace, and
probabilistic DML. Repository evidence additionally contradicts scale as a
relational channel: within-identity norm predicts correctness, but raw dot and
raw Euclidean retrieval lose 24.31 and 2.20 points. A regime cannot manufacture
pairwise information absent from its routing scalar. Candidate 120 is **DEAD AT
GATE 2**.

## 121. Counterfactual mutual-necessity matching: saliency-guided hard attention

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Leave-one-region-out occlusion of the correct-class proxy score would label
identity-necessary regions, and a pair would compare only regions labelled
necessary in both images. This was motivated by MaxSim recovering 6.67 points
over fixed-coordinate regions while remaining 3.6 points below global Proxy
Anchor.

Occlusion necessity is established perturbation attribution. Intersecting two
binary attribution masks is hard pairwise attention, and matching the retained
regions is saliency-guided structural matching. DIML already optimizes and
evaluates explicit feature-map correspondences; self-supervised image-to-region
similarity and attention-based DML already use region importance as retrieval
supervision. The causal provenance of a mask does not change the downstream
operator from masking/weighting correspondences. Candidate 121 is **DEAD AT
GATE 2**.

## 122. Gromov--Wasserstein internal-structure comparison: established correspondence

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The region result motivated a genuinely nonseparable comparison: represent each
image by the metric-measure space of its intermediate regions, then compare two
images by Gromov--Wasserstein distance so matching preserves region--region
relations without requiring fixed coordinates.

Gromov--Wasserstein distance is already the canonical isometry-invariant
comparison of metric-measure spaces and is established for graph matching and
representation alignment. More directly, Im, Liu, and Hong, *Shape-of-You:
Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild*
(CVPR 2026), jointly optimize inter-feature similarity and intra-structural
consistency to match image regions. DIML is the adjacent DML structural-matching
baseline. Using the same operator as a retrieval score or Proxy Anchor companion
changes the benchmark/application, not the comparison mechanism. Candidate 122
is **DEAD AT GATE 2**.

## 123. Per-query augmentation adaptation: established test-time training

**Gate-2/protocol death recorded 2026-08-01; no diagnostic, implementation, or
GPU.** ARCG measured heterogeneous image-specific transformation response, which
suggested adapting the embedding separately for each unseen query using only its
controlled augmented views.

Tursun et al., *Learning Test-time Augmentation for Content-based Image Retrieval*
(2020), already learn transformation policies and aggregate augmented query
features specifically for retrieval. Test-Time Training (Sun et al., 2020)
already updates a model on a single unlabeled test example through self-supervision,
and later retrieval TTA methods adapt online query representations. Choosing an
augmentation-consistency objective for a DML encoder is an application of these
operators, not a new method. Multi-view aggregation also violates the registered
single-view comparison used to reject IDEAL/BLenDeR-style capacity advantages.
Candidate 123 is **DEAD AT GATE 2**.

## 124. Self-supervised gradient signatures: Fisher features and FUNGI prior

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The repository already measures per-image and per-group gradient agreement, so
the next candidate represented an unseen image by the parameter gradient of an
augmentation-consistency objective and compared those gradients alongside the
ordinary embedding.

This is the Fisher-kernel construction: encode an observation by the score of a
model with respect to its parameters and compare score vectors. More directly,
Simoncini et al., *No Train, all Gain: Self-Supervised Gradients Improve Deep
Frozen Representations* (FUNGI, 2024), compute per-image gradients from
self-supervised objectives, project them, and concatenate them with frozen visual
features for nearest-neighbour recognition. Choosing a DML encoder or a different
pretext loss changes the source model, not the representation mechanism.
Candidate 124 is **DEAD AT GATE 2**.

## 125. Acquisition-crossing positives: camera-aware ReID and metadata leakage

**Gate-2/protocol death recorded 2026-08-01; no diagnostic, implementation, or
GPU.** RSPG's accepted In-Shop graph was 89.08% concentrated within filename
acquisition tokens, suggesting an explicit countermeasure: preserve the identity
label but apply positive attraction chiefly across acquisition groups so session,
background, and model-shot shortcuts cannot satisfy it.

Cross-camera positive learning, camera-diversity losses, camera-aware similarity
consistency, and view-confusion training are established person-ReID mechanisms
(including Wu et al., ICCV 2019, and Lee et al., ICCV 2023). More fundamentally,
the In-Shop acquisition token was inferred from filenames and is not part of the
registered generic DML supervision. Treating it as ground truth would exploit a
dataset-specific metadata convention unavailable on CUB/Cars and would not be a
like-for-like similarity-learning result. Candidate 125 is **DEAD AT GATE 2 AND
ON PROTOCOL**.

## 126. Foundation projection with radius control: occupied composite, contaminated scale

**Gate-1/Gate-2 death recorded 2026-08-01; no new GPU.** An inventory of legacy
DGX artifacts found frozen DINOv2-small CUB R@1 0.85466 improving to 0.85770 and
frozen SigLIP Cars196 0.96458 improving to 0.96987 after an 80-step Group SupCon
+ XBM + radius projection. These are one-seed historical results under a
different pretrained-backbone regime.

The delta cannot motivate a novel operator: supervised contrastive grouping,
cross-batch memory, and class-radius regularization are each established, and
the composite artifact has no component attribution. The absolute scale is not
like-for-like with the corrected BN-Inception/ResNet-50 matrix and has no audit
for CUB/Cars presence in foundation-model pretraining. Replicating the increment
could validate a projection recipe but could not establish method novelty.
Candidate 126 is **DEAD AT GATES 1--2**. Full audit:
`docs/historical_artifact_audit_2026-08-01.md`.

## 127. Pairwise conditional description length: established compression distance

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
To escape static descriptor distances, this proposal defined similarity by the
reduction in code length when one image conditions the compression of another.
The intended operator is nonseparable and pair-dependent.

It is nevertheless established. Normalized information/compression distance
defines similarity from individual and joint code lengths; Cerra and Datcu
(2012) use fast compression distance for content-based image retrieval, and
Guha and Ward (2012) measure image similarity through sparse conditional
compressibility. Nikvand et al. (2018) explicitly propose normalized conditional
compression distance for visual similarity, recognition, and retrieval. More
recent descriptive-autoencoding work likewise derives conceptual similarity
from conditional description complexity. Replacing a classical compressor with
a learned one changes the approximation to code length, not the comparison
operator. Candidate 127 is **DEAD AT GATE 2**.

## 128. Hypergraph encounter-order tracking: underidentified seed audit, no new operator

**Gate-1 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Given the historical hypergraph-incidence seed-0 outlier, Claude proposed
correlating sample encounter order with the three seed outcomes and treating a
lucky ordering as a new information source.

The proposal is underidentified. In this implementation incidence is recomputed
inside stochastic minibatches; seed changes initialization, augmentation,
sampling order, optimizer trajectory, and nondeterministic kernels together.
Three outcomes cannot attribute the difference to encounter order. Moreover,
determinizing, stratifying, or averaging batch order is a sampling/training
protocol change, not new supervision or a comparison operator. The already
measured fixed-seed nondeterminism and six-seed CUB variance make the outlier
expected rather than mechanistically diagnostic. Candidate 128 is **DEAD AT
GATE 1**.

## 129. Associative retrieval dynamics: modern Hopfield prior

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
This branch replaced a static pair score with iterative energy minimization: an
unseen query would evolve toward memories learned only from training identities,
and the converged state would be used for inductive retrieval.

Modern Hopfield networks are exactly differentiable content-addressable memories;
their update is attention over stored patterns. k-Hopfield layers retrieve the
k-nearest memories, U-Hop learns a separating feature map for improved memory
retrieval, and Hopfield--Fenchel--Young networks generalize the energy and report
image-retrieval experiments. Storing training samples, prototypes, or proxies
changes the memory contents, not the associative operator. Using the held-out
gallery instead would additionally be transductive. Candidate 129 is **DEAD AT
GATE 2**.

## 130. Score-flow embedding denoising: separable post-processing map

**Algebraic death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The candidate learned a score/mean-shift field over training embeddings and
iteratively denoised each unseen embedding toward the learned identity manifold
before ordinary cosine retrieval.

For a training-fitted deterministic flow `D`, its score is
`cos(D(f(x)), D(f(y)))`. This is exactly a separable embedding map `D o f`; the
iterations can be unrolled or amortized into an additional head. Score matching
is then a generative/denoising auxiliary objective, with diffusion representation
learning, denoising autoencoders, and deep mean-shift priors as direct families.
Conditioning `D` jointly on `(x,y)` would instead be a learned non-metric pair
network. Neither form creates a new comparison or supervision object. Candidate
130 is **DEAD BEFORE GATE 2**.

## 131. Convex-set image embeddings: visual-overlap boxes already exist

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The coexistence of image-specific uncertainty and region ambiguity motivated
representing each image as a box/convex set and scoring two images by normalized
intersection volume, avoiding the failed Tversky arm's sign threshold on dense
coordinates.

Rau et al., *Predicting Visual Overlap of Images Through Interpretable Non-Metric
Box Embeddings* (2020), already map images to boxes and use asymmetric
intersection/containment ratios as an image-matching score. Probabilistic box
embeddings provide differentiable expected intersections, and IDML is the direct
uncertainty-aware DML neighbour. Substituting identity supervision for visible
surface-overlap supervision changes the label semantics, not the set-valued
representation or overlap comparison. Candidate 131 is **DEAD AT GATE 2**.

## 132. Per-image Grassmann subspaces: direct local-feature prior

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The failure of fixed-coordinate region matching, despite the large MaxSim gain,
motivated representing each image by the subspace spanned by its local features
and comparing two images through principal angles or a learned Grassmann
distance. This would preserve several plausible local appearances without
committing to a probability or box-overlap model.

The operator is already explicit in the primary literature. Wei et al.,
*Grassmann Pooling as Compact Homogeneous Bilinear Pooling for Fine-Grained
Visual Classification* (ECCV 2018), convert each image's local CNN activation
matrix to its principal-singular-vector subspace and reduce image similarity to
principal-angle comparison. Hu et al., *Subspace Representation Learning for
Few-shot Image Classification* (2021), likewise represent an individual image
by a subspace in local CNN feature space, learn a weighted subspace distance
between images, and evaluate on CUB. End-to-end training, a retrieval loss, or a
different benchmark changes the recipe rather than the representation and
comparison mechanism. Candidate 132 is **DEAD AT GATE 2**.

## 133. Local-correspondence positive eligibility: occupied retrieval training

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The large gain from local-feature MaxSim, together with the failure of a
trainable region arm, motivated using geometric/local-feature agreement only to
decide which labelled same-class image pairs remain eligible positives. This
would change the supervision graph without requiring local matching at test
time.

That mechanism is occupied. Lee et al., *Correlation Verification for Image
Retrieval* (CVPR 2022), state that non-overlapping same-class landmark pairs can
interfere with learning and preselect overlapping positive pairs using DELF
local-feature evidence. Xu et al., *Deep Asymmetric Metric Learning via Rich
Relationship Mining* (CVPR 2019), more generally reject the all-positive-pairs
constraint and construct a visual-distance minimum spanning tree per class,
allowing visually distant same-class images to remain only indirectly
connected; they evaluate on CUB, Cars196, and SOP. Replacing DELF or embedding
distance with a new local-consistency statistic changes the edge estimator, not
the discrete same-class eligibility mechanism. Candidate 133 is **DEAD AT GATE
2**.

## 134. Cross-trajectory consensus gating: teacher interaction selection

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The observed 1.08-point spread between nominally identical fixed-seed runs
motivated training two independently initialized heads and retaining a labelled
same-class edge only when both heads agree that it is positive. The intended
observable was relation stability across optimization trajectories rather than
hardness within one trajectory.

The mechanism is occupied by co-teaching and, more directly, by Ibrahimi et al.,
*Learning with Label Noise for Image Retrieval by Selecting Interactions*
(WACV 2022). T-SINT uses a teacher-based setup to identify and exclude unreliable
entries of the retrieval distance matrix, including positive and negative
interactions. Noise-resistant DML also imports two-network co-teaching and
ranking-based selection. Using head agreement instead of teacher confidence is
an uncertainty estimator inside the same cross-model interaction-selection
operator; it also adds training cost without identifying the source of this
repository's between-trajectory variation. Candidate 134 is **DEAD AT GATE 2**.

## 135. Mutual gradient-influence positive gating: DML influence prior

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The fixed-seed 1.08-point spread motivated defining a same-class edge by causal
training helpfulness rather than appearance: retain `A--B` only when an update
on `A` reduces `B`'s retrieval loss and the converse also holds, approximated by
gradient dot products or an influence function.

Liu et al., *Debugging and Explaining Metric Learning Approaches: An Influence
Function Based Perspective* (NeurIPS 2022), already design an empirical
influence function for DML, identify training samples responsible for retrieval
generalization errors, and study CUB, Cars196, and In-Shop. TracIn and
gradient-similarity data selection already operationalize training-example
helpfulness. Applying that estimator in both directions and thresholding it at
pair granularity is a composition of influence-based data selection and ordinary
positive mining; symmetry is a policy, not a new supervision principle.
Candidate 135 is **DEAD AT GATE 2**.

## 136. Spectral class connectivity: fragmentation predicts better retrieval

**Preregistered 2026-08-01 before implementation or GPU.** Exact epoch-10
In-Shop embeddings show that 40.33% of within-class 1-NN graphs are disconnected.
A Fiedler-gradient diagnostic differs from ordinary farthest-positive mining in
25.48% of classes, narrowly clearing the registered 20% minimum. The candidate
maximizes the algebraic connectivity of each four-sample class affinity graph,
supervising the existential property that the class has no weak cut rather than
selecting labelled positive edges.

DAMLRRM's fixed per-class minimum spanning tree is the closest DML prior;
Fiedler regularization and spectral clustering apply the mathematics to network
topology or clustering rather than a zero-shot-retrieval class graph. No direct
primary-source collision was found. The qualified novelty case, diagnostic,
matched-IPC4 control, numerical prediction, and falsification rule are in
`docs/spectral_class_connectivity_candidate.md`.

A stronger CPU diagnostic then falsified the premise before the spectral arm
started. Fragmented classes have class-balanced leave-one-out R@1 0.94813 versus
0.93605 for connected classes (+1.208 pt), and exact class-size matching increases
the difference to +3.534 pt. Thus disconnectedness marks legitimate multimodal
structure associated with better retrieval, not a failure needing repair. The
partial IPC4 control was killed after epoch 5 and excluded; no `pa_fiedler`
artifact exists. Candidate 136 is **DEAD AT GATE 1**.

## 137. Weak-cut preservation: occupied intra-class diversity objective

**Gate-2 death recorded 2026-08-01; no implementation or GPU.** Candidate 136's
post-mortem established a real opposite-sign measurement: after exact class-size
matching, In-Shop classes with disconnected within-class 1-NN graphs have
leave-one-out R@1 **+3.534 points** above connected classes. Candidate 137 would
therefore preserve weak cuts, for example by penalizing increases in the class
Laplacian Fiedler value while Proxy Anchor retains identity supervision.

The mechanism is occupied even though the statistic is new. Ranked List Loss
explicitly preserves useful within-class hypersphere structure; Deep
Compositional Metric Learning targets the generalization harm from suppressing
intra-class variation; Self-Expanded Equalization, Deep Disentangled Metric
Learning, and DAAL preserve diverse or adaptively subgrouped within-class
representations. Reverse-contrastive and anti-collapse objectives cover direct
same-class spreading. A Fiedler penalty changes the regularizer used to preserve
diversity, not the supervision object, and its unconstrained optimum isolates
samples. Candidate 137 is **DEAD AT GATE 2**.

## 138. Reconstruction-derived discrepancy supervision: occupied by AdvRF

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The MaxSim recovery and the finding that disconnected class modes retrieve well
motivated replacing category contraction with a training-data-only reconstruction
signal: use residual regions that an image model cannot reconstruct to supervise
category-agnostic visual discrepancies, then retain a single retrieval embedding
at inference.

Wang, Shi, and Li, *Adversarial Reconstruction Feedback for Robust Fine-grained
Generalization* (ICCV 2025), already reformulate fine-grained image retrieval as
visual discrepancy reconstruction. Their reconstruction model exposes residual
discrepancies overlooked by the retrieval model, the two models refine discrepancy
localization adversarially, and the category-agnostic representation is distilled
into the deployed retrieval model. A masked autoencoder, non-adversarial residual,
or cheaper decoder changes capacity and optimization, not the reconstruction-derived
supervision object. Candidate 138 is **DEAD AT GATE 2**.

## 139. Training-only latent visual-attribute supervision: occupied by VAPNet

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The finding that disconnected same-class modes retrieve better suggested deriving
pose, viewpoint, or appearance factors from training images and using those factors
as richer supervision than the category label, with no text encoder, manual
attributes, or external generator.

Wang et al., *Learning to Parameterize Visual Attributes for Open-set Fine-grained
Retrieval* (NeurIPS 2023), already solve this exact problem. VAPNet learns visual
attributes from known-category images without attribute annotations, discovers rich
semantics from local patches, refines the attributes online, and uses them as
supervisory signals to tune a retrieval model for unknown categories. Naming the
latent factors viewpoint or appearance, using deterministic crops, or replacing its
attribute encoder changes the factor estimator rather than the training-only latent
attribute supervision mechanism. Candidate 139 is **DEAD AT GATE 2**.

### Horizon correction forced by candidates 138--139

The primary-source checks did more than kill two mechanisms. VAPNet reports
standard-split, single-model Recall@1 of **0.762 CUB, 0.948 Cars196, and 0.939
In-Shop**; AdvRF reports **0.766 CUB and 0.949 Cars196** and does not evaluate
In-Shop. Both deploy a ResNet-50/GAP 2048-D representation after 200 training
epochs and neither states a seed count or uncertainty. Therefore the previous
language treating 0.9038 In-Shop as an open general ceiling is withdrawn.

The repository's 512-D, corrected, multi-seed matrix remains a legitimate
controlled regime, and 0.9038 remains a cheap In-Shop screening threshold. A
method that clears it has only passed a recipe-matched screen; it has not beaten
the published general benchmark horizon. Missing error bars weaken fine-grained
comparisons but are not grounds to ignore reviewed standard-split results. Full
audit: `docs/open_set_fg_retrieval_horizon_2026-08-01.md`.

## 140. Density-gradient alignment: occupied density attraction

**Gate-1/2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The proposed CPU test would correlate within-class density-gradient direction
with retrieval outcome and, if favourable, attract samples toward dense class
regions. The number had not been measured, so it cannot supply provenance.
Moreover, *On Learning Density Aware Embeddings* (Ghosh et al., CVPR 2019)
already iteratively shifts class centres toward dense regions and attracts
embeddings there. Candidate 140 is **DEAD**.

## 141. Class-size-conditioned margin: scalar class balancing

**Gate-1/2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The proposed correlation between rival cardinality and retrieval outcome is
unmeasured, while the proposed action merely scales an existing margin by class
frequency. Class-balanced loss and adaptive/hierarchical metric margins already
occupy that operation. It changes pressure, not the supervision relation.
Candidate 141 is **DEAD**.

## 142. Perturbation-stable ranking: adversarial ranking defense

**Gate-1/2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Measuring top-k stability under small embedding perturbations would not itself
motivate new supervision; penalising rank changes is robustness regularisation.
Zhou et al., *Adversarial Attack and Defense in Deep Ranking* (ECCV 2020),
already trains retrieval models against ranking perturbations on CUB, Cars196,
and SOP. Candidate 142 is **DEAD**.

## 143. Persistent-homology class repair: topological regularisation

**Gate-1/2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The suggested first-Betti-number diagnostic is unmeasured and its proposed
action is to remove loops from learned class clouds. Persistent-homology
penalties for simplifying learned topology are established, including Chen et
al., *A Topological Regularizer for Classifiers via Persistent Homology*
(AISTATS 2019). Candidate 143 is **DEAD**.

## 144. Local-curvature flattening: manifold regularisation

**Gate-1/2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The proposed local Hessian-curvature correlation is unmeasured, and flattening
the resulting geometry is a scalar manifold regularizer rather than additional
supervision. *Curvature Regularization to Prevent Distortion in Graph
Embedding* (Pei et al., NeurIPS 2020) directly occupies curvature-controlled
embedding. Candidate 144 is **DEAD**.

The full generation and adversarial audit is in
`docs/post_horizon_candidate_batch_2026-08-01.md`.

## 145. Learnable visual-concept decomposition: occupied by VCE

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The 6.67-point local MaxSim recovery motivated learning concept slots from
regional features and aggregating them into one deployed descriptor. Wang et
al., *VCE: Visual Concept Embedding for Open-set Fine-grained Image Retrieval*
(Knowledge-Based Systems 2025), already uses learnable concept vectors,
cross-attention over regional features, independence constraints, and concept
relation modelling for the same open-set FGIR objective. Together with VAPNet,
this closes the learnable local-concept route. Candidate 145 is **DEAD AT GATE
2**.

## 146. Local-similarity-to-global distillation: occupied self-distillation

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The same MaxSim measurement motivated computing an expensive local relation
matrix only during training and teaching a compact global descriptor to
reproduce it. S2SD (Roth et al., ICML 2021) already transfers similarities from
auxiliary high-dimensional embedding and feature spaces into the deployed DML
embedding; Global-Local Self-Distillation (Lebailly et al., WACV 2023) directly
studies local-to-global feature matching. MaxSim changes the teacher kernel,
not the similarity-distillation mechanism. Candidate 146 is **DEAD AT GATE
2**. Full scan: `docs/recent_fg_operator_scan_2026-08-01.md`.

## 147. Trajectory-aligned mode variance: augmentation-response supervision

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Preserving class modes whose members follow similar augmentation trajectories
is the ARCG/AugSelf observable routed through invariance or pair weighting.
Changing the statistic from response agreement to trajectory variance does not
change the operator. Candidate 147 is **DEAD**.

## 148. Direct nearest-neighbour event optimisation: listwise ranking

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Directly optimising the labelled top-1 retrieval event seems to avoid proxies,
but every differentiable top-k surrogate assigns pressure to the winning
positive and negative comparisons. Ranked List Loss, Smooth-AP, and
differentiable ranking already occupy this supervision object. Candidate 148 is
**DEAD**.

## 149. Class-conditional implicit density fields: occupied density DML

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Fitting a scalar density field per labelled training class would preserve
multiple modes, but it remains class-conditional density modelling. Density
Aware Metric Learning and density-adaptive DML occupy the objective; replacing
their estimator with a small implicit network is capacity, not a new
supervision relation. Candidate 149 is **DEAD**.

Claude also proposed an algebraic impossibility proof based on gradient
factorisation. An adversarial follow-up correctly rejected that proof while
still finding no concrete counterexample. The distinction is recorded in
`docs/operator_counterexample_audit_2026-08-01.md`.

## 150. Cross-layer rescue supervision: deep supervision or distillation

**Gate-2 death recorded 2026-08-01; no export, implementation, or GPU.** A
proposed operating-point diagnostic would count final-head errors rescued by
intermediate backbone layers. Every recovery action is already named: feature
fusion/routing uses the intermediate representation at test; a companion loss
is Deeply-Supervised Nets; transferring intermediate similarities into the
512-D head is S2SD-style feature self-distillation; preservation constraints
are regularisation. Because no outcome motivates an unoccupied operator, even
the diagnostic would be purposeless. Candidate 150 is **DEAD**.

## 151. Parity-coded multimodal supervision: supervised hashing and ECOC

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The +3.534-point benefit associated with disconnected In-Shop class graphs
motivated a union of learned image codewords with shared parity checks. Deep
supervised hashing already jointly learns semantic codes and continuous image
representations, while ECOC networks impose algebraically separated class
codes. Multiple codewords per class are discrete sub-centres; discarding the
code head at test makes it auxiliary supervision rather than a new similarity
operator. Candidate 151 is **DEAD**. Full audit:
`docs/reopened_design_batch_150_151.md`.

## 152. Cross-instance transformation algebra: group-action prior art

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
The beneficial disconnected-class result and local MaxSim recovery motivated
learning low-rank same-class transformation operators, enforcing composition
over triples, and sharing transformation atoms across classes. VISALOGY already
learns cross-image visual transformation analogies; SymReg (NeurIPS 2022)
explicitly learns simple latent group actions for unknown input
transformations; global temporal alignment uses cycle-consistent
correspondence as representation supervision. The proposed cycle law and atom
dictionary instantiate that established group-action operator rather than
creating new supervision. Candidate 152 is **DEAD AT GATE 2**. Full audit:
`docs/transformation_algebra_candidate.md`.

## 153. Immune negative selection: Proxy Anchor negative term alone

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Claude initially judged class-specific exclusion detectors live. Algebraic
adversarial review reversed it: the normalized detector loss is exactly Proxy
Anchor's foreign-proxy negative term with the non-collapse positive term
removed. Empirical detector anchoring becomes complementary-label/one-vs-rest
classification or classical real-valued artificial-immune coverage. Candidate
153 is **DEAD**.

## 154. Sheaf transport between class modes: learned sheaf regularisation

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Learned fibers, edge transports, and cycle holonomy are the objects of Neural
Sheaf Diffusion; candidate 152 already established the adjacent latent
group-action/cycle-consistency prior. Applying those objects to a within-class
mode graph changes the application, not the geometric consistency operator.
Candidate 154 is **DEAD**.

## 155. Viability-constrained updates: rank constraints plus gradient projection

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Projecting negative-separation updates into the tangent cone that preserves
same-class kNN ranks combines listwise rank constraints with PCGrad/trust-region
optimization. Enforcing an occupied constraint through projection rather than
a penalty is not new supervision. Candidate 155 is **DEAD**. Full batch:
`docs/cross_field_candidate_batch_153_155.md`.

## 156. Same-class set volume: occupied variance preservation

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Requiring same-class batches to retain determinant volume is an intra-class
diversity constraint. Variance-preserving DML, group-sensitive triplet sampling,
and self-supervised intra-class ranking already occupy the action; log-det is a
different scalar functional, not a new relation. Candidate 156 is **DEAD**.

## 157. Cross-view error-correcting evidence: dropout or hashing

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Requiring matches to survive independent erasure of local evidence is feature or
region-dropout consistency. Explicit parity coordinates make it supervised
hashing/ECOC, already closed by candidate 151. A class-syndrome version also
cannot transfer to unseen identities. Candidate 157 is **DEAD**.

## 158. Residual retrieval-error boosting: occupied by BIER

**Gate-2 death recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Assigning later metric heads the retrieval errors missed by earlier heads is
BIER's online gradient-boosted embedding ensemble. The all-five-miss diagnostic
also found only **15/5,924 = 0.253%** total-query rescues, too little Gate-1
support for a distinct certificate/rescue mechanism. Candidate 158 is **DEAD**.
Full audit: `docs/cross_field_candidate_batch_156_158.md`.

## 159--163. Decorated higher-order constraints: no new observed label

**Gate-2 deaths recorded 2026-08-01; no diagnostic, implementation, or GPU.**
Per-identity matroid rank and tomographic rank reduce to low-rank/nuclear-norm
regularisation. The proposed regional instrumental variable violates the
exclusion restriction and reduces to routed adaptive-margin triplets. SAT
transitivity plus capacity is constrained subclustering. Ecological niche
competition is mixture-of-experts load balancing and diversity regularisation.
None supplies a measured higher-order label. Full audit:
`docs/higher_order_supervision_audit_159_163.md`.

## 164--169. Pixel observables: occupied reconstruction, transformation, and attention

**Gate-2 deaths recorded 2026-08-01; no implementation or GPU.** PID synergy
reduces to regional interaction/local-global aggregation; MDL grammar to
reconstruction likelihood; causal controls and metamorphic relations to
augmentation equivariance; algorithmic teaching to exemplar/prototype
reconstruction. Gradient-attribution alignment is attention consistency with a
different attribution estimator and requires double backpropagation. Li et al.
(2020) already align transformed attention across matched images in metric
learning, while MAMC regulates corresponding attention regions across images.
Full audit: `docs/pixel_observable_audit_164_169.md`.

## 170--173. Cross-domain relations: absent incidence or occupied mixing

**Gate-1/2 deaths recorded 2026-08-01; no implementation or GPU.** Simplicial
learning needs observed higher-order incidence absent from class-labelled image
datasets; neighbourhood-derived simplices are circular. Speaker variability and
protein ensemble imports reduce to nuisance subspaces or learned class densities.
Counterfactual part transplantation is Metrix-style mixed-target DML, Embedding
Expansion, and fine-grained intra-class part swapping unless causal parts are
independently observed. Full audit:
`docs/cross_domain_relations_audit_170_173.md`.

## 174. Orbit-adaptive potential fields: dead on radius reliability

**Gate-1 death recorded 2026-08-02; no OAPF training GPU.** ARCG measured a selective,
non-distance augmentation-response graph on In-Shop but showed that hard
positive replacement self-erases. Candidate 174 transfers each image's measured
augmentation displacement into that image's PFML zero-force plateau when it
acts on a different same-class image, while retaining PFML's proxy field.

The exact composition was not found in the two adversarial prior-art passes, but
all of its pieces are crowded: PFML supplies the field, self-tuning kernels
supply endpoint-local scales, PFE/IDML/ScaleFace supply endpoint uncertainty,
SER-FIQ supplies perturbation dispersion as quality, and ScoreCL supplies
augmentation-change weighting. Its largest risk is a sign/feedback defect:
large displacement may mean instability, so a larger plateau removes precisely
the corrective attraction the image needs.

An adversarial review killed the first proposed diagnostic before execution. Its
held-out outcome, pair weighting, augmentation packs, units, and radius-to-PFML
mapping were undefined. More decisively, BN-Inception has no dropout and only
one independent epoch-10 checkpoint was retained, so the radius cannot be
distinguished from SER-FIQ-style perturbation uncertainty without circularly
reusing the same augmentation dispersion or inventing a stochastic model. The
inverse-radius AUC control was also mathematically vacuous because
`log(1/r) = -log(r)` under an unconstrained fitted model. The process lesson is that an
impressive list of thresholds does not identify a diagnostic: every outcome,
nuisance control, unit conversion, permutation scope, and parameter map must
exist independently of the candidate statistic.

The completed review then supplied a prospective repair before any OAPF data:
two digest-seeded packs from the exact training augmentation distribution; a
radius-independent held-out retrieval-margin event; class-held-out weighted
folds; pack-A RMS dispersion as the explicit ordinary-uncertainty control;
nonnegative-coefficient direction tests; within-class derangements; fixed
distance-decile residual effects; Euclidean units; and a fixed map to PFML's
`delta=0.2`.

The corrected diagnostic then killed the candidate at its first signal gate.
On 25,882 official In-Shop training images and 153,115 same-class pairs, the
held-out binary outcome had valid class-balanced prevalence **0.472582**, but
independent six-view packs produced global radius Spearman only **0.317593** and
within-class-residual Spearman only **0.184057**, versus **0.50** required for
both. The q90 crop displacement is not a reproducible per-image endpoint scale;
most variation changes with the crop draw. All fitted-model gates were correctly
skipped. This kills OAPF before PFML reproduction or training. Full record:
`docs/oapf_candidate.md`.

**Second pre-data audit and baseline correction (2026-08-02).** Claude and an
independent static review attacked the executable diagnostic before it touched
candidate data. Accepted corrections add coherent pack-to-canonical shift to
M0, replace order-preserving cyclic permutations with within-class random
derangements, regress the continuous decile outcome on a continuous baseline,
class-balance every decile stratum, enforce six L2-normalised views, chunk pair
outcomes, bind the exact checkpoint/report digests, and emit explicit failure
artifacts for undefined negative outcomes. The review's broader circularity
claim is rejected: RMS is intentionally an adjacent control for whether q90
tail extent adds information, and pack-B augmentation compatibility is the
held-out phenomenon being predicted rather than a radius-defined label. This
limits any pass to that narrow statement. Full adjudication:
`docs/oapf_claude_audit_2026-08-02.md`.

The same audit cycle found that the historical PFML collapse is invalid as a
baseline result. Local code averaged the raw Eq. 6 total over millions of
ordered pairs while using Adam's coupled weight decay; this divided the data
gradient without dividing the weight-decay gradient and changed the
optimization problem. The loss scale is repaired and unit-tested, but the
local PFML preset still differs from the primary supplement in epochs, batch,
warm-up, head, decay exponent, CUB weight decay, and learning rate. Therefore a
Had Gate 1 passed it would have advanced only to a preregistered PFML
reproduction, not to OAPF training. It did not pass. Full baseline audit:
`docs/pfml_reproduction_audit_2026-08-02.md`.

## 175. Force-conserving response transport: occupied balanced OT weighting

**Gate-2 death recorded 2026-08-02; no diagnostic, implementation, or GPU.**
ARCG's positive-to-unknown graph self-erased, so candidate 175 proposed routing
a fixed amount of same-class attractive mass through a doubly-stochastic
augmentation-response coupling. The conservation constraint prevents a vanishing
positive objective, but the operator is Sinkhorn balanced assignment and the
coupling is soft positive importance. SwAV and simultaneous self-labelling
occupy balanced Sinkhorn codes, while batch-wise optimal-transport metric
learning already uses an OT plan as importance-weighted DML supervision.
Changing the cost to response disagreement is not a new supervision mechanism.
Full audit: `docs/force_conserving_transport_audit_2026-08-02.md`.

## 176. Cross-instance nuisance-tangent quotient: classical tangent distance

**Gate-2 death recorded 2026-08-02; no diagnostic, implementation, or GPU.**
OAPF's stochastic radius was unreliable, so candidate 176 retained deterministic
augmentation directions and proposed attracting same-class pairs only in the
complement of their endpoint nuisance-tangent subspaces. Tangent distance
already minimises distance between two local transformation manifolds/tangent
planes, and adaptive tangent-distance methods learn those local metrics.
Orthogonal projection is the least-squares form of the same operator; combining
it with PFML does not create new supervision. Full audit:
`docs/nuisance_tangent_quotient_audit_2026-08-02.md`.

## 177--182. Post-OAPF batch: ranking, diversity, codes, sets, synthesis, abstention

**All DEAD at Gate 2 on 2026-08-02; no implementation or GPU.** Consensus
intervention ordering is Fu et al.'s augmentation ranking with a response-based
label source and repeats IPSR's failed ordinal mechanism. Determinantal class
volume is established representation-diversity regularisation. Response-code
parity is augmentation-aware auxiliary prediction; complementary evidence
union is set/part supervision; nuisance-delta transplantation is established
feature synthesis; and counterfactual negative abstention is negative mining.
Claude's independent cross-field batch additionally reduced to ECOC, unavailable
causal instruments, variance regularisation, adversarial robustness, or an
externally supplied hierarchy. Full audit:
`docs/post_oapf_candidate_batch_2026-08-02.md`.

## 183. Proxy shell membership: occupied class-radius geometry

**Gate-2 death recorded 2026-08-02; no implementation or GPU.** Replacing a
point proxy by an admissible shell was intended to preserve the measured
beneficial fragmentation while retaining identity membership. PFML's flat
attraction region, chance-constrained DML's class-proxy covering radii, and
non-isotropic probabilistic proxy DML already occupy class-specific feasible
spread around proxies. The shell changes constraint geometry, not the labelled
relation. Audit: `docs/post_oapf_candidate_batch_2026-08-02.md`.

## 184. Reciprocal proxy ownership: symmetric proxy/centroid cross-entropy

**Algebraic Gate-2 death recorded 2026-08-02; no implementation or candidate
GPU.** The motivating In-Shop asymmetry is real: proxy-to-own-centroid ownership
is 99.975%, centroid-to-own-proxy ownership 70.303%, and the image-level
ownership split carries a 7.91-point leave-one-out R@1 gap. But proxy-to-centroid
ranking is Proxy Anchor aggregated by class, while centroid-to-proxy ranking is
Proxy-NCA/class softmax applied to a centroid. Their reciprocal sum is symmetric
cross-entropy on the same similarity matrix; leave-one-out changes only the
centroid estimator. Claude's initial live verdict is rejected on this algebra.
Full audit: `docs/reciprocal_proxy_ownership_audit_2026-08-02.md`.

## 185. Mode pseudo-identities: occupied class splitting and false negatives

**Gate-2 death recorded 2026-08-02 before the fragmentation replication; no
candidate GPU.** Splitting each annotated identity's early 1-NN components into
separate Proxy Anchor labels would turn the measured +3.534-point fragmentation
association into a causal edit. But the operator is clustering-derived
pseudo-labelling. Easy Positive already preserves within-class subclusters by
sparse positive connection, Divide-and-Conquer partitions data and embedding
subspaces, and subcentre/hierarchical DML occupies retaining the parent label.
The only remaining distinction is to repel two modes known to share a true
identity, which injects embedding-derived false negatives rather than new
supervision. Full audit: `docs/mode_pseudoidentity_audit_2026-08-02.md`.

**Fragmentation confounding follow-up (pre-registered, CPU only).** On the same
frozen seed-0 operating pack, matching exact class size plus global quintiles of
mean within-class cosine and nearest-foreign-centroid cosine retained 60.43% of
eligible classes and increased the disconnected-minus-connected R@1 difference
to **+5.875 points**. The marker is not explained away by those coarse controls,
but this is still observational and in-sample. It strengthens the measured
phenomenon, not candidate 185's occupied operator. Independent seeds 1 and 2
remain pending. Full record:
`docs/fragmentation_confounding_preregistration_2026-08-02.md`.

## 186. Within-class subspace protection: occupied gradient surgery

**Gate-2 death recorded 2026-08-02; no implementation or candidate GPU.** The
adjusted In-Shop fragmentation gap (+5.875 points) motivated protecting measured
within-class modes from proxy attraction. Projecting only the attractive gradient
through the complement of a class covariance basis, however, is ordinary
orthogonal gradient projection with a class-conditional estimator. On normalized
embeddings it reduces, up to a radial term, to a projected positive proxy: either
a conservative low-rank class metric or non-conservative selective gradient
surgery. OGD/GPM occupy
protected-subspace projection (GPM obtains bases by SVD of representations),
PCGrad occupies loss-component gradient surgery, NIR already targets proxy-induced
loss of non-isotropic class-local structure, and Easy Positive already loosens
same-class collapse. It also repeats candidate 176 with class PCA substituted for
augmentation tangents. Every labelled relation remains unchanged. The composition
therefore changes optimization geometry, not what supervision exists. Full audit:
`docs/within_class_subspace_protection_audit_2026-08-02.md`.

## 187--194. Post-fragmentation batch: the partition is exhausted

**All DEAD on 2026-08-02; no diagnostic, implementation, or GPU.** Eight
candidates were generated under the tightest budget yet — training pixels and
class labels only, roughly 1x cost, and an explicit requirement to change what
cross-instance supervision exists rather than how it is scored, weighted,
selected, or optimised.

The batch repeatedly reduced to a small routing taxonomy rather than eight new
mechanisms: identity, refinement, coarsening, weighting, or subsetting of the
label partition. This is useful empirical coverage, **not a mathematical
completeness theorem**; higher-order functionals exist, but the proposals here
lacked an independently observed correspondence and collapsed to existing
similarity contrasts.

- **187, directed retrieval obligation** (mutual top-1 **0.567035**; R@1
  **0.969065** versus **0.897733**): with a symmetric cosine the union of ordered
  duties is exactly the per-anchor constraint set of Ranked List Loss/Smooth-AP
  already killed as candidate 148, and no direction is representable at
  inference.
- **188, population-ablation invariance** (centroid-to-proxy ownership
  **70.303%**): identically satisfied by a separable encoder, hence vacuous;
  non-vacuous forms are group DRO or Qi et al.'s robust pair weighting.
- **189, model-free pixel eligibility graph** (regional MaxSim **+6.67** points;
  CUB top-5 positive Jaccard **0.411**): discrete same-class eligibility with a
  new edge estimator, occupied by DAMLRRM. An independent draft incorrectly
  cited Correlation Verification here; primary inspection showed it is a
  geometric re-ranker, and that citation was removed before commit.
- **190, cross-class analogical quadruples** (class-pair variance
  **52.57--58.90%**, interaction **4.75%**): class labels do not observe which
  arbitrary within-class displacement vectors express the same factor. Its loss
  expands to a six-Gram-entry contrast, its vector-equality operator is adjacent
  to TraVeLGAN/Difference Vector Equalization, and the closed in-repo analogue
  lost **7.237** points as TIRD.
- **191, observed mode-count preservation** (**40.33%** disconnected;
  **+5.875** adjusted points): endogenous target, and every relaxation is
  algebraic connectivity or a diversity penalty. Candidate 186's causal-direction
  objection stands — fragmentation correlates **+0.04754** with class R@1 while
  the spent variable correlates **+0.41302**.
- **192, absolute margin distribution** (margin **0.033816** correct versus
  **0.023650** incorrect): the definition of a margin loss, plus loss shaping.
- **193, cardinality-invariant relations** (proxy-centroid cosine correlating
  **0.2760** with class size): class-balanced loss, logit adjustment, adaptive
  margins; candidates 100, 141 and 34 already closed it.
- **194, transitive-closure certificates** (fragmentation plus stable local CUB
  structure, pair-rank Spearman **0.863**): DAMLRRM verbatim, already candidates
  32 and 68.

The verdict is **NONE**. The useful update is that the excluded operator families
covered every executable proposal in this round; this is not an impossibility
proof. Full audit:
`docs/post_fragmentation_candidate_batch_187_194_2026-08-02.md`.

### Fragmentation replication and identity stability (measurement update)

The observation motivating candidates 185--194 is now substantially stronger,
but those candidates remain dead. Two preregistered out-of-sample In-Shop
epoch-10 Proxy Anchor seeds reproduced almost identical fragmentation prevalence
(**40.075%**, **40.025%**) and positive exact size-matched class R@1 gaps
(**+2.932**, **+3.262 points**); their separately adjusted gaps were **+5.966**
and **+5.806 points**. More importantly, fragmentation was not assigned to a
random 40% each run: pairwise cross-seed Cohen kappas were **0.8859**, **0.8870**
and **0.8816**, with **36.20%** of identities fragmented in all three seeds.

The mechanism-level finding is therefore that symmetrized within-class 1-NN
disconnection is a reproducible *identity property* under optimizer reseeding,
not merely an aggregate training accident. It still does not identify semantic
modes or causality. Every run shares images, labels, model, loss and augmentation;
stable label noise, near-duplicate groups, sample composition, or a knife-edge
1-NN bridge can generate the same kappa. The stable-all versus connected-all R@1
gap was only **+1.050 points** and was preregistered as descriptive because that
conditioning changes class composition. Consequently the result reopens a
diagnostic question, not pseudo-identities, sparse positives, subcentres,
topology, diversity or gradient-protection operators already killed at Gate 2.

Process lesson: replication of a prevalence and outcome association is weaker
than replication of the units carrying the exposure. Both were required here;
even their joint pass establishes robustness only to the factor actually varied
(optimizer seed).

**Partition follow-up.** A second preregistered test found that the component
membership itself recurs: mean pairwise ARI **0.8415**, minimum **0.8375**, exact
component-count agreement **67.27%**, and all-three k=2 persistence **56.34%**
among eligible stable-fragmented classes. Pairwise-only cohorts gave the same
ARI, so conditioning on three-seed disconnection did not manufacture it. This
kills the narrow random-edge explanation, but not fixed acquisition clusters,
label noise or near duplicates. No dead Gate-2 operator is revived; the next
step is to test those observable alternatives before naming candidate 195.

**Acquisition-token resolution.** The next preregistered audit found that the
stable components are almost the filename-series partition: series ARI
**0.7541--0.7607**, versus view-descriptor ARI **-0.1425-- -0.1416**, with mean
paired difference **+0.9234**. A visual check of one multi-series identity found
two colourways of the same design, each with its own front/side/additional views;
series is therefore a real appearance grouping in at least that case, not merely
camera pose. But it is already observed in the training filename. Direct use is
hierarchical/subidentity metadata; pixel recovery is clustering/pseudo-label
splitting. Both are occupied mechanisms, so this strong measurement does not
create candidate 195. It instead exposes series composition as the omitted
confounder in the earlier fragmentation-outcome association.

**Series-confounding falsification.** The preregistered prediction that exact
series composition would collapse the outcome gap failed. Matching the complete
sorted series-size signature and prior geometry controls retained **58.2--61.0%**
of classes and produced **+5.476**, **+5.744**, and **+5.587 points** across the
three seeds. Exact size plus series count was essentially identical. Thus series
identifies the repeatable partition but is not the omitted confounder carrying
the retrieval association. This keeps a causal question open, not a method:
direct series hierarchy, cross-series mining/alignment, and pixel-derived series
clusters remain established supervision operators.

## 195--198. Post-series intervention batch

**All DEAD at Gate 2 on 2026-08-02; no implementation or candidate GPU.** The
surviving series-adjusted association motivated four concrete edits. A
cross-series bridge is camera-aware positive emphasis/mining; a series-child plus
identity-parent relation is SoftTriple/hierarchical proxy DML; removing series is
camera/domain-invariant representation learning; and leave-one-series-out
retrieval is episodic meta metric learning or group DRO. The filename token
changes the group estimator, not any of those operators, and has no matched
analogue on CUB or Cars. Full primary-source audit:
`docs/post_series_candidate_batch_195_198_2026-08-02.md`.

The verdict is **NONE**. This is not evidence that the +5.5-point association is
spurious; it is evidence that the currently identifiable interventions are
occupied. The search moves to the faithful Cars RS@k measurement rather than
spending GPU on a renamed camera-aware or hierarchical method.

**Cross-series outcome audit.** A preregistered attempt to replace raw training
leave-one-out R@1 with cross-series-only R@1 was underidentified: 1,259--1,264 of
1,274 multi-series identities were fragmented, leaving 10--15 connected controls
and only 6.83--9.97% matched coverage versus the required 20%. The result is
formally **inconclusive**, despite replicated descriptive gaps of **-22.05 to
-41.92 points** (under-covered matched: -22.36 to -28.01). This does not prove
fragmentation harmful; it proves the binary exposure has almost no common
support among multi-series identities. Do not report the large negative as a
passed prediction.

**Official-split audit.** A separate preregistered metadata analysis found that
**95.604%** of 14,218 In-Shop queries have a same-series gallery positive;
**42.720%** have only same-series positives, 52.884% have both, and just 4.396%
are cross-series-only. Thus official R@1 legitimately but strongly rewards the
series structure recovered by the fragmented graphs. This explains why raw
training leave-one-out is a poor causal outcome for cross-series invariance. It
does not authorize exploiting filenames: such a method would be benchmark-
specific hierarchy/mining and fail the protocol's second-dataset requirement.

## 199. Existential Recall@k: Easy Positive over a smooth rank

**DEAD at Gate 2 on 2026-08-02; no implementation or GPU.** RS@k's positive-
count target differs from benchmark any-positive success, motivating a noisy-OR
over positive rank events. In the zero-temperature limit that operator is
`min_positive rank`: the easiest positive alone satisfies the query. Xuan et
al.'s Easy Positive already proposes exactly that loosened same-class relation
to prevent class collapse. Applying it to Patel et al.'s differentiable rank at
multiple k values is a composition of occupied positive mining and an occupied
surrogate, not a new supervision primitive. Full audit:
`docs/existential_recall_audit_2026-08-02.md`.

## 200. First-hit rank hazard: listwise loss reparameterization

**DEAD at Gate 2 on 2026-08-02; no implementation or GPU.** Because R@k is the
CDF of the first relevant rank, a discrete hazard would avoid independently
counting nested k events. But the hazard is just the finite-difference
parameterization of the same listwise relevance labels. Expected Reciprocal Rank
already models first satisfaction/stopping across ranks, ListNet defines top-k
probabilities, and RS@k provides the DML smooth-rank estimator. Only gradient
allocation changes. Full audit: `docs/first_hit_rank_hazard_audit_2026-08-02.md`.

## 201. Training-proxy response coordinates: Mahalanobis reduction and Classemes prior

**DEAD at Gate 2 on 2026-08-02; no implementation or candidate GPU.** The
99.975% proxy-to-own-centroid ownership measurement motivated treating the
training proxy bank as relational coordinates for unseen images. For proxy
matrix `P`, however, cosine between linear response vectors `Px` and `Py` is
exactly cosine after the fixed map `(P^T P)^(1/2)`: a PSD Mahalanobis remapping
absorbable into the embedding head. Softmax responses break that linear identity
but enter established dissimilarity representations and, most directly,
Classemes (Torresani et al., ECCV 2010), which uses a category-classifier output
vector as a compact descriptor for novel-category retrieval/classification.

The newly found weak-metric Cross-Entropy paper by Mou et al. (PMLR 2025) does
not reopen the route: it evaluates CIFAR with the same class vocabulary at train
and test and uses output dimension equal to the class count. Under class-disjoint
DML, its axes become foreign training-class landmarks, which is precisely the
occupied representation above. Full algebra and primary sources:
`docs/proxy_response_coordinates_audit_2026-08-02.md`.

## 202. Support-multiplicity attraction from fragmented classes

**DEAD at Gate 2 on 2026-08-02; no implementation or candidate GPU.** The
strongest remaining measurement appeared to say that multiple independent
within-class supports help: fragmented classes retained adjusted R@1 gaps of
**+5.806 to +5.966 points** across three seeds, and their component identities
were stable. But requiring a positive from another inferred component is an
order-statistic positive-mining rule over a clustering-derived subidentity. Its
hard form composes occupied mining with pseudo-label/subcentre supervision; its
soft form is positive weighting; and its worst-component form is group DRO.
It does not introduce an observed supervision relation outside candidates
185--198.

The post-mortem also invalidated the motivating measurement as causal
provenance rather than only killing the operator. Fragmentation exposure and leave-one-out R@1 shared the same nearest
same-class neighbour. In a preregistered CPU audit, deleting each query's
nearest same-class partner reversed the locked adjusted gap from **+5.875** to
**-3.910 points** (unadjusted **-2.240**) at unchanged **60.43%** matched
coverage. A subsequent adversarial review noted that this deletion removes both
the shared estimator edge and a real part of the exposure, so it is an
over-correction: the association is **unidentified, not refuted**. The stable
graph property identifies fixed tight-pair structure, but it still supplies no
identified evidence for mode-preserving supervision. Full audit:
`docs/fragmentation_partner_exclusion_preregistration_2026-08-02.md`.

## Source-faithful Cars196 RS@k reference (not a numbered candidate)

The preregistered deciding run passed its reproduction interval: raw best R@1
**0.793260**, selection-corrected **0.788987**, with a **+0.427-point** selection
bonus. Its best R@1/2/4/8 curve was **0.793260 / 0.863608 / 0.912803 /
0.946993**. An independent audit against official revision `ed052029...` found
no remaining source mismatch. This validates RS@k as an occupied Cars196
reference in this codebase; it does not revive candidates 199 or 200 and does
not itself supply a novel supervision variable.

## 203. Resolvable within-class co-occurrence design

**DEAD at Gate 1 on 2026-08-02; no implementation or GPU.** RS@k's pinned
sampler presents four images from every Cars class in each batch, suggesting
that balancing which same-class images co-occur across steps could change what
finite supervision is observed. But the immutable trajectory does not measure
that mechanism. Its within-epoch loss variation jointly contains random
crops/flips, changing parameters, CUDA nondeterminism, and batch composition;
the scalar loss history cannot identify the composition share. A proposed
variance decomposition was therefore rejected before calculation. Without an
outcome-relevant co-occurrence measurement, the edit fails provenance and does
not reach its required prior-art audit against batch design, tuple mining, and
memory methods. Full audit:
`docs/post_rsatk_candidate_audit_203_2026-08-02.md`.

## 204. Near-resolvable positive-pair batch design

**DEAD at Gate 2 on 2026-08-02; no diagnostic, implementation, or GPU.** The
proposal equalized each same-class pair's realized cross-epoch co-occurrence
count under a fixed M-per-class budget. Random sampling already gives every pair
the same inclusion probability, so the block design changes count variance,
not the expected supervision relation. GCBS (Sachidananda et al., ICML 2023)
occupies global contrastive batch assignment over sample permutations;
Combinatorial Designs for Deep Learning (Chisaki et al., JCD 2020) already uses
balanced designs to replace irregular random edge frequencies in neural
training; incomplete-U-statistic work formalizes pair-selection designs for
metric-learning ERM. Cross-epoch statefulness is an application-level residue,
not a defensible new mechanism. Full audit:
`docs/pair_coverage_batch_design_audit_2026-08-02.md`.

## 205--210. Cross-field supervision batch

**All DEAD before Gate 3 on 2026-08-02; no diagnostic, implementation, or
GPU.** Conditional-rate coding reduces to multi-view information bottleneck,
class compression, or mined pairing (205). Rasch specific objectivity is the
same double-centred interaction that TIRD lost **7.237 points** on (206).
Proximal causal controls are unidentified under pixels plus class labels (207).
Good--Turing reserves only **0.18--0.30%** mass and then becomes Proxy Synthesis
(208). Fractional-factorial augmentation is an estimator for the already tried
augmentation-response displacement at about 16x cost (209). Binding-error
composites combine Metrix/CutMix with Proxy Synthesis or uniformity (210).

The round's mechanism-level lesson is that cross-field imports now fail mainly
because their required variable is not identified in the permitted data, not
because the analogy lacks a name. Full audit:
`docs/cross_field_candidate_batch_205_210_2026-08-02.md`.

## 211--213. Adversarial supervision-object batch

**All DEAD at Gate 2 on 2026-08-02; no implementation or GPU.** A controlled
transform identity genuinely identifies intervention-corresponded quadruples,
but enforcing equal displacement across two images is occupied cross-instance
equivariance (AugSelf/EquiMod) and conflicts with ARCG's measured response
disagreement (211). Treating a tight acquisition partner as a distinct relation
type is DAMLRRM-style visual graph construction, graded supervision, or Easy
Positive mining (212). Constraining the full image-to-proxy ownership matrix is
balanced prototype assignment, multi-proxy assignment, or DADA distribution
alignment (213).

The audit corrected two inputs without reopening an operator. Use clean RSPG
In-Shop density **8.66%**, not the contaminated partial-run **8.63%**. And the
fragmentation partner-exclusion reversal makes the +5.875-point association
**unidentified, not refuted**, because deleting the closest partner removes the
edge defining the exposure as well as the shared outcome support. Full audit:
`docs/adversarial_candidate_audit_211_213_2026-08-02.md`.

## 214. Fixed-probe learning-dynamics supervision

**DEAD at Gate 2 on 2026-08-02; no diagnostic, implementation, or GPU.** A
fixed unaugmented panel at fixed checkpoints would validly measure per-image
proxy acquisition and forgetting without candidate 203's transform and batch
confounds. But the statistic is endogenous to the model trajectory. Acting on
it is curriculum/dynamic sampling, interaction selection/co-teaching, temporal
distillation, trajectory-derived relabelling, or regularization. Because test
classes are disjoint, the training-indexed statistic cannot itself become an
unseen-class supervision object. The repository's 1.08-point trajectory spread
also makes a one-run trace unstable; multiple trajectories exceed the cost
budget and reproduce candidate 134. Full audit:
`docs/fixed_probe_trajectory_audit_2026-08-02.md`.

## 215. Class-conditional operating-point supervision

**DEAD at Gate 1 on 2026-08-02; independently occupied at Gate 2; no candidate
GPU.** A missing ICLR 2024 source introduced OPIS, the across-class variance of
utility under one absolute threshold. The preregistered three-seed In-Shop
diagnostic found class-OPIS-contribution versus class-error Spearman rhos
**0.15688 / 0.13546 / 0.18039**. Median **0.15688** and every seed missed the
registered pass thresholds; no seed crossed the <=+0.10 falsifier, so the
result is inconclusive, not a clean refutation. OPIS CV across seeds was
**0.24882**.

The operator is closed independently. TCM uses absolute-margin hard-pair
regularization; OneFace estimates domain thresholds and applies a Threshold
Consistency Penalty; UniTSFace learns a unified threshold inside a
sample-to-sample loss. Classwise alignment is the same threshold-consistency
objective, and using deviations as coefficients is weighting. Full audit:
`docs/threshold_consistency_horizon_audit_2026-08-02.md`.
