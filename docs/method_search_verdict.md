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

Changing *what supervision exists* — the only untried class — points at controlled
generative expansion of intra-class support. That is reportedly already done: BLENDER
(Kolf et al., arXiv 2026) claims +3.7 R@1 on CUB, +1.8 on Cars. Class-semantic
supervision is likewise occupied by Roth, Vinyals & Akata, *Integrating Language
Guidance into Vision-Based Deep Metric Learning* (CVPR 2022).

*Not independently verified.* The BLENDER citation is a 2026 preprint reported by the
reviewer and has not been checked against the actual paper; treat the exact numbers as
unconfirmed until someone opens it. The estimate given was **~20%** that it delivers
≥ +1.0 pt under our digest-pinned recipes, and *effectively zero* that it constitutes a
novel direction.

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
method by being combined. Candidate sixteen remains unregistered and unqueued.

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
supervision conflict. Candidate 1 receives no confirmation run.

The averaging control itself improved raw by **+0.183 pt** and corrected by
**+0.255 pt** over paired seed-0 Proxy Anchor. Its raw gain narrowly missed the
preregistered +0.20 to +0.50 pt range. This remains evidence about an old
weight-averaging method, not a novel-method success.

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
