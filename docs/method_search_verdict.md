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

One measurement claim is still unharvested and needs no GPU: quantifying the
**best-over-training selection bias** from the per-epoch evaluation histories already
saved in every artifact. If the upward bias scales with run noise, then a slice of the
published DML literature is reporting max-statistics rather than improvements.
