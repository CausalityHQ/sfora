# SFORA handoff — state as of 2026-07-31

Everything needed to pick this up on a fresh machine. Written because the laptop is
being reset; nothing here depends on local state except an SSH key for the DGX.

---

## 1. Where things live

| what | where |
|---|---|
| branch | `research-reset-2026-07-27` |
| remote | `git@github.com:CausalityHQ/sfora.git` |
| GPU host | `riomus@100.104.199.68` (Tailscale; a DGX Spark, one GB10) |
| remote repo | `/home/riomus/group-learning` |
| remote venv | `/home/riomus/group-learning/.venv` |
| experiment logs | `/home/riomus/experiment-logs/reference-matrix/` |
| result artifacts | `/home/riomus/group-learning/reports/generated/*.json` |
| datasets | CUB + Cars from the HF cache; In-Shop at `/home/riomus/datasets/inshop`; iNat2018 at `/home/riomus/datasets/inat2018` |

**Remote git HEAD is stale** (`3a1ae15`) with a large dirty tree. The *source* is
kept in sync by `rsync`, not by git, and has been verified byte-identical to the
local branch. Deploy with:

```bash
rsync -az src/sfora/ riomus@100.104.199.68:/home/riomus/group-learning/src/sfora/
scp scripts/run_priority_queue_v30.sh riomus@100.104.199.68:/home/riomus/group-learning/scripts/
```

---

## 2. GPU queue state

**The GPU is idle.** Queue `scripts/run_priority_queue_v30.sh` completed at
2026-07-31T13:19:06+02:00 and intentionally stopped for three-seed averaging
judgement:

```bash
ssh riomus@100.104.199.68 'tail -n 20 /home/riomus/experiment-logs/q30.log'
```

It completed only In-Shop `pa_ema_avg_bnfix` seeds 1 and 2, as ordered. Do not
restart the superseded v26 plan: dual EMA is falsified, averaging did not
replicate as a raw effect, and neither Cars nor the momentum sweep is warranted.
The continuing prior-art-gated search is recorded in
`docs/method_search_verdict.md`.

**Stopped stale controller:** a separately launched v26 controller (PGID
`2754213`) had been waiting while v30 held the GPU and began CUB
`pa_dual_ema` immediately afterward. It was terminated at
2026-07-31T13:35:26+02:00, before any result artifact was written. The stop is
recorded in `q26.log`. Do not treat that partial run as evidence or restart v26.

**Status check** (this is the whole loop):

```bash
ssh riomus@100.104.199.68 'grep -E "DONE|FAIL" /home/riomus/experiment-logs/reference-matrix/controller.log | tail -5'
ssh riomus@100.104.199.68 'cd ~/group-learning && .venv/bin/python scripts/analyze_reference_matrix.py'
ssh riomus@100.104.199.68 'cd ~/group-learning && .venv/bin/python scripts/measure_selection_bias.py "reports/generated/image_end_to_end_cub.*.json"'
```

Run times on the GB10: CUB ≈ 44 min (base) to 55 min (any EMA arm), Cars ≈ 60–75 min,
In-Shop ≈ 2.2 h (Proxy Anchor) to 4.6 h (HIST base).

### 2b. The EMA factorial — resolved (2026-07-31)

An EMA teacher supplies two separable things: a **distillation target**, and an **averaged
copy of the weights to evaluate**. Every arm before `pa_dual_ema` forced one EMA to do
both. CUB seed 0, against `proxy_anchor` 0.6825:

| arm | teacher mom | eval mom | Δ |
| --- | ---: | ---: | ---: |
| `pa_distill` | 0.999 | — (student) | **+0.91** |
| `pa_distill_avg` | 0.99 | 0.99 | +0.52 |
| `pa_ema_avg_fast` | — | 0.99 | +0.45 |
| `pa_distill_fast` | 0.99 | — (student) | +0.30 |
| `pa_ema_avg` | — | 0.999 | +0.07 |

**The two roles want opposite timescales.** As a target, 0.999 beats 0.99 (+0.91 vs
+0.30) — a slow teacher is a more stable thing to regress toward. As the evaluated model,
0.99 beats 0.999 (+0.45 vs +0.07) — a fast average tracks the current solution instead of
dragging 5.3% of the initialisation along. `ema_eval_momentum` decouples them;
`pa_dual_ema` is slow teacher + fast evaluated average.

**Read these numbers with §4b.** Reported deltas understate the averaging arms badly,
because they collect far less best-over-training selection bonus.

The required BN-correct In-Shop screen rejected the combination:

| arm | digest | raw R@1 | corrected R@1 |
| --- | --- | ---: | ---: |
| Proxy Anchor seed 0 | `16a3bc844c81` | 0.9024 | 0.9015 |
| `pa_ema_avg_bnfix` | `80f57f183966` | 0.9043 | 0.9033 |
| `pa_dual_ema_bnfix` | `79f9d35c4eea` | 0.9044 | 0.9040 |

Dual-minus-average was +0.014 pt raw and +0.077 pt corrected. The
preregistered raw threshold was +0.24 pt and absolute dual R@1 had to reach
0.9048; it reached 0.9044. Candidate 1 failed gate 4 and received no
confirmation seeds.

The averaging-only follow-up then completed seeds 1–2. Its three paired raw
deltas were +0.18 / −0.13 / +0.15 pt: mean **+0.068 pt**, sd 0.169,
paired-t p = 0.5589, exact sign p = 0.500. Selection correction increased the
mean to **+0.203 pt**, sd 0.157. This is evidence that best-over-training
under-credits the stable arm, but not a replicated raw method gain. Do not queue
Cars or the momentum sweep.

## 3. The one strong result: H3, the BatchNorm teacher/student mismatch

**This is the finding worth publishing.** It is a bug-finding about a widely-used
training pattern — *not* a rescue of HERD. See the closing caveat below.

The EMA teacher was created with `deepcopy(model)` then `.eval()`, so it normalised
with BatchNorm **running** statistics while the student trained in `.train()` mode
using **batch** statistics. `_update_ema_teacher` also hard-copied buffers instead of
blending them. With BatchNorm frozen the two coincide; with it trainable the teacher
is a systematically different function of the same images.

| In-Shop | seed 0 | seed 1 | seed 2 | mean | σ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HERD (baseline) | 0.8906 | 0.8892 | 0.8900 | 0.8899 | 0.0007 |
| **`herd_bnfix`** | 0.9035 | 0.9048 | 0.9041 | **0.9041** | 0.0007 |

**+1.42 pt, every seed positive, ≈20σ.** The "distillation hurts HIST by 1.39 pt"
result that triggered this whole investigation was an implementation bug.

It then replicated on a **second base it was never developed against**, Proxy Anchor
(0.9029 / 0.9021 / 0.9044 vs `pa_distill` 0.8999 / 0.8994 / 0.8990): **+0.37 pt**
against a −0.41 pt regression. So across both legs:

| leg | recovery | regression it undoes |
| --- | ---: | ---: |
| HIST | +1.42 pt | −1.39 pt |
| Proxy Anchor | +0.37 pt | −0.41 pt |

**Six paired seeds, six positive** — exact sign test p = 2⁻⁵ = 0.031, which carries
no normality assumption (the per-leg t-tests at n=3 do). The proportionality is the
real tell: each leg recovers almost exactly the regression it had to undo, which is
what a bug-recovery predicts and what a method gain has no reason to produce.

Validated three further ways: mechanism proved at unit level; the *null* prediction
(inert under frozen BatchNorm) proved by test; the *positive* prediction confirmed
across seeds and bases.

**State it narrowly.** The fix repairs the bug; it does not make distillation a win.
With it in place `herd_bnfix` is +0.03 pt over HIST and `pa_distill_bnfix` is
−0.04 pt under Proxy Anchor (t = −0.29, p = 0.80) — both indistinguishable from
their bases. Anyone writing this up should resist the pull toward "our method
improves retrieval"; the contribution is the defect and its generality.

The fix lives behind `ema_teacher_train_mode` / `ema_teacher_ema_buffers`, both
defaulting to the historical behaviour so old artifacts still reproduce.

Generalisable claim: *momentum-teacher recipes silently break under trainable
BatchNorm unless the teacher's normalisation mode matches the student's.* Applies to
any MoCo/BYOL/DINO-style teacher on a backbone with updating BatchNorm.

> **⚠️ Prior art, found 2026-07-30 — this technique is not new.** Cai et al.,
> *Exponential Moving Average Normalization for Self-Supervised and Semi-Supervised
> Learning* (CVPR 2021), identify the same teacher/student BatchNorm mismatch and
> propose EMA-ing the normalisation statistics — which is exactly what
> `ema_teacher_ema_buffers` does. Our fix is EMAN, independently rediscovered.
>
> So H3 is **not a novel method**. What remains ours is the *audit* result: this defect
> is present in DML momentum-teacher implementations, and its measured cost is a
> 0.3–1.4 pt regression that recovers proportionally on two independent bases across
> six paired seeds. Frame it as "a published fix that this subfield has not adopted,
> and here is what it costs", never as a new technique. Anyone writing this up must
> cite EMAN in the same breath as the claim.

---

## 4. The measurement result

Nearly as important, and it explains why so much else failed.

- **A 1.08 pt spread at a FIXED seed** (three effectively-identical runs: 0.7183 /
  0.7154 / 0.7075). More than half the apparent "seed variance" was GPU
  nondeterminism, with no seed difference at all. This one is unchanged and is the
  most useful measurement in the project.
- **CUB across-seed σ is 0.57 pt at six seeds** (0.50–0.68 over four arms). The
  σ ≈ 0.88 pt this document used to quote came from three seeds and was 55% too high.
- **Seeds required is a property of the comparison, not the dataset.** Pairing removes
  most seed variance on one leg and none on another, so for a +0.5 pt effect:

  | leg | per-arm σ | paired sd | seeds for 80% power |
  | --- | ---: | ---: | ---: |
  | `pa_distill` − `proxy_anchor` | 0.60 pt | **0.367 pt** | **5** |
  | `herd` − `hist` | 0.68 pt | **0.729 pt** | **17** |

  The old "+0.5 pt → 12–37 seeds" was computed from an unpaired σ and is wrong in both
  directions. **Always quote the paired sd of the specific comparison**; a
  dataset-level σ cannot power a paired test. Three seeds cannot estimate either — the
  3-seed paired sds were 0.153 and 0.322, i.e. 2–4× too small.
- CUB still **cannot** resolve the ~1 pt improvements the literature reports from
  single runs. A quantitative version of Musgrave's *Metric Learning Reality Check*.
- **In-Shop σ = 0.12 pt.** All method questions should be settled there, not on CUB.
- `deterministic: bool` (default False) now removes the nondeterminism entirely —
  `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`.
  **Turn this on for all new experiments.**

Also corrected along the way: the legacy "HERD beats HIST" headline was a **LayerNorm
confound** (the control ran without embedding LayerNorm). Properly paired, HERD and
HIST are indistinguishable on CUB (+0.03, p=0.93), and HIST's 3-seed mean is 0.7107 —
essentially exactly its published 0.714.

---

### 4b. Best-over-training selection bias — measure it before comparing arms

`scripts/measure_selection_bias.py`. Best-over-training is a `max` over ~60 noisy
evaluations, so it overshoots the true curve, and **the overshoot grows with the arm's
noise**. Estimated per run from the selected epoch's *neighbours only*.

| CUB arm | evaluates | selection bonus |
| --- | --- | ---: |
| `pa_distill` | student | 0.836 pt |
| `proxy_anchor` | student | 0.769 pt |
| `pa_ema_avg_fast` | averaged weights | 0.306 pt |
| `pa_ema_avg` | averaged weights | 0.074 pt |

Arms evaluating averaged weights collect **2.5–10× less** bonus, because averaging smooths
the evaluated curve. Removing each arm's own bonus **reverses the ranking**:

| paired | reported | corrected |
| --- | ---: | ---: |
| `pa_ema_avg_fast` − `proxy_anchor` | +0.414 | **+0.732** |
| `pa_ema_avg` − `proxy_anchor` | +0.059 | **+0.610** |
| `pa_distill` − `proxy_anchor` | +0.658 | +0.592 |

So **the standard protocol structurally hides weight averaging.** On a flat simulated
plateau with 0.5 pt evaluation noise the estimator recovers +1.16 pt of pure selection —
larger than most published DML gains, from a truth with no improvement in it. It also
reproduces a known failure as a sanity check: `local_nca` collapsed, peaked in its first
epochs, and best-over-training still reported 0.5733 against a 0.3394 trend.

Caveats: averaging arms are at n=2, the correction is one estimator, and corrected values
are **not** the benchmark metric — a paper quoting them must argue the protocol.

## 5. The cognitive-science candidates — both settled, both failed

Both replaced the **similarity function** rather than adding a loss term, and both
differed from official Proxy Anchor only in their declared recipe delta. Both were
decided on In-Shop, where σ = 0.12 pt makes one seed conclusive:

| method | In-Shop | vs PA 0.9035 | CUB (3 seeds) |
| --- | ---: | ---: | ---: |
| Shepard exponential kernel | 0.8999 / 0.8998 | **−0.36 pt** (~3σ) | 0.6743 — unreadable |
| Tversky contrast similarity | 0.8600 / 0.8543 | **−4.63 pt** | 0.6758 — unreadable |

The general lesson, worth carrying forward: **a similarity function better-motivated
as a model of human judgement is not thereby better as a retrieval score.** Tversky's
model is descriptively correct about people; retrieval is not asking that question.

Tversky's failure size is the informative one — the `x·f_k > 0` membership test
discards the *magnitude* of agreement, which is lossless on the sparse binary
fingerprints where Tanimoto earns its keep in cheminformatics and very lossy on dense
CNN embeddings. Shepard's was small and real: the fatter tail genuinely exists
(2.9× more mass on the far neighbour at PA's operating point), it simply does not
help, which suggests distant true positives are distant *because they should be*.

The rationale as it stood before the runs is preserved below.

**`shepard` / `shepard_l1`** — Shepard's exponential generalisation kernel.
Cosine-softmax is secretly Gaussian: on unit vectors `cos = 1 − d²/2`, so
`exp(cos/T) ∝ exp(−d²/2T)`. Shepard (*Science* 1987) derived `exp(−d)`, linear in
distance; `d = √(2−2cos)` is non-affine in cos so no temperature reproduces it.
Targets the orphan mechanism: under a Gaussian kernel a moderately-distant true
positive's softmax weight collapses and it stops receiving gradient. Measured
tail-weight ratio vs cosine: 1.19× at T=1.0, 1.54× at T=0.2, 2.90× at T=0.05 (Proxy
Anchor's α=32 ≈ T 0.03). Literature is entirely *observational* — nobody trains on it.

**`tversky`** — Tversky's contrast similarity in the bounded ratio form
`S = |A∩B| / (|A∩B| + α|A−B| + β|B−A|)`. Asymmetric when α≠β, which cosine cannot be;
α=β=1 is exactly Tanimoto/Jaccard from cheminformatics. Differentiable Tversky exists
([arXiv:2506.11035](https://arxiv.org/abs/2506.11035), Stanford 2025) but is
evaluated only on closed-set classification and language modelling — **no retrieval,
no unseen classes** — and uses the *unbounded* contrast form.

Honest priors from the literature sweep were ~20% (Shepard), 15–25% (Tversky) — both
were, in the end, roughly calibrated.

---

## 6. What is dead — do not retry

Thirteen candidates, each with a mechanism recorded in `docs/results.md`. The two
cognitive-science imports of §5 are entries twelve and thirteen; the earlier eleven:

| candidate | result |
| --- | --- |
| pairwise EMA distillation (HERD on CUB) | +0.03, p=0.93 — indistinguishable |
| hypergraph distillation (`herd_hg`) | −0.09 |
| prototype-affinity distillation | +0.05, p=0.89 (the "+0.52" was seed noise) |
| balanced sampling IPC=4 | **−2.74** — costs batch class diversity |
| persistent hypergraph (`hist_mem`) | premise refuted by IPC=4 |
| Sinkhorn hyperedge coupling (`hist_shot`) | near-inert by construction |
| Local NCA (`local_nca`) | **−13.7**; diagnostic showed the mechanism never engaged |
| Region Proxy Anchor (`region_pa`) | **−3.6**, σ 0.03 across seeds |
| hubness reduction | correlation −0.82 but **causally negative**: CSLS −0.65, Sinkhorn −3.16 |
| aligned pack fusion (Procrustes) | works (+2.2 pt) but is ensembling — not novel |
| asymmetric query/gallery embeddings | killed on mechanism: hubness is a property of image *content*, identical in either role |

Eliminated on literature before spending GPU: ensemble→single distillation (Yu et al.
CVPR 2019; S2SD), Recall@k surrogates (Patel et al. CVPR 2022), the NCA/SNNL `L_in`
form (Frosst et al. ICML 2019), late interaction (ColBERT — and a read-out probe
measured −1.47 pt).

---

## 7. Gotchas that cost real time

1. **The queue must take the objective from the recipe.** Passing `--objectives`
   from a hardcoded argument silently overrode the recipe, so every new *objective*
   ran as plain `hist`. Reported a result that was never the method.
2. **Three places whitelist objectives by name** and all must be updated for a new
   one: the `EndToEndObjective` Literal, `_uses_metric_proxies`, and
   `_DERIVED_OBJECTIVE_BASE` in `cli.py`. `_loss_for_objective` must also *forward*
   any new kwarg — it accepted one and dropped it.
3. **Never `scp` over a running queue script.** bash reads scripts incrementally by
   byte offset; use a new version number instead.
4. **`pkill -f <pattern>` matches your own ssh command.** Use `run_priority_queue_v1[5]`
   style bracketing.
5. **Artifacts must be matched by recipe DIGEST, not recipe ID.** Superseded In-Shop
   runs share an ID with current ones at 0.844 vs 0.902; mixing them produced a
   spurious +3.4 pt. `analyze_reference_matrix.py` pins on digest.
6. **Screening at n=1 on CUB does not work.** It produced a false positive (+0.52)
   within hours. Minimum 3 seeds, and prefer In-Shop.

---

## 8. Tools worth keeping

| script | what it does |
| --- | --- |
| `scripts/analyze_reference_matrix.py` | paired analysis, digest-pinned, exact sign test **and** paired t (they disagree at n=3 — quote both) |
| `scripts/measure_hubness.py` | hubness skew, cross-class rate, antihub fraction; `--correct` shows corrections *hurt* |
| `scripts/measure_antihubs.py` | antihub stability/cost/headroom |
| `scripts/aligned_pack_fusion.py` | Procrustes-aligned pack fusion |
| `scripts/probe_late_interaction.py` | MaxSim read-out probe |
| `scripts/run_priority_queue_v30.sh` | completed averaging seeds 1–2; queue now idle |
| `scripts/measure_selection_bias.py` | best-over-training selection bonus, per arm and per pair |

---

## 9. Recommended next steps

The iterative search is closed on the current evidence. Candidate 1 failed its
preregistered In-Shop screen; candidate 2 died on retrieval prior art; candidate
3 lacked repository provenance and was independently occupied by BLenDeR;
candidate 4's graded within-class relations were established continuous-label
DML prior art. See `docs/method_search_verdict.md` §§8–12. What survives is
worth writing, and it is not a novel-method claim.

1. **Write up H3.** Six paired seeds across two bases, all positive, with the
   recovery-tracks-regression proportionality as the mechanistic evidence. The claim
   is *momentum-teacher recipes silently lose 0.3–1.4 pt under trainable BatchNorm
   unless the teacher's normalisation mode matches the student's* — it applies to
   MoCo/BYOL/DINO-style setups well beyond DML, and it is checkable in an afternoon
   by anyone with such a codebase. This is the highest-value thing left.

2. **The reproducibility paper is the defensible one**, and it is nearly written
   already: provenance infrastructure that caught **two real confounds** (LayerNorm
   between a method and its control, BatchNorm between teacher and student), the
   quantified noise argument (a 1.08 pt spread at *fixed* seed; σ = 0.57 pt across six
   seeds; seeds-required is 5 or 17 depending on the *comparison*, not the dataset;
   most papers report one run), and a thirteen-entry
   negatives catalogue where every entry has a recorded mechanism rather than just a
   number. A *Metric Learning Reality Check* with the measurement done properly.

3. **Do not queue another method without a new repository measurement.** Another
   regularizer or similarity score repeats the fifteen-candidate failure pattern.
   Cross-trajectory supervision and controlled support expansion—the two
   measurement-adjacent alternatives—are occupied prior art. A new candidate
   must begin again at gate 1 of `docs/search_protocol.md`.

4. **Turn on `deterministic: true`** for everything new. It removes half the observed
   seed noise and costs nothing. Should have been default from the start.

5. **Validate on In-Shop, never CUB.** σ = 0.12 pt vs 0.57 pt. One In-Shop run at
   2.2 h beats six CUB runs at 45 min for any effect under 2 pt. The single most
   expensive lesson of this project — a CUB screen at n=1 produced a false positive
   (+0.52 pt) that survived hours before three seeds killed it.
