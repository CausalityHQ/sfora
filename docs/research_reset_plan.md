# SFORA research reset — status and plan (2026-07-27)

Status summary and forward plan after the 2026-07-20 recipe audit and the
corrected DGX reference-recipe matrix. Written against live evidence from
`riomus@100.104.199.68` (spark-2751, 1× GB10) and the current `main` tree.

---

## 1. Where Codex stopped

Last Codex session on this repo: `2026-07-25T16:21` (rollout
`019f99a6-d860-7fe3-b7b9-52084f62ea51`, 3060 events). It ran as a persistent
monitoring goal over the DGX experiment controller and ended with an explicit
**stop-and-reset recommendation**, not a completion:

> "No — not yet. We cannot honestly claim that HERD is both novel and better
> under the corrected evidence."

Its six recommendations were: finish the current iNat seed, pause the queue
before seeds 1–2 consume days, freeze the "novel"/"universal" claims, implement
S2SD as a direct baseline, pivot to a genuinely hypergraph-native teacher target,
and gate any expansion on a preregistered +0.5 R@1 threshold.

Nothing was committed after `22f7dd6` (2026-07-20). The reset was never executed.

---

## 2. Live state (verified 2026-07-27 08:5x)

| item | state |
| --- | --- |
| DGX | reachable via Tailscale `100.104.199.68`, up 25 days, GPU 96% / 77 °C |
| running job | `sfora image-end-to-end --dataset-name inat2018 --recipe pa_distill --seed 0`, started Jul 26 |
| progress | step ~100,800 / 130,380 (77%), epoch 46/60, ETA ~7 h |
| controller | `run_remote_extended_datasets.sh --controller`, PID 1649975, alive since Jul 20 |
| queue behind it | iNat seeds 0–2 × {PA, PA-distill, HIST, HERD} — **~10 more runs ≈ 12 days** |
| CI | green (`gh run list`: CI + Pages passing) |
| remote code | source files hash-identical to local `main`; but remote git HEAD is `3a1ae15` with a large dirty tree |

---

## 3. The corrected evidence

### 3.1 In-Shop — the only clean paired comparison we own

Best-over-training R@1, official/frozen recipes, 3 seeds each. PA rows are the
`reference` track (official Proxy Anchor In-Shop recipe); HIST rows are a frozen
`selected_extension` (HIST published no In-Shop recipe, so it was selected from
SOP using **training-split-only** scoring — no test leakage).

| arm | seed 0 | seed 1 | seed 2 | mean | Δ vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy Anchor | 0.9024 | 0.9048 | 0.9032 | **0.9035** | — |
| PA + distillation | 0.8999 | 0.8994 | 0.8990 | **0.8994** | **−0.41 pt** |
| HIST | 0.9046 | 0.9037 | 0.9031 | **0.9038** | — |
| HERD (HIST + distillation) | 0.8906 | 0.8892 | 0.8900 | **0.8899** | **−1.39 pt** |

Seed σ ≈ 0.0012. Every paired per-seed difference is negative (PA: −0.25 / −0.54
/ −0.42; HIST: −1.40 / −1.45 / −1.31). The effect is 3–12σ. **This is not noise —
the distillation reliably hurts on In-Shop, on both bases.**

Also note HIST − PA = **+0.03 pt**. The two bases are indistinguishable here.

### 3.2 iNaturalist 2018 — the recipe, not the method, is broken

| arm | best R@1 | best epoch | final R@1 |
| --- | ---: | ---: | ---: |
| Proxy Anchor | 0.2099 | **5** / 60 | 0.1734 |
| PA + distillation (running) | 0.2094 | **5** / 60 | 0.1749 @ ep 46 |

Both arms peak at **epoch 5** and then decay for 55 more epochs. The recipe is
`proxy_anchor.inat2018.selected-from-cars-51db570` — Cars196 hyperparameters
(8k images, 98 classes) transferred to iNat2018 (~450k images, ~8k classes). The
LR/schedule/step-count are simply wrong for the dataset. We are spending ~30
GPU-hours per run to report a number that was reached in the first ten minutes,
and the arms are within 0.05 pt of each other — statistically empty.

### 3.3 The headline claims have no corrected backing

Recipe-backed artifacts on the DGX: **16 total, all In-Shop or iNat**. There are
**zero** reference-recipe artifacts for CUB, Cars, or SOP — yet every headline
claim in `README.md` and `docs/results.md` is CUB (71.6 single / 74.68 pack) or
Cars. Those are all `modified_legacy`.

Worse, the legacy CUB HERD-vs-HIST comparison is **confounded**: in
`run-inat2018-matrix.sh` and its predecessors the `hist` arm ran
`--no-embedding-layer-norm` while the `herd` arm ran `--embedding-layer-norm`.
HERD got LayerNorm *and* distillation; the control got neither. Since official
HIST already enables no-affine embedding LayerNorm, the corrected HERD delta is
EMA-distillation-only — and has never been measured on CUB.

The repo already documents this caveat honestly (`docs/results.md:9`,
`README.md:51`). The problem is that the headline tables above the caveat still
assert the uncorrected claims.

### 3.4 The ensemble result is real but compute-driven

From the existing ablation (`docs/results.md:400`): a pack of plain HIST models
clears reported PFML (0.734) at 4 models and reaches 0.7443 at 5. HERD adds
~0.25 pt at 5 models. So the SOTA-beating number is feature-concatenation
ensembling — an established DML paradigm (BIER and descendants) — not HERD.

### 3.5 Novelty boundary

`_relational_distillation_loss` (`src/sfora/image_end_to_end.py:3428`) is:
row-wise softmax over the batch cosine-similarity matrix, diagonal masked,
cross-entropy against an EMA-teacher's distribution at the same temperature.

- **S2SD** (Roth et al., ICML 2021) already distils row-wise softmax
  distributions of batch cosine-similarity matrices, objective-agnostically, in
  supervised DML. Mathematically very close.
- **STML** (Kim et al., CVPR 2022) already pairs a momentum/EMA teacher with
  relational similarity targets in metric learning.
- **RKD** (Park et al., CVPR 2019) established relation transfer for metric
  learning.
- **HIST** (Lim et al., CVPR 2022) contributes the hypergraph component.

None of these is implemented as an in-harness baseline. "HIST base + generic EMA
relational loss" may be an unreported *combination*, but combination novelty is
weak — and under corrected evidence it is not even an improvement.

---

## 4. Diagnosis

Five problems, in priority order.

**P1 — The decisive experiment was never run.** Official reference recipes exist
in `image_recipes.py` for PA×{cub,cars,sop,inshop} and HIST×{cub,cars,sop}. CUB
and Cars are already cached on the DGX (`~/.cache/huggingface/hub`). Both are
small and cheap. The corrected CUB/Cars matrix that would actually settle the
headline claim has never been launched.

**P2 — Compute is badly misallocated.** The DGX is spending ~30 GPU-h per run on
an iNat recipe that peaks at epoch 5, with ~12 days of identical runs queued
behind it, while the cheap decisive experiment sits unrun.

**P3 — Public claims contradict our own best evidence.** README asserts the
distillation "improves **any** base loss" and "beats Proxy Anchor on every
dataset". The only clean, official-recipe, multi-seed, paired evidence we have
says it hurts both bases, consistently.

**P4 — No prior-art baselines.** Without S2SD and STML in-harness, there is no
defensible novelty argument regardless of the numbers.

**P5 — Reproducibility gap in the remote checkout.** Remote HEAD is `3a1ae15`
(months stale) with a large dirty tree. Source files currently hash-match local
`main`, so results are valid — but artifacts record a `recipe_digest` and no
**code** commit. For a project whose current thesis is provenance safety, this is
the obvious hole.

---

## 5. Hypotheses that may explain the negative result

### H3 — BatchNorm train/eval mismatch between teacher and student (PRIMARY)

**This is the strongest hypothesis and it explains every observation we have.**

The EMA teacher is built and used like this:

```python
ema_teacher = _copy.deepcopy(model)      # image_end_to_end.py:722
ema_teacher.eval()                       # :725  <-- teacher uses BN RUNNING stats
...
ema_embeddings = _normalize(ema_teacher(images), torch).detach()   # :936
...
_update_ema_teacher(ema_teacher, model, momentum=config.ema_momentum)  # :952
```

and `_update_ema_teacher` (`:2772`) EMA-blends **parameters** but **hard-copies
buffers**:

```python
teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
teacher_buffer.data.copy_(student_buffer.data)   # <-- NOT an EMA
```

The student trains in `.train()` mode and therefore normalises with **batch**
statistics. The teacher is in `.eval()` mode and normalises with **running**
statistics.

* When **BatchNorm is frozen**, running stats never change, and train/eval
  normalise identically. The teacher really is "the student, slower". Everything
  is consistent.
* When **BatchNorm is trainable**, teacher and student compute embeddings through
  *different normalisation functions*. The teacher's similarity matrix is no
  longer a lagged copy of the student's — it is a systematically different
  function of the same images. Distilling toward it fights the base loss instead
  of regularising it. The hard-copied buffers make it worse: the teacher's
  normalisation statistics jump to the student's **instantly** while its weights
  lag by ~1/(1−m) ≈ 1000 steps. That is an internally inconsistent model — stale
  weights wearing fresh normalisation statistics.

Now check `freeze_batch_norm` against the sign of every distillation result we
have:

| dataset | arms | `freeze_batch_norm` | distillation effect |
| --- | --- | :---: | ---: |
| CUB | PA, HIST | **True** | **helps** (+1.2, +1.6 legacy) |
| Cars | PA, HIST | **True** | **helps** (+0.8, +1.3 legacy) |
| In-Shop | PA (`reference`) | **False** | **hurts** −0.41 |
| In-Shop | HIST (from SOP recipe) | **False** | **hurts** −1.39 |

Verified directly in the artifacts: all four In-Shop arms record
`freeze_batch_norm=False, freeze_batch_norm_affine=False`. The correlation is
perfect across every measurement, with no exceptions.

Note also that running a momentum teacher in `eval()` is the **nonstandard**
choice: MoCo's key encoder, BYOL's target network and DINO's teacher all run in
train mode. This looks like a plain implementation bug, not a design decision.

**Prediction.** CUB has frozen BN, so the running CUB matrix should show a
*positive* paired delta. If it does, "the method is worthless" is the wrong
conclusion — "the method is broken whenever BN is trainable" is the right one.

**Decisive test (~7 GPU-hours).** Fix the teacher's normalisation consistency
(run it in train mode, and/or EMA the buffers), then rerun In-Shop PA+distill for
3 seeds against the PA baseline we already own. Recipe untouched — this is a bug
fix, not a hyperparameter change. If the sign flips, H3 is confirmed and the
In-Shop "negative result" was never a scientific finding.

### H1 and H2 — secondary

Both are cheaper to test and remain worth running, but H3 subsumes much of their
explanatory power.

**H1 — It is a small-data regularizer, not a universal improvement.**
CUB has 5.9k train images / 100 classes; In-Shop has ~25k / 3997; iNat far more.
A variance-reducing teacher target helps when data is scarce and only costs
capacity when it is not. *Test:* sweep `ema_distill_weight ∈ {0, 0.25, 0.5, 1.0}`
on In-Shop, one seed. If harm is monotone in weight, H1 is supported and the
honest framing becomes "a low-data DML regularizer" — narrower, but defensible
and publishable as such.

**H2 — The loss is mis-specified in two concrete ways.**

1. *No teacher sharpening.* `student_logits` and `teacher_logits` use the **same**
   `tau` (0.1). DINO/STML sharpen the teacher (τ_t < τ_s) to make the target
   informative rather than a near-copy of the student. As written the target
   carries little information beyond the student's own geometry.
   *Test:* separate `ema_distill_tau_teacher` (0.04) from `ema_distill_tau_student` (0.1).
2. *No warmup gate.* The MEAD path gates on `step > warmup_steps`
   (`image_end_to_end.py:892`); the relational path does **not**
   (`:902`, `:937`). From step 0 the EMA teacher is essentially the random-init
   head, so early training distils noise. On a 60-epoch In-Shop run that is
   thousands of poisoned steps; on a 40-epoch CUB run it is far fewer — which
   would also partly explain the CUB/In-Shop divergence.
   *Test:* gate the relational term on `step > warmup_steps`.

If H2 fixes the sign of the In-Shop delta, the method is alive and the previous
result was an implementation bug, not a scientific finding.

---

## 6. Plan

### Phase 0 — Stop the bleeding (today, ~1 h, no GPU cost)

- [ ] Decide the fate of the running iNat PA-distill seed (see §7).
- [ ] **Stop the controller** (PID 1649975) so it cannot auto-launch the remaining
      ~10 iNat runs. Park, do not delete, its state.
- [ ] Pin remote provenance: `git fetch && git checkout` local `main` SHA on the
      DGX, and record the code commit SHA into every future artifact's `config`.
- [ ] Freeze the disputed claims in `README.md` and `docs/results.md` — demote the
      headline tables behind the existing correction notice rather than above it.

### Phase 1 — Run the decisive matrix (~1–2 days GPU)

The experiment that should have run first.

- [ ] Extend `run_remote_extended_datasets.sh` (or a new `run_remote_reference_matrix.sh`)
      to cover `cub` and `cars` with the existing **reference** recipes.
- [ ] Matrix: {PA, PA+distill, HIST, HERD} × {cub, cars} × seeds {0,1,2} = 24 runs.
      Both datasets are small and already cached; expect well under the cost of a
      single iNat run.
- [ ] **Critical:** HIST and HERD must differ *only* in `ema_distill_weight`.
      Both inherit `embedding_layer_norm: True` from `_hist_config`. Verify this
      in the emitted config before trusting any delta — this is the exact confound
      that invalidated the legacy result.
- [ ] Report paired per-seed deltas with a bootstrap CI, not just means.

### Phase 2 — Prior-art baselines and loss fixes (parallel, mostly CPU/dev)

- [ ] Implement **S2SD** as a first-class objective/augmentation in
      `image_end_to_end.py`, registered in the objective registry.
- [ ] Implement **STML** (or at minimum its EMA-teacher relational-similarity term).
- [ ] Implement the H2 fixes behind flags: `ema_distill_tau_teacher` and a warmup
      gate on the relational term. Keep defaults unchanged so existing artifacts
      remain interpretable.
- [ ] Tests for each, following the existing `tests/test_losses.py` /
      `tests/test_image_end_to_end.py` patterns.

### Phase 3 — Cheap diagnostic sweeps (~0.5 day GPU, run after Phase 1 frees the GPU)

- [ ] H1: In-Shop `ema_distill_weight ∈ {0, 0.25, 0.5, 1.0}`, seed 0.
- [ ] H2: In-Shop seed 0 with (a) teacher sharpening, (b) warmup gate, (c) both.
- [ ] CUB: same four-point weight sweep, to locate the crossover in dataset size.

### Phase 4 — Decision gate (preregistered, write it down *before* looking)

**Statistical correction (2026-07-27).** An earlier draft of this gate demanded a
"3-seed paired bootstrap CI excluding zero". That is unsound and has been
withdrawn. A 3-sample bootstrap resamples from 3 points; its CI is an artifact of
that fact. The honest test for paired seeds is an **exact sign/permutation test**,
whose two-sided p-value has a hard floor of `2^-(n-1)`:

| seeds | best attainable two-sided p |
| ---: | ---: |
| 3 | 0.250 |
| 4 | 0.125 |
| 5 | 0.063 |
| **6** | **0.031** |

**Three seeds can never reach p < 0.05, no matter how large the effect.** So the
matrix is a *screening* run, not a publishable one. Verified in
`scripts/analyze_reference_matrix.py`, which reports this floor alongside every
result so the gap cannot be forgotten.

Two-stage gate:

**Stage A — screen (3 seeds, running now).** Continue only if, on CUB **and**
Cars, under official reference recipes with LayerNorm held constant, the paired
distillation delta is **positive on every seed** and the mean is **≥ +0.5 R@1**.
This is a direction check; it carries no significance claim.

**Stage B — confirm (≥ 6 seeds, CUB at minimum).** Only a Stage-A pass earns the
extra compute. Requires all-positive paired deltas across ≥ 6 seeds (exact
p ≤ 0.031), mean ≥ +0.5 R@1, and no regression exceeding −0.2 pt on In-Shop or
SOP. CUB runs at ~90 min/arm, so 6 seeds × 2 arms is ~18 GPU-hours — affordable.

Only a Stage-B pass supports a method claim in the README.

Outcomes:

- **Pass** → the method survives. Then and only then does the novelty question
  matter: beat or materially differ from S2SD/STML, and pivot to the
  hypergraph-native target (§ below) to establish distinctness.
- **Fail, H1 supported** → reframe honestly as a **low-data DML regularizer**.
  Narrow, real, and publishable as a focused study with a clear crossover curve.
- **Fail, H2 fixes it** → the negative result was an implementation bug; rerun
  Phase 1 with the fixed loss before drawing any conclusion.
- **Fail outright** → SFORA becomes a **reproducibility, benchmark and
  negative-results** project. That is a genuinely valuable contribution: strict
  provenance-tracked author recipes across 5 datasets, a clean multi-seed refutation
  of a plausible-sounding distillation idea, and a documented list of ~16 loss-geometry
  changes that do not move the plateau. Do not undersell this outcome.

### Phase 5 — The pivot, if Phase 4 passes or H1 holds

Codex's proposal, and it is the right one: the current teacher target is a
*generic pairwise similarity matrix*, which is exactly what S2SD/STML already do.
A genuinely HIST-native target would distil the teacher's **hyperedge incidence
structure, semantic-tuplet distributions, and uncertainty-weighted higher-order
relations** — quantities that only exist because of the hypergraph, and that no
pairwise method can express. Test on CUB seed 0 first; expand only on the Phase 4
gate.

---

## 7. Open decision — the running iNat seed

The PA-distill iNat run is at 77%, ~7 h remaining, and will produce a number that
is (a) statistically tied with the PA arm and (b) determined by epoch 5 of 60.

- **Let it finish** — completes one clean paired iNat data point; costs ~7 h GPU
  and delays Phase 1 by that much.
- **Kill it now** — frees the GPU immediately; loses a data point that is already
  known to be uninformative.

Either way the **controller must be stopped** so the remaining ~12 days of iNat
runs do not launch. If iNat is worth keeping at all, it needs its own recipe
work first (the epoch-5 peak is a schedule bug, not a result).

---

## 8. Memory correction required

`memory/relational-distillation-universal.md` records "EMA-teacher distillation
improves ANY base; best-base+distill beats PA everywhere". The corrected In-Shop
evidence contradicts this. That memory must be rewritten or deleted once Phase 1
lands.
