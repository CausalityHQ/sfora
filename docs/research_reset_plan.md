# SFORA research reset — status and plan (2026-07-27)

Status summary and forward plan after the 2026-07-20 recipe audit and the
corrected DGX reference-recipe matrix. Written against live evidence from
`riomus@100.104.199.68` (spark-2751, 1× GB10) and the current `main` tree.

---

## 0. Where the thesis stands (2026-07-27, late)

Updated as results landed. Three of the ideas this document was built on are now
dead or downgraded, and the surviving lead is not the one it was written to pursue.

| claim | status |
| --- | --- |
| HERD (pairwise EMA distillation) beats HIST | **dead** — the legacy +1.6 was a LayerNorm confound; corrected paired delta is **−0.27** |
| The distillation is a universal improvement | **dead** — it is a *regulariser*; it helps overfitting bases and hurts well-regularised ones |
| A hypergraph-native (n-ary) target is the novelty | **falsified** — the provably n-ary arm scored **−0.09**; the non-n-ary control scored **+0.52** |
| H3: BatchNorm teacher/student mismatch | mechanism proved at unit level; In-Shop intervention queued |
| SHOT (entropic-OT coupling) | untested; motivation weakened by batch composition (§Phase 5b) |
| Prototype-affinity KD from an EMA teacher | **retracted** — +0.52 was seed noise; 3-seed mean is 0.7112 (σ = 1.1 pt) |
| Balanced sampling (IPC=4) activates the hypergraph | **dead** — 0.6909, −2.74 pt |

**The measurement problem now dominates the method problem.** CUB seed noise in this
harness is σ ≈ 1.1 pt (3 seeds of `herd_hg_incidence`; the legacy 9-seed HERD figure
was σ ≈ 0.6). Required seeds per arm for 80% power at α = 0.05,
`n = ((z_{α/2}+z_β)·σ/Δ)²`:

| effect Δ | σ = 0.6 | σ = 1.1 |
| ---: | ---: | ---: |
| +0.5 pt | 12 | **37** |
| +1.0 pt | 3 | 10 |
| +2.0 pt | 1 | 3 |

So **the preregistered +0.5 pt gate on CUB is unreachable in practice** — 12–37
seeds per arm is 8–25 GPU-hours *per arm*, before any ablation. Screening variants
at one seed, as this plan did, cannot work, and it produced a false positive within
hours. (An earlier draft of this paragraph said "≈ 10 seeds"; that used a paired
approximation and understated it. The one-sample figure above is the right one.)

This is worth stating beyond this project, because it is a quantitative version of
Musgrave et al.'s reality check: at σ ≈ 0.6–1.1 pt, **the ~1 pt improvements the CUB
literature routinely reports from single runs are not resolvable.** A claimed +1.0 pt
needs 3–10 seeds to distinguish from noise, and most papers report one.

Practical consequence for this project: stop screening on CUB. Either commit to ≥ 12
seeds per arm there, or move method search to In-Shop, where σ = 0.12 pt against a
−1.39 pt effect — a signal-to-noise ratio roughly **80× better**, and enough that a
single seed already resolves it.

The honest shape of the work is no longer "we invented a hypergraph distillation
method". It is: *a controlled study of when EMA distillation helps in deep metric
learning, with a mechanism that explains the field's inconsistent results, plus the
provenance machinery that made the study trustworthy.* The +0.52 arm, if it
survives multi-seed, is a useful empirical finding inside that study — Hinton-style
dark-knowledge KD over Mahalanobis-proxy logits — not a novel mechanism.

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

Every paired per-seed difference is negative (PA: −0.25 / −0.55 / −0.41; HIST:
−1.40 / −1.45 / −1.31).

**Statistics, stated correctly.** An earlier draft said "seed σ ≈ 0.0012, so the
effect is 3–12σ". That used the wrong denominator — the across-seed spread of a
single arm, rather than the standard error of the *paired differences* — and
implied a z-score where n=3 gives df=2. Corrected, from
`scripts/analyze_reference_matrix.py`:

| comparison | mean Δ | paired t (df=2) | p (t-test) | p (exact sign) |
| --- | ---: | ---: | ---: | ---: |
| PA + distill vs PA | −0.406 | −4.75 | **0.042** | 0.250 |
| HERD vs HIST | −1.387 | −33.9 | **0.0009** | 0.250 |

Read this honestly:

* The **HIST/HERD leg is robust** — a 1.39 pt regression with a t of −34.
* The **PA leg is marginal**, not overwhelming: p ≈ 0.04 *and only under a
  normality assumption that three points cannot evidence*. The assumption-free
  exact sign test cannot go below 0.25 at n=3 for either leg.

So: "distillation hurts HIST on In-Shop" is well supported; "distillation hurts
PA on In-Shop" is suggestive and consistent, but rests on 3 seeds and a
distributional assumption. Do not oversell the PA leg.

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

**Confounds, and why they do not explain the sign.** The four cells above differ
in more than `freeze_batch_norm`, so each rival explanation was checked against
the artifacts:

| rival explanation | ruled out by |
| --- | --- |
| **Backbone** (In-Shop PA uses `bn_inception`, CUB/Cars use `resnet50`) | In-Shop **HIST** is `resnet50` and regresses *hardest* (−1.39). Backbone held constant, sign still flips. |
| **Batch size** (HIST 32 vs PA 180 → 31 vs 179 negatives in the row-softmax) | CUB HIST and In-Shop HIST **both** use batch 32. Same batch size, opposite sign. |
| **Learning rate / optimizer** | CUB HIST and In-Shop HIST both use Adam at 1e-4, batch 32, ResNet-50. |
| **Dataset size / class count** | *Not* ruled out — this covaries perfectly with `freeze_batch_norm` across our cells, and is exactly hypothesis H1. |

The clean contrast is **Cars HIST vs In-Shop HIST**: same backbone, same
optimizer, same batch size, same learning rate, same loss — differing in
`freeze_batch_norm` (True → False) and dataset. Distillation helps in the first
and inflicts our largest regression in the second.

That said, these are **not four independent draws**. `freeze_batch_norm` is never
randomised — it is baked into each method's published protocol per dataset — so
the table is really two correlated dataset-clusters, {CUB, Cars} vs {In-Shop}.
For the PA arm specifically, backbone and `freeze_batch_norm` are perfectly
collinear in this codebase. **H3 is not established by the table — it is
established or killed by the direct test below.**

**The mechanism's channel is verified, though.** Independently checked across both
backbones: `torchvision` ResNet-50 is BatchNorm2d throughout; `bn_inception.py`
has dozens of `BatchNorm2d` and **zero Dropout**; the embedding head adds only
`Linear` + optional `LayerNorm(elementwise_affine=False)`, neither of which is
mode-sensitive. So BatchNorm is not merely *a* source of teacher/student
divergence — it is architecturally the **only possible** one. That matters for
the intervention: since the fix flips exactly two booleans and nothing else, and
BatchNorm is the only channel they can act through, a sign change cannot be
attributed to anything else.

### What is proved at unit level, and therefore not worth GPU time

Four tests in `tests/test_image_end_to_end.py` discharge parts of this by proof
rather than measurement:

1. **The mechanism is real.** With trainable BatchNorm, an eval-mode teacher and a
   train-mode student compute *different* outputs from identical weights and
   identical input; setting `ema_teacher_train_mode` makes them agree exactly.
2. **The fix is inert under frozen BatchNorm.** Both teacher modes force backbone
   BN to eval, so outputs are bit-identical. H3's null prediction on CUB is
   therefore a theorem, not an experiment — **the CUB `_bnfix` arms were dropped
   from the queue**, reclaiming ~2.5 GPU-hours for the In-Shop tests that
   actually discriminate.
3. **`ema_teacher_ema_buffers` is inert under frozen BatchNorm** (to within one
   ulp — the blend is the identity in exact arithmetic but rounds in float).
4. **A train-mode teacher ignores its running buffers entirely**, since BatchNorm
   in train mode normalises with *batch* statistics. This retires a plausible
   worry that the fixed teacher's normalisation statistics might "lag" behind the
   student at `momentum=0.999`: there is no lag, because those buffers never
   reach the teacher's output. `ema_teacher_ema_buffers` is consequently inert in
   *both* configurations of interest; `ema_teacher_train_mode` does all the work.

**Two opposing predictions (this is what makes it a test, not a story).**

* On **CUB**, backbone BN is frozen, so teacher and student already normalise
  identically and the fix must be **inert**. `herd_bnfix` ≈ `herd`.
* On **In-Shop**, BN is trainable, so the fix should **flip the sign**.

A hypothesis that predicts *no effect* in one arm and a *sign reversal* in
another is falsifiable in both directions. Both arms are queued.

**Decisive test (~13 GPU-hours, queued).** Rerun In-Shop with the teacher's
normalisation made consistent, against baselines we already own. Ordered by how
much the result can carry:

1. **`herd_bnfix`, 3 seeds** vs HERD's 0.8906 / 0.8892 / 0.8900. This is the
   **robust** leg (−1.39 pt, t = −33.9). Run it first — testing the fix where the
   effect is solid discriminates far better than where it is borderline.
2. **`pa_distill_bnfix`, 3 seeds** vs PA's 0.9024 / 0.9048 / 0.9032 (the marginal
   leg).

Recipes are otherwise untouched: this flips two booleans, not a hyperparameter.

### H3 is a contribution in its own right, whichever way the R@1 lands

Worth stating separately from HERD's fate, because it survives either outcome:

> Momentum-teacher recipes silently break under trainable BatchNorm unless the
> teacher's normalisation mode is made consistent with the student's — and this
> bug can flip the sign of a paired multi-seed result.

That is mechanistic, generalisable, and falsifiable with a built-in decisive test.
It applies to any MoCo/BYOL/DINO-style momentum teacher grafted onto a metric-
learning backbone with updating BatchNorm — a common thing to try. It is an
engineering/reproducibility finding rather than a modelling one, and it is
better-shaped than most entries in the "did not work" catalogue.

It does **not**, however, manufacture novelty for the loss: the fix does not touch
`_relational_distillation_loss` at all, so the mechanism claim is unchanged from
§3.5. If the fixed version works, the honest framing is "HIST base + STML-style
EMA-relational distillation, validated where the original comparison was
confounded" — not a new distillation mechanism.

### H4 — the teacher and student never see different views (structural gap)

Confirmed at every call site: the non-MEAD path calls `ema_teacher(images)` and
`model(images)` on the **same tensor**; the MEAD path calls
`ema_teacher(global_images)` on the same `global_images` the student embeds.

So the relational term receives **zero cross-view supervision**. It is pure
temporal EMA-lag self-consistency on a fixed input — not the augmentation-
invariance objective that makes relational distillation informative in
DINO/BYOL/STML, where the teacher sees a *different* crop. A same-view, same-τ
EMA target can carry very little beyond smoothed noise, because no
invariance pressure is being exploited at all.

This is more fundamental than the temperature question (H2.1): it limits what the
loss can express *in principle*, not merely how peaked its target is. It also
independently supports the novelty verdict in §3.5 — without cross-view
supervision this is a strictly weaker relative of STML, not an extension of it.

### H5 — the distillation weight has never been calibrated (open, needs instrumentation)

Measured from the logged `loss_history` of the completed In-Shop runs
(digest-pinned, same step counts within each pair):

| arms | base final loss | with distillation | implied distill term | ratio |
| --- | ---: | ---: | ---: | ---: |
| HIST → HERD | 0.226 | 3.121 | 2.895 | **12.8×** |
| PA → PA+distill | 0.680 | 5.688 | 5.008 | **7.4×** |

At `ema_distill_weight = 1.0` the distillation term is 7–13× the base objective
*in value*, and the dominance ratio orders the same way as the damage (HIST 12.8×
→ −1.39 pt; PA 7.4× → −0.41 pt).

**This is a flag, not a finding, and must not be quoted as one.** Cross-entropy
decomposes as `CE(q,p) = H(q) + KL(q‖p)`. Only the `KL` term carries gradient;
`H(q)` is an additive constant w.r.t. the student. The logged total cannot
separate them, so a large loss *value* is entirely compatible with a small
gradient contribution — which is what one expects early on, since the teacher
starts as an exact `deepcopy` of the student (`KL = 0` at step 0). The
dominance-ratio correlation is also two points.

**Resolve by instrumentation, not argument:** log teacher entropy `H(q)` and
`KL(q‖p)` separately from the relational loss. Cheap, and it converts this from
speculation into a measurement. Deliberately deferred rather than rushed into the
training loop mid-matrix, so all arms run identical code.

### H2.2 (no warmup gate) — demoted

The EMA time constant is ~1/(1−m) = 1000 steps at m = 0.999. Against actual step
counts: In-Shop PA runs 8,640 steps, so the catch-up window is ~12% of training;
In-Shop HIST runs 49,349 steps, so it is ~2%. HIST therefore has *far less*
relative exposure to a near-init teacher — yet HERD regresses 3× harder than
PA+distill. If missing warmup were the dominant mechanism the ranking would run
the other way.

Still worth an ablation (it is nearly free), but it cannot explain the HIST/PA
asymmetry and should not be presented as the leading explanation.

### H2.1 (no teacher sharpening) — kept, but recalibrated

Factually correct: `student_logits` and `teacher_logits` both use
`config.ema_distill_tau`. But the framing was loose. Teacher sharpening in DINO
works together with explicit **centering** (which this code has none of) to
prevent entropy collapse; and an EMA teacher being correlated with its student is
by design in MoCo/BYOL/DINO, not a defect. Expect a magnitude lever worth perhaps
0.1–0.3 pt, not a sign flip. H4 is the better explanation of why the target is
low-information.

### H1 — small-data regularizer

Unchanged, and *not* separable from H3 by the current data: dataset size covaries
perfectly with `freeze_batch_norm` across our four cells. The In-Shop weight
sweep separates them — if harm is monotone in `ema_distill_weight` *after* the H3
fix, H1 survives on its own.

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

### Phase 5 — The pivot: a hypergraph-native teacher target

The current teacher target is a *generic pairwise similarity matrix* — precisely
what S2SD/STML already do. A defensible target must be one that **only exists
because HIST builds a hypergraph**.

**The non-pairwise argument, made precisely.** In `_hist_loss`:

```python
incidence   = one_hot + exp(-alpha * distance) * (1 - one_hot)   # H, (N, E)
edge_degree = incidence.sum(dim=0)                               # d_e, sums over the WHOLE BATCH
propagate   = Dv^-1/2 @ H @ W @ De^-1 @ H.T @ Dv^-1/2            # G, (N, N)
logits      = propagate @ theta2(lrelu(bn1(theta1(features))))   # (N, C)
```

`G_ij = Σ_e H_ie·H_je / (d_v(i)·d_v(j))^{1/2}·d_e(e)` is **not** a function of
`(z_i, z_j)` alone: its normalisation `d_e(e) = Σ_k H_ke` runs over every sample
currently in the batch. RKD/S2SD/STML all compute targets of the form `s(z_i,z_j)`
for a fixed pairwise metric; none can reproduce a quantity whose normalisation
depends on how many *other* samples share a hyperedge. That is a real, checkable
novelty boundary — unlike the current loss, which has none.

**The infrastructure already exists.** `_attach_hist_module` (line 646) runs
*before* `ema_teacher = copy.deepcopy(model)` (line 733), and `_update_ema_teacher`
iterates `.parameters()` — so **`ema_teacher.hist_module` is already a live,
correctly EMA-updated copy** of the Gaussian prototypes and HGNN weights. It is
simply never read: `_relational_distillation_loss` only consumes
`ema_teacher(images)` backbone embeddings. Every candidate below is "write a loss
that reads a module already being maintained", not "build EMA infrastructure".

Candidates, ranked:

1. **HGNN propagated-logit distillation** (strongest). Distil the teacher's
   pre-CE `logits` tensor: `q_i = softmax(logits^t_i / τ_t)`, stop-grad;
   `L = CE(softmax(logits^s_i / τ_s), q_i)`. Fully hypergraph-native via `G^t`.
   Falsifier: CUB seed 0, HIST base, this term only.
2. **Incidence-row distillation** (cheapest safe test). Distil `H^t_{i,:}`
   restricted to `class_within`. Needs only `hist_module.means`/`log_vars` — it
   never touches `bn1`, so it sidesteps the BatchNorm question entirely. Run this
   **first**. It is informationally a subset of (1), making the pair a clean
   decomposition: if (2) reproduces most of (1)'s effect, the gain is in the
   affinity; if not, it is in the propagation.
3. **Full-catalog Gaussian-prototype distillation.** Soft targets over *all* `C`
   classes using the prototype bank. Honest caveat: this bypasses `H`/`G`
   entirely and is prototype/proxy distillation — closer to classic dark-knowledge
   KD than to anything hypergraph-native. Weaker novelty claim.
4. **Edge-degree distillation** — control only, to test whether any gain in (1)
   comes from fine structure or merely a coarse batch-composition prior.

**A correction to note when implementing.** It has been suggested that the
teacher's `hist_module.bn1` running statistics are "frozen at deepcopy time and
never updated". That is **false** for the current code: `_update_ema_teacher`
hard-copies *all* buffers from the student, `hist_module.bn1` included, so they
track the student exactly. It *does* become true if `ema_teacher_ema_buffers=True`
(the H3 fix) is enabled, which introduces deliberate lag. Inert today because the
teacher's `hist_module` is unused — but any candidate above that forwards through
`bn1` must revisit it, and should recall the earlier bug where miscalibrating that
exact BatchNorm dropped training loss while collapsing zero-shot retrieval.

**Budget.** Cap the pivot at ~4 days and a single-seed CUB smoke test per
candidate, under the same preregistered +0.5 R@1 gate as everything else — no
grading on a curve because a candidate is new. Honest prior: each of these is
roughly a 20–30% shot, and they all still bet on the same family of mechanism
("EMA-smooth a noisy relational statistic") that the In-Shop result just went
1-for-1 against. The argument for a better prior is mechanistic rather than
wishful: `G` has far fewer effective degrees of freedom than an `N×N` pairwise
matrix (roughly the number of classes present in the batch), so its per-step
estimate is noisier — and a noisier statistic is exactly where an EMA teacher has
more variance to remove.

### Phase 5b — SHOT: Sinkhorn Hyperedge Optimal Transport

A second, independent bet, arrived at from a literature survey rather than from
the existing code. Unlike the distillation family it is a **structural** change to
HIST, not an added regularisation term, and it involves no EMA teacher at all.

**Target, restated honestly.** Published CUB numbers are not a trustworthy bar.
Musgrave et al., *A Metric Learning Reality Check* (ECCV 2020), showed that a
decade of claimed DML advances were "marginal at best" once backbone, optimizer,
augmentation and embedding size were held constant, and documented specific papers
comparing against competitors on weaker backbones. Our own harness agrees: we
reproduce HIST **above** its published number (0.7183 vs 0.714) while PFML — the
nominal 73.4 SOTA — collapses entirely. So the bar we hold ourselves to is
**beat 0.7183 in-harness**, everything else held constant. That is a harder and
more meaningful target than beating a number we cannot reproduce.

**The gap.** Optimal transport appears in the hypergraph literature only as a
metric *between* hypergraphs (HyperCOT; Wasserstein hypergraph coarsening), while
hypergraph message passing normalises the incidence matrix by degree (HGNN, HNHN's
`D_V^-1`, SNALS's one-sided `D_E^-1`). Nothing replaces the degree normalisation
*inside* node↔hyperedge message passing with an entropic-OT coupling. Separately,
SwAV established that a Sinkhorn equipartition constraint prevents representation
collapse, and its objective is explicitly a free energy — an energy term plus an
entropy term scaled by a temperature `ε`.

**The defect being fixed.** HIST propagates with

```
G = Dv^-1/2 · H · W · De^-1 · H^T · Dv^-1/2,
H_ie = 1{y_i = e} + exp(-alpha·d_ie)·(1 - 1{y_i = e})
```

**Correction to an earlier overstatement.** A first draft of this section called
`De^-1` an ad-hoc defect that lets broad-Gaussian classes dominate propagation.
That is not right, and the claim is withdrawn: `De^-1` divides each hyperedge by
its total soft degree, so a broad class accumulating mass is *already*
down-weighted, and `Dv^-1/2 ... Dv^-1/2` is the standard symmetric hypergraph
Laplacian normalisation. HIST is not doing something naive here.

The accurate framing is weaker but still worth testing: degree normalisation and a
marginal-constrained transport coupling are **two different normalisations of the
same incidence matrix**. Degree normalisation rescales; entropic OT additionally
*constrains* the mass each sample distributes and each hyperedge receives, which
ties propagation to the batch's true class composition. With `samples_per_class: 0`
in the reference config, batches are not class-balanced, so the two normalisations
genuinely differ. Whether that helps is an empirical question, not a bug fix.

**A design weakness found by testing, and what it forced.** With the
HIST-compatible cost the true class has zero cost, so once classes separate the
incidence goes near-one-hot — and `one_hot / N` *already* satisfies the
class-population marginals, leaving Sinkhorn almost nothing to do. The safe variant
is therefore close to inert exactly where training spends most of its time. Hence a
second cost mode, `geometric`, which keeps the true-class term so the coupling stays
a live geometry-driven soft assignment with labels entering only through the
marginal. `test_geometric_cost_stays_active_when_classes_are_well_separated` pins
the difference. Expect `hist_shot` ≈ HIST and treat `hist_shot_geometric` as the
real test of the idea.

**The method.** Replace `H` with the coupling that minimises the free energy

```
P* = argmin_P  <C, P>  -  eps·H(P)      s.t.  P·1 = 1/N,  P^T·1 = c
```

solved by Sinkhorn iterations in the log domain, with `C_ie = alpha·d_ie` masked to
zero on the true class and `c_e` = class `e`'s true share of the batch.

Each domain does real work here:

* **Physics.** `F = U − TS` with temperature `eps`; `P*` is the Gibbs equilibrium
  coupling and the duals `u, v` are two coupled partition functions. This
  repository's own catalogue records that raw potentials "collapse without a
  partition-function (softmax) normaliser" — entropic OT supplies precisely that,
  applied to the *higher-order* structure rather than to pairwise potentials.
* **Biology.** The column marginal is competitive exclusion: every class holds
  exactly its niche share of the batch's representational mass and none may
  monopolise it. We use the *true* class share rather than SwAV's uniform prior,
  which is known to degrade under long-tailed data.

**Why this is a stronger bet than the distillation family.** It is not
variance-reduction on a noisy target — the mechanism the In-Shop result went
1-for-1 against. It changes what the hypergraph *computes*.

**Strict generalisation, and why that matters.** The cost is defined so that
`exp(-C)` **is** HIST's incidence. Therefore at `iterations=0, epsilon=1.0` the
coupling degenerates to `H` and the loss equals `_hist_loss` exactly —
`test_sinkhorn_hist_reduces_to_hist_without_balancing` asserts it. Plain HIST is a
provable special case, the balancing is a single interpretable knob, and any
measured difference is attributable to the transport constraint alone. The
`hist_shot_uniform` arm ablates the marginal choice.

**Batch composition, and what it does to the marginal choice.** Official HIST sets
`--IPC default=0`, i.e. balanced sampling *off*, and our `samples_per_class: 0`
matches it (another fidelity check passed). But that has a consequence worth
recording, because it is also a real observation about HIST itself: CUB has 5,864
training images over 100 classes, so a random batch of 32 contains roughly 27
distinct classes with **one or two samples each**. Almost every hyperedge therefore
has a single true member.

Two implications:

1. HIST's "higher-order" structure on CUB is carried almost entirely by the *soft*
   memberships `exp(-alpha·d)`, not by multiple true members sharing a hyperedge.
   The hypergraph is doing soft label propagation more than it is doing genuine
   n-ary grouping.
2. With `n_e ∈ {1, 2}` the class-population marginal is already nearly uniform, so
   `hist_shot` and `hist_shot_uniform` will be close to each other. The biology
   framing (competitive exclusion via unequal niche shares) has much less to bite
   on here than it would on a dataset with real batch imbalance. That is an honest
   weakening of the motivation, and it further concentrates the bet on
   `hist_shot_geometric`.

Same preregistered gate as everything else: ≥ +0.5 R@1 over HIST on CUB seed 0 to
earn a multi-seed run, and ≥ 6 seeds before any claim. The queue applies this
automatically (`escalate_if_promising`), so a losing arm costs one run rather than
three.

### Phase 6 — The fallback, which should be prepared in parallel, not after

If the loss-innovation track fails, the honest and still-valuable contribution is
already 80% written:

* **(a) The provenance system.** Digest-verified, publication-backed recipes
  across 5 datasets, with track classification and manifest verification. Most DML
  papers ship nothing like it, and it is precisely what caught the recipe drift
  and the LayerNorm confound that invalidated the legacy headline.
* **(b) The negative-results catalogue.** ~16 loss-geometry changes that do not
  move the CUB plateau, plus this multi-seed refutation of a plausible
  distillation idea — and, if the pivot fails, the hypergraph candidates too.
  Failed experiments become evidence rather than waste.
* **(d) Train-clean compression.** 2560→2048 dims at 100% retrieval retention,
  fit only on disjoint train classes, with the correct explanation for *why*
  un-centered projection is required under cosine retrieval. Small, clean, and
  it would survive review.

This is a real paper: *"what does not move a same-architecture DML plateau, and
how to know your recipe is not lying to you"*. Do not treat it as a consolation
prize — on current evidence it is the higher-expected-value deliverable.

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
