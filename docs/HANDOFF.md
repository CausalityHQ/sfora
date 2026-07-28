# SFORA handoff — state as of 2026-07-28

Everything needed to pick this up on a fresh machine. Written because the laptop is
being reset; nothing here depends on local state except an SSH key for the DGX.

---

## 1. Where things live

| what | where |
|---|---|
| branch | `research-reset-2026-07-27` (64 commits ahead of `main`) |
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
scp scripts/run_priority_queue_v16.sh riomus@100.104.199.68:/home/riomus/group-learning/scripts/
```

---

## 2. What is running right now

Queue `scripts/run_priority_queue_v16.sh`, launched detached on the DGX:

```bash
ssh riomus@100.104.199.68 'cd /home/riomus/group-learning && \
  setsid nohup bash scripts/run_priority_queue_v16.sh > /home/riomus/experiment-logs/q.log 2>&1 &'
```

Order: In-Shop `pa_distill_bnfix` ×3 → CUB `shepard` ×3 + `shepard_l1` → CUB
`tversky` ×3 → remaining CUB baselines.

**Status check** (this is the whole loop):

```bash
ssh riomus@100.104.199.68 'grep -E "DONE|FAIL" /home/riomus/experiment-logs/reference-matrix/controller.log | tail -5'
ssh riomus@100.104.199.68 'cd ~/group-learning && .venv/bin/python scripts/analyze_reference_matrix.py'
```

Run times on the GB10: CUB ≈ 35–50 min/arm, In-Shop ≈ 2.2 h (Proxy Anchor) to 4.6 h
(HIST base).

---

## 3. The one strong result: H3, the BatchNorm teacher/student mismatch

**This is the finding worth publishing, and it rescues HERD rather than burying it.**

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

Validated three ways: mechanism proved at unit level; the *null* prediction (inert
under frozen BatchNorm) proved by test; the *positive* prediction confirmed across
three seeds. Fix lives behind `ema_teacher_train_mode` / `ema_teacher_ema_buffers`,
both defaulting to the historical behaviour so old artifacts still reproduce.

Generalisable claim: *momentum-teacher recipes silently break under trainable
BatchNorm unless the teacher's normalisation mode matches the student's.* Applies to
any MoCo/BYOL/DINO-style teacher on a backbone with updating BatchNorm.

---

## 4. The measurement result

Nearly as important, and it explains why so much else failed.

- CUB seed noise σ ≈ 0.88 pt **and a 1.08 pt spread at fixed seed** (three
  effectively-identical runs: 0.7183 / 0.7154 / 0.7075). More than half the "seed
  variance" was GPU nondeterminism.
- Seeds needed for 80% power at α=0.05: **+0.5 pt → 12–37 seeds**; +1.0 pt → 3–10;
  +2.0 pt → 1–3. So **CUB cannot resolve the ~1 pt improvements the literature
  routinely reports from single runs.** A quantitative version of Musgrave's
  *Metric Learning Reality Check*.
- **In-Shop σ = 0.12 pt.** All method questions should be settled there, not on CUB.
- `deterministic: bool` (default False) now removes the nondeterminism entirely —
  `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`.
  **Turn this on for all new experiments.**

Also corrected along the way: the legacy "HERD beats HIST" headline was a **LayerNorm
confound** (the control ran without embedding LayerNorm). Properly paired, HERD and
HIST are indistinguishable on CUB (+0.03, p=0.93), and HIST's 3-seed mean is 0.7107 —
essentially exactly its published 0.714.

---

## 5. Live candidates (running now)

Both replace the **similarity function** rather than adding a loss term, and both
differ from official Proxy Anchor only in their declared recipe delta.

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

Honest priors from the literature sweep: ~20% (Shepard), 15–25% (Tversky).

---

## 6. What is dead — do not retry

Eleven candidates, each with a mechanism recorded in `docs/results.md`:

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
| `scripts/run_priority_queue_v16.sh` | current queue |

---

## 9. Recommended next steps

1. **Let the queue finish.** `shepard` and `tversky` at 3 seeds each is the immediate
   question. Given CUB's noise, only a ≥2 pt effect is interpretable at n=3.
2. **Move method validation to In-Shop.** σ = 0.12 pt makes a single seed decisive.
   Both new recipes need In-Shop variants (the HIST In-Shop recipe is a frozen
   `selected_extension` via `reports/generated/recipe_selection.inshop.hist.json`).
3. **Turn on `deterministic: true`** for everything new — it makes paired comparisons
   exact and costs nothing.
4. **Write up H3 regardless of what the new methods do.** It is a confirmed ~20σ
   bug-finding with a generalisable claim, and it is currently the strongest result
   in the project.
5. If both new candidates fail, the defensible paper is the reproducibility one:
   provenance infrastructure that caught two real confounds (LayerNorm, BatchNorm),
   the quantified noise/headroom argument, and an eleven-entry negatives catalogue
   with mechanisms.
