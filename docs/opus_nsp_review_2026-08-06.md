I have what I need. Writing the review.

---

# Independent adversarial review — NSP (Null-Space Provisioning), pass 30

**Verdict: DEAD.** Earliest failing gate: **mathematical validity (§2, the proof-level degeneracy attack)** — before novelty, identifiability, executability, or forecasting. The single most decisive mechanism-level reason is stated in §4 below and is *internal*: the property NSP must have to defeat class-copying (Degeneracy 3) is exactly the property that makes its provisioned energy a subtractive term in the deployed cosine metric.

All numbers below are my own, computed on CPU in this session; scripts are in `/tmp/nsp_audit/`.

---

## 1. The rank-`C−1` premise

The proposal's §0/§2 claim: *"every loss of the form `L(f(x),y)` shapes the descriptor only inside the span of the class centroids"*, hence `d−C+1` directions get no identity-relevant gradient.

**This conflates a rank identity with an optimization constraint.** `rank(between-class scatter) ≤ C−1` is trivially true. "The optimizer can only move features inside a fixed `(C−1)`-dim subspace" does not follow. Case by case:

**Proxy losses.** `L_PA` depends on `z` only through `s(z,p_c)`, so `∇_z L ∈ span{p_1..p_C}` — the **proxy** span, not the **centroid** span. NSP builds `Q` from `μ` (an EMA, lagging ~100 steps, refreshed every 50), while PA trains proxies at **100× base LR**. Nothing couples them. Second, with L2 normalization `∇_{z̃}L = (I−zzᵀ)∇_z L/‖z̃‖`, and the `−(zᵀp_c)z` term is along `z`, which by NSP's own energy floor has 25% of its mass in `range(Q)`.

Measured (`g_paleak.py`, `d=512`, `C=100`, `r₉₉≈54`):

| `‖Qz‖²` (γ) | proxy off-span energy | `‖Q ∇L_PA‖² / ‖∇L_PA‖²` |
|---|---|---|
| 0.00 | 0% | 0.013 |
| 0.25 | 0% | **0.077** |
| 0.00 | 20% | **0.218** |
| 0.25 | 20% | **0.239** |

The supervised loss puts **8–24% of its gradient energy into the subspace NSP declares label-free** — and NSP's own energy floor *increases* the leak from 1.3% to 7.7%. The two subspaces are not decoupled; the mechanism claimed "in one line" (§1.4) is one-directional at best.

**Pair/tuple losses.** `span{z_i − z_j : y_i=y_j}` is the within-class scatter span, rank up to `min(n−C, d) = d`. The proposal's "contained in Φ in expectation" is a category error: `E[z_i−z_j | y_i=y_j] = 0`, and the loss is driven by realized second moments, not means. Even pure PA contracts within-class scatter outside Φ: for two same-class samples, `δz₁ − δz₂ ∝ (z₂z₂ᵀ − z₁z₁ᵀ)p_c ∈ span{z₁,z₂}`, not `span{p_c}`.

**Sample-specific gradient.** Each sample's gradient lies in a *sample-dependent* subspace `span{p_c} ⊕ span{z_i}`. Union over a batch: up to `min(C+N, d) = 280` dims on CUB; over an epoch, `d`.

**Nonlinear encoder.** `∇_θL = Jᵀ∇_{z̃}L`; the induced feature change at another input is `J(x')J(x)ᵀ∇_{z̃}L` — an NTK block, not a scalar. It maps `range(Q)` out of `range(Q)`. Verified in §3.

**Where the premise does hold.** In `c3_nullspace_gradient.py` (linear encoder, identity carried purely by first moments in a shared signal subspace) unseen-class R@1 rose 0.054 → 0.106 while unseen R@1 *inside* `range(Q)` stayed at chance (0.021 → 0.026, chance 0.024). So the error mode is **real in the first-moment-only regime** and the proposal deserves credit for it. It is not a general theorem, and real fine-grained data is not that regime.

---

## 2. The EMA-centroid projector

**`Q` is not a label-null subspace; it is a first-moment-null subspace.** Constructed two classes with identical means and different variance along `e₁` (`f_projector.py`, F1): `‖Qe₁‖ = 0.9997`, and a **one-parameter probe on `|(Qz)₁|` classifies seen classes at 0.923** (chance 0.500). Centroid-orthogonality guarantees `E[wᵀz|y]=0` and nothing else. Class-conditional variance/shape structure — which is exactly what fine-grained DML lives on — survives inside `range(Q)`. Every sentence in the proposal reading "label-null" is unearned.

**`r ≈ 97–99` on CUB is not attainable.** `r₉₉` is a 99%-*energy* rank of the empirical centroid scatter of `C=100` rows. Even for a perfectly isotropic latent, Marchenko–Pastur spread puts `r₉₉ = 77`; realistic decay gives 32–75 (F2). My trained toy gave `r₉₉ = 44` from `C=80` (algebraic max 79).

| centroid spectrum decay `a` | CUB `r₉₉` (`1−r/d`) | SOP `r₉₉` (`1−r/d`) |
|---|---|---|
| 0.0 | 77 (0.850) | 504 (0.016) |
| 1.0 | 68 (0.867) | 476 (0.070) |
| 1.5 | 53 (0.896) | 337 (0.342) |
| 2.0 | 32 (0.938) | 89 (0.826) |

Two consequences. (i) On CUB/Cars `1−r₉₉/d ≈ 0.85–0.94` for **every** spectrum — the "dose" is fixed by `C/d`, so the dose–response law (§5.2) is a one-point calibration with no independent predictive content on the datasets where it predicts a signal. (ii) On SOP the dose swings by a factor of 20 with an unmeasured spectral exponent, so "the method self-disables on SOP" is not derivable a priori. The proposal also states both `r → d, Q → 0` (§1.2) and `1−r₉₉/d ≈ 0.10–0.25` (§5.2) — i.e. `Q` has 51–128 dims. Those are inconsistent.

**Stale/unvisited rows and a hard NaN.** `μ` initialization is unspecified. If `μ=0`, "then row-normalized" is `0/0` with no ε — on SOP (`C=11318`, ~45 classes/step, first SVD at step 50) roughly **80% of rows are NaN at the first refresh**, propagating through the SVD into `Q` and into the total loss. This is a crash, not a subtlety. If instead `μ` is initialized nonzero: with `init=0`+20% visited, `r₉₉ = 456` vs 476 fully-populated; with random-unit init, 503 (F3). The rank effect is modest; the contamination of `V` is not — with `init=0` all unvisited rows centre to the *same* vector `−μ̄`, injecting one spurious direction and biasing `μ̄`.

**Row-normalize-then-centre.** Normalizing inside the EMA recursion gives a class seen once the same unit weight as a class seen 1000 times. On SOP (~5 images/class) every row is a unit-normalized single-batch mean of ~4 images — the table is an estimation-noise matrix, and its 99%-energy rank is inflated toward `min(C−1,d)`. The predicted `r → d` on SOP is plausibly an **estimator artifact**, not a property of the representation. That matters because "the gain must vanish on SOP" is one of the three pre-registered falsifiers.

**Ties and the moving subspace.** Rotations *inside* the retained block are harmless (`P = V_{:r}V_{:r}ᵀ` depends only on the subspace). The instability is at the boundary: with a continuum spectrum, `σ_r² ≈ σ_{r+1}²` is generic, so `r` flips across refreshes and whole directions migrate between `P` and `Q`. Over 200 epochs on CUB there are ~128 refreshes. Content deposited in `range(Q_t)` at time `t` sits inside `range(P_{t'})` at `t' > t`. **"Structurally incapable of rotating the discriminative subspace" is therefore false over the training trajectory even granting the instantaneous claim.**

**Detached-`Q` feedback.** `μ` is an EMA of `z`, and `z` now carries the provisioned content. Any non-exchangeable part of it is absorbed into `μ`, rotates `P` toward it, and is then evicted from `Q` at the next refresh — forcing perpetual re-provisioning of fresh directions. No stability analysis is offered.

---

## 3. The gradient claims

**Correct, and I confirm it:** `∇_{z̃}L_aux ∈ range(Q)` exactly. Measured `‖P∇_{z̃}L_aux‖²/‖∇_{z̃}L_aux‖² = 5.3×10⁻¹⁵`.

**The stated Jacobian is the ε=0 form.** For `u = v/(‖v‖+ε)`, `∂u/∂v = (1/(‖v‖+ε))[I − ûûᵀ·‖v‖/(‖v‖+ε)]`, not `(I−uuᵀ)/‖v‖`. Cosmetic: both `v` and `û` lie in `range(Q)`, so the `range(Q)` conclusion survives. At `v=0` the NT-Xent gradient vanishes (all logits equal), so no blow-up — but `Q` for steps 1–49, before the first SVD, is **unspecified**. If `Q=I` there, the first 50 steps *are* control C3 (full-space SimCLR) at the most plastic point of training. The `L_e` gradient is also misstated: `∇_{z̃}L_e = −(4/N)·1[e<γ_t](γ_t−e)Qz̃/sg(‖z̃‖²)`; the proposal drops a factor 2 and the `1/N`. Absorbed into `η`, but this is a document claiming "executable mathematics."

**Feature-space orthogonality does not survive the optimizer.** One step, then measure how much of the induced change in `z̃` on a *held-out* batch lands in `range(P)` (`d_leakage.py`; `range(P)` is 12/64 dims, isotropic baseline 0.1875):

| backbone | optimizer | energy of induced `Δz̃` in `range(P)` |
|---|---|---|
| frozen | SGD | 0.0000 |
| frozen | **AdamW** | **0.2117** |
| trained | SGD | 0.0084 |
| trained | **AdamW** | **0.0504** |

The claim holds **only** for a linear head with a frozen backbone under plain SGD. NSP trains the backbone under AdamW. Adam's diagonal preconditioner destroys the rank-1-with-left-factor-in-`range(Q)` structure of the head gradient — leakage *exceeds* the isotropic-random baseline. Moment estimates are also shared, so `L_aux` changes the preconditioner `L_PA` is stepped with. Add the shared nonlinear backbone (`W K_f Wᵀ Q g` is an arbitrary PSD map of `range(Q)`) and the "structural" guarantee is gone.

**Weight decay closes the loop the detached denominator was meant to open.** `sg(‖z̃‖²)` correctly prevents the *loss* from being minimized by shrinking `Pz̃` — that is good design and I credit it. But `∇L_e` is purely along `Qz̃`, so `L_e` monotonically inflates `‖z̃‖`; under AdamW's decoupled, isotropic weight decay the new equilibrium has a **smaller** `‖Pz̃‖` than base. The failure arrives through the optimizer instead of the loss.

---

## 4. Provisioning degeneracies — the decisive section

**Degeneracy 1 (Welch bound) is refuted.** `max_{i≠j}⟨u_i,u_j⟩ ≥ −1/(N−1)` bounds the *unattained global optimum*; it is not a lower bound on learned rank. Minimizing NT-Xent (`N=180`, `τ=0.2`, perfect alignment) over configurations constrained to rank `k` (`a_rank.py`):

| `k` | NT-Xent min | gap vs simplex |
|---|---|---|
| 32 | 1.3760 | 0.1683 |
| 48 | 1.2994 | 0.0917 |
| 64 | 1.2617 | 0.0540 |
| 96 | 1.2224 | 0.0147 |
| 179 (simplex) | 1.2077 | 0 |

At rank 64 the loss is within 4.5% of the global optimum; weighted by `β=0.3` the entire penalty for collapsing 179→64 is **0.016 total-loss units** against an `O(1–10)` `L_PA`. There is no barrier. Empirically, contrastive SSL *does* dimensionally collapse ([Jing et al., ICLR 2022](https://openreview.net/forum?id=YevsQ05DEN7)). "Provable lower bound on provisioned rank: 179" is false.

**Degeneracy 3 (`E[u|y]=0`) is not a corollary — and it is the kill.** Same-class negatives are pushed apart *exactly as hard* as different-class negatives; the loss is indifferent to which. NT-Xent's uniformity term acts on the **marginal** ([Wang & Isola, ICML 2020](https://arxiv.org/abs/2005.10242)), not on class conditionals. A configuration where classes occupy separated caps with instances spread inside each cap has a near-uniform marginal, near-optimal uniformity, and `E[u|y] ≠ 0` — and the *shared backbone under a strong class loss* actively produces that geometry. Treating same-class items as negatives is the textbook false-negative pathology ([Chuang et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)), not an exchangeability mechanism.

**Now the internal contradiction.** Deployed cosine decomposes as `cos(z_q,z_g) = (1−γ)cos_P + γ·cos_Q`. If the blind channel *is* class-exchangeable (what Degeneracy 3 needs), `cos_Q` has equal means for positives and negatives — it contributes zero identity signal and pure variance. Simulated at CUB-like difficulty (100 unseen classes × 58, correlated fine-grained prototypes, base R@1 = 0.706; `e4.py`):

| `a` = identity content of `Q` | `Q`-alone R@1 | γ=0.05 | γ=0.10 | γ=0.15 | **γ=0.25** | γ=0.40 |
|---|---|---|---|---|---|---|
| **0.00** (exchangeable) | 0.008 | 0.702 | 0.692 | 0.672 | **0.606** | 0.411 |
| 0.10 | 0.021 | 0.713 | 0.709 | 0.702 | 0.660 | 0.498 |
| 0.20 | 0.126 | 0.739 | 0.761 | 0.783 | 0.793 | 0.756 |
| 0.30 | 0.611 | 0.775 | 0.837 | 0.877 | 0.929 | 0.956 |

At NSP's stated `γ = 0.25`, a channel satisfying its own Degeneracy-3 requirement costs **−10.0 R@1 points** against a forecast of **+2.0**. Break-even requires the "blind" channel to reach a standalone unseen-class R@1 of roughly 0.13–0.61 *by itself* — i.e. to be a strong class-informative retriever, which is precisely the class-copying Degeneracy 3 claims the negative set forbids. **Degeneracy 3 and the +2.0 forecast are mutually exclusive.** This is the decisive mechanism-level reason.

The same contradiction appears in the pre-registered evidence: §4 expects *seen*-class probe from `u` flat and *unseen*-class probe from `u` up ≥3 pts. On CUB the train/test split is an arbitrary partition of bird species sharing the same attribute vocabulary; a direction informative for unseen species is almost certainly informative for seen ones. The two predictions are close to self-contradictory.

**Degeneracy 4 (shortcuts) is worse than stated.** The negative set is 179 in-batch images. Discriminating 180 random natural images is trivially solvable from colour/texture statistics — SimCLR needs batch 4096 and MoCo a 65k queue precisely because small negative sets make the task too easy. CUB has 5864 training images total. Worse, at 128 px after an RRC scale-0.2 crop, fine-grained bird detail is gone and background/habitat dominates. And NSP **deploys the object SimCLR discards**: [Chen et al., 2020](https://arxiv.org/abs/2002.05709) found the representation *before* the projection is >10% better downstream because the contrastive loss removes information. There is no projection head here; `u` is a linear readout of the deployed descriptor.

**Can the energy floor be met degenerately? Yes — by moving the split.** `r` is the 99%-*energy* rank, a soft rank the network controls. Making the centroid spectrum peakier lowers `r`, enlarges `range(Q)`, and raises `e` **for free with no change in content**. Peaky embedding spectra are the documented natural state — the exact pathology [ρ-spectrum regularization (Roth et al., ICML 2020)](https://proceedings.mlr.press/v119/roth20a.html) was invented to fight. NSP creates gradient pressure in the direction known to *hurt* generalization, and this degeneracy is not considered anywhere in §2. Head *scaling* alone does not work (the ratio is scale-invariant), which is a genuine point in the design's favour.

**Is the floor even active?** Unknown, and unmeasured. In my trained toy the natural `E‖Qz‖²` fell from 0.162 → 0.042 — a `γ=0.25` floor there would be a **6× forced reallocation of the deployed metric**, not a gentle guard. If on real CUB the base value sits above 0.25, C5 and C7 are null by inactivity and prove nothing. This single number is measurable from a base checkpoint in under an hour and should have been a precondition for the whole design.

---

## 5. Prior art — judged on supervision object and action

**[DiVA, ECCV 2020](https://arxiv.org/abs/2004.13458) occupies the object and the action.** From the paper: the class-discriminative embedding is *"specialized to features which help **only** in separating among training classes, and may not correctly translate to unseen test classes"* — NSP's §2 error mode, stated verbatim in 2020. DiVA's remedy: dedicated sub-embeddings `φ_disc, φ_shared, φ_intra, φ_nce`, where `φ_nce` is a **sample-specific instance-discrimination (NCE) task**, allocated 128 of the 512 dims, **concatenated into the deployed descriptor**. So DiVA already deploys ~25% of the descriptor carrying instance discrimination in a partition disjoint from the class task. NSP's differences are the *wrapper*: learned-vs-fixed partition, energy floor vs dimension quota. The proposal's own **C4 predicts the fixed-block version recovers 50% of the gain** — it concedes that published prior art delivers half the effect, leaving a claimed novel margin of +1.0 pt against ±0.5 pt seed noise.

Other occupied ground the proposal misses or misjudges:

- **[Sharing Matters (TPAMI 2020)](https://arxiv.org/abs/2004.05582)** — same error mode, *contradictory* remedy: the transferable characteristics are the ones **shared across training classes**, i.e. living inside the class-discriminative solution, not orthogonal to it. NSP's projector excludes exactly those directions. This is prior art *and* a mechanism objection.
- **[MIC (ICCV 2019)](https://arxiv.org/abs/1909.11574)** — auxiliary head explicitly to explain away intra-class variation. Same object.
- **[Islam et al., ICCV 2021](https://arxiv.org/abs/2103.13517)** — joint self-supervised contrastive + supervised loss **improves transferability** with no subspace machinery. This makes C3 a *published positive result*, not a null. NSP's C3 prediction (+0.3 to −0.5) contradicts it.
- **[GPM (ICLR 2021)](https://arxiv.org/abs/2103.09762)** and **[Adam-NSCL (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Training_Networks_in_Null_Space_of_Feature_Covariance_for_Continual_CVPR_2021_paper.html)** — SVD of representations → null space of task A → learn task B inside it. That is NSP's action, in parameter space. Far nearer than the cited PCGrad; not mentioned.
- **[Domain Separation Networks (NeurIPS 2016)](https://proceedings.neurips.cc/paper/6254-domain-separation-networks.pdf)** and **[Liu et al., ACL 2017](https://aclanthology.org/P17-1001/)** — shared/private subspaces of one encoder, orthogonality-constrained, different objective per subspace. NSP's "structural rather than penalised" is a real but narrow distinction.
- **Neural collapse** — [Papyan, Han & Donoho, PNAS 2020](https://www.pnas.org/doi/10.1073/pnas.2015509117) establishes the `C−1` simplex; [Galanti et al., ICLR 2022](https://arxiv.org/abs/2112.15121) and [Kornblith et al., NeurIPS 2021](https://arxiv.org/abs/2010.16402) establish that stronger class separation *hurts* transfer. Genuine support for NSP's error mode; the standard remedies regularize collapse rather than fill the complement, which is the one place NSP is doing something less common.

Fair distinctions I confirm: [S2SD](https://arxiv.org/abs/2009.08348) (no teacher here), BIER/A-BIER, [INLP](https://arxiv.org/abs/2004.07667) (remove vs populate), Barlow Twins/VICReg, Matryoshka, AdvProp-as-component.

The proposal's closing novelty claim — no prior work does (a) online centroid-spectrum subspace *and* (b) an energy floor on its complement — is true as a conjunction of two implementation details. Judged on supervision object and action, as it must be, **NSP is DiVA's instance-discrimination branch with a learned partition and an energy hinge.**

---

## 6. Reduction, controls, and the objective

**The PA reduction is not what it says.** PA's official ResNet-50/512 recipe is `batch 120, warm 5, bn-freeze 1, step LR decay 5/10`, reporting **CUB 69.9 / Cars 87.7**. The proposal uses batch 180, warm 1, cosine decay, 200 epochs — five deviations — and then forecasts its own "re-measurement" at **0.699 / 0.882**, i.e. exactly the published CUB number it claims not to inherit (§1.2, §5.5). The base row carries no independent information. Also: PA's ResNet-50 recipe **freezes BN**; the proposal never addresses this, and "separate auxiliary BN affine + running stats" is incoherent against a frozen-BN base.

**C3 does not isolate the mechanism.** It changes three things at once: the subspace, the aux task's effective energy share (100% vs NSP's 25%), and the absence of `L_e`. A matched C3 would equalize the energy share.

**C4 does not isolate "learned subspace."** `h_φ` is a free linear map; it can rotate the centroid span into dims 1–99 and make the fixed block functionally identical to NSP. If C4 loses 1.0 pt the cause is optimization dynamics, and no mechanism is offered. Whether C4 includes `L_e` is unstated — another confound.

**C7 is uninterpretable without the base `e`.** As is C5. Both are trivially ≈0 if the floor never binds.

**C6 is the one informative control, and its predictions are self-defeating:** if removing same-class negatives yields +1.0 *and* raises the seen-class probe, then class information in `Q` helps retrieval, Degeneracy 3 collapses as a defense, and NSP reduces to a supervised-contrastive variant.

**Compute matching 200 → 266 epochs is not valid as a null.** It gives C2 33% more optimizer steps and a stretched cosine schedule (a different intervention), 266 epochs on CUB is ~4.4× PA's published schedule and deep in the overfitting regime where "≤+0.2" is guaranteed and uninformative, and the true wall-clock ratio is ~1.4–1.6× (below), so a wall-clock match is ~280–320 epochs.

**Statistics.** SOP is forecast at +0.3 with ±0.3 s.d. per arm — the "law" itself spans +0.24 to +0.60, and the falsifier is ">0.5 pt above the law." None of these are separable at 5 seeds. "The gain must vanish on SOP and In-Shop" is not a testable falsifier as specified. In-Shop is absent from the §5.1 table entirely.

**The objective is not met on the proposal's own arithmetic.** §5.3 forecasts CUB 0.729 vs frontier 0.734 and Cars 0.908 vs 0.927 (−1.9 pt, far outside ±0.006). The only crossing path is conditional on a faithful PFML reproduction whose loss form, expansion, pooling, LR/epoch schedule, batch size, augmentation, and even whether 0.734 is GAP or concat-pooled are **all admitted unknown** (§5.5), with the proposal's own `P(cross) ≈ 0.25–0.30`. A method that forecasts below the frontier and can only reach it by stacking on an unreproducible base does not satisfy a frontier objective.

---

## 7. Cost, legality, leakage

Measured on the real ResNet-50 graph (`h_cost.py`, retained activation tensors, fp32, batch 180):

| quantity | measured | proposal |
|---|---|---|
| activations @224 | 21.48 GiB | — |
| activations @128 | 7.02 GiB (ratio 0.3266) | — |
| FLOP ratio (fwd+bwd) | **1.3265×** | 1.33× ✓ |
| **peak memory ratio** | **1.322×** (28.90 vs 21.86 GiB) | **1.20× ✗** |
| `C×512` EMA table (SOP) | 22.1 MiB | 23 MB ✓ |
| aux BN params | ~53 K | ✓ |

Peak memory is understated by ~10 points because both branches' graphs are alive through a single `backward()` on the summed loss, and activations are ~98% of the footprint. 28.9 GiB fp32 does not fit a 24 GB card. Thin SVD (`11318×512` ≈ 3 GFLOP, ~6.6×/epoch on SOP) is genuinely negligible — that claim is fine. All-pairs NT-Xent (`360²`) is negligible.

Wall-clock will exceed 1.33×: the 128 px branch has poor arithmetic intensity in stages 4–5, and the second augmentation pipeline (colour jitter + `GaussianBlur` at p=0.5) is CPU-bound on 180-image batches. Estimate 1.4–1.6×.

Two-resolution training with only BN separation is not sufficient: AdvProp's aux-BN was designed for same-resolution adversarial inputs, whereas a 224→128 change shifts the whole activation-statistics regime ([Touvron et al., FixRes, NeurIPS 2019](https://arxiv.org/abs/1906.06423)). The shared conv weights still take gradient from both distributions.

**Legality / leakage.** Deployment is clean (single 224 view, no probe, no aux BN, no reranking, cosine NN) — Lane A compliant. Two real gates:

1. **No validation split is specified.** The sensitivity grid is `β(3) × γ(3) × τ(2) × K_P(3) = 54` configurations plus 8 controls at 5 seeds. CUB/Cars/SOP have no standard val split, and test-set tuning is the confound [Roth et al., ICML 2020](https://proceedings.mlr.press/v119/roth20a.html) built their protocol to remove. With ±0.5 pt seed noise and a +2.0 target, a 54-point test-set grid is a leakage channel of the same magnitude as the claimed effect. Fix: carve held-out validation *classes* out of train and select there.
2. **The unseen-class probe is test-label-supervised and is elevated to a falsification gate** (§5.4). As a post-hoc diagnostic it is fine; as a go/no-go criterion it is test-conditioned selection. Move it to held-out training classes. ("Fraction of test NN decisions changed by zeroing `Q`" is a pure diagnostic — fine.) Also, `u` is undefined for C3 (no projector), so the "+3 pts vs C3" comparison is unspecified.

The ImageNet↔CUB/Cars overlap is acknowledged, and the random-init / SSL-init contamination probe is good practice.

---

## 8. Verdict

**DEAD.**

**Earliest protocol gate:** mathematical validity — §2's proof-level degeneracy attack. All three load-bearing arguments fail before novelty, identifiability, executability, or forecasting are reached: the Welch bound is a bound on an unattained optimum and gives no rank floor (rank-64 is within 4.5% of the simplex optimum); `E[u|y]=0` is not a corollary of the negative set (NT-Xent's uniformity acts on the marginal); and `range(Q)` is a first-moment-null, not a label-null, subspace (0.923 seen-class probe accuracy inside it).

**Single most decisive mechanism-level reason:** NSP's two central claims are mutually exclusive. If the blind channel is class-exchangeable — the property Degeneracy 3 requires to defeat class copying — then at `γ = 0.25` it contributes zero identity signal and pure variance to the deployed cosine, costing **−10.0 R@1 points** on a CUB-calibrated retrieval simulation against a forecast of +2.0. The only regime in which the provisioned energy pays for itself is one where the blind channel is a strong class-informative retriever on its own — i.e. exactly the class-copying degeneracy the proposal claims is structurally forbidden. No implementation change reconciles these; they are properties of the objective and of the energy floor, not of the wrapper.

**Not the primary reason, but independently sufficient:** the supervision object and action (allocate a disjoint region of the deployed descriptor to instance discrimination, to recover unseen-class-relevant structure that class-discriminative training discards) is [DiVA](https://arxiv.org/abs/2004.13458), which states NSP's error mode verbatim and deploys 128/512 dims of sample-specific NCE. The proposal's own C4 concedes half the effect to the fixed-block version.

### Valid subcomponents, preserved separately from the verdict

These stand on their own and are worth keeping:

1. **`∇_{z̃}L_aux ∈ range(Q)` is exactly correct** (verified to 5×10⁻¹⁵), including under the ε-normalization despite the stated Jacobian being the ε=0 form. As a *feature-space* construction this is sound.
2. **The detached denominator in `e = ‖v‖²/sg(‖z̃‖²)` is correct and well-motivated** — it does prevent the floor being satisfied by shrinking `Pz̃` at the loss level. (It does not prevent the same outcome arising through AdamW's isotropic weight decay.)
3. **`r₉₉/d` is a genuinely useful, cheap diagnostic** and pre-registering a dose–response relation on it is good methodology, even though the specific law has no predictive content on CUB/Cars.
4. **The error mode is real in the first-moment-only regime** — my controlled experiment reproduced it (unseen R@1 rose 0.054→0.106 in `range(P)` while staying at chance in `range(Q)`).
5. **The FLOP arithmetic (1.33×), the `C×512` table size (23 MB), and the SVD cost estimate are all correct.**
6. **C8 is a well-designed inertness null**, and the honesty of §5.3/§5.5 (explicit non-crossing, explicit source ambiguity), the 5-seed paired-delta plan, and the random-init contamination probe are above field norm.

### If someone wants to try to resurrect a variant

One measurement, no GPU-days required (~1 GPU-hour from an existing base checkpoint), decides whether any version of this is worth running: **the base model's natural `E‖Qz‖²` and the standalone unseen-class R@1 of `Qz` alone.** From the table in §4, a variant is only viable if `Qz` on its own already achieves unseen-class R@1 ≳ 0.15 at the intended `γ`. If it does not, the energy floor is subtractive at every setting and the design is unrecoverable. Note that this measurement would also convert C5/C7 from uninterpretable to decisive.

**Sources:** [DiVA](https://arxiv.org/abs/2004.13458) · [Sharing Matters](https://arxiv.org/abs/2004.05582) · [MIC](https://arxiv.org/abs/1909.11574) · [S2SD](https://arxiv.org/abs/2009.08348) · [Roth et al., Revisiting DML](https://proceedings.mlr.press/v119/roth20a.html) · [Proxy-Anchor (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [Proxy-Anchor repo](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020) · [SimCLR](https://arxiv.org/abs/2002.05709) · [Wang & Isola](https://arxiv.org/abs/2005.10242) · [Jing et al., dimensional collapse](https://openreview.net/forum?id=YevsQ05DEN7) · [Chuang et al., debiased contrastive](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) · [Islam et al.](https://arxiv.org/abs/2103.13517) · [GPM](https://arxiv.org/abs/2103.09762) · [Adam-NSCL](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Training_Networks_in_Null_Space_of_Feature_Covariance_for_Continual_CVPR_2021_paper.html) · [Domain Separation Networks](https://proceedings.neurips.cc/paper/6254-domain-separation-networks.pdf) · [Liu et al., adversarial MTL](https://aclanthology.org/P17-1001/) · [Papyan, Han & Donoho](https://www.pnas.org/doi/10.1073/pnas.2015509117) · [Galanti et al.](https://arxiv.org/abs/2112.15121) · [Kornblith et al.](https://arxiv.org/abs/2010.16402) · [INLP](https://arxiv.org/abs/2004.07667) · [FixRes](https://arxiv.org/abs/1906.06423)

**Uncertainty I want on record.** My retrieval-dilution and rank experiments are synthetic and calibrated to CUB-like difficulty, not run on real CUB embeddings; the magnitudes (−10.0 pts at γ=0.25, break-even at `a≈0.2–0.3`) are model-dependent, though the *sign* follows directly from `cos = (1−γ)cos_P + γ·cos_Q` with `E[cos_Q|pos] = E[cos_Q|neg]` and does not depend on calibration. The `r₉₉` figures assume power-law centroid spectra; real values must be measured. I did not verify PFML's existence or numbers, and I did not inspect the repository's ledger, prior reviews, or any failure catalogue, as instructed. One asymmetry worth flagging for whoever runs this anyway: background colonization of `Q` on CUB would produce *positive* same-class correlation (same species, same habitat), so a positive CUB result would be consistent with the shortcut rather than the claimed mechanism — the pre-registered background probe is therefore load-bearing, not optional.
