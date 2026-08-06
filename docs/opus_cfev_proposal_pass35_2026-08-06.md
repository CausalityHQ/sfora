# Frozen blind proposal — CFEV: Cross-Fitted Extreme-Value Retrieval Risk

**Provider:** claude-opus (direct proposer fallback; Fable credits exhausted; no consultation dispatched)
**Frozen before consultation start:** 2026-08-06 UTC
**Role:** constructive proposer, not adjudicator.
**Verdict:** ONE method (not NONE).
**Lane:** **Lane A only** (ResNet-50, 512-D normalized descriptor, ~224 px, single-view cosine
retrieval, 200 epochs). Lane B numbers (AdvRF, VAPNet) and CRT's MiT-B2 row are **not used** for any
forecast, comparison, or frontier arithmetic anywhere below.

---

## 0. One-paragraph statement of the method

Every loss in the deep-metric-learning literature — contrastive, triplet, N-pair, multi-similarity,
circle, proxy-NCA, ProxyAnchor, PFML — is a functional of *individual pairwise similarities observed
inside the minibatch*. But zero-shot Recall@1 is not a functional of individual pairs: it is the
probability that the **maximum** of ~10³–10⁵ negative similarities exceeds the positive similarity,
i.e. an **extreme order statistic against a gallery two to three orders of magnitude larger than any
minibatch**. CFEV trains directly on that quantity by (i) fitting a **Generalized Pareto (peaks-over-
threshold) tail** to the negative-similarity population inside the batch, (ii) using the fitted tail to
**extrapolate the exceedance probability out to gallery size M = |D_train|**, far beyond the observed
support, and (iii) **cross-fitting across an identity-disjoint split of the batch classes** — the tail
law is estimated on class-half A and applied to anchors in class-half B — so that the objective can
only be reduced by making the negative-similarity *law* transferable across an identity boundary, not
by memorizing pair-specific separations. The gradient reaches the network through the differentiable
probability-weighted-moment estimator of the Pareto shape ξ and scale σ, through the smooth exceedance
rate ζ, and through the hardest positive similarity. Nothing is added at deployment: a single
ResNet-50, one view, one 512-D L2-normalized descriptor, plain cosine nearest neighbour.

The mechanism makes a **quantitative, falsifiable cross-dataset prediction that no competing account
makes**: the gain must scale with log(M/m), the log-ratio of gallery size to batch size, so it must be
largest on SOP, then In-Shop, then Cars, then CUB — and it must largely vanish in a matched-compute
control where M is set to the batch size.

---

## 1. Executable mathematics

### 1.1 Objects, shapes, and the deployed model

| symbol | object | shape | learned? | gradient? |
|---|---|---|---|---|
| `φ_θ` | ResNet-50 trunk, ImageNet-1K init, global average pool | image → ℝ²⁰⁴⁸ | yes | yes |
| `W` | embedding layer, no bias | ℝ^{512×2048} | yes | yes |
| `z_i` | `e_i = W φ_θ(x_i)`, `z_i = e_i/‖e_i‖₂` | S⁵¹¹ | — | yes |
| `p_c` | one ProxyAnchor proxy per training class, `p̂_c = p_c/‖p_c‖₂` | ℝ⁵¹², c=1..C | yes | yes |
| `ū` | EMA peaks-over-threshold cut-point | scalar buffer | no (buffer) | **stop-grad** |
| `ξ̂_A, σ̂_A, ζ̂_A` | GPD shape / scale / exceedance rate on half A | scalars | no (computed) | **yes** (into `z`) |
| `M` | extrapolation gallery size, set to `|D_train|` | scalar constant | no | no |

**Deployed at test time:** `z = W φ_θ(x)/‖·‖₂` only. Proxies, `ū`, `ξ̂`, `σ̂`, `ζ̂`, the class split, and
the whole tail machinery are **discarded**. One 224×224 centre crop, one forward pass, one 512-D
vector, cosine nearest neighbour. Zero extra deployed parameters, zero extra deployed FLOPs.

### 1.2 Base loss — ProxyAnchor, reproduced exactly as published

CFEV is an additive term on top of ProxyAnchor (Kim, Kim, Cho, Kwak, CVPR 2020), which I use because
its mathematical reduction and recipe are disclosed in a primary source I can reproduce. With batch
`X`, `P⁺` the set of classes present in the batch, `X_c⁺ = {i : y_i = c}`, `X_c⁻ = {i : y_i ≠ c}`:

```
L_PA = (1/|P⁺|) Σ_{c∈P⁺} log(1 + Σ_{i∈X_c⁺} exp(−α (z_iᵀp̂_c − δ)))
     + (1/|P|)  Σ_{c∈P}  log(1 + Σ_{i∈X_c⁻} exp(+α (z_iᵀp̂_c + δ)))
```

with **α = 32, δ = 0.1**, one proxy per class, `|P| = C` (the sum in the second term runs over **all**
proxies and is normalized by the total proxy count, not by `|P⁺|` — this is PA as published, and it
makes the second term's scale explicitly dataset-dependent through C; see §1.8, this is *not* treated
as harmless).

**PA primary-source recipe, reproduced:** AdamW; base lr 1e-4; **proxy lr ×100 (= 1e-2)**; weight decay
1e-4; batch 180 sampled as 45 classes × 4 images (balanced sampler); embedding 512-D; RandomResizedCrop
to 224 + RandomHorizontalFlip at train, Resize 256 + CenterCrop 224 at test; embedding layer warm-up
with the trunk frozen for the first epoch.

**Stated departures and ambiguities (§7 lists all):** the lane mandates **200 epochs**, which is longer
than PA's own reported budget; I therefore add a **cosine lr decay to 0 over 200 epochs** and apply the
*identical* schedule to every baseline, control, and CFEV run, so all comparisons are internally
matched. Proxy init: `kaiming_normal_(fan_out)`. Weight decay is applied to trunk and `W` only —
**not** to proxies and **not** to BN affine parameters. BN running statistics are **frozen** at their
ImageNet values from epoch 1 (a standard DML reproduction choice); identical in all arms.

### 1.3 The identity-disjoint split (cross-fitting)

At **every** optimizer step, draw a uniformly random partition of the 45 batch classes into
`A` (23 classes, 92 images) and `B` (22 classes, 88 images). Resampled independently each step, so over
an epoch every class occupies both roles. This is the sample-splitting device of double/debiased machine
learning (Chernozhukov et al. 2018) transplanted to *identities*: the nuisance object (the tail law)
is estimated on classes that are disjoint from the classes on which the risk is evaluated.

### 1.4 Negative-similarity pool and the peaks-over-threshold cut

```
S_ij = z_iᵀ z_j
N_A  = { S_ij : i,j ∈ A, i<j, y_i ≠ y_j }          |N_A| ≈ 4048  (92·91/2 − 23·6 = 4186 − 138)
N_B  = { S_ij : i,j ∈ B, i<j, y_i ≠ y_j }          |N_B| ≈ 3696
```

Threshold: a **global EMA quantile**, not a per-batch quantile, so the network cannot game the cut by
reshaping one batch:

```
q_t   = Quantile_{1−p₀}( N_A ∪ N_B )            p₀ = 0.10          (no gradient)
ū_t   = (1−ρ) ū_{t−1} + ρ q_t                   ρ = 0.05, ū_0 = q_0
u     = stopgrad(ū_t)
```

**Smooth exceedance rate** (this is where gradient reaches *sub-threshold* negatives):

```
ζ̂_A = (1/|N_A|) Σ_{s∈N_A} sigmoid((s − u)/κ)          κ = 0.01
```

**Exceedance set** (hard selection, values keep gradients — selection is piecewise-constant, so this is
differentiable a.e.):

```
Y_A = sort_asc({ s − u : s ∈ N_A, s > u })            k_A = |Y_A| ≈ 405
```
If `k_A < 64`, the CFEV term is skipped for that half this step (guards the early-training regime).

### 1.5 Differentiable GPD fit — probability-weighted moments (Hosking & Wallis 1987)

With ascending order statistics `y_(1) ≤ … ≤ y_(k)`:

```
â₀ = (1/k) Σ_{j=1..k} y_(j)
â₁ = (1/k) Σ_{j=1..k} [(k−j)/(k−1)] · y_(j)
D  = â₀ − 2â₁                            (clamped: D ← sign(D)·max(|D|, 1e−6))
ξ_raw = 2 − â₀/D
σ_raw = 2 â₀ â₁ / D
ξ̂ = 0.45 · tanh(ξ_raw / 0.45)            (soft clamp to (−0.45, 0.45))
σ̂ = softplus(σ_raw) + 1e−4
```

*Correctness check.* For an exact exponential tail with mean σ (the ξ=0 case), `a_r = E[X(1−F)^r] =
σ/(r+1)²`, so `a₀ = σ`, `a₁ = σ/4`, `D = σ/2`, giving `ξ_raw = 2 − 2 = 0` ✓ and `σ_raw = 2σ(σ/4)/(σ/2)
= σ` ✓. The estimator is closed-form, requires no inner optimization, and is differentiable through
both the values and the (piecewise-constant) sort permutation. PWM is chosen over MLE precisely because
MLE for the GPD requires an inner Newton loop whose gradient path is fragile and whose fixed-point is
not guaranteed for ξ < −1/2.

### 1.6 The extrapolated Recall@1 risk

For anchor `i ∈ B`, the **hardest-positive** similarity — R@1 succeeds iff the *best* positive beats
*every* negative:

```
s⁺_i = max_{j ∈ B, j≠i, y_j = y_i} S_ij
w_i  = (s⁺_i − u) / σ̂_A
```

Smooth, everywhere-finite GPD survival, extended below the threshold by a softplus so that the
objective is defined and differentiable for anchors whose positive is not yet above the cut:

```
w̃_i = κ_w · softplus(w_i / κ_w)                     κ_w = 0.05   (w̃ ≥ 0, w̃ ≈ w for w ≫ 0)
ℓ_i  = −(1/ξ̂_A) · log(1 + ξ̂_A w̃_i)                  (|ξ̂| < 1e−3 ⇒ use the limit ℓ_i = −w̃_i)
q_i  = ζ̂_A · exp(ℓ_i)                               ≙ P̂(a random gallery negative beats s⁺_i)
```

**Per-anchor extrapolated R@1 error against a gallery of M negatives**, under the working assumption
of approximately independent gallery negatives:

```
R̂_i = 1 − (1 − q_i)^M  ≈  M q_i          (q_i ≪ 1 in the operating regime)
```

**CFEV loss** (log1p wrapper: ≈ `M q_i` — i.e. the actual R@1 error — when the risk is small, and
logarithmic, hence bounded-gradient, when it is large):

```
L_{B|A} = (1/|B|) Σ_{i∈B} log(1 + M · q_i)
L_CFEV  = ½ ( L_{B|A} + L_{A|B} )
```

`L_{A|B}` is the mirror term: tail fitted on `N_B`, anchors taken from `A`.

**Tail-transfer agreement penalty** (encodes the exchangeability claim of §2 directly):

```
L_agree = (ξ̂_A − ξ̂_B)² + (log σ̂_A − log σ̂_B)²
```

### 1.7 Total objective, gradient paths, and schedule

```
L = L_PA + λ_t · ( L_CFEV + λ_g · L_agree )        λ_g = 0.05
```

**Gradient paths — all four are load-bearing and each is separately ablated in §4:**

1. `∂L/∂s⁺_i` — pushes the hardest positive up. Saturates automatically once `M q_i ≪ 1` (unlike PA,
   which keeps pushing); this is a feature, see §2.3 (D4).
2. `∂L/∂σ̂_A` and `∂L/∂ξ̂_A` → through `â₀, â₁` → to **every exceedance** `s ∈ N_A` above `u`, with PWM
   weight `(1/k)[1 − 2(k−j)/(k−1)]` for `∂/∂â₀ − 2∂/∂â₁` combinations. The weight is **O(1/k)** per
   pair and is *signed*: raising a mid-tail value and lowering the extreme tip both reduce ξ̂.
   This is the mechanistic opposite of hard-negative mining, which puts O(1) gradient on one pair.
3. `∂L/∂ζ̂_A` → `sigmoid'((s−u)/κ)/κ` on negatives **near** the threshold — a soft, localized push of
   the bulk-tail boundary downward.
4. `∂L_agree/∂(ξ̂,σ̂)` on both halves — penalizes class-subset heterogeneity of the similarity law.

**Schedule.** `λ_t = 0` for epochs 1–5 (PA warm-up regime; the GPD fit is meaningless before the
similarity distribution has organized), then linear ramp `λ_t = λ* · (t−5)/20` over epochs 6–25, then
constant `λ*` for epochs 26–200. `ū` is updated from epoch 1 so it is converged when the ramp starts.

**λ\* is not tuned per dataset.** It is set by a disclosed **gradient-norm matching rule** with a single
global constant `r = 0.5`: at epoch 25, on 20 held-out training batches, compute
`λ* = r · ‖∂L_PA/∂Z‖_F / ‖∂(L_CFEV + λ_g L_agree)/∂Z‖_F` (Z = the 180×512 batch embedding matrix), then
freeze `λ*` for the remaining 175 epochs. One constant, four datasets, no per-dataset search.

### 1.8 The scale question, stated explicitly (not assumed harmless)

`L_PA`'s second term is normalized by `|P| = C`, so its magnitude — and therefore the *ratio* of loss
gradient to AdamW's decoupled weight decay — differs by roughly `log C` across CUB (C=100), Cars
(C=98), In-Shop (C=3997) and SOP (C=11318). Adding `λ_t L_CFEV` changes the total gradient magnitude
and hence the effective decay-to-signal ratio on the trunk. Two consequences are handled, not waved:

- The **gradient-norm matching rule** of §1.7 fixes the *relative* scale to `r = 0.5` on every dataset,
  so the CFEV weight is not a covert per-dataset tuning knob.
- Control **C1** (§4) runs `λ_t = 0` with `L_PA` multiplied by `(1 + r)` so that the total gradient
  norm matches the CFEV arm. Any gain that C1 reproduces is a **scale effect, not a mechanism effect**,
  and must be subtracted from the claimed effect. I commit to reporting `CFEV − C1`, not `CFEV − C0`,
  as the headline mechanism estimate.

### 1.9 Complete hyperparameter table (frozen)

| name | value | role |
|---|---|---|
| α, δ | 32, 0.1 | PA, as published |
| batch | 180 = 45 cls × 4 | PA, as published |
| lr / proxy lr / wd | 1e-4 / 1e-2 / 1e-4 | PA, as published |
| epochs / schedule | 200 / cosine→0 | lane mandate |
| split | 23 / 22 classes, resampled every step | cross-fitting |
| p₀ | 0.10 | POT tail fraction |
| ρ | 0.05 | EMA rate for ū |
| κ | 0.01 | exceedance-rate temperature |
| κ_w | 0.05 | sub-threshold softplus width |
| ξ clamp | ±0.45 | GPD stability |
| k_min | 64 | fit-validity guard |
| **M** | **\|D_train\|** = 5864 / 8054 / 25882 / 59551 | extrapolation gallery size |
| r | 0.5 | global grad-norm ratio → λ* |
| λ_g | 0.05 | tail-agreement weight |
| ramp | 0 (ep 1–5), linear (6–25), flat (26–200) | schedule |

**M uses no test-set information.** It is the cardinality of the official *training* image set. This is
deliberate: setting M from the test gallery size would be a (small but real) contamination channel, and
I refuse it. The training-set size happens to be the same order as each benchmark's gallery, which is a
property of the benchmarks, not knowledge I inject.

---

## 2. The causal zero-shot error mode, and a proof-level attack on the degeneracies

### 2.1 The error mode: **sample-max/gallery-max gap under pair memorization**

Zero-shot R@1 for a query of an unseen class fails exactly when
`max_{j ∈ Gallery⁻} s(q, j) > max_{j ∈ Gallery⁺} s(q, j)`, a comparison against
`M ≈ 6·10³–6·10⁴` negatives. Every batch-based DML loss instead controls
`max_{j ∈ batch⁻} s`, a maximum over `m⁻ ≈ 176` negatives, **all of seen classes**.

Two distinct failures compound:

1. **Order-statistic gap.** If negative similarities have survival `1−F`, the expected sample max over
   n draws sits near the `(1−1/n)` quantile. Training controls the `1−1/176` quantile; deployment is
   scored at the `1−1/59551` quantile on SOP. Under an exponential-type tail with rate `1/σ`, the gap
   between those two quantiles is `σ · log(59551/176) = σ · 5.83`. A loss that is *blind* beyond its
   batch's own extreme has **no gradient at all** on the region of the distribution that decides
   SOP R@1. It can drive the batch-max down to any target while the far tail is unconstrained.
2. **Pair memorization makes (1) unfixable by bigger batches alone.** The seen-class objective is
   satisfiable by `O(C²)` pair-specific separations that the network can encode class-conditionally
   (it can identify the seen class first — cheap, since seen classes are memorizable — then apply a
   class-specific correction). This solution family has capacity `Θ(C²)` while the objective imposes
   only `Θ(C²)` constraints, so it is *exactly* satisfiable and carries **zero information about the
   similarity law of a class never seen**. Hard-negative mining, which concentrates all gradient on
   the single hardest observed pair, is the most efficient possible driver *into* this family: each
   gradient step services one pair.

CFEV attacks both. Against (1), the GPD is the *asymptotically correct* parametric family for
threshold exceedances (Pickands 1975; Balkema–de Haan 1974), so fitting it to the top 10% of a batch's
negatives and evaluating the survival at the positive gives an estimate of the far-tail exceedance
probability that is **valid outside the observed support**. Against (2), the objective is a functional
of only four scalars per half — `(ξ̂, σ̂, ζ̂, u)` plus the positives — with `O(1/k)` gradient on any single
pair, and the four scalars are estimated on classes **disjoint** from the classes being scored.

### 2.2 Why the fitted quantity transfers to unseen identities — the exchangeability argument

Model the benchmark's identities as i.i.d. draws `c ~ 𝒞` from a class population, with the class-
conditional embedding law `μ_c = Law(z | y = c)` a measurable function of `c`. Then the negative-
similarity law
`F(s) = P_{c≠c', z∼μ_c, z'∼μ_{c'}}( zᵀz' ≤ s )`
is a **population functional of 𝒞 and of the network**, and its empirical counterpart from a sample of
C classes is a **U-statistic of order 2 over classes**. Standard U-statistic concentration gives
deviation `O_P(C^{-1/2})` (driven by the *first* Hájek projection, i.e. by the number of **classes**,
not pairs). Split-half train/test identities are exchangeable draws from the same 𝒞, so `F` — hence
`(ξ, σ, ζ)` — is a train-estimable, test-valid object. By contrast, the pair-specific separations
`{s(c,c')}` are *not* a low-dimensional population functional: their empirical values carry no
constraint on an unobserved pair. **This is the precise sense in which CFEV optimizes something that
generalizes across the identity boundary and hard-negative losses do not.**

Cross-fitting converts this from an argument into an *enforced constraint*: `L_{B|A}` can only fall if
the tail law estimated on A actually bounds the risk of anchors in B. Because the partition is redrawn
every step, a solution that is tail-transferable on average over `C(45,23)`-many random identity splits
is, by construction, one whose similarity law does not depend on which identities were used to fit it.

**Honest limit of this argument.** It shows the *estimand* transfers; it does not by itself prove the
*optimum* transfers, because the network still chooses `μ_c` for seen `c` only. What the argument does
establish is that CFEV's target is in the class of functionals that admit `O(C^{-1/2})` transfer, and
that pairwise-max targets are not. I flag this as the single largest theoretical gap in the proposal.

### 2.3 Proof-level attack on the cheapest degeneracies

**(D1) Total collapse** `z_i ≡ z*`. Then all `S_ij = 1`, so `N_A` is the constant 1, `Y_A` is empty
(no `s > u` once `ū → 1`), `k_A < k_min = 64`, CFEV is skipped, and `L_PA`'s second term diverges as
`log(1 + |X_c⁻| e^{α(1+δ)})` ≈ `α·1.1 = 35.2` per class. **Blocked with margin ≫ any CFEV gain.**

**(D2) Degenerate-tail exploit — the one that actually threatens the method.** Make all negatives
equal to a constant `c₀ < s⁺`. Then `â₀ = â₁ = 0` on exceedances, `D → 0`, and the clamp `|D| ≥ 1e−6`
makes `σ̂ → 0⁺`, `w → ∞`, `q → 0`, `L_CFEV → 0`. So a *perfectly* separated, zero-variance seen-class
configuration drives CFEV to its floor. Three independent blocks:
 - `ζ̂` is computed on **all** of `N_A` with a sigmoid, so it does not vanish unless the whole negative
   population sits below `u`; but `u` is an **EMA quantile of the same population**, so `ζ̂` is pinned
   near `p₀ = 0.1` by construction. The exploit cannot reduce `ζ̂` — it is a self-normalizing anchor.
   This is the structural reason for using an EMA *quantile* rather than a fixed absolute threshold.
 - `k_min = 64` forces at least 64 exceedances to exist for the term to be active; a batch that
   degenerates loses the CFEV gradient entirely and is left with `L_PA` alone, which does *not* reward
   degeneracy.
 - The `softplus(σ_raw) + 1e−4` floor bounds `w ≤ (s⁺−u)/1e−4`, and `ℓ` is `log`-compressed, so the
   attainable floor of `log(1 + M ζ̂ e^{ℓ})` is bounded below by `log(1 + M·p₀·e^{−(s⁺−u)·10⁴})`, which
   is a genuine (not vacuous) target only when the *distributional* separation is real.
 Residual risk: a partial version of D2 (over-concentrating the negative tail to shrink σ̂ without
 lowering it) is not fully excluded. Control **C8** (σ̂-only vs ξ̂-only gradient) is the diagnostic.

**(D3) Estimator gaming by outlier injection.** Because `ξ̂ = 2 − â₀/(â₀−2â₁)`, a few very large
exceedances raise `â₀` faster than `â₁` (whose weights `(k−j)/(k−1)` vanish at the top of the ascending
order), which *raises* `D` and lowers `ξ̂` — i.e. the network could fake a light tail by adding extreme
outliers. Blocked structurally: adding extreme negatives raises `s` above `s⁺` for those pairs, which
directly increases the *true* R@1 error, and the same values raise `ζ̂` (they are above `u`) and raise
`q` through `ζ̂` linearly. The PWM path's benefit is `O(1/k) = O(1/405)`; the `ζ̂` path's cost is
`O(1/|N_A|) = O(1/4048)` per pair but is *multiplicative* on `q` and hence on the whole loss. Also the
`±0.45` tanh clamp caps the achievable gain from ξ̂ manipulation at a bounded amount. **Not a free
exploit, but the least-bounded of the four; C3 (empirical-tail control) detects it.**

**(D4) Positive saturation / intra-class collapse.** `∂L/∂s⁺_i ∝ M ζ̂ e^{ℓ_i} ℓ'_i/(1 + M ζ̂ e^{ℓ_i})`,
which decays like `e^{−w_i}` once `M q_i ≪ 1`. So CFEV **stops pulling positives together once the
extrapolated risk is negligible** — structurally unlike ProxyAnchor/PFML, whose positive term keeps
applying `exp(−α(s−δ))` pressure. This is a genuine advantage on unseen-class generalization
(over-tightened seen classes are the classic over-fitting signature) and it is *testable*: C5 freezes
the tail parameters so only the `s⁺` path is live; if C5 alone reproduces the gain, the mechanism claim
collapses to "a better-shaped positive schedule" and I would report the method as such.

**(D5) Split-degenerate solution.** The network could try to make the tail law depend on class index
in a way that happens to be constant across the *particular* halves. Blocked: the partition is redrawn
i.i.d. every step, so satisfying `L_{B|A}` on average requires transferability over an exponentially
large family of splits; and `L_agree` penalizes any residual `(ξ,σ)` heterogeneity between halves.

**(D6) Trivially satisfying the risk by shrinking `M`.** `M` is a frozen constant read off the training
set, not a learned or tuned quantity, so there is no gradient path to it.

---

## 3. Adversarial primary-source novelty search

I searched inside DML (ranking surrogates, distribution-matching losses, proxy losses, tail/long-tail
losses) and outside it (extreme-value statistics, open-set recognition, econometrics, face-recognition
FAR-targeted training, distributionally-robust optimization). Nearest works, each with the mechanism
distinction in one sentence:

**Inside DML / retrieval**

| nearest work | mechanism distinction from CFEV |
|---|---|
| **Recall@k Surrogate loss** (Patel, Tolias, Matas, CVPR 2022) — smooth sigmoid rank surrogate, solves the batch-size problem by brute-force ~4000-sample batches + similarity mixup | RS smooths the **empirical** rank *inside the observed batch* and buys gallery realism by paying for an enormous batch; CFEV never evaluates an empirical rank, it fits a GPD to the batch tail and **extrapolates the exceedance probability analytically to M ≫ m** at ordinary batch 180. |
| **Smooth-AP** (Brown et al., ECCV 2020) | Same distinction: a sigmoid-relaxed *empirical* ranking statistic, with no parametric tail and no out-of-support extrapolation. |
| **Histogram loss** (Ustinova & Lempitsky, NeurIPS 2016) — *the closest work in spirit* — estimates the positive/negative similarity **distributions** by soft histograms and minimizes P(pos < neg) | Histogram loss targets a **first-order, whole-support** functional (the AUC-type reversal probability, a *mean*), estimated **non-parametrically and therefore only on the observed support**, with no gallery-size dependence and no identity split; CFEV targets the **M-th extreme order statistic** via a parametric tail whose whole purpose is validity *outside* the observed support, and its value changes by design when M changes. |
| **Multi-Similarity / Circle / hard-negative mining** | These concentrate `O(1)` gradient on the hardest *observed* pairs; CFEV spreads signed `O(1/k)` gradient across ~400 tail pairs with PWM weights and has *no* notion of "the hardest pair". |
| **ProxyAnchor / ProxyNCA++ / PFML (Potential-Field DML, CVPR 2025)** | All are functionals of individual sample–proxy or sample–sample interactions (PFML: a continuous attractive/repulsive potential field over embeddings, augmented by 15 proxies/class on CUB-Cars, 2 on SOP); none contains an order-statistic, a tail-index, or a gallery-size parameter, and none estimates a nuisance quantity on an identity-disjoint subset. |
| **Proxy Synthesis** (Gu & Ko, AAAI 2021), **Metrix** (ICLR 2022), **Embedding Expansion** (CVPR 2020) | These *synthesize* pseudo-unseen classes/points to broaden the seen-class support; CFEV synthesizes nothing and instead extrapolates the *risk functional* of the existing support. |
| **Divide-and-Conquer the embedding space** (Sanakoyeu et al., CVPR 2019), **BIER** | Split the data/embedding into groups to train separate learners; CFEV's split is not a learner partition — it is a **statistical cross-fit** whose only purpose is to estimate a nuisance parameter on identities disjoint from the scored ones. |
| **Long-tailed recognition losses** (LDAM, Balanced-Softmax, Pareto/focal reweighting of the *loss* distribution) | "Tail" there means the tail of the **class-frequency** distribution and reweights examples; CFEV's tail is the upper tail of the **similarity-value** distribution and is used to compute a probability, not a weight. |

**Outside DML**

| nearest work | mechanism distinction from CFEV |
|---|---|
| **EVM / OpenMax / Meta-Recognition** (Scheirer, Rudd, Boult) — EVT (Weibull/GPD) on recognition scores for open-set rejection | These fit EVT **post hoc at test time to calibrate a rejection threshold on a frozen network**; CFEV puts the EVT fit **inside the training graph** and backpropagates through ξ̂ and σ̂ into the network weights, changing the geometry rather than calibrating it. |
| **SPOT / DSPOT streaming anomaly detection**, **DeepExtrema**, Pareto-loss forecasting | EVT for thresholding or for predicting extreme *targets*; none makes an extreme-order-statistic *retrieval risk* the training objective. |
| **CVaR / superquantile / DRO training** | Optimizes the tail of the **per-example loss** distribution over the empirical sample, still within-sample; CFEV optimizes the tail of a **similarity** distribution and explicitly extrapolates beyond the sample to a specified population size M. |
| **Double/debiased ML cross-fitting** (Chernozhukov et al., 2018) | There, sample splitting removes own-observation bias in Neyman-orthogonal moment estimation for treatment effects; here, the split is over **identities** and its purpose is to block a *memorization shortcut*, not to debias a moment. |
| **Face-recognition TAR@FAR-targeted losses** (e.g. distribution-distillation losses) | These match or shift similarity distributions between easy/hard cohorts at a fixed operating point; none fits a GPD, none has a gallery-size extrapolation term, and their operating point is a threshold rather than a max-order-statistic. |
| **Uniformity / alignment** (Wang & Isola, ICML 2020), MHE, MMCR | Global entropy- or energy-type functionals of the *whole* embedding distribution, class-agnostic and gallery-size-agnostic; CFEV constrains only the upper 10% of the inter-class similarity law and its value is a monotone function of M. |

**Honest novelty assessment.** The individual ingredients are all standard in their home fields
(POT/GPD, PWM estimation, sample splitting, ProxyAnchor). What I claim is new, and what I could not
find in a primary source, is the **composite estimand**: a train-time loss equal to the GPD-extrapolated
probability that the maximum of M unobserved negatives beats the positive, with the tail estimated on
an identity-disjoint half of the batch. The nearest genuine threat is Histogram loss (same "optimize a
distributional overlap functional" family) and Recall@k-Surrogate (same "optimize the retrieval metric"
goal); §4's controls C3 and C6 are designed specifically so that a reviewer can decide empirically
whether CFEV is anything more than those two.

---

## 4. Decisive matched-compute controls

All arms: identical ResNet-50/ImageNet-1K init, 512-D, 224 px, batch 180, AdamW, 200 epochs, cosine
decay, identical augmentation, identical BN-freeze, **5 seeds**, paired by seed. All arms cost within
1% of each other in epoch time. Every control is designed to **kill** a specific simpler explanation.

| # | control | what it changes | simpler explanation it kills |
|---|---|---|---|
| **C0** | ProxyAnchor alone, λ=0 | — | establishes *my own* reference; I do not inherit PFML's or DADA's numbers |
| **C1** | λ=0, `L_PA × (1+r)` | total gradient norm matched to CFEV arm | **"it's just a larger effective loss scale interacting with AdamW decay"** — headline effect is reported as `CFEV − C1` |
| **C2** | CFEV with `M := m = 180` | removes the extrapolation, keeps everything else | **"it's just another batch-level ranking loss"** — this is the single most decisive control |
| **C2′** | M-sweep: `M ∈ {180, 1.8e3, 1.8e4, \|D_train\|, 10\|D_train\|}` | one dataset (SOP), 5 seeds each | tests the *predicted unimodal peak near \|D_train\|*; a flat curve falsifies the gallery-extrapolation account within a single dataset, removing the SOP-vs-CUB dataset confound |
| **C3** | replace GPD survival by the **empirical** survival (soft-rank of `s⁺` among `N_A`), same `log(1+M·q)` wrapper, same cross-fit | removes the parametric tail only | **"it's Recall@k-Surrogate / Smooth-AP in disguise"** and simultaneously detects D3 outlier-gaming |
| **C4** | fit tail on the **same** half as the anchors (A→A, B→B) | removes cross-fitting only | **"the identity split is decoration"**; if C4 ≥ 0.9·CFEV I will drop cross-fitting from the method |
| **C5** | `stopgrad` on `ξ̂, σ̂, ζ̂` (only the `s⁺` path live) | removes all negative-side gradient | **"it's just a better-saturating positive-pull schedule"** (degeneracy D4) |
| **C6** | replace the tail term with a penalty on the **mean** negative similarity at matched grad-norm | first moment instead of tail | **"any negative-suppression term at this weight would do"**; also isolates CFEV from Histogram-loss-type mean functionals |
| **C7** | PA + Wang–Isola uniformity at matched grad-norm (τ swept) | global geometry instead of tail | **"it's uniformity/spreading by another name"** |
| **C8** | gradient-mask so only `ξ̂` (resp. only `σ̂`) receives gradient | isolates shape vs scale | tells us whether the effect is tail-*shape* or merely tail-*scale* shrinkage (relevant to residual degeneracy D2) |
| **C9** | PA + hardest-negative term at matched grad-norm | concentrated vs distributed tail gradient | **"tail pressure = hard-negative mining"** |
| **C10** | `λ_g = 0` | removes tail-agreement | tests whether the homogeneity penalty helps or hurts (I expect small; if negative, it is dropped) |

**Pre-registered decision rule.** The mechanism claim survives only if, on SOP with 5 paired seeds:
`CFEV − C1 ≥ +0.8` R@1 points **and** `CFEV − C2 ≥ +0.6` **and** `CFEV − C3 ≥ +0.4` **and**
`CFEV − max(C5, C6, C7, C9) ≥ +0.4`. If C3 is within ±0.2 of CFEV, the honest conclusion is that CFEV
is a reparameterization of an empirical recall surrogate and should be reported as such, not as a new
mechanism.

---

## 5. Frozen forecasts, falsification thresholds, and frontier arithmetic — Lane A only

### 5.1 The mechanism's own quantitative law (this is the strongest falsifiable claim)

CFEV predicts the gain over a matched ProxyAnchor baseline scales with the **log gallery-to-batch
ratio**, because that is exactly the width of the order-statistic gap the method closes (§2.1):

```
Δ R@1  ≈  γ · log(M / m),      m = 180,   γ frozen at 0.5 R@1 points per nat
```

| dataset | M = \|D_train\| | log(M/m) | predicted Δ |
|---|---|---|---|
| CUB-200-2011 | 5,864 | 3.48 | +1.7 |
| Cars196 | 8,054 | 3.80 | +1.9 |
| In-Shop | 25,882 | 4.97 | +2.5 |
| SOP | 59,551 | 5.80 | +2.9 |

### 5.2 Frozen point forecasts (R@1, mean of 5 seeds, Lane A)

| dataset | **C0** (my PA repro) | **CFEV** median | CFEV 80% interval | Lane-A reference |
|---|---|---|---|---|
| **SOP** | 0.803 ± 0.004 | **0.832** | [0.820, 0.843] | PFML **0.829 ± 0.003**; DADA 0.810 |
| **In-Shop** | 0.914 ± 0.004 | **0.939** | [0.928, 0.948] | PA+DADA **0.930** (seeds unreported) |
| **Cars196** | 0.882 ± 0.005 | **0.901** | [0.888, 0.913] | PFML **0.927 ± 0.003**; DADA 0.921 |
| **CUB** | 0.695 ± 0.005 | **0.712** | [0.699, 0.726] | PFML **0.734 ± 0.003**; DADA 0.729 |

### 5.3 Explicit frontier-crossing arithmetic

- **SOP.** CFEV 0.832 − PFML 0.829 = **+0.003**. Paired-difference sd `√(0.004² + 0.003²) = 0.005`;
  the arms are unpaired (different papers), so a 2σ crossing needs 0.839, i.e. `γ ≥ 0.62` pts/nat.
  **P(point-estimate crossing) ≈ 0.55; P(2σ crossing) ≈ 0.18.** Against DADA's matched-cost SOP row
  (0.810) the margin is **+0.022**, and `P(beating 0.810) ≈ 0.95`.
- **In-Shop.** CFEV 0.939 − PA+DADA 0.930 = **+0.009**. The reference has no reported uncertainty, so
  no significance test is possible and I claim only a point-estimate crossing.
  **P(point-estimate crossing) ≈ 0.62.** This is the most favourable frontier target for CFEV.
- **Cars196.** CFEV 0.901 − PFML 0.927 = **−0.026**. Crossing would require `γ ≈ 1.18` pts/nat, 2.4×
  the frozen value. **P(crossing) ≈ 0.05. I forecast CFEV does not cross the Lane-A Cars frontier,
  and also does not beat DADA's 0.921.**
- **CUB.** CFEV 0.712 − PFML 0.734 = **−0.022**. Crossing needs `γ ≈ 1.13` pts/nat.
  **P(crossing) ≈ 0.08. I forecast CFEV does not cross the Lane-A CUB frontier, and also does not
  beat DADA's 0.729.**

**Why I forecast losing on CUB/Cars and state it rather than hiding it.** (i) Their `log(M/m)` is the
smallest of the four, so the mechanism has the least headroom by its own law. (ii) Their dominant
zero-shot error mode is *intra-class multimodality* (viewpoint, pose, illumination produce multi-modal
class manifolds), which is exactly what PFML's **15 proxies per class** on CUB/Cars is built to absorb —
and note PFML drops to **2 proxies on SOP**, where CFEV's advantage is largest, which is consistent
with the two mechanisms attacking different error modes. CFEV has one proxy per class and does nothing
about multimodality. A CFEV + sub-centre-multi-proxy composition is the obvious repair and I expect it
would recover most of the CUB/Cars gap, but **it is a different method and I do not claim it in this
frozen forecast.**

### 5.4 Falsification thresholds (pre-registered, any one suffices to reject the mechanism)

1. `CFEV − C1` on SOP `< +0.8` points → **the mechanism does not exist above the loss-scale effect.**
2. `CFEV − C2` (M = batch) `< +0.6` points → **gallery extrapolation, the core claim, is falsified.**
3. C2′ M-sweep on SOP is flat (max − min `< 0.8` points across `M ∈ [180, 10|D_train|]`) or peaks at
   `M ≤ 1.8e3` → **falsified within a single dataset, with the dataset confound removed.**
4. The fitted slope `γ̂` from regressing the four measured Δ's on `log(M/m)` is `< 0.2` pts/nat or
   negative → **the scaling law is falsified**, even if the mean gain is positive.
5. `CFEV − C3` `< +0.4` → **CFEV is an empirical recall surrogate, not a tail-extrapolation method.**
6. SOP CFEV `< 0.820` (below the 80% interval floor) → the point forecast is wrong regardless of
   mechanism.
7. Fitted `ξ̂` at convergence sits at a clamp boundary (`|ξ̂| > 0.44`) on any dataset → the GPD fit is
   not identified and the reported "tail shaping" is an artefact of the clamp.

I commit to reporting all seven, including negative outcomes, and to reporting `CFEV − C1` (not
`CFEV − C0`) as the headline mechanism estimate.

---

## 6. Cost, and benchmark / contamination risk

### 6.1 Training cost

Per step, on top of ProxyAnchor: two `≤92×92` cosine Gram blocks (`2 × 92² × 512 ≈ 8.7` MFLOP), two
sorts of ~4·10³ scalars, and ~15 elementwise kernels. ResNet-50 forward+backward on 180 images at
224² is ≈ `3 × 180 × 4.1` GFLOP ≈ **2.2 TFLOP**. The added arithmetic is `≈ 4·10⁻⁶` of the step; the
real cost is kernel-launch latency.

- **Epoch time: ≈ 1.00–1.005×** baseline (measured overhead expected below timing noise).
- **Memory: +180² × 4 B × 2 ≈ 260 kB** activations; **1.00×** peak.
- **Extra learned parameters: 0.** (Proxies come from PA, not from CFEV.)
- **No auxiliary network, no second view, no teacher, no reconstruction branch, no adversary.**

For context within Lane A: PA+DADA costs ≈1.06× epoch time and 1.01× memory. CFEV is materially
cheaper than the Lane-A cost reference. (Lane-B systems — AdvRF's training-only ResNet-34/U-Net
reconstruction plus distillation, VAPNet's attribute machinery — are far heavier, but they are Lane B
and I make no comparison to them.)

### 6.2 Deployment cost

**Identical to a plain ResNet-50 retrieval model.** One model, one view, one 512-D descriptor, cosine
NN. No reranking, no transduction, no gallery fitting, no test-time augmentation.

### 6.3 Contamination and benchmark risks — stated, not minimized

- **No test-side information is used anywhere.** `M = |D_train|` by construction (§1.9). No test images,
  no gallery statistics, no external data, no text/VLM encoder, no extra annotations.
- **ImageNet-1K pretraining overlaps CUB (many bird classes) and Cars.** This is a real contamination of
  the "zero-shot" framing that CFEV shares with *every* reference cited in this lane, including PFML and
  DADA. It preserves internal validity (all arms share the same init) but means absolute numbers should
  not be read as true zero-shot transfer. I flag this rather than inherit it silently.
- **Small-test-set variance.** CUB (5,924 gallery) and Cars (8,131) show ≈±0.4–0.6 R@1 points of seed
  noise; forecast differences of 1–2 points there are only ~2–3σ with 5 seeds. Reported as mean ± std
  over 5 seeds to match PFML's protocol.
- **Reference-uncertainty asymmetry.** PFML reports ±0.003 over five runs; **PA+DADA's In-Shop 0.930
  reports neither seed count nor uncertainty**, so my In-Shop "crossing" can only ever be a
  point-estimate claim. I do not convert it into a significance claim.
- **I do not inherit PFML's frontier.** I could not obtain PFML's primary-source recipe, so I cannot
  reproduce it; PFML is used strictly as an *external reference line*, and every mechanism claim is
  made against my own C0/C1 reproductions under a single disclosed recipe.
- **Statistical risk in the fit itself.** Cosine similarity has bounded support, so the true tail is in
  the **Weibull** domain of attraction (`ξ < 0`) — which is where PWM is well-behaved and GPD-MLE is
  not, so the estimator choice is principled rather than convenient; but it also means the fitted `ξ̂`
  will sit in the negative range and falsification test #7 (clamp saturation) is a live concern.
- **The i.i.d.-gallery assumption is wrong.** Gallery images of the same class are correlated, so the
  *effective* number of independent negatives is smaller than `|D_train|`. This biases the ideal `M`
  downward by an unknown factor. C2′ (the M-sweep) measures it. If the empirical optimum is near
  `M ≈ m`, falsification test #3 fires and the method should be rejected.
- **SOP/In-Shop have ~5.6 and ~6.5 images per training class**, so the hardest-positive `s⁺` is a max
  over ≤3 in-batch positives — a noisier statistic there than on CUB/Cars. This cuts *against* the
  datasets where I forecast the largest gains, and is a reason my SOP interval is wide.

---

## 7. Unresolved source ambiguities (stated in full)

1. **PFML (CVPR 2025, Potential-Field-based Deep Metric Learning).** I have its reported R@1 values
   (CUB 0.734, Cars 0.927, SOP 0.829 over five runs; 15 proxies/class on CUB-Cars, 2 on SOP) but not its
   full training recipe from primary source. Consequence: **no inheritance** — all mechanism claims are
   against my own matched C0/C1.
2. **PA+DADA (AAAI 2024) In-Shop 0.930** — seed count, uncertainty, and whether the 0.930 row uses the
   identical 512-D/224 px/200-epoch configuration are unreported to me. Treated as a point reference.
3. **ProxyAnchor's exact schedule.** PA's published recipe (AdamW, lr 1e-4, proxy lr ×100, wd 1e-4,
   batch 180, 1-epoch embedding warm-up, α=32, δ=0.1) is disclosed, but its **epoch count and lr-decay
   schedule** are not unambiguous to me and in any case differ from this lane's 200-epoch mandate.
   I therefore specify cosine-to-zero over 200 epochs and apply it identically to every arm. **Any
   comparison of my C0 to PA's published numbers is therefore not valid; only within-study comparisons
   are.**
4. **Proxy initialization** (`kaiming_normal_(fan_out)`) and **BN-freeze policy** are reproduction
   choices I state rather than inherit; both are held identical across all arms.
5. Whether `L_agree` helps at all is genuinely unknown to me (C10 decides it); I include it at
   `λ_g = 0.05` because it encodes the §2.2 exchangeability claim directly, not because I have evidence.

---

## 8. Where I think this most likely fails (proposer's own view)

1. **C3 comes out flat.** The parametric extrapolation may buy little over a well-tuned empirical
   soft-rank, because 405 exceedances already characterize a bounded-support tail reasonably well. This
   is my single largest worry and it is exactly what C3 is for.
2. **γ is smaller than 0.5 pts/nat.** The transfer argument in §2.2 bounds the *estimand*'s
   generalization, not the optimum's; the realized γ could be 0.2, which would leave SOP at ~0.815 —
   above DADA, below PFML, and a negative result against the frontier.
3. **Residual degeneracy D2/D3.** Tail-scale shrinkage without genuine separation is bounded but not
   provably excluded; C8 is the diagnostic and a positive C8 σ̂-only result would be bad news.
4. **The forecast is honestly a partial one.** I forecast a Lane-A frontier crossing on **SOP
   (point-estimate, ~0.55) and In-Shop (point-estimate, ~0.62)** and an explicit **failure to cross on
   CUB and Cars**. I state this rather than inflating four numbers, because the mechanism's own scaling
   law forbids a uniform win and a uniform-win forecast would be the first thing a competent reviewer
   should disbelieve.
