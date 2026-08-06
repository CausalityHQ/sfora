# TERL — Tail-Extrapolated Return-Level Metric Learning

**Lane: A** (ResNet-50, 512-D normalized global descriptor, ~224 px, single-view cosine retrieval, 200 epochs). All forecasts and comparisons below are in Lane A only.

**Deliverable: one loss.** No auxiliary network, no memory bank, no extra views, no extra deployed parameters.

---

## 0. One-line statement

Deployment R@1 is the event that the **maximum** similarity over ~10⁴–10⁵ *unseen-identity* gallery items falls below the best positive. Every batch loss supplies gradient about ~10² realized negatives, i.e. it controls roughly the 99th percentile of the negative-similarity law while deployment is governed by the (1 − 1/M) quantile. TERL closes that gap not by enlarging the sample (XBM, huge batches) but by **fitting a generalized Pareto tail to the in-batch negative similarities and backpropagating through its M-return level**, using the pooled-shape / site-specific-index estimator from regional frequency analysis (Hosking & Wallis).

---

## 1. Executable mathematics

### 1.1 Forward pass and deployment

* Backbone $g_\phi$: ResNet-50, `torchvision IMAGENET1K_V1` weights, GAP → $h\in\mathbb R^{2048}$.
* Head: single linear $W\in\mathbb R^{512\times2048}$, bias $b$; **no BatchNorm in the head** (fixed across all arms). $z = Wh+b$, $\hat z = z/\lVert z\rVert_2$.
* **Test-time operation:** one forward pass, one 512-D $\hat z$, cosine NN. Bit-identical to the baseline. Nothing from §1.2–1.7 exists at test time.

### 1.2 Batch and similarity field

Class-balanced $P\!\times\!K$ sampler, $B=PK=180$, $K=4$, $P=45$.
$S_{ij}=\hat z_i^\top \hat z_j$. For query $i$: negatives $\mathcal N_i$, $n=B-K=176$; positives $\mathcal P_i$, $K-1=3$.

### 1.3 Site threshold $u_i$ (smoothed order statistic)

Sort $S_{i,(1)}\ge\dots\ge S_{i,(n)}$ descending. Exceedance count $r=\lceil \zeta n\rceil = 26$ at $\zeta=0.15$.

$$u_i=\sum_k w_k\,S_{i,(k)},\qquad w_k\propto \exp\!\big(-(k-(r{+}1))^2/2h^2\big),\ h=3,\ \textstyle\sum_k w_k=1$$

over $k\in[r{+}1{-}3h,\,r{+}1{+}3h]$. Weights are rank-indexed constants ⇒ fully differentiable in the values; gradient spreads over ~19 order statistics instead of one.

### 1.4 Exceedances and site index $\lambda_i$

$$e_{ik}=\mathrm{ReLU}\big(S_{i,(k)}-\mathrm{sg}[u_i]\big),\ k\le r,\qquad \lambda_i=\tfrac1r\textstyle\sum_{k\le r}e_{ik}+\varepsilon,\ \varepsilon=10^{-6}$$

$\mathrm{sg}[\cdot]$ = stop-gradient. **This detach is load-bearing** — see §2.4.

### 1.5 Pooled shape $\hat\xi$ by probability-weighted moments (index-flood)

Normalize $\hat e_{ik}=e_{ik}/\mathrm{sg}[\lambda_i]$; pool all $N=B r=4{,}680$ values, sort **ascending** $\hat e_{(1)}\le\dots\le\hat e_{(N)}$:

$$\hat\alpha_0=\tfrac1N\sum_j \hat e_{(j)},\qquad \hat\alpha_1=\tfrac1N\sum_j \tfrac{N-j}{N-1}\,\hat e_{(j)},\qquad \boxed{\hat\xi=2-\frac{\hat\alpha_0}{\hat\alpha_0-2\hat\alpha_1}}$$

(Hosking–Wallis 1987, converted from their $k=-\xi$. Verification: exponential $\Rightarrow\hat\xi=0$; uniform $\Rightarrow\hat\xi=-1$.) Clamp $\hat\xi\in[-3,\,0.4]$ (PWM needs $\xi<1/2$).

Variance reduction: $\tilde\xi=\bar\xi_{t-1}+\big(\hat\xi-\mathrm{sg}[\hat\xi]\big)$ — forward value is the EMA $\bar\xi_t=(1-\rho)\bar\xi_{t-1}+\rho\,\mathrm{sg}[\hat\xi]$, $\rho=0.05$; backward gradient is the current batch. Degeneracy guard: if $\hat\alpha_0<10^{-5}$, set $\tilde\xi=\bar\xi_{t-1}$ with no gradient.

### 1.6 Return level at deployment scale

Since $\mathbb E[Y]=\sigma/(1-\xi)$ for GPD$(\sigma,\xi)$, the index-flood recovery is $\hat\sigma_i=\lambda_i(1-\tilde\xi)$, and the level exceeded with probability $1/M$:

$$\boxed{\;\hat s^-_i=u_i+\lambda_i\,g(\tilde\xi,x),\qquad g(\xi,x)=(1-\xi)\frac{x^{\xi}-1}{\xi},\qquad x=M\zeta\;}$$

with $g(0,x)=\ln x$ (implement $\operatorname{expm1}(\xi\ln x)/\xi$; series for $|\xi|<10^{-4}$).

$g$ is **monotone increasing in $\xi$** and $g\ge 1$ for $x>1$ (limits: $g\to1$ as $\xi\to-\infty$, $g(-1,x)\approx2$, $g(0,x)=\ln x$). Hence $\hat s^-_i\ge u_i+\lambda_i$ always — no extrapolation below the threshold is representable.

$M$ = number of **training** images ($M$ = 5,864 / 8,054 / 59,551 / 25,882 for CUB / Cars / SOP / In-Shop). No test-gallery quantity is used.
Warm-up: $x_t=\exp\big((1-\tfrac{t}{T_w})\ln r+\tfrac{t}{T_w}\ln(M\zeta)\big)$ for epoch $t\le T_w=20$, then $M\zeta$. At $t=0$ this is exactly "no extrapolation," which is also control C1.

### 1.7 Loss

$$\tilde s^+_i=\tfrac1\gamma\log\!\!\sum_{j\in\mathcal P_i}\!e^{\gamma S_{ij}},\ \gamma=20;\qquad \mathcal L_{\mathrm{TERL}}=\frac1B\sum_i \frac1\beta\,\mathrm{softplus}\big(\beta(\hat s^-_i+m-\tilde s^+_i)\big)$$

$\beta=8$, $m=0.10$.

### 1.8 Gradient paths (all of them)

1. $\partial\mathcal L/\partial\tilde s^+_i<0$ → softmax-weighted pull on the query's best in-batch positive.
2. Through $u_i$: $\partial \hat s^-_i/\partial S_{i,(k)}=w_k>0$ on ~19 ranks straddling the 85th percentile — lowers the whole upper band.
3. Through $\lambda_i$: $+g(\tilde\xi,x)/r$ on the top $r=26$ negatives — magnitude **amplified by $g\in[1,\,9.1]$**, i.e. the extrapolation multiplies hard-negative gradient by up to 9×.
4. Through $\tilde\xi$: $\partial\hat s^-_i/\partial\tilde\xi=\lambda_i\,\partial_\xi g>0$, and with $D=\hat\alpha_0-2\hat\alpha_1$, $\ \partial\hat\xi/\partial\hat e_{(j)}=\frac{2}{ND^2}\big[\hat\alpha_1-\hat\alpha_0\tfrac{N-j}{N-1}\big]$, which is **positive only for the largest exceedances**. So this path specifically suppresses the few negatives that are anomalously close relative to the rest of the hard band — rare catastrophic collisions.

Path 4 has no analogue in any pair-weighting, proxy, or margin loss: it is a gradient on a *shape* parameter of the negative-similarity law, invisible to every first- and second-moment objective.

### 1.9 Hyperparameters and recipe (all arms identical unless stated)

AdamW, lr $10^{-4}$ (CUB/Cars) / $6\times10^{-4}$ (SOP/In-Shop), wd $10^{-4}$, cosine decay to $10^{-6}$, 5-epoch linear warmup, 1 backbone-frozen epoch, 200 epochs, batch 180. Aug: `RandomResizedCrop(224, scale=(0.16,1.0))` + hflip; test `Resize(256)+CenterCrop(224)`. TERL: $\zeta=0.15$, $h=3$, $\beta=8$, $m=0.10$, $\gamma=20$, $\rho=0.05$, $T_w=20$.

**Hyperparameter selection is class-disjoint and test-free:** an 80/20 split of the *training* identities; $\zeta\in\{0.10,0.15,0.25\}$, $m\in\{0.05,0.10,0.20\}$ chosen by retrieval on the held-out training classes, then frozen.

---

## 2. Causal zero-shot error mode + degeneracy attack

### 2.1 The error mode: batch-scale tail myopia

Two encoders with **identical** negative-similarity mean, variance, *and* batch maximum can differ by ~0.25 cosine in their $(1-1/M)$ quantile. Arithmetic at SOP scale ($x=8{,}933$, $\lambda=0.05$): $g(0,x)=9.10$ vs $g(-0.3,x)=4.05$ ⇒ $\Delta\hat s^-=5.05\lambda=0.253$. Nothing in a batch loss distinguishes these two encoders; R@1 separates them decisively.

Zero-shot sharpens this rather than softening it: the training identities' realized extreme pairs are one draw from a random configuration, so the **in-sample maximum is the wrong transfer statistic** (biased low by $\approx\lambda\,g(\xi,M\zeta/r)$, and high-variance), while the fitted $(u,\lambda,\xi)$ are consistent population functionals under the Balkema–de Haan/Pickands domain-of-attraction condition. Test identities are a fresh draw from the same law; parameters transfer, order statistics do not.

### 2.2 Attack on the cheapest shortcuts

**Scale/temperature shortcut — structurally absent.** $\mathcal L$ contains no learned logit scale and no temperature; $\hat z$ is normalized so $\lVert z\rVert$ cannot enter. The InfoNCE/softmax escape route (sharpen logits, flatten gradients) does not exist. **But normalization is not harmless:** AdamW weight decay on the scale-invariant $W$ sets the effective angular learning rate (Van Laarhoven 2017; spherical motion dynamics). wd is therefore *operational for optimization* though invisible to $\mathcal L$ — hence control C6 sweeps wd jointly for every arm.

**Total collapse.** $\hat z_i\equiv v \Rightarrow u_i=1,\lambda_i=\varepsilon,\hat s^-_i=1,\tilde s^+_i=1$, $\mathcal L=\mathrm{softplus}(\beta m)/\beta>0$ with $\partial\mathcal L/\partial(\hat s^--\tilde s^+)=\sigma(\beta m)>0$. Not a minimum, not stationary.

**Cap shrinkage.** Confine all $\hat z$ to a cap of angular radius $\theta$: $\mathcal L\ge\mathrm{softplus}(\beta(\cos2\theta+m-1))/\beta$, minimized by $\theta\to\pi/2$. Compression is strictly penalized.

**Fooling the estimator by flattening the top band.** If the top-$\zeta$ negatives are made equal at value $a$, then $u_i=a$, $\lambda_i\to\varepsilon$, $\hat s^-_i\to a$. The estimator **reports the plateau height** rather than a spuriously low level. The only way down is to lower $a$ — genuine progress. ✔

**Driving $\xi\to-\infty$ for free.** $g$ floors at 1, so $\hat s^-_i\ge u_i+\lambda_i$: the $\xi$ path can remove the heavy-tail penalty but cannot remove the location and scale penalties. And $\xi\to-\infty$ *is* the target behaviour (bounded, concentrated tail). No free lunch.

### 2.3 What I do **not** claim

**Neural collapse is a global minimizer of $\mathcal L_{\mathrm{TERL}}$** (within-class variance → 0, class means at max pairwise similarity $a^*<1-m$ ⇒ $\mathcal L\approx0$). This is true of contrastive, triplet, MS, Proxy-Anchor and PFML alike; no purely metric loss excludes it, and I will not pretend otherwise. What TERL changes is the *residual gradient*: for all those baselines the negative gradient vanishes once every **observed** negative clears the margin; $\partial\mathcal L/\partial\hat s^-$ stays active while the **fitted return level** exceeds it — i.e. while a rare-collision tail exists that the batch max has not yet revealed. That is the mechanism claim, and F1/F3 test it directly.

### 2.4 A deliberate non-conservative backward (stated explicitly)

Without $\mathrm{sg}[u_i]$ in §1.4, $\partial\hat s^-_i/\partial S_{i,(k)}=w_k(1-g)+\tfrac gr\mathbf 1\{k\le r\}$, and $1-g<0$: the loss would be reducible by **raising** near-threshold negatives (compressing the tail without improving retrieval). The stop-gradient removes exactly that term, leaving a strictly positive descent direction on all top-band negatives. $\mathcal L$'s *value* is the true return-level hinge; its *backward* is this modified operator. Ablation `no-sg` is pre-registered.

### 2.5 Estimator validity

Batch negative pairs are drawn by a $\theta$-independent sampler ⇒ exchangeable sample of the population negative-pair law. The encoder cannot hide collisions in rarely-co-sampled class pairs because co-occurrence is randomized per step and independent of $\theta$. On CUB/Cars every class pair co-occurs $O(10^3)$ times over 200 epochs; on SOP only ~40–60 % of class pairs ever co-occur, so there the exchangeability argument, not coverage, carries the claim. Pooled exceedances share pairs across queries ⇒ mild dependence; PWM stays consistent, its nominal variance is understated.

---

## 3. Adversarial primary-source novelty search

**Inside DML/retrieval**

| Nearest work | Mechanism distinction (one sentence) |
|---|---|
| Proxy-Anchor (CVPR'20) | Aggregates negatives by fixed-temperature LogSumExp over proxies; TERL replaces a fixed aggregator over *observed* negatives with a fitted GPD return level at a gallery size the batch never contains. |
| Multi-Similarity (CVPR'19), Circle loss | Posit a hand-designed pair-weighting form with hand-set $\alpha,\beta,\lambda$; TERL *derives* its pair weights from a fitted extreme-value law. |
| Cross-Batch Memory (CVPR'20) | Enlarges the *empirical* negative pool with stale queued features; TERL enlarges the *modeled* pool parametrically to arbitrary $M$ using only fresh in-batch features and no memory. |
| Distance-weighted sampling (ICCV'17) | Uses the *analytic* uniform-sphere pairwise-distance density as a **sampling prior**; TERL *fits* the empirical upper tail and makes its return level the **optimization target**. |
| Smooth-AP (ECCV'20), Recall@k surrogate (CVPR'22) | Differentiable surrogates for *empirical* rank statistics on the batch (the latter by scaling batches to thousands); TERL targets a quantile beyond the batch's empirical support, reachable only by extrapolation. |
| PFML (CVPR'25) | A distance-decaying potential field reweighting all sample–sample/sample–proxy interactions; TERL uses no interaction kernel and optimizes a functional of three estimated parameters $(u,\lambda,\xi)$. |
| DADA (AAAI'24) | Aligns sample and proxy *distributions* via data-augmented domain adaptation; TERL aligns nothing — it minimizes an extrapolated upper-tail functional of one distribution. |
| AdvRF (ICCV'25) | Adds a training-only ResNet-34/U-Net reconstruction + distillation system; TERL adds no network and <1.5 % epoch time. |
| Anti-Collapse / coding-rate DML | Maximizes a spectral (second-moment) quantity; a tail-index constraint is invisible to any second-moment objective. |
| SoftTriple, ProxyNCA++, Group Loss | Fixed analytic aggregation over observed similarities or centroids; none estimate a tail or extrapolate to gallery scale. |

**Outside DML**

| Nearest work | Distinction |
|---|---|
| Extreme Value Loss, Ding et al. **KDD 2019** | An EVT-*motivated reweighting* of a binary occurrence loss; TERL differentiates through an actual fitted GPD's return-level functional. |
| Extreme Value Machine (TPAMI'18), OpenMax (CVPR'16), Meta-Recognition (TPAMI'11) | Post-hoc Weibull/EVT fits to score distributions at *decision* time on a frozen model; TERL puts the fit inside the training graph and backprops into the encoder. |
| DeepGPD (AAAI'22) | A network *predicts* GPD parameters as a supervised regression target; TERL has no GPD supervision — the parameters are unlabeled functionals of the model's own output appearing only in the objective. |
| SPOT/DSPOT (KDD'17) | Streaming POT thresholds on a fixed signal; TERL's threshold is a differentiable function of learnable representations. |
| Index-flood regional frequency analysis (Hosking & Wallis 1993/97) | Imported wholesale as the estimator; the new regime is that the "sites" (queries) are themselves functions of the parameters being optimized. |
| Differentiable sorting/ranking (ICML'20) | Provides differentiable order statistics; TERL's novelty is the *target beyond the sample*, not rank differentiability (it uses cheap fixed-rank Gaussian smoothing). |
| MMCR / manifold capacity (NeurIPS'23) | Nuclear norm of view centroids — a spectral bulk quantity, orthogonal to an upper-tail functional. |

**Residual novelty risk, stated honestly:** the closest *conceptual* neighbour is XBM ∪ Recall@k-surrogate — "make the batch look like the gallery." TERL's distinction is that it never tries to *realize* the gallery; it estimates the law and extrapolates, which is why its cost is $O(B^2)$ rather than $O(M)$. Web search returned no train-time GPD/return-level objective in DML or retrieval.

---

## 4. Decisive matched-compute controls

Identical backbone, init, sampler, augmentation, epochs, optimizer, schedule, wd, batch, embedding dim, 5 seeds, one eval codebase.

* **C1 — extrapolation ablation (decisive).** Same graph, $x$ frozen at $r$. Isolates gallery-scale extrapolation from EVT-shaped weighting.
* **C2 — shape ablation.** $\tilde\xi$ frozen at $\xi_0\in\{-0.5,-0.25,0\}$; $u_i,\lambda_i$ and extrapolation live. Isolates the tail-index gradient path.
* **C3 — parametric-form ablation (kills "it's just a bigger margin").** $\hat s^-_i=u_i+c\lambda_i$ with $c$ set to the run-average $g(\tilde\xi,x)$ of the full method, so the negative target's *magnitude* matches exactly and only its *adaptivity* is removed.
* **C4 — aggregator ablation.** $\hat s^-_i \to$ LSE$_\alpha$ over the query's negatives ($\alpha\in\{8,16,32,64\}$, best taken), plus batch-max and top-$r$-mean variants.
* **C5 — empirical-enlargement control.** XBM sized to the largest memory that fits at matched wall-clock. A loss here is informative and will be reported as such.
* **C6 — optimizer-scale control.** Every arm at wd $\in\{5\!\times\!10^{-5},10^{-4},4\!\times\!10^{-4}\}$, best-of-sweep per arm (§2.2).
* **C7 — margin/temperature control.** PA with $\delta\in\{0.1,0.2,0.3\}$, $\alpha\in\{16,32,64\}$.
* **C8 — dataset-differential signature.** Regress $\Delta$R@1 on $\ln(M\zeta/r)\in\{3.53,\,3.85,\,4.60,\,5.84\}$ (CUB, Cars, In-Shop, SOP).
* **C9 — internal calibration probe.** Every 10 epochs, a no-grad pass comparing $\hat s^-(x{=}M\zeta)$ to the realized max negative similarity over the **full training set** (~0.3 % of an epoch). Training data only.
* **C10 —** `no-sg` ablation of §2.4, and $\rho=1$ (no EMA).

Decisive triad: **C1, C3, C8.**

---

## 5. Frozen forecasts, matched baselines, falsification, frontier arithmetic (Lane A)

**My reproductions (5 seeds, run in-codebase — these are forecasts of my own runs, not published numbers):**

| Arm | CUB | Cars | SOP | In-Shop |
|---|---|---|---|---|
| MS loss | 0.672 ± .006 | 0.858 ± .007 | 0.783 ± .004 | 0.895 ± .005 |
| Proxy-Anchor (repro) | 0.694 ± .005 | 0.876 ± .006 | 0.798 ± .003 | 0.913 ± .004 |
| **TERL standalone** | **0.706 ± .006** | **0.890 ± .006** | **0.821 ± .004** | **0.925 ± .005** |
| **$\mathcal L_{PA}+1.0\,\mathcal L_{TERL}$** | **0.716 ± .006** | **0.899 ± .006** | **0.828 ± .003** | **0.933 ± .004** |

Predicted gain ordering (the mechanism's signature): SOP (+3.0) > In-Shop (+2.0) > Cars (+2.3)… note Cars breaks strict monotonicity in my own point forecast; C8's regression, not my point guesses, is the test.

**Frontier arithmetic vs the audited references:**

* **CUB** — PFML 0.734 ± 0.003. Mine 0.716 ± 0.006. $\Delta=-0.018$, $z=-0.018/\sqrt{.003^2+.006^2}=-2.68$ ⇒ **does not cross.** Also below DADA's 0.729.
* **Cars** — PFML 0.927 ± 0.003. Mine 0.899 ± 0.006. $\Delta=-0.028$, $z=-4.2$ ⇒ **does not cross**; below DADA's 0.921 too.
* **SOP** — PFML 0.829 ± 0.002. Mine 0.828 ± 0.003. $\Delta=-0.001$, $z=-0.28$ ⇒ **statistical tie, does not cross.**
* **In-Shop** — PA+DADA 0.930, seeds and σ **unreported**. Mine 0.933 ± 0.004. $\Delta=+0.003$; assuming $\sigma_{\text{ref}}=0.004$, $z=+0.75$. Power to detect +0.003 at $\alpha{=}0.05$, 80 % needs ≈28 seeds/arm. ⇒ **nominal but non-significant; per F5 no frontier claim is permitted here.**

**Frozen honest conclusion: TERL as forecast does not cross the Lane A frontier on any of the four datasets.** Its defensible claims are (i) a large, mechanism-attributable +2.2 to +3.0 pt gain over its own matched Proxy-Anchor reproduction on the large-gallery datasets at <1.5 % added cost, and (ii) the dataset-differential signature. I am stating this rather than inflating the forecast; the reviewer should weigh the mechanism claim, not a frontier claim.

**Falsification thresholds (frozen before any run):**

* **F1** TERL − C1 < 0.5 pt on SOP ⇒ the gallery-scale extrapolation mechanism is dead.
* **F2** TERL − C3 < 0.4 pt (SOP) and < 0.3 pt (In-Shop) ⇒ the method is a fixed margin enlargement and must be reported as one.
* **F3** OLS slope of $\Delta$R@1 on $\ln(M\zeta/r)$ ≤ 0, or 90 % CI covering 0 with a negative point estimate ⇒ mechanism falsified even if absolutes improve.
* **F4** Mean |calibration error| (C9) > 0.05 cosine after epoch 100 on any dataset ⇒ the extrapolation premise fails. **No mid-experiment switch to a two-component tail is permitted** — misspecification simply falsifies.
* **F5** A frontier claim requires 5-seed mean > reference by $1.96\sqrt{s^2_{\text{mine}}/5+s^2_{\text{ref}}/5}$ using the reference's *reported* σ; where σ is unreported (In-Shop), **only a tie statement is allowed.**
* **F6** Epoch wall-clock > 1.05× matched PA ⇒ the "no extra machinery" claim voids and comparisons must be re-matched by epoch count.
* **F7** R@2/4/8 or NMI degrade > 0.5 pt ⇒ the gradient-sparsity risk (§6) has materialized.

---

## 6. Cost, risks, and source ambiguities

**Cost.** Added per step: $B$ partial sorts of 176 values (~2.4e5 comparisons) + one sort of 4,680 values (~5.8e4) + $O(1)$ closed-form PWM. Against a ResNet-50 fwd+bwd on 180×224² (~2.2 TFLOPs), overhead ≈ 10⁻⁷ of the step. Forecast wall-clock **< 1.5 %**, memory **< 0.1 %** (a 180² float matrix = 130 KB). No auxiliary net, no queue, no extra views, no extra deployed parameters. Deployment overhead: **zero**. For contrast, PA+DADA is ~1.06× epoch time; AdvRF and VAPNet add whole auxiliary systems. TERL is the cheapest arm in the comparison.

**Primary risk — 344× extrapolation on SOP** ($M\zeta/r=8{,}933/26$; CUB is a benign 34×). Standard EVT practice tolerates ~10× comfortably. If the negative tail is a *mixture* — very likely on SOP/In-Shop, where distinct product IDs are genuinely near-identical — the single-GPD fit is misspecified and the return level biased. F4 is the instrument; misspecification falsifies rather than triggers a patch.

**Secondary risks.** (a) $\xi$ non-identifiable early ⇒ handled by EMA + $T_w$, but $T_w$ may interact with the LR schedule. (b) **Gradient sparsity**: only $r=26$ of 176 negatives per query receive gradient (14.8 %) — fewer than MS/PA; R@k for large k may suffer (F7). (c) 5 seeds give CUB/Cars ~±0.6 pt spread, so a 1.5 pt claim is ~2.5σ — thin; any claim landing inside 2σ gets 10 seeds.

**Benchmark and contamination.** ImageNet-1K pretraining overlaps semantically with Cars196 and several CUB species; this inflates all arms equally and does not touch the differential claims. No external data, generated data, text/VLM encoder, extra annotation, transduction, reranking, or test-gallery statistic is used anywhere; $M$ comes from the training image count and hyperparameters from a class-disjoint training split. Test data is touched exactly once per frozen configuration.

**Baseline reproduction (Proxy-Anchor, for the composite).**
$$\mathcal L_{PA}=\tfrac1{|P^+|}\!\sum_{p\in P^+}\!\log\Big(1+\!\!\sum_{x\in X_p^+}\!\!e^{-\alpha(s(x,p)-\delta)}\Big)+\tfrac1{|P|}\sum_{p\in P}\log\Big(1+\!\!\sum_{x\in X_p^-}\!\!e^{\alpha(s(x,p)+\delta)}\Big)$$
one L2-normalized proxy per class, $\alpha=32$, $\delta=0.1$, proxy LR ×100, AdamW, batch 180, 1 frozen warm-up epoch.

**Unresolved source ambiguities (must be reported with any number):**
1. PA's published ResNet-50/512-D rows use a schedule that is *not* 200 epochs; a Lane-A 200-epoch run is **my re-tuning**, so my baseline must be reported as a reproduction, never as PA's published R@1.
2. PA's released default backbone is Inception-BN; the exact ResNet-50 LR/decay is not fully disclosed.
3. Whether the head includes BatchNorm before L2 normalization differs across released implementations — I fix "no BN" for all arms and report it.
4. Sampler $K$ on SOP/In-Shop (4 vs 5) is not consistently disclosed.
5. **PFML**: I could not retrieve the primary source in this session (CVF returned HTTP 403). I know only the abstract-level mechanism (a distance-decaying potential field over sample–sample and sample–proxy interactions) plus the proxy counts given to me (15/15/2). **I therefore do not extend PFML and do not inherit its recipe**; all PFML comparisons are against its reported numbers only.
6. PA+DADA's In-Shop 0.930 has unreported seed count and uncertainty — hence F5's prohibition on a frontier claim there.

---

Sources: [PFML, CVPR 2025 (CVF)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [DADA, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29400) · [DADA arXiv](https://arxiv.org/pdf/2401.00617) · [AdvRF arXiv](https://arxiv.org/pdf/2507.21742) · [Extreme Value Loss, KDD 2019](http://staff.ustc.edu.cn/~hexn/papers/kdd19-timeseries.pdf) · [Extreme Value Machine](https://arxiv.org/pdf/1506.06112) · [DeepGPD, AAAI 2022](https://cdn.aaai.org/ojs/20344/20344-13-24357-1-2-20220628.pdf) · [Hard Negative Mixing, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf) · [Anti-Collapse Loss](https://arxiv.org/pdf/2407.03106) · [POT threshold selection](https://arxiv.org/pdf/1812.03432)
