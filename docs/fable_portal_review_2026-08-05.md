# Independent cold review of PORTAL — 2026-08-05

- Durable consultation: `fa1510780f0840f5`
- Frozen review prompt: `docs/fable_portal_review_prompt_2026-08-05.txt`
- Frozen prompt SHA-256: `4d33619268e18e652f393bc70ac384f9a0125d0efebc33787ed9179a930f5b35`
- Caller attribution: `sfora/emafactorial`
- Provider path: Fable failed during tool startup; the configured same-job Claude Opus fallback produced the complete review below.
- Durable job status: `failed` / exit 1 after streaming a coherent complete answer.
- Result truncation: false.
- Preservation: exact durable `result` field; no local edits to reviewer text.

I'm the cold adversarial reviewer for the frozen PORTAL proposal. I'll first verify the mathematics by hand (PWM estimators, tail asymptotics, return-level arithmetic, dataset-size claims), then run targeted web checks on the prior-art claims — especially the WEINCE citation, which the proposal itself flags as decisive-but-unverified. Loading web tools now.
I'll review this cold, from the frozen text plus primary literature. Loading web tools.## VERDICT: **DEAD**

---

## The single strongest decisive reason

**The deployment scale $N$ has identically zero gradient. PORTAL contains no extrapolation; its return-level term is algebraically a softplus margin loss on observed order statistics.**

Write $x_i=(s_i^+-u_i)_+$. Then

$$\log\Lambda_i=\log(Np)-\tfrac1{\hat\xi}\log\!\big(1+\hat\xi x_i/\hat\sigma\big),\qquad \mathcal{L}_{\mathrm{ret},i}=\log(1+\Lambda_i)=\operatorname{softplus}\!\big(\log(Np)-\psi_{\hat\xi,\hat\sigma}(x_i)\big).$$

$N$ and $p$ enter **only** through the additive constant $\log(Np)$ inside a softplus. Every other quantity — $s_i^+$, $u_i$, $\hat\sigma$, $\hat\xi$ — is a statistic of the *observed* batch∪queue similarities. At the design point the proposal itself asserts ($\hat\xi\approx\xi^\star\approx-0.004$, "nearly exponential"):

$$\mathcal{L}_{\mathrm{ret},i}=\operatorname{softplus}\!\left(\frac{\hat\sigma\log(Np)-\big(s_i^+-u_i\big)}{\hat\sigma}\right),$$

a softplus margin loss between the pooled positive and the 95th-percentile negative, with **margin $\hat\sigma\log(Np)$ and temperature $\hat\sigma$**. Per-anchor weighting is $w_i=\Lambda_i/(1+\Lambda_i)=\varsigma(\log(Np)-x_i/\hat\sigma)$ — a sigmoid gate on a margin violation, i.e. hardness reweighting. This holds for *any* $\hat\xi$, not just $\hat\xi\to0$, because $\log N$ is an offset in the softplus argument regardless of the warp.

Consequences that cannot be repaired without a new proposal:

- The headline claim — *"a quantile that is never realized in any batch and is reachable only by extrapolation"* — is false. No gradient ever depends on anything outside the observed queue.
- *"Temperature and margin stop being hyperparameters"* and *"PORTAL tunes neither"* are false. $N$ and $p$ **are** the margin; $\hat\sigma$ **is** the temperature. A tuned fixed margin reproduces the identical gate exactly.
- **C4/F4 is a margin sweep, not an extrapolation test.** Sweeping $N\in\{10^2,\dots,10^5\}$ is exactly sweeping the margin by $\hat\sigma\log 10^3\!\approx\!6.9\hat\sigma$. F4 will "pass" for a reason orthogonal to the mechanism.
- **C2 does not control this**: it gives the *baseline* a $(\tau,m)$ grid on a different loss form. There is no arm "PORTAL loss form, fixed margin and fixed scale." C3 freezes $\hat\xi$ *and* $\hat\sigma$ *and* their curriculum simultaneously, so it cannot separate adaptive-scale from the (nonexistent) extrapolation.

---

## Every other false, inconsistent, or underdefined operation

### A. The anti-collapse mechanism does not exist (§2, "the load-bearing property")

**A1 — $\hat\xi$ is scale-invariant, so the claimed "designed tension" is false.** $\hat\xi=2-a_0/(a_0-2a_1)$ is a ratio of degree-1 homogeneous statistics: under $y\to cy$, $\hat\xi$ is *exactly unchanged* while $\hat\sigma\to c\hat\sigma$. So $\mathcal{L}_\xi$ constrains tail **shape** only and places **no** constraint on tail scale. Meanwhile $\partial\mathcal{L}_{\mathrm{ret}}/\partial\hat\sigma=\frac{\Lambda}{1+\Lambda}\cdot\frac{x}{\hat\sigma^2}>0$: descent shrinks $\hat\sigma$ with force growing as $\hat\sigma^{-2}$. Compressing the top-5% excesses toward a common value sends $\Lambda\to0$ and $\mathcal{L}_{\mathrm{ret}}\to0$ with $\hat\xi$ pinned at target — **the atomic-similarity configuration is an unbounded descent direction, reached along the exact axis the exclusion mechanism is blind to.** §2's *"$\mathcal{L}_{\mathrm{ret}}$ prefers small $\hat\sigma$, $\mathcal{L}_\xi$ forbids the atomic limit"* is refuted by one line of algebra.

**A2 — the neural-collapse proof contradicts the guard stated one paragraph earlier.** At an exact simplex ETF all negatives sit at $-1/(C-1)$ with zero spread, so $u_i$ equals that value and the excess set $\{y>0\}$ is **empty**: $m=0$, $a_0=a_1=0$, denominator clamped to $\varepsilon_a$, $\hat\xi=2-0/\varepsilon_a=+2$ — not $-\infty$. In the near-atomic case (all excesses $=y_0$): $\frac1m\sum_k(1-\frac{k-0.35}{m})=\frac12-\frac{0.15}{m}$, so $a_0-2a_1=0.3y_0/m$; with $m=2.4\times10^4$, $y_0\sim10^{-2}$ that is $1.25\times10^{-7}\ll\varepsilon_a=10^{-4}$. **The guard is always active in precisely the regime the proof invokes**, and the proof's "$\hat\xi=2-y_0/0^+\to-\infty$" silently drops it.

**A3 — with the guard active the gradient has the wrong sign.** Clamped, $\hat\xi=2-a_0/\varepsilon_a$ with $\partial\hat\xi/\partial(a_0-2a_1)=0$: the *dispersion* channel, the only anti-collapse channel, is dead. The surviving term $\partial\mathcal{L}_\xi/\partial a_0=+\lambda_\xi\delta/\varepsilon_a=2\times10^4$ drives the mean excess **down** — toward the atomic configuration. And "Guard $a_0-2a_1\ge\varepsilon_a$" is **underdefined**: hard clamp / additive floor / softplus give three different *signs* of the load-bearing behaviour.

### B. The tail-index anchor targets an inapplicable asymptote

**B1.** The EVI derivation is correct *as an asymptotic*: $f(s)\propto(1-s^2)^{(d-3)/2}\Rightarrow\bar F(1-\epsilon)\propto\epsilon^{(d-1)/2}\Rightarrow\xi=-2/(d-1)=-0.003914$ at $d=512$. But that index describes the limit **threshold $\to$ endpoint $s=1$**. $\hat\xi$ is estimated at $p=0.05$, where $u\approx1.645/\sqrt{512}=0.073$ — the shoulder of a near-Gaussian ($s\approx\mathcal N(0,1/d)$), nowhere near the endpoint. There the *penultimate* shape parameter of a Gaussian-type tail is $O(-1/\log m)\approx-0.1$ for $m\sim10^4$ (Cohen 1982; Gomes 1984). **A perfectly isotropic embedding yields $\hat\xi\approx-0.1$, ~25× the target in magnitude**, so $\mathcal{L}_\xi$ with $\lambda_\xi=10$ pushes geometry *away* from isotropy. This is exactly the "can a bounded cosine tail use the stated asymptotics" question: it cannot, at this threshold.

**B2 — pooling biases $\hat\xi$ the wrong way and changes what is being penalized.** Excesses are pooled across anchors with per-anchor thresholds but a single $(\xi,\sigma)$. A mixture of exponential tails with heterogeneous scales is hyperexponential — decreasing hazard, heavier than exponential — so pooling biases $\hat\xi$ **upward**, toward 0, the upper clip 0.4, and the PWM validity boundary $\xi<1/2$ the proposal itself cites. In practice $\mathcal{L}_\xi$ penalizes **inter-anchor tail-scale heterogeneity**, a different mechanism than claimed, controlled by nothing.

**B3 — one pooled fit cannot identify per-anchor or unseen-gallery risk.** With shared $(\hat\xi,\hat\sigma)$, every anchor's tail is asserted to be a *translate* of every other's; only $u_i$ varies. Nothing in the construction references unseen classes or any transfer property, so per-anchor and unseen-gallery exceedance risk are not identified. The proposal concedes tail shapes "may differ" — but the deeper problem is that the estimand is a translate family.

**B4 — support-endpoint violation is reported as a convenience.** With $\hat\xi\approx-0.004$, $\hat\sigma\approx0.03$, the fitted endpoint is $u_i+\hat\sigma/|\hat\xi|\approx u_i+7.7$: the model assigns mass to $s>1$, impossible for cosine. The text reports this ("finite endpoint at $x\approx12.8\gg2$") as evidence $\mathcal{L}_\xi$ "never binds," when it is the diagnostic that the bounded-support premise generating $\xi^\star$ is not satisfied by the fit.

**B5 — the clip decouples the two terms exactly where coupling is needed.** $\hat\xi$ clipped to $[-1,0.4]$ *inside* $\mathcal{L}_{\mathrm{ret}}$ is gradient-free outside the range. At the lower clip, $\bar F_i=p(1-x/\hat\sigma)_+$ is **exactly zero** for $x>\hat\sigma$: zero loss and zero gradient for every such anchor.

### C. Estimator correctness and executability

**Correct, credited:** the PWM formulas are exact Hosking–Wallis (1987). With $\alpha_r=\sigma/[(r+1)(r+1-\xi)]$: $\alpha_0/(\alpha_0-2\alpha_1)=2-\xi$ and $2\alpha_0\alpha_1/(\alpha_0-2\alpha_1)=\sigma$ — both estimators are exactly consistent. The stated exponential check ($a_0=\sigma$, $a_1=\sigma/4\Rightarrow\hat\xi=0,\hat\sigma=\sigma$) is exact. The constant-embedding degeneracy check is correct ($\Lambda=N$, loss maximal). The gradient formula $\partial_x\log(1+\Lambda)=-\Lambda/[(1+\Lambda)\sigma(1+\xi x/\sigma)]$ is right.

**C1 — ordering direction of $y_{(k)}$ is never stated; the formula is only HW for *ascending* order.** Descending gives $\hat\alpha_1'=\alpha_0-\alpha_1$, hence $\hat\xi=2-a_0/(2a_1-a_0)$ — negative denominator for $\xi<0$, undefined. Not executable as written.

**C2 — independence is violated at the level that matters.** $m\approx2.4\times10^4$ counts $112$ anchors × ~210 exceedances, but $K=4$ anchors per class see nearly the same queue and *all* 112 share it. Effective independent sample size is orders of magnitude smaller; the low-variance claim and the $\xi<1/2$ asymptotics are unsupported. The extremal index is raised only for the gallery Poisson bound, never for the estimator itself.

**C3 — the $u_i$ gradient route is ~97.5% dead.** $u_i$ is one selected order statistic among ~4200 negatives of which 4096 are the stop-gradient queue: with probability $\approx0.975$ the selected element is a queue vector and $\partial u_i/\partial\theta=0$. "Four routes ($s_i^+,u_i,a_0,a_1$)" overstates; live gradient is anchor-side only.

**C4 — $\beta=8$ contradicts its own justification.** With $|\mathcal P(i)|=3$ and the $1/|\mathcal P|$ normalization, $s_i^+$ is a soft *mean* $+\tfrac\beta2\mathrm{Var}$. Positives $(0.90,0.85,0.80)$ give $s_i^+=0.857$ vs mean $0.850$, max $0.900$. "R@1 needs the *best* positive to win" requires $\beta\gtrsim10^2$; at $\beta=8$ the surrogate optimizes the average positive.

**C5 — branch kink.** At $s_i^+=u_i$ the GPD branch gives $\bar F_i=p$ with slope $-p/\hat\sigma\approx-1.7$; the empirical branch at temperature $0.01$ has slope set by the local negative density. Values roughly agree, derivatives do not — an unstated first-order discontinuity firing on every hard anchor early in training.

**C6 — "standalone and parameter-free" is false.** PORTAL adds $p$, $N$, the 10-epoch log-linear $N$ ramp, $\varepsilon_a$, the $[-1,0.4]$ clip, $\delta=0.2$, $\lambda_\xi=10$ with a 5-epoch ramp, $\beta$, $\gamma$, $\delta_p$, $\lambda_p$, and the $0.01$ sigmoid temperature — strictly more hyperparameters than the margin loss it replaces.

**C7 — label noise sits exactly in the fitted region.** SOP/In-Shop contain the same object under distinct ids; CUB contains near-indistinguishable congeners. False negatives *are* the top-5% exceedances: they set $u_i,a_0,a_1$, hence both $\hat\sigma$ and $\hat\xi$, and receive a bounded non-vanishing $1/\hat\sigma$ push apart. Named only as a contamination caveat for reading small deltas; never as an estimator problem.

### D. Controls and forecasts

**D1 — the mechanism covariate is the wrong one, and its ordering is reversed.** The split arithmetic is right ($5864/100{=}58.6$, $8054/98{=}82.2$, $25882/3997{=}6.47$, $59551/11318{=}5.26$; logs $4.07/4.41/1.87/1.66$). But $N_{\mathrm{tr}}/C_{\mathrm{tr}}$ is just **images per training class** — a *proxy-coverage* quantity, applicable only against proxy baselines. PORTAL's own extrapolation distance is $\log(N_{\mathrm{tr}}/|\mathcal N|)$ with $|\mathcal N|\approx4200$: **0.33 (CUB), 0.65 (Cars), 1.82 (In-Shop), 2.65 (SOP)** — *exactly the reverse* of the forecast gain ordering. On CUB the queue already covers 72% of the claimed deployment scale, yet CUB is forecast the largest gain. F2 therefore cannot falsify the stated mechanism, and the ordering it does predict (CUB/Cars ≫ SOP/In-Shop) is the generic ordering nearly every DML method exhibits — a 3–4 point Spearman a null method passes.

**D2 — controls do not isolate extrapolation.** Given the decisive reduction, the live alternatives are adaptive temperature, margin offset, and quantile-based hard-negative emphasis. C1/F6 rules out "more negatives" only; C2 tunes the *baseline*; C3 confounds shape, scale and curriculum; C4 is a margin sweep; C5 tests only $\rho$; C6/C7 are orthogonal. No arm isolates the claimed mechanism.

**D3 — uncertainty arithmetic.** The $z$-values are internally consistent (I recomputed 4.89 / 4.19 / 2.64 and 3.07 / 2.30). But: they are $z$-tests on $n=5$ — under $t_8$, SOP's 2.64 gives $p\approx0.03$ two-sided, before any multiplicity correction over 3 datasets × 2 references. And the stated SEM-vs-std sensitivity is **wrong**: if PFML's ±0.003 is SEM, its std is $0.003\sqrt5=0.00671$, pooled SE $=\sqrt{0.005^2/5+0.003^2}=0.00374$ vs $0.00261$, so $z$ shrinks by **1.43×, not "~2.2×"** ($\sqrt5$ was applied to the pooled SE instead of the one component).

**D4 — the base is not measured.** "PFML my matched repro 0.728 ± 0.004" is a forecast, and §6 concedes PFML's full recipe "is not established." Every in-codebase delta is a difference between two predictions. Requirement 7 ("derive from measured premises and a reproduced matched base") is not met.

### E. Prior art and mechanism equivalence

**E1 — WEINCE.** I read the abstract directly (arXiv:2606.00262, 29 May 2026): *"Using extreme value theory, we show that this [softmax] assumption is often misaligned with the normalized embedding setting used in modern contrastive learning… WEINCE… uses **anchor-wise online batch statistics** to blend the usual softmax logits with an **endpoint shortfall correction**, adding **no trainable parameters**."* Every framing element PORTAL claims is already present: EVT on normalized-embedding similarities, the softmax-fails-at-the-top motivation, anchor-wise tail statistics, the bounded-support **endpoint** premise from which $\xi^\star=-2/(d-1)$ is derived, and parameter-free-ness. PORTAL's residual novelty is the $\log(Np)$ offset (a zero-gradient constant, §decisive) plus the $\xi^\star$ anchor (an inapplicable asymptote, §B1). The proposal concedes its WEINCE reading was "a model-summarized PDF, not equation-level inspection" and that the distinction "needs direct verification" — the abstract does not support it.

**E2 — the mechanism, correctly named, is occupied.** Once reduced to a softplus margin on (pooled positive − 95th-percentile negative) with adaptive scale, PORTAL sits inside quantile/top-k/CVaR margin surrogates and rank-metric surrogates (Recall@k Surrogate, Patel et al., CVPR 2022, arXiv:2108.11179; Smooth-AP), and inside temperature-as-hardness-weighting for contrastive losses. Notably, the isotropic-sphere similarity density $\propto(1-s^2)^{(d-3)/2}$ that generates $\xi^\star$ is *itself* prior art inside DML: Wu et al., ICCV 2017 (arXiv:1706.07567) derive and use exactly that density for distance-weighted sampling. The novelty table's distinctions against XBM, OpenMax/EVM, $\rho$-spectral, alignment/uniformity and chance-constrained DML are fair; the distinctions that matter (WEINCE, quantile/top-k margin, adaptive temperature) are not established.

### F. What survives clean

PWM algebra and the exponential reduction are exact. The constant-embedding degeneracy is correctly excluded. Split statistics and $z$ arithmetic are internally consistent. Cost claims are plausible (<2% step time, +8 MB queue, proxy matrix removed). The deployment surface is compliant: one ResNet-50, one view, fixed 512-D, cosine NN, no reranking, no test-gallery statistics; setting $N=N_{\mathrm{tr}}$ correctly avoids the gallery-size protocol violation, and the refusal to claim the In-Shop frontier given unknown PA+DADA uncertainty is right. No data, contamination, tuning, compute, capacity, or benchmark-protocol violation found beyond the ordinary ImageNet-pretraining caveat the references share.

---

**Why DEAD rather than UNRESOLVED.** The decisive reason is exact algebra on the load-bearing claim, requires no experiment to settle, and simultaneously invalidates the control (C4/F4) designed to protect it. It is independently reinforced by A1 (the anti-collapse tension does not exist — one line of homogeneity), A2–A3 (the "load-bearing" collapse proof contradicts its own guard and inverts sign when the guard fires), B1 (the boxed target is an endpoint asymptote used at a $p=0.05$ threshold), and D1 (the mechanism's own predictive ordering is reversed under its own configuration). Any of A1, A2 or B1 alone would be decisive; together with the $N$-has-no-gradient result there is nothing left of the claimed mechanism to test. Repairing any of these is a new proposal.

**Sources:** [WEINCE, arXiv:2606.00262](https://arxiv.org/abs/2606.00262) · [Hosking & Wallis, *Technometrics* 29(3), 1987](https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488243) · [Diebolt et al., PWM validity beyond γ<1/2](https://www.numdam.org/item/CRMATH_2004__338_8_629_0.pdf) · [Cohen, *The penultimate form of approximation to normal extremes*](https://www.cambridge.org/core/journals/advances-in-applied-probability/article/abs/penultimate-form-of-approximation-to-normal-extremes/C090649B555017DCCD108B53318B38B4) · [Gomes, *Penultimate approximations in EVT*](https://link.springer.com/article/10.1023/A:1009971120468) · [Wu et al., *Sampling Matters in Deep Embedding Learning*, ICCV 2017](https://arxiv.org/abs/1706.07567) · [Patel et al., *Recall@k Surrogate Loss*, CVPR 2022](https://arxiv.org/pdf/2108.11179)

