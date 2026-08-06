# Independent cold review of PORT

Consultation ID: `ec5bbd01382d4b00`  
Provider/model: Claude Opus  
Caller: `sfora/emafactorial`  
Status: completed, exit 0  
Review prompt: `docs/opus_port_review_prompt_2026-08-06.txt`  
Review-prompt SHA-256: `2b6094e8e3f1020c141087554ec50fde2f9189312ac19397912e49787648314c`  
Native result truncated: false  
Protocol note: reviewer accidentally opened unrelated `RSPG_TASK.md`, declared it irrelevant, and opened no other repository evidence.

## Verdict: **DEAD**

Five independent defects are decisive; three of them are provable from the frozen formulation without any experiment. One sub-fact (PFML's numeric table) is genuinely BLOCKED, but it does not gate the verdict because the proposal itself declines to claim that row.

*(Note: I opened `RSPG_TASK.md` before recognising it describes an unrelated candidate. It has no bearing on PORT and I read nothing else in the repository.)*

---

## 1. What is algebraically correct

Reported first so the negatives are not confused with sloppiness.

- **Hosking–Wallis PWM closed form is correct**, including the ascending-order weight $(k-j)/(k-1)$. With $F(x)=1-(1-\kappa x/\alpha)^{1/\kappa}$, $\alpha_r=\alpha/[(r+1)(r+1+\kappa)]$, so $a_0/D = 2+\kappa$ and $\hat\xi = 2-a_0/D$, $\hat\sigma = 2a_0a_1/D$ recover $(\xi,\sigma)$ exactly. The exponential check verifies: $a_0=\sigma,\ a_1=\sigma/4,\ D=\sigma/2 \Rightarrow \hat\xi=0,\hat\sigma=\sigma$.
- **All five partials in §1.6 are correct**: $\partial\hat\sigma/\partial a_0=-4a_1^2/D^2$, $\partial\hat\sigma/\partial a_1=2a_0^2/D^2$, $\partial\hat\xi/\partial a_0=2a_1/D^2$, $\partial\hat\xi/\partial a_1=-2a_0/D^2$, and the PWM rank weights.
- **$E'(\xi)>0$ is correct**: $g(\xi)=\xi Le^{\xi L}-e^{\xi L}+1$, $g(0)=0$, $g'(\xi)=\xi L^2e^{\xi L}$ has the sign of $\xi$, so $0$ is a strict minimum.
- **Lemma 1's domination statement is correct** (its stated consequence is not — §4 below).
- Declustering bias bound $\le \log K/\beta_d$ (0.0578, 0.0289) is correct. The $L_{\max}$ table is arithmetically correct. $\Delta \propto M^{\xi}$ and "$10\times$ batch to halve at $\xi=-0.3$" is correct ($2^{1/0.3}=10.1$). The fp32 cancellation hazard on $D=a_0-2a_1$ is correctly identified. FLOP accounting is right to order ($3.15$ TFLOP for R50 fwd+bwd at $B{=}256$).
- **WEINCE is real and correctly characterised.** arXiv 2606.00262, *"When Softmax Fails at the Top: Extreme-Value Corrections for InfoNCE"*, Erol, Evren, Ozel, Morgan, Ryu, Zheng, 29 May 2026, ICML 2026. Verbatim: *"we treat $(\lambda_i,\hat\beta_i)$ as stop_grad statistics (no backprop through the tail fits)"*; endpoint fixed at $x_F=1$; benchmarks CIFAR-10/100, STL-10, ImageNet-32, Tiny-ImageNet under linear-probe/kNN; no CUB/Cars/SOP/In-Shop. The proposal's two stated disclaimers are accurate.
- **PFML is real** (Bhatnagar & Ahuja, CVPR 2025, arXiv 2405.18560) and evaluates **only** Cars-196, CUB, SOP — confirming "PFML does not evaluate In-Shop."
- No "PORTAL" EVT tail-risk DML method surfaced; that prior-art item appears absent.

---

## 2. Decisive defect A — the gradient is provably a 2-parameter affine-in-rank reweighting of *observed* order statistics

$\hat q$ depends on the batch only through $(u, a_0, a_1)$. Both PWM are linear in the order statistics with rank weights $\partial a_0/\partial v_{(i)} = 1/k$ and $\partial a_1/\partial v_{(i)} = (i-1)/(k(k-1))$. Therefore, for **any** function of $(a_0,a_1)$:

$$\frac{\partial \hat q}{\partial v_{(i)}} \;=\; A(\hat\xi,L,k) \;+\; B(\hat\xi,L,k)\cdot\frac{i-1}{k-1},\qquad i=1..k$$

**exactly affine in rank.** I verified this numerically at nine $(\xi,L,k)$ points; residual against a straight line is $2\text{–}13\times10^{-16}$, machine precision.

§1.6 claims: *"PORT's per-rank gradient weights are **sign-mixed and determined by the fitted tail, not by rank**… Every occupied weighting is single-signed and rank-determined… This is directly measurable and is the cleanest empirical handle on whether the claimed mechanism is the one operating."*

**This is false.** The weights are exactly rank-determined — a straight line — and the fitted tail supplies only the two scalars $(A,B)$. Uniform-over-top-$k$ (CVaR) is the $B=0$ member of PORT's own family. The stated fingerprint is not a fingerprint; the mechanism cannot produce any profile outside a 2-D affine family that already contains an occupied method.

**Consequently PORT does not supervise an unobserved population order statistic.** There is no random variable over unseen identities anywhere in the loss or its gradient. $\hat q$ is a deterministic smooth functional $\Phi(v_{(1)},\dots,v_{(k+1)})$; equivalently $\hat q = u + a_0\,h(a_1/a_0)$, since $\hat\xi=(1-4\rho)/(1-2\rho)$ and $\hat\sigma=2a_0\rho/(1-2\rho)$ with $\rho=a_1/a_0$. That is *mean excess × a function of one L-moment ratio* — an adaptive margin on observed order statistics. Population size $N$ enters only through $A$, $B$, and the scalar $\sigma(\beta(\hat s^--\lambda))$.

Note further that the rank weights are **scale-free in $a_0$**. So $\hat\sigma$ affects the gradient *only* through the hinge-saturation scalar $\sigma(\beta(\hat s^--\lambda))$ — a per-anchor loss weight. D5(ii)'s "that coupling is the mechanism" is wrong in the regime where gradients actually flow; the $\hat\sigma$ channel *is* per-anchor adaptive weighting.

---

## 3. Decisive defect B — $N$ is inert at the operating point; the object is the fitted right endpoint

At $\xi=-0.3$, $\sigma=0.03$, $u=0.40$:

| $N$ | 127 | 500 | **2491** ($\Lambda$-cap) | 11,316 | $10^9$ |
|---|---|---|---|---|---|
| fraction of $u\to$ endpoint | 64.6% | 76.5% | **85.5%** | 90.8% | 99.7% |

At the implemented SOP/In-Shop operating point $\hat q$ is already 85.5% of the way to $u+\hat\sigma/|\hat\xi|$. Moving $N$ from 2491 to $10^9$ — six orders of magnitude of "identity population" — moves $\hat q$ by 14% of that span, $\approx 0.014$ cosine units.

So the supervision object is, numerically, **the estimated finite right endpoint of the negative-similarity law**. That is precisely WEINCE's object (endpoint shortfall, Weibull domain, bounded cosine), which WEINCE fixes at $x_F=1$. The residual distinction collapses to: *estimate the endpoint instead of fixing it at 1, and differentiate through the estimate*. The proposal's own residual-risk paragraph anticipated a narrowing to "out-of-sample identity-population extrapolation for zero-shot DML." Neither "out-of-sample" (defect A) nor "identity-population" (this defect) survives.

Also: the schedule extrapolates to $N=e^{6.434}/0.25=2491$, not $N_{\text{test}}=11{,}316$. The motivating arithmetic in §2 uses the latter.

---

## 4. Decisive defect C — the dominant force is repulsion on the *threshold* class, with net attraction on the hard tail

Finite differences on an exact-GPD profile ($\xi=-0.3$, $\sigma=0.02$, $k=31$, $L=6.434$):

$$\partial\hat q/\partial u = +11.77,\qquad \partial\hat q/\partial v_{(1)} = +0.72,\qquad \partial\hat q/\partial v_{(k)} = -1.42$$

At exact population PWM values: $w_u=+9.84$, $\sum_{i\le k} w_i = -8.84$, total $=1.000$ (location equivariance, as it must be). Ranks 1–11 repulsive, **ranks 12–31 attractive**. The single largest force in the entire negative term is a repulsion on $v_{(k+1)}$ — the 32nd-hardest class, the *least hard* one retained.

Conditioning worsens with $\hat\xi$, and $\hat\xi$ is inside the admissible clamp band throughout:

| $\hat\xi$ | $-0.95$ | $-0.6$ | $-0.3$ | $0.0$ | $+0.2$ |
|---|---|---|---|---|---|
| $w_u$ | $+2.13$ | $+4.52$ | $+9.84$ | $+23.09$ | $+40.31$ |
| $\sum_{i\le k}w_i$ | $-1.13$ | $-3.52$ | $-8.84$ | $-22.09$ | $-39.31$ |

Monte Carlo gives $\mathrm{sd}(\hat\xi)=0.25$ at $k{=}31$ and $0.34$ at $k{=}15$ — comparable to the full clamp width of 1.25. So the cancellation ratio swings between $\sim2$ and $\sim40$ step to step.

D6 concedes *"some ranks receive an attractive contribution."* That understates it by an order of magnitude: ~92% of the total weight mass is cancelling, two-thirds of the retained tail is attractive, and C7 ("zero every attractive component") therefore does not ablate the mechanism — it replaces the estimator with a different one. C7 has no pre-registered threshold.

---

## 5. Decisive defect D — the mechanism is provably inert on CUB/Cars while gains are forecast there

Monte Carlo, $M{=}63$, $K{=}4$, $\kappa{=}0.25$, $L_{\max}{=}3.219$ (4000 draws):

- mean $\hat q - v_{(1)} = \mathbf{-0.0000}$
- the floor $\max(\hat q, v_{(1)})$ **binds on 45.6% of anchors** — the EVT block receives exactly zero gradient there (`max` is a hard selector)
- effective extra margin $\mathbb{E}[\max(\hat q - v_{(1)},0)]$: $+0.0046$ at $L_{\max}$ vs $+0.0011$ at $L_0$ — a net $+0.0035$ cosine units, i.e. $0.18$ in the softplus argument at $\beta{=}50$

SOP is better ($+0.0350$ vs $+0.0016$; floor binds 18.3%) but the CUB/Cars rows of the frozen forecast attribute $+0.013$ and $+0.014$ R@1 to a mechanism whose mean increment is zero, with C1 already accounting for $+0.003$. §7.5 says the mechanism "barely engages" on CUB/Cars; §5 forecasts gains there anyway. These cannot both stand.

---

## 6. Decisive defect E — §2's motivating magnitude is computed at a tail index the proposal itself rules out

§2 claims a gap of **0.07–0.13** cosine units on SOP, via $\Delta\to\sigma\log(N/M)$ — the $\xi=0$ branch. §1.3 insists *"cosine similarity is bounded above by 1, so the true tail is Weibull-domain, $\xi<0$ — precisely our regime,"* and §1.6/§3 operate at $\xi\approx-0.3$.

At $\xi=-0.3$, the proposal's own formula $\Delta=(\sigma/\xi)(\kappa M)^\xi[(N/M)^\xi-1]$ gives $0.013$–$0.026$. The implemented schedule on an exact-GPD profile gives $\hat q - v_{(1)} = +0.0154$. The headline is inflated **5–8×**, and at the true operating point the increment ($0.015$) is *below* the softplus transition half-width $1/\beta = 0.02$. "Comparable to the entire MS hinge band… not a second-order correction" does not survive.

Separately, $\hat\sigma = O(0.015\text{–}0.03)$ is asserted as *"measured"* with no measurement reported; it is the only quantity converting the theory into cosine units.

---

## 7. Further false, non-executable, or underdefined claims

**D1 is not executable as frozen.** At $f\equiv\text{const}$: $a_0=a_1=D=0$, so the guard weight $\mathrm{sigmoid}((D/a_0-0.05)/0.02)=\mathrm{sigmoid}(0/0)=\text{NaN}$ and $\hat\sigma=2a_0a_1/D=0/0=\text{NaN}$; $w\cdot\text{NaN}+(1-w)\cdot 0=\text{NaN}$. The guard is undefined at the exact degeneracy it guards. Also $v_j = 1+\log K/\beta_d = 1.058 > 1$, so the declustered "similarity" exits the cosine range and D1's "$\hat s^-=1$, the maximum on the reachable set" is wrong by 0.058.

**D4 is false.** $a_1$'s PWM weight on the largest excess is exactly $(k-k)/(k-1)=0$. The spike profile $e=(e_{\max},0,\dots,0)$ therefore gives $a_1=0$, hence $\hat\sigma=2a_0a_1/D = \mathbf{0}$ exactly, hence $\hat q=u$ and $\hat s^-=v_{(1)}$: PORT $\equiv$ hard mining, increment zero. D4 asserts *"The estimator penalizes exactly this cliff-shaped profile."* It does the opposite. (Descent from the operating point pushes $\rho\uparrow$ toward the flat profile and the $\xi_{lo}$ clamp instead — $\hat q - u$ is non-monotone in $\rho$ with an interior local minimum near $\rho\approx0.33$ — but the frozen claim is false as written.)

**Lemma 1's consequence is false.** *"There exists no parameter setting that lowers PORT by making the estimator lie."* Domination bounds the loss below by $L_{\text{hard}}$; it says nothing about the increment, which is the *only* thing the EVT machinery contributes and which the shape channel above zeroes exactly.

**The $\xi$ upper clamp bounds nothing physical.** At $\hat\xi=+0.30$, $L=6.434$: $E=19.64$, so the admitted increment is $19.6\hat\sigma \approx +0.59$ cosine units at $\hat\sigma=0.03$ — potentially exceeding the cosine range, in a method whose premise is that cosine is bounded by 1. Nothing enforces $\hat q\le 1$. Empirically the p99 increment on SOP is $+0.164$ — 8× the softplus half-width — in a simulation where **all anchors are statistically identical**. That falsifies §2's third consequence: the anchor-dependence of the increment is estimation variance, not "heavy-tailed crowded neighbourhood" signal.

**Declustering is inert.** With vs without the $\beta_d{=}24$ LSE: $\hat\xi$ $-0.278$ vs $-0.266$; $\hat\sigma$ $0.0367$ vs $0.0372$; increment $+0.0323$ vs $+0.0334$; floor 18.3% vs 18.7%. Indistinguishable. And *"a constant absorbed into the threshold"* is false — the bias is a per-class function of within-class spread, ranging over $[0,0.058]$ at $K{=}4$, i.e. **2–4× the entire scale parameter $\hat\sigma$ being estimated**. It is a similarity-dependent perturbation of the very order statistics being fitted. "$M=P-1$ approximately independent block maxima" is also unestablished: CUB draws $P{=}64$ of $C_{\text{train}}{=}100$ classes without replacement — a 64% sampling fraction, so "an identity population the batch never contains" contains 64% of it, and $L_{\max}$ is capped at exactly $C_{\text{train}}$.

**Threshold choice sits at the worst point of the POT bias–variance tradeoff.** $\kappa=0.25$ puts $u$ at the 75th percentile of 63/127 values — inside the bulk, where second-order regular-variation error is first-order and is then amplified by $e^{\xi L}$. $k=15/31$ gives $\mathrm{sd}(\hat\xi)=0.34/0.25$. No threshold-bias term is bounded; no mean-excess or threshold-stability diagnostic is pre-registered. §7.3 concedes this for CUB only; SOP's 0.25 is where the only claimable result lives.

**$D/a_0 \in [0.34,0.67]$ is wrong.** $1/(2-\xi)$ at $\xi=0.3$ is $0.588$, not $0.67$. Empirically $D/a_0$ reaches $0.166$, below the claimed floor. (The guard threshold at $0.05$ is genuinely far away, so this is cosmetic.)

**Mixtures and transfer point the wrong way.** The per-anchor negative law is a mixture over identity types; at $\kappa=0.25$ no single GPD domain holds. Worse, §2's transfer argument is asserted *against its own dynamics*: training actively thins the seen-identity negative tail while unseen identities receive no such pressure, so $(\hat\sigma,\hat\xi)$ systematically **under**-states the unseen tail — more so the better PORT works. §7.2 notices the symptom and labels it saturation.

**§6 and §7.1 contradict each other on contamination.** §6: *"Compliance: no test data… Clean."* §7.1: fit $\hat\xi,\hat\sigma$ on **test** identities and use the result as a go/no-go gate *before* the sweep. That is selection conditioned on test-split statistics.

**Error-bar semantics are inconsistent.** §6 states seed $\sigma\approx0.5$–$0.9$ pt (an SD) and the table's $\pm0.006$ on CUB matches that SD; the "$\approx3.3\sigma$" SOP arithmetic treats $\pm0.003/\pm0.002$ as standard errors of 5-seed means. Same symbol, two meanings. And F2 requires $\Delta$ separations of $0.004$ on CUB/Cars while §6 states $\Delta<0.010$ there is uninterpretable at 5 seeds — F2 is not resolvable by the stated design.

**Cost claims:** $3\cdot B\cdot P = 0.098$ MFLOP at $P{=}128$, not $0.05$; "+32 KB" undercounts the $B\times P$ matrix ($256\times128$ fp32 $=131$ KB). Both immaterial. The C8 claim *"~1/10 the memory"* is wrong by ~16×: 2491 XBM slots $\times$ 512 fp32 $=5.1$ MB against a claimed 32 KB — and 2491 slots is far below XBM's standard SOP configuration, so C8 undersizes the comparator.

---

## 8. Controls: what is and is not isolated

| Confound | Isolated? |
|---|---|
| Extrapolation vs tail weighting | **No.** C6 still fits $\hat\sigma$ from PWM, so it remains an affine-in-rank weighting; on the full exponential branch ($\hat\xi=0\Rightarrow\hat\sigma=a_0$), $\hat q = u + L(\mathrm{CVaR}_\kappa - u)$ — *exactly scaled CVaR*. C6 and C4 then differ only by the scalar $L$ and the $u$-channel. The control that would isolate extrapolation — per-anchor $(A,B)$ affine rank weights chosen without EVT — is absent. |
| Hard-negative mining | **No.** PORT replaces MS's log-sum-exp over all negatives with a softplus of one scalar, deleting the multiplicity term. C1 is the correct reference; the headline $\Delta$ is quoted against C0, inflating the attributed effect ~25%. |
| Adaptive margin / temperature | **No.** $\hat\sigma$ acts on the gradient only through $\sigma(\beta(\hat s^- -\lambda))$, a per-anchor loss weight — i.e. it *is* adaptive weighting (§2 above). |
| Batch composition | **No.** $P,K$ are set per dataset family (64×4 vs 128×2), changing $k$, $M$, $L_0$ and the declustering bias — so F2's "signature" is confounded with a per-family configuration change, contradicting "no per-dataset tuning." |
| Queue size | Partially; C8 undersized and its memory claim wrong by ~16×. |

**Falsifier coverage is thin.** C1, C3, C4, C7, C10 carry **no** pre-registered threshold — including C4 (CVaR), which my derivation shows is the $B=0$ member of PORT's own weight family and is therefore the single most decisive control, and C3/C10, which are the only probes of the anchor-dependence claim that §8 shows is estimation noise. F6 fires the upper clamp on 0.1% (SOP) / 2.2% (CUB) against a 20% threshold, and F7 sets >3% wall-clock against a <1% forecast — both non-binding by construction.

---

## 9. Baseline / frontier / BLOCKED

By the proposal's own accounting PORT crosses no Lane-A frontier (CUB $-0.030$, Cars $-0.029$, SOP $-0.008$, In-Shop $-0.005$); the only real claim is a delta over a self-reported MS reproduction. The conditional PFML-base row is explicitly unclaimed and correctly so.

**BLOCKED sub-fact:** PFML's exact numbers (0.734 ± 0.003 / 0.927 ± 0.003 / 0.829 ± 0.002) and its undisclosed recipe. CVF returned 403 to me as it did to the proposal's author, and the arXiv PDF exceeds my fetch limit. Cars 0.927 at R50/512/224 sits at the very top of the published range and I could not confirm it. This is the only genuinely unresolvable item, and it does not gate the verdict.

---

## 10. What survives

The PWM algebra, the five §1.6 partials, the $E'>0$ monotonicity, the domination inequality itself, the $M^\xi$ scaling, the fp32 hazard, the WEINCE and PFML characterisations, and the honesty of §7 and of the frontier arithmetic. The engineering is careful and the self-criticism is real.

What does not survive is the supervision object. PORT's gradient is a two-parameter affine-in-rank reweighting of observed order statistics whose extrapolation target is, at the operating point, the fitted right endpoint — WEINCE's object with the endpoint estimated rather than fixed. The population size $N$ that names the method moves that target by 14% across six orders of magnitude. The claimed "fingerprint" that was to distinguish the mechanism empirically is provably a straight line containing CVaR. The mechanism is inert on half the benchmarks it forecasts gains on, its motivating magnitude is inflated 5–8× at its own tail index, and its gradient's largest single force is a repulsion on a class that is by construction not hard, against net attraction on the hardest 20.

Wrapper novelty is not at issue here; the object is occupied and the mechanism does not do what the one-sentence claim says. **DEAD.**

Sources: [WEINCE, arXiv 2606.00262](https://arxiv.org/abs/2606.00262) · [WEINCE full text](https://arxiv.org/html/2606.00262v1) · [PFML, arXiv 2405.18560](https://arxiv.org/abs/2405.18560) · [PFML, CVPR 2025 (403 to me)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [PFML CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/33305) · [Extreme Value Machine](https://arxiv.org/pdf/1506.06112) · [Extrapolation in Statistical Learning with EVT](https://arxiv.org/html/2605.01909v1)

