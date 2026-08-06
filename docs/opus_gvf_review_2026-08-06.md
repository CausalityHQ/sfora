## VERDICT: **DEAD** — earliest failed section **§2.2**

**Decisive mechanism-level reason:** GVF's entire added mechanism is the amplitude $\theta_k$ of the shared frame, and §2.2 derives its behaviour from sample→proxy attraction alone. PFML's verbatim energy — which §0 reproduces correctly and GVF inherits byte-for-byte — also contains **attraction between same-class proxies with no same-class repulsion** ($\Psi_{j,\text{att}}$ includes $\sum_k\psi_{\text{att}}(r,p_{j,k})$, and $U$ evaluates $\Psi_j$ *at* every $p_{j,k}$; confirmed at the primary source). At $M=15$ that is 225 same-class proxy–proxy ordered terms against $2n_jM=120$ sample–proxy terms under the proposal's own 32×4 sampler, and the proxy–proxy pairs sit at distance $\approx\theta$ where $\psi_{\text{att}}$ is ~225× stiffer than at the sample distance. The frame amplitude is therefore crushed by a term §2.2 never writes down.

Exact scalar algebra on the sphere (all distances depend only on $\phi,\theta_k,a_k$; $g_k\!=\!1$, i.e. best case for GVF), with a **genuinely live** direction $\mathbb{E}[a_1^2]=0.287$ against the derived bar $0.200$:

| $\alpha$=3, $\cos\phi$=0.7 | $\theta_1^*$ per §2.2 (samples only) | $\theta_1^*$ under PFML's verbatim $U$ |
|---|---|---|
| $\delta$=0.1 | 0.645 rad | **0.000 rad** |
| $\delta$=0.2 | 0.650 rad | **0.010 rad** |
| $\delta$=0.3 | 0.615 rad | **0.150 rad** $=\arcsin(\delta/2)$ |

Insensitive to the one undisclosed knob that could rescue it: $\theta_1^*=0.010$ at 4, 8, 16 and 32 images/class. So over two-thirds of PFML's disclosed range $\delta\in[0.1,0.3]$ a live direction lands **below GVF's own F2 kill line of 0.02 rad**, and at $\delta$=0.3 it saturates a $\delta$-set ceiling covering 19% of the residual it models — 4.7% after §1.5's $\beta\!\to\!0.25$ anneal, which governs the epochs that produce the deployed weights. At $\theta\!\to\!0$ all $2K$ proxies coincide with $\hat\mu_j$, so GVF degenerates to PFML with one proxy of weight $M$ — below C1, not equal to it.

## Second, independent kill (same section)

§2.2's threshold is **per frame direction**: $a_k^2>1/(\alpha+2)$, with $a_k=\langle n,n_k(j)\rangle$, $n$ a *unit* residual direction. I verified the closed form against exact spherical distances to 5–6 digits, and the $\phi\!\to\!0$ limit is exactly $1/(\alpha+2)$. But $\{n_k\}$ stay orthonormal through tangent projection (max $|\langle n_k,n_l\rangle|=1.9\times10^{-3}$ at $d$=512), so $\sum_k\mathbb{E}[a_k^2]\le1$ and **at most $\alpha+2$ directions can ever escape**. $K$=7 — chosen in §1.3 *only* so $M=2K+1=15$ matches PFML — requires $E_7>7/(\alpha+2)$: impossible for every $\alpha\le4$, and needs 87.5–100% of within-class residual energy in 7 of 511 dimensions at $\alpha\in\{5,6\}$.

The pretest built to catch this cannot: F0 thresholds the **aggregate** $E_7$ at the **per-direction** bar. Every realistic spectrum passes while 0–1 directions are live:

```
power law i^-1.0   E_7=0.380  F0 PASS   live k = 0
power law i^-2.0   E_7=0.920  F0 PASS   live k = 1
exp decay 0.8^i    E_7=0.790  F0 PASS   live k = 0
hard rank-7        E_7=0.409  F0 PASS   live k = 0
```
Pooled-covariance estimation at ~58.6 images/class ($d/n\approx0.087$, MP edge 1.71×) biases $E_7$ further **upward**. F2 is equally blind — six dead directions plus one at 0.15 gives mean 0.021, passing. The proposal names this as its load-bearing uncertainty (`:289`) but measures it against a bar ~5× too lenient and a statistic that green-lights unconditionally, so the caveat does not rescue the mechanism.

## Other confirmed defects

- **§2.3 self-certification fails.** $\psi_{\text{rep}}$ is a *short-range hard core*: constant $\delta^{-\alpha}$ for all distances $>\delta$, gradient only inside $\delta$. The $-\mathbb{E}_{\text{neg}}[\omega_{\text{rep}}n'n'^\top]$ term supplying the minus sign is identically flat except for negatives within $\delta$ of a proxy. "Signal-aligned frames are maxima… it cannot converge to the shortcut" (`:153`) does not follow; the globally active attraction term rewards any high-residual-variance direction regardless of between-class content. Writing a sparse hard-core term as an expectation over negatives materially overstates it. (The same clipping *does* legitimately justify dropping repulsion at $\theta$=0 — a correct result the proposal reaches without stating the reason.)
- **§2.1 capacity argument is not a function-class proof and is self-refuting.** 14 free proxies/class = 7,168 parameters vs 58.6 images × 512 = 29,696 numbers — 4× too few to "memorize the embedding of every training image" (`:124`). And 716,800→3,591 relocates class-indexing capacity rather than rationing it; the 25M-parameter backbone retains it. C8 (frozen unused tensor) does not test this. Per the decisive finding, PFML's free proxies are themselves contracted into a $\delta$-ball, so the R2 spreading mechanism the argument assumes does not exist.
- **Per-class adaptation is inert.** $c_k=\langle\hat\mu_j,\hat w_k\rangle\sim\mathcal N(0,1/512)$ ⇒ $g_k(j)\in[0.995,1]$ always. The gate is pole-free but never fires, so every class receives literally identical displacement geometry — the tie is stronger than §1.3 presents.
- **§5's frozen table contradicts its own probabilities.** Table +1.5 over C1 (se 0.179) implies P(Δ≥0.5)≈1.000 and P(CUB≥73.9)≈0.87; stated values are 0.55 and 0.28. Reconciling P=0.55 with mean +1.5 needs sd ≈ 8 points; P(CUB≥73.9)=0.28 implies a median of 73.4–73.7, not 74.1. The stated probabilities are mutually coherent (P(both)=0.18); it is the frozen number of record that is not the author's belief.
- **C9 does not de-confound $M$ from $K$.** Duplicated pad proxies sit at zero mutual distance (clipped constant) and simply multiply one location's attraction weight; that is not $M$ distinct proxies.
- **§1.3/§1.8 do not execute on SOP.** $M=2K+1$ is always odd; "unsigned $K$=1" appears nowhere in the boxed construction or the pseudocode. Separately, `qr(Omega).Q` does not enforce §1.2's positive-diagonal convention — harmless for the signed case (sign is a gauge, $p_{k,\pm}$ swap) but the unsigned variant is genuinely ill-defined without it.
- **Planned order is inadmissible.** Screening on In-Shop first tests a dataset where GVF self-forecasts 92.4 against a 93.0 reference and explicitly claims no crossing is even *claimable* (`:235`). The screen kills the proposal before reaching CUB/Cars, the only arms where it forecasts crossings. Admissible order is CUB → Cars; the proposal does not flag the conflict.

## Correct subcomponents (preserve independently of the verdict)

1. **§0's reduction is reproduced exactly.** $\psi_{\text{att}},\psi_{\text{rep}},\Psi_j,U$, $M$=15/15/2, lr 5e-4, 100× proxy lr, 200 epochs, $\alpha\!\in\!\{0..6\}$, $\delta\!\in\![0.1,0.3]$, 73.4/92.7/82.9 all match the primary source. §7's list of undisclosed items is accurate and complete.
2. **§2.2's expansion and closed form are algebraically correct** (verified to 5–6 digits; $\phi\!\to\!0$ gives exactly $1/(\alpha+2)$). The ±-pair first-order cancellation is real, and the mandatory nonzero init follows correctly.
3. **The pole-free gate and the rejection of parallel transport** for its $1/(1+\langle o,\mu\rangle)$ blow-up is a genuinely correct geometric point.
4. **D4 is sound**: QR orthonormality makes redundancy impossible by construction, and the SoftTriple contrast is accurate — without its regularizer SoftTriple does hold a set of similar centres.
5. **§6 trap 2 is correct and non-obvious**: $Q(\Omega D)=Q(\Omega)$ for positive diagonal $D$, so column renormalization is a legitimate gauge fixing that bounds $\partial Q/\partial\Omega\sim1/\|\Omega\|$ while leaving $Q$ unchanged. Traps 1 and 3 are also correct.
6. **All arithmetic checks out**: 122/87 params-per-image, 0.61, ~200×, 11,296×; se 0.224/0.190/0.126; thresholds 73.85/93.08/83.15; $z$=3.1/2.6; the SOP non-crossing concession is correct.
7. **The SFT overlap does not break the novelty claim.** I read the ECCV camera-ready: $A$ is computed in closed form by Gram–Schmidt + Rodrigues from the two class means, is nonparametric, acts on samples for feature augmentation, and learns no shared subspace. §3.5's distinction holds as stated.
8. **§6's contamination/selection audit is unusually honest** — the +0.5 to +0.9 best-epoch inflation being the same order as the claimed gain is correct and important.

**Novelty, resolved:** SFT is not the break, but §3 is thinner than it claims. SFT names **Yin et al., FTL (CVPR 2019)** as "the most similar work to ours" — "the covariance between all classes is the same, thus shared… each feature is the summation of the class-dependent mean and a class-independent variance" — which is GVF's §2.1 premise in a learned deep setting, and it is uncited. SVDL and the shared intra-class-variant-basis line in single-sample face recognition are likewise absent. GVF's object remains a **reparameterization of an existing supervision object plus a train-time scaffold**, not a new supervision object or action. A demerit, not the kill.

**Uncertainty:** PFML's per-dataset $\alpha,\delta$ are genuinely undisclosed, so I report the decisive result across the whole disclosed grid rather than at a point. My simulation fixes $\mu$ and varies one $\theta_k$ with the rest at init; a co-adapting backbone moves samples but cannot change the 225× static stiffness ratio, and the sampler sweep confirms this. I did not verify PA+DADA's In-Shop 93.0, AdvRF, or SFT's Sec 4.2 and supplement. This is decidable on the disclosed material — not BLOCKED.

Sources: [PFML full text](https://arxiv.org/html/2405.18560v2) · [Spherical Feature Transform, ECCV 2020 camera-ready](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640409.pdf) · [SoftTriple](https://arxiv.org/pdf/1909.05235) · [Intra-class variant basis shared across subjects](https://mdpi.com/1424-8220/15/1/1071/htm) · [Low-rank class-specific + intra-class variant dictionary](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142403)
