# Independent Opus review: DOIR (Pass 28)

Consultation ID: `c97fba8c9fa94e66`

I have what I need. Here is the review.

---

# Independent adversarial review — DOIR (Pass 28)

## VERDICT: **DEAD**

**Earliest protocol gate it fails:** the pre-GPU mechanism/degeneracy gate — §2's degeneracy audit (D1–D5) jointly with §4's control-validity check. Both fail on paper. No GPU is required to reject.

**Single most decisive mechanism-level reason:** *the barrier's global minimum is reachable by moving proxies alone, so $\mathcal R$ imposes no constraint on the deployed embedding.* I froze $z$ (so R@1 is mathematically pinned — not approximately, exactly) and ran the proposal's own objective $\mathcal L_{\text{base}}+0.05\,\mathcal R$ on the proxies only, at the proposal's own proxy lr of $10^{-2}$:

| iter | $\mathcal R$ | erank$(F)$ | base loss | mean $q_y$ |
|---|---|---|---|---|
| 0 | 1.1941 | 154.9 | 2.07392 | 0.767 |
| 80 | 0.01297 | 504.4 | 0.000030 | 1.0000 |
| 200 | **0.00002** | **512.0** | 0.000027 | 1.0000 |

$\mathcal R$ reaches its global minimum ($\tilde F=I$), erank reaches the full 512, and the base loss *improves* — with the deployed model bit-identical. Every signature DOIR proposes to measure is produced by a path that cannot move R@1 by one image. For contrast, the same proxies-only descent with $\lambda=0$ drives erank *down* to 38 and $\mathcal R$ *up* to 2.30 — so the entire erank effect is attributable to the barrier acting on discarded parameters.

The root cause is structural: **$\ell_i$ is exactly a function of the single scalar $q_{i,y_i}$**, while $\mathcal R$ depends on the whole $q_{i,\cdot}$ and $m_{i,\cdot}$. This is enabled and amplified by the proposal's own deliberate recipe choices — proxy lr 100× the backbone's, and explicitly **no proxy weight decay** (§1.4) — and it is not covered by D1–D5.

---

## 1. Re-derivation: what is correct, and what the rank actually is

**Correct, verified independently:**

- $\nabla_z\log q_c=s(m_c-\bar m)$. From $\nabla_z\pi_j=s\pi_j(p_j-\bar m)$, $\nabla_z q_c=sq_c(m_c-\bar m)$. ✓
- $F_i=s^2\sum_c q_c(m_c-\bar m)(m_c-\bar m)^\top$ is the exact Fisher of the categorical $c\sim q(\cdot|z)$, and $\sum_c q_c(m_c-\bar m)=0$ as a score must. ✓
- The pairwise form $s^2\sum_{c<c'}q_cq_{c'}(m_c-m_{c'})(m_c-m_{c'})^\top$ is the standard weighted-variance identity. ✓
- $\mathcal R\ge0$ with equality iff $\tilde F=I$ (Jensen on $\log$ under $\sum\tilde\lambda_k=d$). ✓
- $\mathcal R_{\max}=4.5939$ at $d{=}512,\varepsilon{=}10^{-2}$, rank-1. The stated "≈4.59" is right. ✓
- $G=\frac{d}{\operatorname{tr}F}(A-\frac{\operatorname{tr}(A\tilde F)}{d}I)$ and $\operatorname{tr}(GF)=0$ — I re-derived both; correct. But this is just **Euler's theorem for a 0-homogeneous function**; §1.4's framing ("not an assumed-harmless normalization") oversells a triviality, and the consequence it *doesn't* draw is that $\operatorname{tr}F$ is left completely unconstrained. Measured, it collapses from $1.45\times10^{2}$ to $1.03\times10^{-3}$ (see §5).

**Rank — the prompt's hypothesis is half right, and the proposal is right where it matters.**

The per-sample $\Pi_iF_i\Pi_i$ has rank exactly $C-1$: I measured **99** on a CUB-shaped configuration ($C{=}100$, $K{=}15$, $d{=}512$). If that were the whole story, $\mathcal R$ would have a hard floor of $3.4066$ nats (I computed the rank-limited optima: $r{=}99\to3.407$, $r{=}98\to3.417$, $r{=}31\to4.167$), leaving only $4.59-3.41=1.18$ nats of reachable range instead of the 4.59 D4 assumes.

**But the batch average is not rank-limited.** I measured $\operatorname{rank}(F)=512$ for $B{=}90$ in a converged-like regime. So "a $C$-class Fisher can never be full rank in $d{=}512$" is **false for the object DOIR actually uses**, and the fixed-zero-eigenvalue attack does not land. Credit to the proposal.

The residue is a real problem for D2, though: the extra $512-99=413$ directions come *entirely* from across-sample variation of $m_{ic}$ within each class — i.e. sample-dependent within-class proxy responsibilities. That is precisely the channel D2 claims to have neutralized "by construction." D2 neutralizes only *static* redistributions that preserve every $m_{ic}$; it does not neutralize the sample-indexed variation that fills 81% of the spectrum.

**Lemma 1 — the span claim holds; the mechanism does not.**

The span claim is correct, and more strongly than stated. Writing $\kappa=e^{-s\delta}$ and $Z=\sum_{j'}e^{s\langle z,p_{j'}\rangle}$, the numerator is $\kappa Zq_y$ and the denominator $\kappa Zq_y+Z(1-q_y)$, so

$$\ell_i=-\log\frac{\kappa\,q_{y}}{\kappa\,q_{y}+1-q_{y}}$$

— a function of $q_{i,y_i}$ alone (verified numerically to $|{\rm diff}|=0.00\times10^{0}$). Hence $\nabla_z\ell_i\parallel\nabla_z\log q_{y}$, and $u^\top F_iu=0\Rightarrow u^\top\nabla_z\ell_i=0$. ✓

Two failures follow:

1. **The absorbing set is generically empty.** $u^\top F_iu=0\ \forall i$ requires $u^\top m_{ic}=u^\top\bar m_i$ for all $i,c$, which for generic responsibilities forces $u^\top p_j\equiv\text{const}$ across all $CK$ proxies. With $CK=1500$ (CUB), $1470$ (Cars), $22{,}636$ (SOP) — all $\gg d=512$ — no such $u$ exists almost surely. The set the method is built to escape does not exist under the proposal's own configuration.

2. **The decay is quantitatively nil.** Lemma 1's dynamical content is "under weight decay $\gamma$ that component decays as $e^{-\eta\gamma t}$." With the proposal's own AdamW, lr $10^{-4}$, wd $10^{-4}$ (decoupled decay rate $\eta\gamma=10^{-8}$/step):

   | dataset | steps (200 ep, $B{=}180$) | decay factor | idle direction removed |
   |---|---|---|---|
   | CUB | 6,600 | 0.99993 | 0.0066% |
   | Cars | 9,000 | 0.99991 | 0.0090% |
   | SOP | 66,200 | 0.99934 | 0.066% |

   An idle direction is not driven to zero; it is *neutral*, retaining its initialization. "Absorbing and self-reinforcing" is off by roughly four orders of magnitude.

**Lemma 2 — two errors.** With the trace normalization carried through, $\frac{\partial\mathcal R}{\partial\lambda_j}=-\frac{1}{\operatorname{tr}F}\big[\frac{1}{\tilde\lambda_j+\varepsilon}-\frac1d\sum_k\frac{\tilde\lambda_k}{\tilde\lambda_k+\varepsilon}\big]$.

- "bounded below by $\tfrac1{d(1+\varepsilon)}$ times the largest": since $\tilde\lambda$ ranges up to $d$, the true ratio is $\varepsilon/(d+\varepsilon)\approx2\times10^{-5}$, not $1/(d(1+\varepsilon))\approx1.9\times10^{-3}$ — ~100× off.
- "the flow is pushed out of it at a rate that does not vanish as $\lambda_k\to0$": **false as a statement about the parameter flow.** $\lambda_u$ is quadratic in the $u$-components, so $\partial\mathcal R/\partial\theta=0$ *exactly* at an idle point. It is a saddle, not a repeller. What survives is a linearized instability whose rate rewards whichever parameter supplies $u$-components most cheaply — and that is the proxies. **Lemma 2, stated correctly, predicts the degeneracy rather than the mechanism.**

---

## 2. The five degeneracy arguments, tested

**D1 (temperature) — sound. Preserve.** $s$ is fixed, $z$ and $p$ normalized, and $\mathcal R(\alpha F)=\mathcal R(F)$ exactly by 0-homogeneity.

**D2 (within-class scatter) — overstated.** True for redistributions preserving every $m_{ic}$; false for the sample-indexed within-class responsibility variation that supplies 413 of 512 directions (§1). EXP B settles it empirically: proxies alone carry $\mathcal R$ to 0.

**D3 (parked junk proxies) — two arithmetic errors, both widening the parking zone.**

- The stated criterion "$\lambda\gtrsim\varepsilon\operatorname{tr}F/d$" is right, but it reduces to relative mass $\lambda/\operatorname{tr}F\ge\varepsilon/d=1.95\times10^{-5}$, **not** the stated "relative mass $\ge10^{-2}$." A factor-512 error.
- "$q_cq_{c'}\lesssim e^{-2s\Delta}$" assumes both classes are displaced by $\Delta$. For a junk proxy $\Delta$ below the best logit with $q_c\approx1$, the bound is $\lesssim Ke^{-s\Delta}$.

Conservatively correcting only the first: $e^{-32\Delta}\ge1.95\times10^{-5}\Rightarrow\Delta\le0.339$, versus the claimed $\Delta\le0.072$ — **4.7× wider**. Including $K{=}15$ pushes it to $\Delta\le0.42$. "Parking is arithmetically excluded" does not follow. Separately, the threshold is measured against $\operatorname{tr}F$, which I measured collapsing five orders of magnitude, so the absolute bar falls with it. D3 also directly contradicts Lemma 2: D3 says near-zero directions don't count; Lemma 2 says the barrier pushes hardest exactly there. Both cannot be the operative story.

**D4 (uniform $q$) — conclusion sound, and stronger than stated. Preserve.** With the margin, driving $q_y\to1/C$ costs $\log(1+(C-1)/\kappa)=7.79$ nats on CUB, not 4.61. But D4 defends only against the gross degeneracy; the operative one costs **0**, not $\log C$, so the "20×/40× safety factor" is irrelevant to the actual failure mode. (D4's budget figure is also inflated: it uses the full 4.59 range.)

**D5 (noise) — declared, and its designated detector is a no-op.** See C8.

**Label permutation (new, not in D1–D5).** $F_i$ **has no dependence on $y_i$ whatsoever**, and $\sum_c$ runs over all classes, so $F$ is *exactly* invariant under any relabeling of classes. Measured: $|\Delta\mathcal R|=6.7\times10^{-16}$, machine precision. This kills C8 (below).

**The exact null-space degeneracy (new, decisive).** Because $\ell_i=f(q_{i,y_i})$, the base loss is exactly blind to (a) the shape of the residual mass $1-q_{i,y_i}$ across non-target classes and (b) the split of $q_{y}$ across the $K$ target proxies — both of which $\mathcal R$ depends on strongly. Optimizing strictly inside that null space:

| iter | $\mathcal R$ | erank | base loss | drift |
|---|---|---|---|---|
| 0 | 1.1941 | 154.9 | 2.0739198 | 0 |
| 200 | **0.3458** | **355.7** | 2.0739198 | **0.00e+00** |

Machine-exact zero drift for 200 steps. (This rank-1-per-sample dependence is generic to softmax CE, not caused by the margin; what is specific here is that $\mathcal R$ has $O(1)$ nats of freedom in that null space while the entire barrier term is $\lambda_{\max}\mathcal R\le0.23$ nats.)

**Honest negative — one attack of mine failed.** I predicted proxies could fake isotropy by acquiring components orthogonal to the batch embedding span at negligible logit cost. Tested at $a\in\{0.05,0.1,0.2\}$, it **raised** $\mathcal R$ (+0.0005/+0.0010/+0.0057) and cost base loss. Net negative at every amplitude. That attack does not work; D3's confusion gating genuinely bites there.

**Anneal-to-zero — internally inconsistent.** If $\lambda_t\to0$ "returns the endpoint to a pure task optimum," erank at $t=T$ need not be elevated; yet falsifier #4 demands exactly that at the end. And since the weight-decay absorbing force is $10^{-8}$/step, there is nothing for the final 40 epochs to re-collapse. The anneal makes DOIR a basin-selection effect, which C6 (matched compute) does not control for.

---

## 3. Base loss, margin, $K$, and separability from the recipe

**Exact class marginal — yes, and genuinely distinct. Preserve.** $q_c=\sum_{j\in c}\pi_j$ is exactly the class marginal of the proxy-level categorical, unlike SoftTriple's soft-max-weighted pooled similarity. The claim that the Fisher's model is the trained model's parametric family holds.

**The margin is not a margin.** Because *every* target proxy receives $\delta$, the loss collapses to the monotone $\ell_i=-\log\frac{\kappa q_y}{\kappa q_y+1-q_y}$, $\kappa=e^{-3.2}=0.0408$. Consequences:

- **No decision-boundary effect at all.** The minimizer set is identical to marginless $-\log q_y$. CosFace/ArcFace/SoftTriple margins *do* change the boundary because their softmax is over individual or pooled class logits, not a marginal. Calling this "additive cosine margin $\delta=0.1$" is incorrect.
- What it actually is: a **confidence-amplifying gradient reweighting**. $|g'(q)|\big/(1/q)=1+\frac{q(1-\kappa)}{1-(1-\kappa)q}\to1/\kappa=e^{s\delta}=24.5$ as $q\to1$ — the opposite of margin saturation. At $q=0.99$ the gradient is 20× the CE gradient.

This is a specification defect, not fatal on its own, but it means $s$ and $\delta$ are not doing what C7 assumes they do.

**$K=15/15/2$ — verified correct.** From PFML's primary text: *"We use M = 15 for the CUB-200 and Cars-196 datasets, while M = 2 is used for the SOP dataset."* Credit where due; §7.1's recollection is accurate. Two caveats: PFML is a **potential-field** loss (attraction/repulsion superposition), not a softmax, so matching $M$ does not give cost parity — different losses, different per-step cost structure. And PFML uses Adam lr $5\times10^{-4}$ against the proposal's AdamW $10^{-4}$; the recipes are not matched. "matched to PFML's disclosed proxy counts for cost parity" is a non-sequitur. (Also: PFML is CVPR **2025** — the arXiv preprint is 2024. The proposal's citation is right.)

**DOIR's gain cannot be separated from the base recipe.** §1.2 is the author's own novel construction: exact class marginal + a mis-specified margin + zero proxy weight decay + a specific lr split, forecast at 0.716 CUB against PFML's 0.734. Every $\Delta$ is paired against *that* base. C6 controls capacity and time; **nothing controls the recipe**. The loss form, $\delta$, and the zero-proxy-decay choice were chosen jointly with DOIR and are confounded with it — and zero proxy decay is precisely the choice that opens the degenerate path. A base-only ablation over (margin form, proxy decay) is absent.

---

## 4. Prior art

I searched Fisher-information regularization, determinant/D-optimality objectives, information-matrix isotropy, spectral/orthogonality proxy regularizers, log-det metric learning, natural-gradient and active-learning analogues, and DML spectrum methods.

**The novelty claim at the object level survives.** I did not find prior work targeting the Fisher of a *class-marginal* posterior in *embedding* coordinates, trace-normalized and isotropized by a log-det barrier. §3's distinctions from VICReg/Barlow Twins, EWC, K-FAC, and Wang–Isola are correct as stated.

**But two material gaps in the search, and one understatement:**

- **MCR² is missing entirely** ([Yu, Chan, You, Song, Ma, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html)) — a log-det coding-rate objective on between/within class scatter, explicitly for diverse, non-collapsed, discriminative representations. This is the closest "log-det of a class-structured scatter matrix as the training objective" prior art and belongs in §3.
- **ITML** (Davis, Kulis, Jain, Sra, Dhillon, ICML 2007) — LogDet-divergence metric learning — occupies the log-det-in-metric-learning slot. (Cited from memory; I did not verify a link this session.)
- **The object understates its own pedigree.** For $K{=}1$, $F_i=s^2M^\top(\operatorname{diag}q-qq^\top)M$ is *exactly* the generalized Gauss–Newton / K-FAC output-side factor of a softmax head pulled back to feature space — the matrix K-FAC already forms. §3's "K-FAC uses it as a preconditioner; DOIR makes it the objective" is fair but undersells how standard the matrix is. The multi-proxy marginalization is a modest generalization.

Also confirmed occupied and correctly distinguished: [dimensional/rank collapse and spectrum-flattening regularizers](https://arxiv.org/abs/2110.09348) (on the covariance — C1 is the right control); [Fisher spectra of DNNs](https://arxiv.org/abs/1806.01316); [non-isotropy regularization](https://arxiv.org/pdf/2203.08547); D-optimality as max-log-det of the FIM (textbook optimal design).

**Assessment:** a new supervision *object*, but wrapper-adjacent — a known matrix (GGN block) under a known scalarization (D-optimality log-det) with a known normalization. Not a renamed occupied object. But it is not a new supervision *signal* either: EXP B shows what it actually supervises is the proxy set, which is discarded at test time.

---

## 5. Compute and memory

**Correct:** forming $F$ as $180\times32\times512^2=1.51$ GMAC ✓; Cholesky $512^3/3=0.045$ GFLOP ✓. The **1.02–1.04× wall-clock forecast is plausible** — a fp32 $512\times512$ Cholesky plus solve is ~1–2 ms latency-bound against ~45–90 ms for ResNet-50 fwd+bwd at $B{=}180$. **Preserve.**

**Wrong or unsupported:**

- **$C'=32$ applies only to SOP/In-Shop.** On CUB ($C{=}100$) it is $180\times100\times512^2=4.7$ GMAC $\approx9.4$ GFLOP — 2–3× the stated "3–5 GFLOP."
- **"Cost becomes $C$-independent" is false.** Selecting the top-$C'$ requires all $C$ values of $q_{ic}$, hence the full segment-sum over all $CK$ proxies: $O(B\!\cdot\!CK\!\cdot\!d)=2.09$ GMAC on SOP. That term is irreducible and dominates. Only the $d^2$ outer products become $C'$-limited.
- **The truncation bound is in the wrong units.** "$\le8\rho s^2$" is an *absolute* perturbation; the barrier sees $\tilde F=dF/\operatorname{tr}F$. Relative error is $\sim8\rho/(\mu_0\delta_0^2)$, and I measured $\operatorname{tr}F$ collapsing to $10^{-3}$. The bound is vacuous where it matters. Per-sample rank is also capped at $C'-1=31$ on SOP, whose best-conditioned floor is $\mathcal R\ge4.167$.
- **"$\rho<10^{-3}$ ... measured" is unsupported.** §7 declares the proposal blind with no experiments; §6(v) says the bound "must be measured, not assumed" — directly contradicting §1.5's claim that it was. This is an asserted empirical result with no experiment behind it.
- **The $10^{-6}I$ jitter is not safe.** At the barrier's own optimum I measured $\operatorname{tr}F=1.03\times10^{-3}$, so $\operatorname{tr}(10^{-6}I)=5.12\times10^{-4}$ — **~50% of $\operatorname{tr}F$.** Half the "information matrix" would be jitter. The proposal never states whether jitter is applied before or after trace normalization: **before**, and $\tilde F\to I$ and the barrier silently dies; **after**, it is redundant with $\varepsilon=10^{-2}$. Undecidable as written, and one reading makes the method inert.
- Memory 1.02× is defensible only if $F$ is formed as $G^\top G$ with $G\in\mathbb R^{BC'\times d}$ (11.8 MB) rather than materializing per-sample $F_i$ ($180\times512^2\times4$ B $=189$ MB). Unspecified.

---

## 6. Controls and falsifiers

**Well-posed — preserve:** C1, C2, C5, C6, C7, C9.

**C3 is a no-op.** $F$ is a function of $q$ and $m$ *only*. Detaching both detaches $F$ entirely: $\partial\mathcal R/\partial\theta=0$ for every parameter. C3 as described is bit-identical to the base run, not "gradient only into proxies." The intended control requires detaching $z$, not $q,m$. As written it can only ever "match DOIR" in the trivial sense.

**C4 is well-posed but its interpretation is inverted.** It is the *only* arm that tests the intended mechanism. Given EXP B, the informative comparison is the reverse of the one stated: if C4 $\ll$ DOIR, the gain came from the proxy path, which is discarded at test time.

**C8 is mathematically vacuous.** $F$ is exactly invariant under any class permutation ($|\Delta\mathcal R|=6.7\times10^{-16}$). The only reading with content — randomly *re-partitioning* proxies into groups of $K$ — is not what is written, and I measured that it *lowers* baseline $\mathcal R$ (1.075→0.742, erank 178→293), so a null there would be confounded by a different starting point rather than by semantics. **Falsifier #3 depends entirely on C8 and therefore cannot fire.**

**C10 cannot discriminate, and is ambiguous about test labels.** Computing $F$ needs class posteriors, which need proxies; unseen test identities have none. Reading (a) — training proxies at test embeddings — is legal but does not measure information about unseen identities. Reading (b) — fitting proxies to test classes — uses test labels; the wording "measured on unseen test identities" reads as (b) while §1.6 promises no test data touches selection. Since falsifier #4 is a *reporting* criterion, not a selection criterion, this is **not selection contamination** under either reading — but under (b) it is undisclosed test-label use for a headline mechanism claim.

**Either way C10 is diagnostically empty:** EXP B raises erank from 155 to 512 (3.3×, far above the 1.5× bar) with the embedding frozen. Both readings would *confirm* falsifier #4 under a mechanism that provably cannot move R@1.

**Falsifier #4's sign is likely backwards.** Along a natural convergence trajectory I measured erank$(F)$ falling monotonically as fit improves — 357 → 309 → 178 → 137 → 136 — while $\mathcal R$ rises 0.371 → 1.968, because confusion mass concentrates on fewer classes as classes separate. "erank rises 1.5× while R@1 rises" runs against the natural direction.

**Falsifier #6 is statistically vacuous.** With $n=5$, the one-sided critical Pearson $r$ at $\alpha=0.05$ is $r=0.805$; $r=0.5$ gives $t=1.00$, $p=0.196$. The threshold confirms mediation ~20% of the time under the null and has essentially no power to reject. **Five seeds cannot support a mediation correlation.**

**Falsifier #2** (C1 ≥ 80% of Δ) is underpowered: distinguishing 80% from 100% of a 1.9-pt Δ is ~0.4 pt against a paired SE of ~0.2–0.3 pt. **Falsifier #5** (wall-clock > 1.10×) is 2.5–5× looser than the 1.02–1.04× forecast — it cannot fire.

**Frontier arithmetic vs. the project objective.** The proposal's own numbers concede parity on CUB ($t=0.45$) and Cars ($t=1.58$), claiming one crossing on SOP ($t=3.16$, $p\approx0.013$). I reproduced all three $t$ values. Two problems:

- The Cars "needs ≥0.0064" is wrong: at $\alpha=0.05$ two-sided, df=8, the minimum detectable Δ is **0.0044**. (0.0064 corresponds to $\alpha\approx0.01$.)
- More fundamentally, an **unpaired two-sample $t$ against a number transcribed from another paper is not a valid test.** It models only within-implementation seed variance and ignores between-implementation variance, which dominates in DML (Musgrave et al., *A Metric Learning Reality Check*, ECCV 2020). §7.1 concedes PFML's recipe was not reproduced, so the SOP "crossing" is a 0.4-pt cross-paper difference with no shared pipeline.

**The proposal does not forecast meeting the objective of outperforming an existing method — and says so honestly** ("I am not forecasting a broad frontier crossing"). Beating the DADA matched-cost rows is beating a control, not the frontier.

---

## 7. Valid subcomponents, preserved separately from the method verdict

1. The Fisher derivation: $\nabla_z\log q_c=s(m_c-\bar m)$, $F_i=s^2\sum_c q_c(m_c-\bar m)(m_c-\bar m)^\top$, and the pairwise $q_cq_{c'}$ form — all correct.
2. $\mathcal R\ge0$ iff $\tilde F=I$; $\mathcal R_{\max}=4.5939$ — correct.
3. $G$ and $\operatorname{tr}(GF)=0$ — correctly derived.
4. The **marginless** exact class-marginal multi-proxy softmax is a clean, correctly motivated base, genuinely distinct from SoftTriple.
5. D1 is genuinely blocked.
6. D4's conclusion is right and stronger than stated (7.79 nats, not 4.61, on CUB).
7. Per-sample rank is $C-1$ but batch-averaged $F$ is full rank — measured 512/512. The proposal is right here; the "$C<d$ so $F$ can't be full rank" objection does not land.
8. The 1.02–1.04× wall-clock forecast is plausible.
9. $K=15/15/2$ is correctly recalled from PFML's primary text.
10. C1, C2, C5, C6, C7, C9 are well-posed; C4 is the right arm despite an inverted interpretation.
11. §6's contamination, label-noise, and R@1/R@10/NMI reporting discipline is good practice.

---

## Scope and uncertainty

My experiments are synthetic (isotropic-Gaussian class centres, $B{=}90$, $C{=}100$, $K{=}15$, $d{=}512$, fp32), not ImageNet-pretrained ResNet-50 features. They establish that the barrier's zero set is reachable in proxy space alone under the proposal's own hyperparameters, and that $\mathcal R$'s degrees of freedom sit in the base loss's exact null space — both structural facts that do not depend on feature realism. They do **not** prove DOIR yields exactly zero gain in real training; $z$ does move, and transient dynamics could still shift the basin. What they do prove is that the proposal cannot *attribute* any gain it observes: every stated mechanism signature ($\mathcal R\!\downarrow$, erank$\uparrow$, C10, falsifiers #4 and #6) is fully explained by a path with zero R@1 effect, and the controls as written (C3 a no-op, C8 a no-op, C10 confounded) cannot separate them.

The strongest thing that would change my verdict, obtainable without a GPU: a corrected §2 that blocks the proxy path *by construction* — e.g. defining $F$ with $P$ detached (making C4 the method rather than the control) — plus a base loss whose per-sample gradient is not rank-1 in $q_y$. That is a different method, not a repair.

**Sources**

- [Potential Field Based Deep Metric Learning (PFML), CVPR 2025 — arXiv:2405.18560](https://arxiv.org/abs/2405.18560) · [CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- [Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction (MCR²), NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html)
- [Understanding Dimensional Collapse in Contrastive Self-Supervised Learning, ICLR 2022](https://arxiv.org/abs/2110.09348)
- [Universal Statistics of Fisher Information in Deep Neural Networks, AISTATS 2019](https://arxiv.org/abs/1806.01316)
- [Non-isotropy Regularization for Proxy-based Deep Metric Learning, CVPR 2022](https://arxiv.org/pdf/2203.08547)
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)
- [Towards Improved Proxy-based Deep Metric Learning via Data-Augmented Domain Adaptation (DADA)](https://arxiv.org/pdf/2401.00617)
- [Metric Learning With HORDE, ICCV 2019](https://arxiv.org/abs/1908.02735)

