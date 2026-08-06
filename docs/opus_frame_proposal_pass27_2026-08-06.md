# Opus blind proposal, Pass 27

Consultation ID: `1b818489ae28406a`  
Provider/model: Claude Opus  
Caller: `sfora/emafactorial`  
Status: completed, exit 0  
Prompt: `docs/opus_blind_prompt_pass27_2026-08-06.txt`  
Prompt SHA-256: `3e4040980b388bd9d899a1c7f574ccb721eef99cb59525778d2b0f05ae60d04d`  
Native result truncated: false

## Lane declaration

**Lane A** (matched 512-D CNN): ResNet-50, 512-D L2-normalized global descriptor, 224 px, single-view cosine retrieval. All forecasts, baselines, and controls below are in Lane A only. Lane B (AdvRF, VAPNet) and CRT's transformer lane are not used for any comparison.

---

# FRAME — Factorized Residual Anisotropy for Metric Embeddings

**One-sentence statement.** Force the *class-conditional within-class covariance operators* of the deployed descriptor to **commute** (equivalently: be simultaneously orthogonally diagonalizable), while holding the mutual information between class identity and the coordinate-wise distribution of within-class variance inside a derived band — the lower bound blocking the "make all classes equally noisy" cheat, the upper bound blocking the class-detector shortcut.

The point that makes this a *method* and not a regularizer import: in a DML objective with learnable proxies and a cosine readout, **the entire training objective is exactly equivariant under a global rotation of the embedding space.** So an off-diagonal penalty in the network's own output frame is not a coordinate-dependent penalty — the optimizer is free to place the frame, and the attainable minimum of the penalty is $\min_{U\in O(d)}\sum_c\|\mathrm{offdiag}(U^\top\Sigma_c U)\|_F^2$, a *rotation-invariant* functional of the family $\{\Sigma_c\}$ that is zero iff the $\Sigma_c$ pairwise commute. The same-looking penalty in a closed-set classifier (where the labeled softmax pins the frame) means something entirely different and has no such reading. This is the mechanism.

---

## 1. Executable mathematics

### 1.1 Learned objects and deployment

| Object | Shape | Learned? | Used at test? |
|---|---|---|---|
| $\theta$ — ResNet-50, ImageNet-1K init | — | yes | yes |
| $W\in\mathbb{R}^{512\times 2048},\ b\in\mathbb{R}^{512}$ — embedding head | — | yes | yes |
| $\{\hat p_c\}$ — base-loss proxies | $K\times 512$ | yes | **no** |
| $V\in\mathbb{R}^{C\times 512}_{>0}$ — EMA within-class variance profiles | table | **buffer, no gradient** | **no** |
| $\bar S\in\mathbb{R}_{>0}$ — EMA covariance-energy scale | scalar | **buffer** | **no** |
| $\hat I_{\rm null}$ — sampling-noise floor for the MI term | scalar | recomputed each epoch, no gradient | **no** |

Forward: $z=Wg_\theta(x)+b$, $\hat z=z/\|z\|_2$. **Deployment is exactly $\hat z$**: one model, one 224 px centre-crop view, one 512-D descriptor, cosine NN. Every FRAME object above is discarded after training. Test-time cost delta is identically zero.

### 1.2 Batch construction

PK sampler: $P=30$ classes $\times$ $m=4$ images $=120$ per batch (matches the official Proxy-Anchor ResNet-50 batch size of 120). Augmentation: `RandomResizedCrop(224)` + horizontal flip only. Classes with $<4$ images are excluded from FRAME terms (relevant only for SOP; see §6).

### 1.3 Term 1 — conditional off-diagonal energy via an order-4 U-statistic

The obstacle: $\Sigma_c\in\mathbb{R}^{512\times512}$ from $m=4$ samples. The plug-in estimator $\|\mathrm{offdiag}\hat\Sigma_c\|_F^2$ is hopeless (rank 3, and its bias is a function of the diagonal, which couples the penalty to collapse). Instead, use **exactly unbiased** pairwise contrasts.

For class $c$ with samples $\hat z_{c,1..4}$, take the three disjoint pairings $\mathcal{P}=\{(12|34),(13|24),(14|23)\}$. For pairing $p$ set $\delta_p=\hat z_{a}-\hat z_{b}$, $\delta'_p=\hat z_{a'}-\hat z_{b'}$ and

$$
t_p=\tfrac14\Big[(\delta_p^\top\delta'_p)^2-\textstyle\sum_i \delta_{p,i}^2\delta'^2_{p,i}\Big],
\qquad
T_c=\tfrac13\sum_{p\in\mathcal P}t_p ,
\qquad
S_c=\tfrac13\sum_{p\in\mathcal P}\tfrac14(\delta_p^\top\delta'_p)^2 .
$$

**Proposition 1 (unbiasedness).** For i.i.d. draws within class $c$, $\mathbb{E}[\delta\delta^\top]=2\Sigma_c$ and $\delta\perp\delta'$, so
$\mathbb{E}[(\delta^\top\delta')^2]=\mathbb{E}_\delta[\delta^\top(2\Sigma_c)\delta]=4\,\mathrm{tr}(\Sigma_c^2)=4\|\Sigma_c\|_F^2$ and
$\mathbb{E}[\sum_i\delta_i^2\delta'^2_i]=\sum_i(2\Sigma_{c,ii})^2$. Hence

$$
\mathbb{E}[T_c]=\sum_{i\neq j}\Sigma_{c,ij}^2=\|\mathrm{offdiag}\,\Sigma_c\|_F^2\ \ge 0,
\qquad
\mathbb{E}[S_c]=\|\Sigma_c\|_F^2 .\qquad\blacksquare
$$

Averaging the three pairings symmetrises the kernel over the 4-tuple, so $T_c$ is *the* order-4 U-statistic for this functional (Rao–Blackwellised; minimum variance among unbiased estimators based on the 4 samples). Cost: $O(m^2 d)$, no $d\times d$ matrix is ever formed. Individual $t_p$ can be negative; the *expectation* is $\ge 0$ for every $\theta$, so SGD on an unbiased estimator has its expected minimum exactly at conditional decorrelation. There is no noise-exploitation channel — only variance.

Scale handling (the collapse defence, §2.2):

$$
\boxed{\ \mathcal{L}_{\rm od}=\frac{\sum_{c\in\mathcal B}T_c}{\mathrm{sg}\!\left[\alpha\sum_{c\in\mathcal B}S_c+(1-\alpha)P\bar S\right]+\varepsilon}\ },\qquad
\alpha=0.7,\ \varepsilon=10^{-8},
$$
$\bar S\leftarrow(1-\eta_S)\bar S+\eta_S\cdot\tfrac1P\sum_c S_c$, $\eta_S=0.1$, detached. Gradient flows **only** through the numerator.

Explicit gradient (one pairing): $\partial t_p/\partial\delta=\tfrac12[(\delta^\top\delta')\delta'-\delta\odot\delta'^{\odot2}]$, $\partial t_p/\partial\hat z_a=+\partial t_p/\partial\delta$, $\partial t_p/\partial\hat z_b=-\partial t_p/\partial\delta$; then into $z$ via $J=(I-\hat z\hat z^\top)/\|z\|$, then into $W,b,\theta$. **Proxies receive no FRAME gradient.**

### 1.4 Term 2 — the modulation-information band

Per-class per-coordinate variance estimate (unbiased for $\Sigma_{c,ii}$):
$$
\hat v_{c,i}=\tfrac12\cdot\tfrac16\sum_{1\le a<b\le4}(\hat z_{c,a,i}-\hat z_{c,b,i})^2 .
$$
EMA table with a straight-through gradient path:
$$
\tilde V_{c}= (1-\eta_v)\,\mathrm{sg}[V_c]+\eta_v\,\hat v_c,\qquad \eta_v=0.05,\qquad V_c\leftarrow \mathrm{sg}[\tilde V_c],\quad V\ \text{init}=1/d .
$$
(The gradient into $\theta$ is therefore $\eta_v\,\partial\hat v_c/\partial\theta$; $\lambda_{\rm mod}$ is defined post-hoc to absorb this factor.)

Profiles and the modulation information:
$$
\pi_{c,i}=\frac{\tilde V_{c,i}+\epsilon_v}{\sum_j(\tilde V_{c,j}+\epsilon_v)},\quad
\bar\pi_i=\tfrac1P\sum_c\pi_{c,i},\quad
I \;=\; \tfrac1P\sum_{c}\sum_i \pi_{c,i}\log\frac{\pi_{c,i}}{\bar\pi_i}\ \in[0,\log P],
$$
with $\epsilon_v=10^{-6}$. $I$ is exactly the mutual information $I(C;\mathrm{Idx})$ between a uniform class variable and a coordinate index drawn from $\pi_C$: *how much does knowing which coordinate carries the within-class variance tell you which class it is?*

$$
\boxed{\ \mathcal{L}_{\rm mod}=\frac{\big[\max(0,\iota_--I)\big]^2+\big[\max(0,I-\iota_+)\big]^2}{(\log P)^2}\ }
$$

**Both bounds are derived, not tuned:**

- *Upper.* If a fraction $\phi$ of class $c$'s within-class variance sits in coordinates private to $c$, then $\mathrm{KL}(\pi_c\|\bar\pi)\ge\phi\log P$. Budgeting $\phi\le0.10$ gives $\iota_+=0.10\log P=0.340$ nats at $P{=}30$.
- *Lower.* $\iota_-=\max\big(0.02\log P,\ 2\hat I_{\rm null}\big)=\max(0.068,\,2\hat I_{\rm null})$. $\hat I_{\rm null}$ is the value of $I$ under the null "all classes share one $\Sigma$", simulated once per epoch from the chi-square sampling law of the EMA estimator (no forward passes). Arithmetic: per batch $\nu_1=3$ d.o.f.; EMA effective d.o.f. $\nu_{\rm eff}\approx\nu_1(2-\eta_v)/\eta_v\approx117$; $\mathrm{Var}(\log\hat v)\approx2/\nu_{\rm eff}=0.017$; $I_{\rm null}\approx\tfrac12\mathrm{Var}(\log\hat v)\approx0.008$ nats. So $2\hat I_{\rm null}\approx0.017\ll0.068$: **the noise floor sits a factor of 4 below the binding lower bound**, and the admissible band $[0.068,\,0.340]$ is a factor-5 window. A representation with log-variance modulation of std $0.4$ (typical cross-class ratio $\approx1.5\times$) sits at $I\approx0.08$ — inside, near the bottom.

### 1.5 Total objective, schedule, recipe

$$
\mathcal{L}=\mathcal{L}_{\rm base}+\lambda_{\rm od}(t)\,\mathcal{L}_{\rm od}+\lambda_{\rm mod}\,\mathcal{L}_{\rm mod}
$$

$\lambda_{\rm od}^{\max}=0.5$, $\lambda_{\rm mod}=2.0$. Schedule: both terms held at 0 for epochs $[0,5)$ (class means not yet formed; $\Sigma_c$ is still ImageNet structure), cosine ramp to full over epochs $[5,20]$, then $\lambda_{\rm od}$ cosine-annealed to $0.5\lambda_{\rm od}^{\max}$ over the final 20% of training (the constraint is an *identification* constraint; once the family commutes, holding it rigidly taxes discriminability). $\lambda_{\rm mod}$ is constant after ramp-in.

$\mathcal{L}_{\rm base}$ — **Proxy Anchor** (CVPR 2020), reproduced exactly as published:
$$
\mathcal{L}_{\rm PA}=\frac{1}{|P^+|}\sum_{p\in P^+}\log\Big(1+\!\!\sum_{x\in X^+_p}\!\! e^{-\alpha_{\rm PA}(s(x,p)-\delta_{\rm PA})}\Big)+\frac1{|P|}\sum_{p\in P}\log\Big(1+\!\!\sum_{x\in X^-_p}\!\! e^{\alpha_{\rm PA}(s(x,p)+\delta_{\rm PA})}\Big)
$$
$s(x,p)=\hat z_x^\top\hat p$, $\alpha_{\rm PA}=32$, $\delta_{\rm PA}=0.1$, one proxy per class. Disclosed ResNet-50 recipe (official repo command): AdamW, lr $10^{-4}$, **proxy lr $\times100$**, weight decay $10^{-4}$, batch 120, embedding 512, BN frozen, 5 warm-up epochs (embedding layer only), lr decay $\times0.5$ every 5 epochs. **Deviation for the lane:** the lane specifies 200 epochs; PA's disclosed budget is 60 with step decay. I run **both** budgets for baseline *and* FRAME and report both (see §7, ambiguity A3). I also replace PA's default random sampler with the PK($30{\times}4$) sampler that FRAME requires — **this is a recipe change and the baseline is re-run under the identical sampler** (control C0).

### 1.6 Hyperparameter selection protocol (contamination control)

$\lambda_{\rm od},\lambda_{\rm mod},\eta_v,\alpha,\phi$ are selected **only** on a pseudo-unseen split carved from *training* classes (CUB: train on classes 1–80, validate retrieval on 81–100), then frozen and the model retrained on all training classes. Test classes are touched once, at the end, for the five final seeds. No test-set model selection, no early stopping on test.

---

## 2. Causal zero-shot error mode and degeneracy attacks

### 2.1 The error mode: **nuisance parking**

**Proposition 2 (proxy losses are blind off the proxy span).** Let $\mathcal{L}_{\rm base}=F(\{\hat z_a^\top\hat p_c\})$ for any proxy loss. Then
$\nabla_{\hat z_a}\mathcal{L}_{\rm base}=\sum_c\frac{\partial F}{\partial s_{ac}}\hat p_c\in S:=\mathrm{span}\{\hat p_c\}_{c=1}^{K}$.
Hence the component of $\Sigma_c$ in $S^\perp$, namely $\Pi_{S^\perp}\Sigma_c\Pi_{S^\perp}$, receives **exactly zero direct gradient** from the base loss. $\blacksquare$

Arithmetic in Lane A with one proxy per class: CUB $K=100$, $d=512$ → **$\ge412$ of 512 directions carry no direct training pressure**; Cars $K=98$ → $\ge414$. Cosine retrieval on unseen classes uses all 512. Whatever the trunk parks in $S^\perp$ is uncontrolled, and unseen-class discriminative directions are not confined to $S$.

(Caveats stated: pressure reaches $S^\perp$ *indirectly* through the normalisation Jacobian and weight sharing; and $S$ rotates as proxies learn. The claim is zero *direct* pressure, not zero influence. On SOP/In-Shop $K>d$ so $S=\mathbb{R}^d$ generically — see the differential prediction in §5.)

### 2.2 Why *commuting* conditional noise is the right target

**Proposition 3 (worst-case leakage).** For unseen classes $a,b$ with means $\mu_a,\mu_b$ and query noise $\epsilon_q\sim(0,\Sigma_a)$, the retrieval margin $M=\langle z_q,z_+\rangle-\langle z_q,z_-\rangle$ has $\mathbb{E}[M]=\langle\mu_a,\mu_a-\mu_b\rangle$ and $\mathrm{Var}(M)\ge \Delta^\top\Sigma_a\Delta$ with $\Delta=\mu_a-\mu_b$. Since $\Delta$ is **unknown at training time** (it involves only unseen classes), the training-time-controllable risk is $\sup_{\|\Delta\|=1}\Delta^\top\Sigma_a\Delta=\lambda_{\max}(\Sigma_a)$.

**Proposition 4 (diagonal attains the floor).** For any PSD $\Sigma$ with diagonal $D$, $\lambda_{\max}(\Sigma)\ge e_i^\top\Sigma e_i=D_{ii}$ for all $i$, hence $\lambda_{\max}(\Sigma)\ge\max_i D_{ii}$; and $\lambda_{\max}(D)=\max_i D_{ii}$. So among all conditional covariances with a given per-coordinate variance budget, **the diagonal one uniquely attains the minimum possible worst-case leakage.** $\blacksquare$

Second, independent justification (generative): if within-class variation is produced by a *shared* set of physical mechanisms (pose, illumination, part deformation, sub-type) whose *strengths* vary by class, the true $\Sigma_c$ share an eigenbasis and therefore commute. **Commutativity of $\{\Sigma_c\}$ is a testable consequence of "the same physics explains within-class variation in every class."** A class-detector solution — which entangles nuisance with identity — violates it. Term 1 is that consistency test, run as a training constraint.

**Frame uniqueness is deliberately *not* enforced.** The naive nonlinear-ICA reading would demand pairwise-distinct modulation profiles (the AJD uniqueness condition) so the frame is identified. I discard that requirement: cosine retrieval is rotation-invariant, so an unidentified rotation inside a degenerate eigen-block is harmless. Only the *coordinate-free* content — commutativity, and non-degeneracy of the joint spectra — is enforced. This is why FRAME has two terms and not four.

### 2.3 Proof-level attack on the cheapest degeneracies

| # | Degeneracy | Cheapest form | Why it is blocked |
|---|---|---|---|
| **D1** | **Within-class collapse.** $\Sigma_c\to0$ makes $T_c\to0$ | $\|\mu_c\|\to1$; on the sphere $\mathrm{tr}\Sigma_c=1-\|\mu_c\|^2$ exactly | $\mathcal{L}_{\rm od}$ is degree-4-homogeneous in the within-class scale in **both** numerator and detached denominator ⇒ **exactly scale-invariant at $\alpha=1$**; at $\alpha=0.7$ a shrink buys at most a 30%-weighted transient over the $\eta_S=0.1$ EMA window. Independently: under collapse all $\tilde V\ll\epsilon_v$ ⇒ every $\pi_c\to$ uniform ⇒ $I\to0<\iota_-$ ⇒ $\mathcal{L}_{\rm mod}$ fires at its lower bound. Two independent blocks. |
| **D2** | **One-hot / class-code shortcut.** $z$ becomes a class indicator — the canonical supervision-collapse solution and a global optimum of $\mathcal{L}_{\rm PA}$ when $d>C_{\rm tr}$ | Coordinate $i$ carries variance only in class $i$ | **This passes Term 1** (all $\Sigma_c$ are diagonal ⇒ they commute ⇒ $\mathcal{L}_{\rm od}=0$). It is blocked *only* by Term 2: $I\to\log P\gg\iota_+=0.10\log P$. This is the reason Term 2 exists and why $\iota_+$ is set by a private-variance budget rather than by tuning. |
| **D3** | **Homogenisation.** Identical matrices trivially commute, so $\Sigma_c\equiv\Sigma$ is a global optimum of Term 1 | Encoder suppresses genuine class-specific attribute variability, or adds class-independent noise | Blocked by $\iota_-$: $\Sigma_c\equiv\Sigma\Rightarrow I=0<0.068$. Note this is exactly DVML's modelling assumption, so D3 is not hypothetical — it is a published prior that FRAME's lower bound forbids. |
| **D4** | **Dead dimensions.** Kill $d-k$ coordinates | Dead coordinates have $V_{c,i}\approx\epsilon_v$ for every $c$ | Dead coordinates have *identical* profiles across classes ⇒ they contribute 0 to $I$ while inflating the denominator ⇒ they push $I$ toward the $\iota_-$ violation. Anti-dimensional-collapse is a *consequence*, not a bolted-on term. |
| **D5** | **Unbounded reward hacking of Term 2** | Drive $I\to$ its ceiling | $\mathcal{L}_{\rm mod}$ is a two-sided hinge: it is exactly 0 on $[\iota_-,\iota_+]$ and offers **no gradient reward** for exceeding either bound. Bounded loss ⇒ bounded incentive. |
| **D6** | **Estimator exploitation.** Push $T_c$ negative using sampling noise | $t_p<0$ is attainable for single tuples | $\mathbb{E}[T_c]=\|\mathrm{offdiag}\Sigma_c\|_F^2\ge0$ **for every $\theta$**. The expected loss is bounded below by 0 and attains it only at conditional decorrelation. Noise costs variance, not bias. |

**The residual tension, stated openly.** Prop. 4 says leakage is minimised by a *flat* diagonal (isotropy), while $\iota_-$ demands non-flat, class-varying diagonals. They are reconciled by the size of the band: $\iota_-=0.068$ nats requires log-variance modulation of std $\approx\sqrt{2\times0.068}=0.37$, i.e. cross-class ratios $\approx1.45\times$, so $\lambda_{\max}(\Sigma_c)\lesssim1.45\cdot\mathrm{tr}\Sigma_c/d$ — within 45% of the isotropic floor while retaining modulation. **Whether that residual 45% matters more than the modulation is an empirical question, and control C1 adjudicates it decisively.** I do not claim to know the answer a priori.

---

## 3. Adversarial novelty search

I searched inside DML and outside it, and I report the hits that damage the claim as well as those that do not.

### 3.1 The nearest work, inside classification (the most damaging hit)

**Choi & Rhee, "Utilizing Class Information for Deep Network Representation Shaping" (arXiv 1809.09307, AAAI 2019) — cw-CR / cw-VR.** cw-CR penalises off-diagonal entries of *class-conditional* covariance at hidden layers; cw-VR penalises class-wise variance. This is the closest antecedent to Term 1 and I could not extract its exact formula (the PDF returned binary; see ambiguity A5). Taking the published description at face value, the distinctions are:

1. **Coordinate-free vs coordinate-dependent.** cw-CR sits under a labeled softmax, which *pins* the frame; the penalty there is genuinely a coordinate-dependent decorrelation of hidden units. Under a learnable-proxy cosine objective the whole loss is $O(d)$-equivariant, so the attainable minimum is $\min_U\sum_c\|\mathrm{offdiag}(U^\top\Sigma_cU)\|_F^2$ — a rotation-invariant commutativity functional. Same-looking penalty, different mathematical object.
2. **Estimator.** cw-CR uses plug-in class covariances, which require many same-class samples per batch; FRAME's order-4 U-statistic is exactly unbiased from $m=4$ in $d=512$ at $O(m^2d)$ cost and never forms a $d\times d$ matrix. Without this, the constraint is not estimable in the DML batch regime at all.
3. **Sign.** cw-VR *reduces* class-wise variance; FRAME's $\iota_-$ *forbids* homogenising class-conditional variance. Opposite direction on the same quantity.
4. **Task.** Closed-set classification, no retrieval or zero-shot metric-learning evaluation.

I state plainly: **Term 1 in isolation is close enough to cw-CR that I would not defend it as novel.** The claim is the conjunction — commutativity read coordinate-freely under a rotation-equivariant readout, made estimable by the tetrad U-statistic, and bounded on both sides by a derived modulation-information band whose lower bound is opposite in sign to cw-VR.

### 3.2 Nearest works, one sentence of mechanism distinction each

| Work | Mechanism distinction |
|---|---|
| **PFML** (Potential Field DML, CVPR 2025) — CUB .734 / Cars .927 / SOP .829, 15 proxies on CUB/Cars, 2 on SOP | PFML restructures the *first-order* interaction field between samples and multiple proxies (mean geometry); FRAME leaves the mean geometry to the base loss and constrains the *second-order conditional* geometry, which Prop. 2 shows the mean geometry cannot reach. |
| **PA + DADA** (AAAI 2024) — In-Shop .930 | DADA performs data-augmented domain adaptation across proxy/sample domains at 1.06× epoch time; FRAME adds no domain, no adaptation, and no extra forward pass. |
| **NIR — Non-isotropy Regularization** (CVPR 2022) | NIR matches each class-conditional *density* to a shared non-isotropic target with a normalizing flow (extra learned network, density estimation); FRAME imposes a second-order commutativity constraint with a closed-form $O(d)$ unbiased statistic and no auxiliary network. |
| **DVML** (ECCV 2018) | DVML *assumes* intra-class variance is class-independent and samples synthetic embeddings from it; FRAME's $\iota_-$ makes exactly that assumption a penalised degeneracy (D3), and synthesises nothing. |
| **MIC** (ICCV 2019) | MIC learns a second branch on surrogate clusters to *subtract* intra-class characteristics from the embedding; FRAME adds no branch and *keeps* intra-class structure, factorising it into a commuting family. |
| **$\rho$-spectral regularization** (Roth et al., ICML 2020) | $\rho$-reg flattens the *global marginal* singular-value spectrum by negative-flipping in the miner; FRAME never touches the marginal spectrum and operates on per-class conditional covariances (falsifier F6 tests whether the gain is merely $\rho$ in disguise). |
| **VICReg / Barlow Twins / DeCov** | These decorrelate the *marginal* covariance toward identity; FRAME decorrelates the *class-conditional* covariance and explicitly *forbids* cross-class uniformity — an oppositely-signed requirement. |
| **HIB / PFE / probabilistic embeddings** | Per-*instance* predicted Gaussians altering the readout; FRAME predicts nothing, changes no readout, and deploys a plain point descriptor. |
| **Sub-center ArcFace, multi-proxy methods** | Multiple *means* per class (first-order multi-modality); orthogonal to and additive with a conditional second-order constraint (tested in C6). |
| **CPCANet** (deep-unfolding common PCA for domain generalization, 2026) | Penalises off-diagonal energy of basis-transformed covariances across *domains* inside an unfolded CPCA architecture for DG classification; FRAME uses *class*-conditional covariances in an unmodified ResNet-50 for zero-shot retrieval, with a modulation band and no explicit basis variable. (Source verified only via search snippet — ambiguity A6.) |

### 3.3 Outside DML — the principled imports, named honestly

- **Common Principal Components** (Flury 1984) and **approximate joint diagonalization** (JADE, Cardoso & Souloumiac 1996; SOBI, Belouchrani et al. 1997; Pham & Cardoso 2001). These diagonalise a *fixed* family of covariance matrices by choosing $U$. FRAME inverts the problem: the family itself is the learnable object, and the residual AJD objective is used as a training pressure on the encoder.
- **Nonlinear ICA with auxiliary variables** (Hyvärinen, Sasaki & Turner, AISTATS 2019; Khemakhem et al., iVAE, AISTATS 2020). Identifiability from conditional independence + variance modulation under an auxiliary variable. FRAME uses *training class identity* as that auxiliary variable — freely available and never used this way in DML — but **discards the frame-uniqueness half of the theory** because the retrieval readout is rotation-invariant (§2.2). To my knowledge no prior work imports the variability condition as an information *band* with both a floor and a ceiling.
- **Fourth-order U-statistics for covariance functionals** (dCov/HSIC U-centering; sphericity and bandedness tests). Estimator kinship for $T_c$; those are hypothesis tests on fixed data, here it is a differentiable training signal.

**What I did not find,** after adversarial search: any work that (a) treats the class-conditional within-class covariance operators of a *metric-learning* embedding as a family required to **commute**, (b) exploits the rotation-equivariance of proxy-DML to give that penalty coordinate-free meaning, (c) estimates it unbiasedly from $m=4$ at $d=512$, or (d) constrains $I(C;\text{variance-coordinate})$ to a two-sided derived band.

---

## 4. Decisive matched-compute controls

FRAME's compute overhead is $\sim10^{-7}$ of a training step (§6), so "matched compute" is trivially satisfiable; the controls below are designed to separate the **mechanism** from occupied alternatives, which is the harder job.

| ID | Control | What it kills if it matches FRAME | Prediction |
|---|---|---|---|
| **C0** | **Sampler-matched baseline.** PA with PK(30×4) sampler, no FRAME. *Mandatory* — FRAME requires this sampler, so the published-recipe PA number is not a valid baseline. | "The gain is from the PK sampler." | $\le\pm0.3$ pt vs random sampler |
| **C1** | **Conditional isotropy.** Replace $\mathcal{L}_{\rm od}$ with a penalty driving $\Sigma_c\to(\mathrm{tr}\Sigma_c/d)I$, same weight. | "The mechanism is conditional isotropy, not commutativity + modulation." **This is the decisive control** for the tension in §2.3. | isotropy $\approx+0.5$; FRAME $\approx+1.6$ on CUB |
| **C2** | **Marginal decorrelation.** Same tetrad statistic applied to *cross*-class differences (Barlow/VICReg-style). | "Conditional vs marginal doesn't matter." | marginal $\approx+0.3$ |
| **C3** | **Homogenisation (DVML prior).** Replace Term 2 with a penalty forcing $\Sigma_c$ equal across classes. | "FRAME rediscovers DVML's class-independent variance." | $\le$ baseline; possibly below |
| **C4** | **Rotation control (zero-cost, logical).** Take the trained *baseline*, apply the AJD rotation $U^\star$ post-hoc, re-evaluate. R@1 must be **bit-identical** (cosine is rotation-invariant). | Proves any FRAME gain is optimization pressure, not coordinates. If FRAME's gain were "nicer axes," this control would already deliver it. | identical R@1 |
| **C5** | **Term ablation.** {Term1}, {Term2}, {Term1+Term2}, {Term2 upper-only}, {Term2 lower-only}. | Superadditivity is the claim; if Term1-only $\approx$ full FRAME, Term 2 (the anti-D2 term) is decoration and the method reduces toward cw-CR. | T1 $+0.6$, T2 $+0.4$, both $+1.6$; upper-only $+1.1$, lower-only $+0.7$ |
| **C6** | **Proxy-count control.** PA with $K\in\{1,5,15\}$ proxies/class vs PA($K{=}1$)+FRAME, plus PFML(15)+FRAME. | "Multi-proxy already covers this" — PFML's own mechanism. Prop. 2 becomes vacuous at $K{=}15$ ($1500>512$), so this is the strongest competing explanation. | multi-proxy and FRAME partially additive; FRAME retains $\ge55\%$ of its delta on top of $K{=}15$ |
| **C7** | **Estimator-quality control.** Replace $T_c$ with the naive plug-in $\|\mathrm{offdiag}\hat\Sigma_c\|_F^2$ from $m=4$. | "The U-statistic is decoration." | naive notably worse; its bias is diagonal-coupled and re-opens D1 |
| **C8** | **Noise-injection control.** Isotropic Gaussian noise on $\hat z$ with variance matched to FRAME's measured $\Delta\,\mathrm{tr}\Sigma_c$. | "FRAME is just regularizing noise." | $\le+0.3$ |
| **C9** | **Weight-decay control.** All of the above at $\mathrm{wd}(W)=0$, with $\mathbb{E}\|z\|$ logged per epoch. | Tests whether the scale is operational (see below). | delta preserved within $\pm0.4$ |

**On loss normalization and operational scale.** I do not claim FRAME's normalization is harmless. Both $\mathcal{L}_{\rm PA}$ and FRAME reach $z$ through the same Jacobian $(I-\hat z\hat z^\top)/\|z\|$, so their *relative* weighting is scale-free — but the absolute data-gradient magnitude scales as $1/\|z\|$, and AdamW's **decoupled** weight decay does not. The balance between data gradient and weight decay therefore shifts with $\|z\|$, and FRAME alters $\|z\|$'s trajectory (through D1's exact identity $\mathrm{tr}\Sigma_c=1-\|\mu_c\|^2$). C9 plus per-epoch $\mathbb{E}\|z\|$ logging is required, not optional.

**Diagnostics (all measured on held-out *training*-class images, never test classes):** relative off-diagonal energy $\bar\rho=\sum_c\|\mathrm{offdiag}\Sigma_c\|_F^2/\sum_c\|\Sigma_c\|_F^2$; normalised commutator $\frac{1}{|C|^2}\sum_{c,c'}\|[\Sigma_c,\Sigma_{c'}]\|_F^2/(\|\Sigma_c\|_F\|\Sigma_{c'}\|_F)$; $I$; parking fraction $\kappa=\mathrm{tr}(\Pi_{S_\epsilon^\perp}\bar\Sigma)/\mathrm{tr}\bar\Sigma$; and the known correlates $\pi$-ratio and $\rho$-spectral-decay.

---

## 5. Frozen forecasts, falsification thresholds, frontier arithmetic

**Lane A. ResNet-50 / 512-D / 224 px / single-view cosine / 5 seeds. All numbers are 5-seed means of R@1; $\pm$ is seed SD. Frozen before any run.**

### Baselines I must produce myself

| | CUB | Cars196 | SOP |
|---|---|---|---|
| PA, C0 sampler-matched (my repro) | $0.700\pm0.006$ | $0.880\pm0.005$ | $0.799\pm0.003$ |
| PFML (my repro) | $0.729\pm0.005$ | $0.922\pm0.005$ | $0.826\pm0.003$ |
| PFML (published reference) | 0.734 ± 0.003 | 0.927 ± 0.003 | 0.829 ± 0.002 |
| DADA same-lane (matched-cost control) | 0.729 | 0.921 | 0.810 |

### FRAME deltas (frozen; 90% intervals)

| Base | CUB | Cars196 | SOP |
|---|---|---|---|
| on PA | **+1.6** pt (+0.6, +2.6) | **+1.3** pt (+0.3, +2.3) | **+0.2** pt (−0.3, +0.7) |
| on PFML | **+0.9** pt (0.0, +1.8) | **+0.7** pt (−0.1, +1.5) | **+0.1** pt (−0.4, +0.6) |

### Resulting absolutes

| | CUB | Cars196 | SOP |
|---|---|---|---|
| FRAME + PA | **0.716** | **0.893** | 0.801 |
| FRAME + PFML | **0.738** | **0.929** | 0.827 |

### Frontier-crossing arithmetic — stated against my own interest

- **FRAME+PA does not cross PFML.** 0.716 vs 0.734 (−1.8 pt CUB); 0.893 vs 0.927 (−3.4 pt Cars). I forecast this explicitly. The defensible claim from that arm is a **mechanism claim with a matched baseline**, not a frontier claim.
- **FRAME+PFML crosses only marginally.** CUB $0.738$ vs published $0.734$ → **+0.4 pt**. Whether this is significant depends on an ambiguity I cannot resolve: if PFML's "±0.003 over five runs" is an SD, SEM $=0.0013$ and +0.4 pt is $\approx3$ SEM (significant but thin); **if it is already an SEM, +0.4 pt is $1.3$ SEM and is not significant.** Cars $0.929$ vs $0.927$ → **+0.2 pt**, not significant under either reading.
- Probabilities I will hold myself to: P(FRAME+PA delta $\ge+0.5$ pt on *both* CUB and Cars) $\approx0.60$. P(FRAME+PFML 5-seed CUB mean $>0.734$) $\approx0.45$. P(crossing PFML on **both** CUB and Cars) $\approx0.25$.

**Honest summary of the forecast:** FRAME is forecast to be a real, replicable, near-zero-cost mechanism gain of $+1$ to $+2$ points on the datasets where its precondition holds, and it is **not** forecast to confidently establish a new Lane A frontier. I state that rather than inflate the delta, because the frozen-proposal reviewer will discover it either way and because a mechanism that survives C1–C9 is worth more than a contested +0.4.

### The differential prediction (the strongest evidence available)

Prop. 2's arithmetic predicts an **ordering**, not a uniform gain: the mechanism should scale with the size of the unconstrained subspace $d-K$ and with the estimability of $\Sigma_c$.

| Dataset | train classes | $d-K$ (1 proxy) | imgs/class | predicted delta |
|---|---|---|---|---|
| CUB | 100 | 412 | ~30 | largest |
| Cars196 | 98 | 414 | ~82 | large |
| In-Shop | 3,997 | 0 | ~6.5 | small |
| SOP | 11,318 | 0 | ~5.3, many with 2 | ~neutral |

**A uniform gain across all four datasets would falsify the causal story even if R@1 improves**, because it would mean the gain is not coming from the parked subspace.

### Falsification thresholds (pre-registered)

- **F1** — Effect. If the 5-seed FRAME+PA delta is $<+0.5$ pt on **both** CUB and Cars: mechanism falsified.
- **F2** — Mechanism identity. If $\bar\rho$ (train-class held-out) does not drop by $\ge2\times$ vs C0: the term is not doing what it claims, and any R@1 gain is *not* attributable to the mechanism, even if F1 passes.
- **F3** — Rotation. C4 must yield bit-identical R@1. If not, the evaluation is wrong and everything downstream is void.
- **F4** — Ordering. If the delta ordering is not CUB $\approx$ Cars $>$ In-Shop $\ge$ SOP: causal story wrong (weaker falsifier).
- **F5** — Compute. If baseline given equal *wall-clock* recovers $\ge60\%$ of the delta: compute artifact.
- **F6** — Not-a-known-correlate. If FRAME's R@1 delta is fully predicted by its change in $\rho$-spectral-decay or $\pi$-ratio (partial correlation with $\bar\rho$ and $I$ not significant at $p<0.05$ across the C5 ablation grid): FRAME is a re-parameterisation of a known correlate, not a new mechanism.
- **F7** — Additivity. If FRAME retains $<25\%$ of its delta on top of PFML's 15 proxies (C6): the mechanism is subsumed by multi-proxy mean structure.

---

## 6. Cost, and benchmark / contamination risks

### Cost

**Training FLOPs.** Term 1: $3$ pairings $\times\,3d$ ops $\times\,30$ classes $\approx1.4\times10^5$. Term 2: $\approx1.1\times10^5$. Total $\approx2.5\times10^5$ MACs/step against ResNet-50 fwd+bwd at $224^2$: $\approx3\times4.1\,\mathrm{GFLOP}\times120\approx1.5\times10^{12}$. **Ratio $\approx1.7\times10^{-7}$.** Realistic wall-clock overhead is dominated by kernel-launch latency, not arithmetic: **expect $\le1\%$**, to be measured and reported (compare: DADA 1.06× epoch time, 1.01× memory; AdvRF and VAPNet add whole auxiliary networks).

**Training memory.** EMA table $C\times512$ fp32: CUB 0.2 MB, Cars 0.2 MB, In-Shop 8.2 MB, SOP 23.2 MB. No extra activations, no extra views, no extra networks, no extra backward pass.

**Deployment.** Identical to the baseline — same weights, same single view, same 512-D descriptor, same cosine NN. **Delta is exactly zero**, which is the main practical argument for FRAME over Lane B's auxiliary-system methods.

### Risks

- **SOP structural limitation, stated up front.** Many SOP classes have 2 images, so they cannot supply $m=4$ and are excluded from FRAME terms; and $\Sigma_c$ for a 4-image class is largely *augmentation* covariance rather than genuine intra-class covariance. FRAME on SOP measures a weaker nuisance model. I forecast near-neutral there and treat any large SOP gain as suspicious.
- **Cars196 saturation.** +0.7 pt $\approx$ 57 of 8,131 test queries. Five seeds and reported SD are mandatory; single-seed Cars results are uninterpretable at this margin.
- **"Zero-shot" is only w.r.t. the DML label set.** ImageNet-1K contains ~59 bird classes and multiple car classes; ImageNet initialization is permitted by the protocol but means CUB/Cars unseen classes are not semantically novel to the network. This affects FRAME and every reference equally, but it caps the strength of any transfer claim.
- **CUB/Cars label noise and near-duplicates**, and SOP R@1 partly determined by trivial near-duplicate product photos.
- **In-Shop reference has no uncertainty.** PA+DADA 0.930 has unreported seed count; no significance test against it is possible. I therefore make no In-Shop frontier claim.
- **Test-set selection risk** is the most likely way this proposal silently fails. Mitigated by the frozen pseudo-unseen-class protocol in §1.6; violating it would invalidate the forecast.
- **Recipe-mismatch risk.** Comparing my 200-epoch runs to references trained under different budgets is a known contamination route for frontier claims. Mitigation: every comparison is against *my own* reproduction under the identical recipe, sampler, and budget, with the published number reported separately as a reference line.

---

## 7. Unresolved source ambiguities

- **A1.** Whether PFML's "±0.003 over five runs" is an SD or an SEM. This changes the CUB frontier conclusion from "significant but thin" to "not significant." I cannot resolve it and the §5 arithmetic reports both readings.
- **A2.** PFML's venue year: the prompt and the arXiv landing page for 2405.18560 say CVPR 2025; a secondary review source says CVPR 2024. Immaterial to the math, noted for citation accuracy.
- **A3.** PFML's exact loss form, proxy initialization, optimizer, LR schedule, pooling, and epoch budget were not retrievable in full (the PDF exceeded the fetch limit). I therefore designate **Proxy Anchor as the primary base**, whose ResNet-50 recipe is fully disclosed, and treat FRAME+PFML as a secondary arm contingent on a successful matched PFML reproduction. If my PFML reproduction lands more than 1.0 pt below published on CUB, **the FRAME+PFML arm is void** and no frontier claim is made from it.
- **A4.** Proxy Anchor's disclosed budget is 60 epochs with step decay, not the lane's 200. Both budgets are run for baseline and FRAME; conclusions must hold at the lane budget.
- **A5.** The exact cw-CR / cw-VR formulas (arXiv 1809.09307) could not be extracted — the PDF returned as binary and the summary page carried only the abstract. §3.1's distinction rests on the published description of the method, not on its equations. **If cw-CR turns out to already include a cross-class heterogeneity term or an unbiased small-sample estimator, §3.1's distinction weakens materially and Term 1's contribution reduces to the coordinate-free reading plus the estimator.**
- **A6.** CPCANet was seen only through a search snippet; I could not verify its objective, venue, or whether it is peer-reviewed. Its distinction in §3.2 is provisional.
- **A7.** Third-party ResNet-50/512-D Proxy Anchor numbers vary across reproductions (~0.69–0.70 CUB); the original paper uses Inception-BN. My baseline forecast carries that spread in its stated SD.
- **A8.** DADA's In-Shop 0.930 seed count and uncertainty are unreported.

---

## What would make me withdraw this

If C5 shows Term-1-only $\approx$ full FRAME, the method collapses toward cw-CR under a different readout and the novelty case rests only on the estimator. If C1 (conditional isotropy) matches FRAME, the modulation band is dead weight and the correct method is simply conditional whitening. If F2 fails — $\bar\rho$ does not fall — then whatever produced any gain is not the stated mechanism, and the honest report is a null on the mechanism regardless of R@1. I would rather surface those three outcomes now than have them found later.

**Sources:**
- [Potential Field Based Deep Metric Learning (PFML), arXiv 2405.18560](https://arxiv.org/abs/2405.18560)
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [official implementation](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
- [Utilizing Class Information for Deep Network Representation Shaping (cw-CR / cw-VR), arXiv 1809.09307](https://arxiv.org/pdf/1809.09307)
- [Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning, AISTATS 2019](https://proceedings.mlr.press/v89/hyvarinen19a.html)
- [On the Identifiability of Nonlinear ICA: Sparsity and Beyond, arXiv 2206.07751](https://arxiv.org/pdf/2206.07751)
- [Non-isotropy Regularization for Proxy-based Deep Metric Learning, CVPR 2022](https://arxiv.org/pdf/2203.08547)
- [Revisiting Training Strategies and Generalization Performance in Deep Metric Learning, ICML 2020](https://arxiv.org/abs/2002.08473) · [code](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch)
- [VICReg, arXiv 2105.04906](https://arxiv.org/pdf/2105.04906)
- [Towards Improved Proxy-based DML via Data-Augmented Domain Adaptation (DADA), arXiv 2401.00617](https://arxiv.org/pdf/2401.00617)
- [Simultaneous Diagonalization and SVD of Commuting Matrices, arXiv 2006.16364](https://arxiv.org/abs/2006.16364)
- [Fast Computing for Distance Covariance (fourth-order U-statistics), arXiv 1410.1503](https://arxiv.org/pdf/1410.1503)
- [CPCANet: Deep Unfolding Common PCA for Domain Generalization](https://arxiv.org/pdf/2605.05136) *(verified only via search snippet — ambiguity A6)*

