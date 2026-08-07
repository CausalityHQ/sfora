# Frozen Pass 77 proposal (blind proposer; verbatim)

+ You're out of usage credits. Run /usage-credits to keep using Fable 5 or /model to switch models.
 ## Method: **Nuisance‑Anisotropic Margin (NAM)** — worst‑case‑over‑class‑region Proxy Anchor
 
 **Where I searched and why I landed here.** Architecture/activation/optimizer changes in this benchmark family are mostly compute‑confounded and hard to control honestly; sampling changes are largely subsumed by Proxy Anchor's design. The one axis where the train objective is provably *silent* is the second moment of the class‑conditional embedding distribution. That is a supervision‑construction × representation‑geometry intervention, adds zero inference cost, and admits an exactly matched control. That's where I went.
 
 ---
 
 ### 1. Causal failure mode
 
 Proxy Anchor supervises **instance‑to‑class‑mean** similarity. Zero‑shot Recall@K is decided by **instance‑to‑instance** similarity among classes that have no proxy at all. The gap between the two quantities is governed entirely by the within‑class residual distribution — and PA models each class by its first moment only, so no term in the objective controls that gap. Two consequences:
 
 - The *closest plausible instance* of a negative class can be far closer to a query than that class's proxy is. PA never penalizes this, yet it is exactly what produces an R@1 error.
 - The systematic gradient on an embedding, `∂s(z,p)/∂z = (p̂ − s·ẑ)/‖z‖`, lies in `span({p}) ∪ {ẑ}`. With C ≈ 100 training classes and d = 512, the class‑discriminative pressure occupies at most a ~C‑dimensional subspace; the residual geometry in the other ~400 directions is whatever the backbone happened to produce.
 
 The load‑bearing hypothesis: **within‑class nuisance directions (pose, viewpoint, illumination, crop) are largely class‑agnostic and therefore transfer to unseen identities**, while the class means do not. So the *pooled* second moment is the part of the training‑class distribution that is legitimately extrapolable to test identities — and it is currently unused.
 
 ### 2. Training object
 
 Maintain a pooled, tangential, within‑class second moment over training embeddings (stop‑grad, EMA, `m ∈ {0.9, 0.99}`):
 
 ```
 r_i  = ẑ_i − (ẑ_i·p̂_{y_i}) p̂_{y_i}                    # tangent residual to own proxy
 Σ̄   ← m Σ̄ + (1−m)·(1/B) Σ_i r_i r_iᵀ
 Σ̄_λ = (1−λ) Σ̄ + λ (tr Σ̄ / d) I                       # shrinkage, λ ∈ {0.1, 0.3}
 ```
 
 Replace each point proxy by an ellipsoidal **class region** `Q_p = {p̂ + u : uᵀΣ̄_λ⁺u ≤ ε²}` and take the worst case over it. The support function of an ellipsoid is closed form, `max_{u∈εU} ẑᵀu = ε√(ẑᵀΣ̄_λ ẑ)`, so:
 
 ```
 s⁻(z,p) = ẑ·p̂ − ε σ(ẑ)     (own class: farthest plausible instance)
 s⁺(z,p) = ẑ·p̂ + ε σ(ẑ)     (other class: closest plausible instance)
 σ(ẑ)    = √(ẑᵀ Σ̄_λ ẑ)
 ```
 
 Both fold **exactly** into a per‑sample margin. Using the mean‑preserving parameterization (`σ̃_i` = `σ(ẑ_i)` divided by its stop‑grad batch mean, so `mean(σ̃)=1`):
 
 $$\delta_i \;=\; \delta\big((1-\beta) + \beta\,\tilde\sigma(\hat z_i)\big),\qquad \beta\in[0,1]$$
 
 $$\mathcal{L}_{\text{NAM}}=\frac{1}{|P^+|}\sum_{p\in P^+}\log\Big[1+\!\!\sum_{z\in Z_p^+}\!\!e^{-\alpha(\hat z\cdot\hat p-\delta_i)}\Big]\;+\;\frac{1}{|P|}\sum_{p\in P}\log\Big[1+\!\!\sum_{z\in Z_p^-}\!\!e^{+\alpha(\hat z\cdot\hat p+\delta_i)}\Big]$$
 
 `β = 0` recovers Proxy Anchor **exactly**, and mean margin is `δ` for every `β`. Gradient through `σ̃` adds `−αδβ·Σ̄_λẑ/σ` — a term outside `span(P)` that pushes embeddings off high‑nuisance directions. Warm up `β: 0 → β_max` over the first 10% of steps (proxies are random early, so `Σ̄` is meaningless).
 
 Three lines of code; `α`, `δ`, backbone, optimizer, sampler, epochs all unchanged from the PA recipe. Pooled (not per‑class) `Σ̄` is required both for transfer *and* for estimability — In‑Shop has ~5 images per class, where per‑class covariance is hopeless. Residuals are taken against the proxy, so no balanced‑sampling constraint is introduced.
 
 ### 3. Pre‑flight falsifier (run this first, ~10 GPU‑min, no training)
 
 Take one existing PA checkpoint. Compute `Σ̄` from train‑class residuals and `Σ_w` from test‑class residuals. **Pre‑register: if Spearman correlation of their eigenvalue spectra under the train eigenbasis is < 0.5, or the top‑50 principal‑subspace overlap is at chance, the transferability hypothesis is false and NAM is predicted not to work — abandon before spending compute.** Second cheap kill: in a 100‑step pilot, if `std(σ̃)` across the batch < 0.1, the margin modulation is a no‑op.
 
 ### 4. Expected effect, forecast, falsifiers
 
 Pre‑registered Δ R@1 vs the `β=0` arm, mean of 5 seeds, matched HPO budget (80% intervals):
 
 | | CUB | Cars196 | In‑Shop |
 |---|---|---|---|
 | ΔR@1 | **+0.7** [−0.2, +1.6] | +0.4 [−0.4, +1.2] | +0.15 [−0.15, +0.5] |
 
 Mechanism prediction on **test** identities: `tr(S_w)/tr(S_b)` drops ≥10% vs baseline; test‑embedding energy along the top‑5 train‑`Σ̄` eigendirections drops.
 
 Three controls, all required, all cheap — a win that fails any of them is not the claimed mechanism:
 
 1. **Margin‑matched:** `β = 0` (identical mean margin). Guards against "bigger/softer margin."
 2. **Spectrum‑matched random rotation:** `Σ̄ → RΣ̄Rᵀ`, `R` a fixed random orthogonal matrix. Same spectrum, destroyed eigenbasis. Guards against "any anisotropic margin works."
 3. **Within‑batch shuffled `σ̃`:** same marginal distribution, association with the sample destroyed. Guards against "margin noise / label‑smoothing."
 
 **Primary falsifier:** if NAM does not beat arm 1 by more than one pooled seed‑std on CUB R@1 over ≥5 seeds, or if arms 2/3 match it, the mechanism claim is dead. **Secondary:** if R@1 rises but the `tr(S_w)/tr(S_b)` prediction does not move, report as "effect real, mechanism unconfirmed." **Degeneracy diagnostic:** if `|ρ(σ̃, ‖z‖)| > 0.7`, NAM has collapsed into MagFace‑style quality weighting; report it as such.
 
 **Honest prior:** P(pre‑flight passes) ≈ 0.55; P(mechanism‑confirmed CUB win | it passes) ≈ 0.5. **Net ≈ 25–30%.** Gains of this size in this literature frequently do not survive matched HPO, which is the point of the design below.
 
 ### 5. Deployment constraints & honest reporting
 
 - **Inference unchanged:** L2‑normalized 512‑d output, cosine retrieval, single model / single view / no TTA. `Σ̄` is discarded after training. Zero added inference cost or parameters.
 - **Train cost:** one `d×d` EMA (≈1 MB fp32) and one `B·d²` matvec per step (~2.4e7 FLOPs at B=90) — **<1% wall‑clock**, so matched compute is near‑exact; the baseline still gets identical steps, schedule, and **equal HPO trial count**.
 - **Data legality:** `Σ̄` uses only training‑split images and labels; no test identities, no external generative/text models. Assumes the standard ImageNet‑pretrained backbone — if that is disallowed, all baselines shift and this forecast is void.
 - **Selection correction is mandatory, not decorative.** With K configs at seed noise σ, max‑over‑configs inflation is ≈ σ√(2 ln K) ≈ **1.1 R@1** at K=12, σ=0.5 — larger than the predicted effect. Protocol: tune `β, λ, m` on a **class‑disjoint** validation split carved from training classes (e.g. CUB classes 81–100), then one final run on the official split. Report per‑seed raw values, the raw best‑config number, the selection‑corrected estimate, and a paired‑across‑seeds test with CI.
 - **Budget:** ~200 GPU‑h total (CUB+Cars: 4 arms × 5 seeds × 2 sets ≈ 40–60 h; In‑Shop: 4 arms × 3 seeds ≈ 100 h; plus HPO).
 - **Stability risk:** `α=32` makes the LSE sensitive to margin changes; a difficulty‑correlated margin can act as an unintended mining schedule. Mitigate with the warmup and grad clipping; report loss curves.
 
 ### 6. Closest primary‑literature risks (stated plainly)
 
 - **Wang et al., ISDA (NeurIPS 2019)** — *the nearest relative*. Class‑conditional feature covariance yields a closed‑form augmented‑CE bound. Differences: ISDA is closed‑set CE with a **quadratic** term in weight differences and **per‑class** Σ (unestimable and non‑transferable for unseen classes); NAM is a **worst‑case support function** (√ form, a margin in cosine units) with **pooled** Σ, which is precisely what the zero‑shot argument requires. Reviewers will call this adjacent — concede it and lean on the pooled/transfer distinction and the controls.
 - **Roth et al., Non‑isotropy Regularization for Proxy‑based DML (CVPR 2022)** — closest by topic name; uses a learned normalizing flow over proxy‑residual structure. NAM is parameter‑free and closed‑form and allocates *margin*, not residual density. Adjacent, not identical.
 - **Shivaswamy, Bhattacharyya & Smola (JMLR 2006); Ben‑Tal & Nemirovski** — the `wᵀΣw` second‑order‑cone robust margin is **classical**. The math is not the contribution; the transplant to proxy DML with a pooled, trace‑normalized, mean‑preserving margin is.
 - **Qian et al., SoftTriple (ICCV 2019); Gu & Ko, Proxy Synthesis (AAAI 2021)** — multi‑center proxies and synthetic‑proxy novel‑class simulation. Same failure mode intuition, different mechanism; both are natural additional baselines.
 - **MagFace (CVPR 2021), CurricularFace (CVPR 2020), AdaCos (CVPR 2019)** — adaptive margins keyed on norm/difficulty. Control 3 exists specifically to show NAM is not a relabeling of these.
 - **Hatch et al., WCCN (2006)** — post‑hoc within‑class whitening; NAM bakes the pressure into training so the deployed descriptor needs no transform.
 
 **Recommendation:** run the §3 pre‑flight check first. It costs almost nothing and it is the single most informative experiment in this proposal — it tests the one assumption everything else rests on.
