# Frozen Pass 79 proposal (blind proposer; verbatim)

 You're out of usage credits. Run /usage-credits to keep using Fable 5 or /model to switch models.
 ## Search summary (axes considered, then discarded)
 
 | Axis | Candidate | Why discarded |
 |---|---|---|
 | Geometry | Polyspherical / worst-case-block descriptor (train on soft-min over per-block cosines, which exactly sandwiches the deployed cosine) | Fatal: `min_g cos_g ≤ mean_g cos_g` with equality iff blocks are identical, so the objective *rewards* block collapse. Same defect for max-over-blocks on negatives. |
 | Supervision | Random label-code / ECOC sub-embeddings | Collides hard with DREML, BIER, Divide-and-Conquer. |
 | Geometry | Nonnegative top-k sparse "parts" descriptor (SPLADE-for-DML) | Attractive deployment story (inverted index), but the dominant attractor is one-hot class-indicator atoms, and honest forecast is neutral-to-negative R@1 on 6k-image training sets. |
 | Optimizer | Orthogonalized (Muon-style) updates on the embedding head to prevent head rank collapse | Real but small; reads as "we swapped the optimizer," and matched-search controls likely eat the gain. |
 | Supervision | Episodic pseudo-unseen classes; noisy-prototype perturbation | Overlaps Proxy Synthesis / meta-DML; weak differentiation. |
 
 ## Proposal: GCNP — Geometric-Consensus Nuisance Projection (name provisional)
 
 **One-line.** Estimate, online during Proxy Anchor training, the low-rank subspace along which *every* training class spreads (a log-det **geometric** consensus over per-class within-class covariances, not the pooled arithmetic one), and hard-project it out of the descriptor. No new loss term, no loss weight, zero inference cost.
 
 ### Mathematical training object
 
 Backbone $f_\theta(x)\in\mathbb R^{2048}$ (GAP), head $W\in\mathbb R^{2048\times512}$, $z=Wf_\theta(x)$, $\hat z=z/\|z\|$.
 
 Frame $U\in\mathrm{St}(512,r)$ (stop-grad), strength $\gamma\in[0,1]$. Descriptor and proxies:
 
 $$e(x)=(I-\gamma UU^{\!\top})z,\quad \hat e=e/\|e\|,\qquad \tilde p_c=(I-\gamma UU^{\!\top})p_c,\ \hat p_c=\tilde p_c/\|\tilde p_c\|$$
 
 Loss is **unmodified** Proxy Anchor on $s(x,c)=\langle\hat e(x),\hat p_c\rangle$:
 
 $$\mathcal L=\tfrac1{|P^+|}\sum_{c\in P^+}\log\Big(1+\!\!\sum_{x\in X_c^+}\!\!e^{-\alpha(s-\delta)}\Big)+\tfrac1{|P|}\sum_{c\in P}\log\Big(1+\!\!\sum_{x\in X_c^-}\!\!e^{\alpha(s+\delta)}\Big)$$
 
 $U$ is an alternating-optimization variable, refreshed every $T$ steps from EMA per-class statistics computed on the **unprojected** $\hat z$ (so $U$ can be revised):
 
 $$\mu_c\!\leftarrow\!(1-m)\mu_c+m\,\overline{\hat z},\quad S_c\!\leftarrow\!(1-m)S_c+m\,\overline{(\hat z-\mu_c)(\hat z-\mu_c)^{\!\top}},\quad \tilde\Sigma_c=\tfrac{S_c+\varepsilon I}{\mathrm{tr}(S_c+\varepsilon I)}$$
 
 $$U\leftarrow\arg\max_{U^{\!\top}U=I_r}\ \tfrac1C\sum_{c=1}^{C}\log\det\!\big(U^{\!\top}\tilde\Sigma_c U\big),\qquad \nabla_UJ=\tfrac2C\sum_c\tilde\Sigma_c U\,(U^{\!\top}\tilde\Sigma_c U)^{-1}$$
 
 20 warm-started Riemannian steps (Cayley/QR retraction) every $T{=}200$ steps.
 
 **Why log-det and not the pooled arithmetic mean.** $\log$ diverges to $-\infty$ as a class's variance along $v$ approaches zero, so a direction is selected only if it is noisy for *every* class. A direction that is high-variance inside a few classes but tight inside the rest — precisely the profile of a *transferable discriminative attribute* — is rejected. The arithmetic pooled covariance (i.e., standard supervised whitening) has no such protection; it is dominated by the loudest classes. This is the entire novelty claim, and it is isolated by one ablation arm.
 
 ### Causal mechanism
 
 Proxy Anchor drives within-class variance to ~0 along seen-class discriminative directions. The residual within-class scatter therefore concentrates on nuisance (pose, viewpoint, illumination, shop-vs-consumer domain). For seen classes these directions are already compressed, so deleting them barely moves the training loss. For **unseen** classes no proxy has compressed anything, so those same directions inject raw noise into every cosine score and flip nearest-neighbor decisions. GCNP deletes them and forces the head to reallocate capacity into the complement. The predicted signature is a pure generalization effect with a near-flat training loss delta.
 
 ### Causal failure modes
 
 1. **Consensus nuisance that is unseen-class signal.** Male/female Northern Cardinal differ enormously in plumage color; color is within-class noise there but is the primary discriminator between many unseen species. If color survives into $U$, CUB R@1 drops. The log-det criterion mitigates this only if color is tight in a decent fraction of classes — an empirical question, hence falsifier F1. Cars196 is the clean case (paint color is high-variance inside essentially every model), which is why it is the preregistered primary target.
 2. **Alternating-projection chase.** Project out $U$ → residual scatter migrates → $U$ rotates → oscillation. Mitigation: $\gamma$ ramp over epochs 1–5, $T{=}200$, Procrustes-aligned EMA on $U$. Diagnostic: principal angle between successive frames must decay; if it plateaus above ~20°, the method is unstable and that is the reported result.
 3. **Rank-deficient per-class covariance.** In-Shop has ~5 images/item, so $\mathrm{rank}(S_c)\le4$ and $\log\det(U^\top\tilde\Sigma_cU)$ is pure shrinkage noise for $r>4$. Requires Ledoit–Wolf shrinkage toward the pooled covariance and $r\le16$; In-Shop is a stress test, not primary.
 
 ### Numeric forecast (ResNet-50, 512-d, 224², 5 seeds, seed-mean raw R@1)
 
 | | Reproduced PA baseline | GCNP predicted Δ | 90% interval |
 |---|---|---|---|
 | **Cars196** (primary) | ~87.5 ± 0.4 | **+1.2** | [+0.4, +2.0] |
 | In-Shop (stress) | ~91.9 ± 0.2 | +0.7 | [+0.1, +1.3] |
 | CUB (secondary) | ~69.5 ± 0.5 | +0.2 | [−0.6, +1.0] |
 
 Selection-corrected estimates: subtract ~0.3–0.5 from each. Expected best config: $r{=}32,\gamma{=}1.0$ on Cars/In-Shop; $r{\le}16,\gamma{\le}0.8$ on CUB. **The differential CUB-vs-Cars prediction is itself a test of the mechanism** — a uniform gain across datasets would suggest a generic regularization effect, not nuisance removal.
 
 ### Preregistered falsifiers
 
 - **F1 (mechanism, <1 GPU-hour, protocol-clean).** Split the 100 training classes 75/25. Estimate $U$ on the 75; on the held-out 25 measure removed-within-class-energy ÷ removed-between-class-energy. If ratio < 1.3, **abandon before the sweep**.
 - **F2 (primary).** Cars196 seed-mean gain < +0.4 R@1 against a matched-compute, matched-search-budget PA baseline ⇒ refuted.
 - **F3 (novelty).** If arm-4 (arithmetic pooled-covariance projection = supervised whitening, trained online) matches arm-3 within 0.2 R@1 on all datasets, the contribution collapses to known whitening ⇒ report as a null-novelty result.
 - **F4 (train-time claim).** If post-hoc projection applied to a *frozen* baseline recovers ≥80% of the gain, this is a post-processing trick, not a training method — report it as such.
 
 ### Experimental design
 
 Ablation ladder: (0) PA baseline; (1) post-hoc Radenović-style whitening; (2) post-hoc consensus projection; (3) **online consensus projection (full)**; (4) online arithmetic-mean projection; (5) $r,\gamma$ sweeps.
 
 **Matched compute.** GCNP overhead: per-sample $512^2$ outer products ≈ 33 MFLOP/step vs ~512 GFLOP/step for the backbone (<0.1%); frame refresh amortized to <1% wall-clock. Compute matching is therefore trivial, but **search budget matching is not** — the baseline must receive an equal-count hyperparameter search (e.g. 12 lr/$\alpha$/$\delta$ configs × 2 seeds) under the identical selection rule.
 
 **Honest / selection-corrected reporting.** Model selection uses leave-class-out folds *inside the training split only* (Musgrave et al.'s class-based cross-validation), never test identities; the selected config is then retrained on all training classes. Report: full grid distribution, seed-mean ± std, paired bootstrap CIs over query images, R@1/2/4/8 **and MAP@R**, plus baseline best-of-$N$ under the same $N$ so the selected-vs-selected comparison is like-for-like. Any test-class-based mechanism analysis is reported separately and explicitly labeled post-hoc.
 
 Budget ≈ 58 runs/dataset ≈ 20–40 A100-hours/dataset.
 
 ### Deployment constraints
 
 - Descriptor stays 512-d float, $\ell_2$-normalized, cosine/inner-product — FAISS IVF-PQ / HNSW compatible.
 - $(I-\gamma UU^{\!\top})$ folds into the head at export ($W'\!=\!(I-\gamma UU^{\!\top})W$): **identical inference cost and parameter count**, no architecture change.
 - At $\gamma{=}1$ the descriptor spans rank $512-r$ (480 at $r{=}32$) — a real deviation from full-rank 512-d that must be disclosed; $\gamma<1$ preserves full rank at some cost in effect size.
 - Training memory: full covariances need $C\!\cdot\!512^2\!\cdot\!4$B (CUB/Cars ≈ 105 MB, fine); In-Shop ($C{=}3997$) requires rank-8 streaming sketches (≈65 MB) — the main engineering risk.
 - Data: benchmark training images + labels only, plus the standard ImageNet-pretrained backbone. No generative or text models, no test identities.
 
 ### Closest primary-literature risks
 
 1. **Radenović, Tolias & Chum, TPAMI 2018 (supervised whitening from matching pairs); Mikulík et al. 2010; Gordo et al. 2016.** *Highest risk.* Difference: arithmetic pooled covariance + full whitening + post-hoc, versus geometric consensus + low-rank hard projection + online co-adaptation. Arms 1 and 4 exist specifically to adjudicate this; if they win, the pass reports a negative novelty result rather than a claim.
 2. **Roth et al., ICML 2020** (ρ-spectral regularization; less compression ⇒ better transfer). Risk of being read as another anti-compression regularizer. Difference: targeted removal of identified directions with a differential dataset prediction, not global spectrum flattening.
 3. **Dorfer et al., ICLR 2016 (Deep LDA)** and discriminant-analysis losses. Difference — arguably the opposite: LDA shrinks within-class scatter globally (which is the mechanism *causing* failure mode 1); GCNP deletes only a low-rank consensus subspace and leaves the remaining within-class variance intact.
 4. **Ravfogel et al., ACL 2020 (INLP, iterative null-space projection).** Same operator, different discovery: INLP needs labeled nuisance attributes; here the subspace is unsupervised-in-the-nuisance and derived from class-consensus scatter.
 5. **DiVA / MIC / DVML** — shared-vs-class-specific factor disentanglement. Motivational overlap; mechanically unrelated (no auxiliary branches, no adversarial decorrelation, no extra loss term).
 6. **Arsigny et al. (Log-Euclidean SPD means)** — the underlying tool, not a competitor. **SoftTriple / ProxyGML / HIST** are orthogonal (multi-proxy) and could compose.
