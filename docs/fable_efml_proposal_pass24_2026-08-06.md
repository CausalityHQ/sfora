# Fable EFML proposal — blind Pass 24 — 2026-08-06

- Durable consultation: `020da7dacb704133`
- Frozen prompt: `docs/fable_blind_prompt_pass24_2026-08-05.txt`
- Prompt SHA-256: `b7c988ae88b7c956e6656829d64ce838d09009497a5347eeaf6c9eae9058a3eb`
- Caller attribution: `sfora/emafactorial`
- Provider: Fable only; completed with exit 0.
- Result truncation: false.
- Preservation: exact durable `result` field; no local edits to proposal text.

All references and threats are now pinned. I have what I need to freeze the proposal. Composing the complete deliverable now.

---

# Frozen proposal — EFML: Exchangeable-Fiber Metric Learning

**Decision: ONE method, Lane A** (matched 512-D CNN lane: ResNet-50, 512-D L2-normalized descriptor, 224-px single view, cosine NN, ≤200-epoch budget).

**Essence.** Zero-shot nearest-neighbour retrieval transfers only if distances are *commensurable across identities*: the law of within-identity displacement around an identity's center must be the same for every identity, including unseen ones. Standard proxy losses leave this law unconstrained per class (they bound sample-proxy similarities, not conditional laws), and their gradient pressure drives it toward heterogeneous near-collapse. EFML adds a single distribution-level symmetry constraint to Proxy-Anchor: **the transported within-class displacement distribution, expressed in a common tangent gauge on the embedding sphere, must be identical across classes (kernel two-sample penalty against a pooled reference) and must have a pinned, non-zero dispersion (frozen at warm-up)**. Zero new learned parameters, no synthetic features, no auxiliary networks, no test-time change.

---

## 1. Executable mathematics

### 1.1 Base objective (named published baseline, reproduced)

Backbone: ImageNet-1K-pretrained ResNet-50, global-average-pool features h(x) ∈ R^2048; linear head W ∈ R^{512×2048}; z = Wh(x); deployed descriptor ẑ = z/‖z‖₂. Learned proxies p_c ∈ R^512, one per training class, p̂_c = p_c/‖p_c‖₂.

Proxy-Anchor loss (Kim et al., CVPR 2020), exact reduction, s(x,p)=cos(ẑ,p̂):

L_PA = (1/|P⁺|) Σ_{p∈P⁺} log(1 + Σ_{x∈X_p⁺} e^{−α(s(x,p)−δ)}) + (1/|P|) Σ_{p∈P} log(1 + Σ_{x∈X_p⁻} e^{α(s(x,p)+δ)}), with α=32, δ=0.1.

Primary-source recipe (official repo, ResNet-50/512): CUB — batch 120, lr 1e-4, warm-up 5 epochs (head+proxies only), BN frozen, lr-decay step 5; Cars — same with decay step 10; published R@1 69.9 (CUB), 87.7 (Cars). **Unresolved source ambiguities, stated:** (i) the README does not restate optimizer family, weight decay, decay γ, epochs, or input pipeline — my protocol fixes AdamW, wd 1e-4, proxy-lr ×100, γ=0.5, 60 epochs, resize-256/random-crop-224/horizontal-flip (test: center-crop 224), which matches the official train.py defaults to my best knowledge but must be re-pinned from that file before running; (ii) **no official ResNet-50/512 recipe exists for SOP/In-Shop** (README documents Inception-BN only) — I define and disclose mine: batch 180, lr 6e-4, warm 1, decay step 20 γ=0.25, BN unfrozen, 60 epochs; (iii) the repo's random sampler is replaced by a class-balanced sampler (30 classes × 4 for CUB/Cars; 90 × 2 for SOP/In-Shop) because EFML needs ≥2 samples/class — **the matched baseline PA-repro uses the identical sampler**, so this is not a confound. All deltas are against my PA-repro under this frozen protocol; I do not inherit published frontier rows after recipe changes. Loss weights below are operative *given* wd 1e-4 and these lrs (no "harmless normalization" claim).

### 1.2 EFML constraint

Geometry on S^511 (all closed-form, autodiff-friendly; clamp cosines to ±(1−10⁻⁷)):

- Log map at base a: for unit ẑ, φ = arccos⟨a,ẑ⟩, u = ẑ − ⟨a,ẑ⟩a; Log_a(ẑ) = φ·u/‖u‖ (0 if φ=0).
- Gauge: fixed random unit pole o (frozen at init). For each class c present in the batch, R_c = the minimal rotation taking p̂_c to o (closed form in the span{p̂_c,o} plane; identity off-plane).
- **Transported fiber displacement** of sample i in class c: ξ_i = R_c · Log_{p̂_c}(ẑ_i) ∈ T_oS ⊂ R^512. Gradients flow through both ẑ_i and p̂_c; queue/buffer entries are detached.

Estimation machinery (no learned objects): fixed random Johnson–Lindenstrauss projection P_J ∈ R^{64×512}, entries N(0,1/64), frozen; u_i = P_J ξ_i. Pooled FIFO queue Q of detached u's (4096 for CUB/Cars; 16384 for SOP/In-Shop); per-class FIFO buffers B_c of detached u's (16 for CUB/Cars; 8 for SOP/In-Shop) to stabilize small per-class batch counts. Gaussian kernel k_σ with σ = median pairwise distance in Q, refreshed every 200 steps without gradient; per step, subsample Q̃ ⊂ Q, |Q̃|=512.

**Shape-equality term** (per-class two-sample statistic against the pooled law, log-sum-exp-aggregated so gradient focuses on the most deviant classes, mirroring PA's own LSE design):

L_hom = (1/β) log[ (1/C_B) Σ_{c∈batch} exp(β · MMD²_k(A_c, Q̃)) ], A_c = {u_i: i∈c, with grad} ∪ B_c, β = 5.

**Dispersion pin** (blocks the collapse solution): m̂ = batch mean of ‖ξ_i‖² (full 512-D, radians²); L_sc = (m̂/m* − 1)², with m* frozen as the running mean of m̂ over the final warm-up epoch.

**Total:** L = L_PA + λ_hom·L_hom + λ_sc·L_sc, λ_hom = 2.0, λ_sc = 0.5 (defaults frozen; the only permitted tuning is λ_hom∈{1,2,4}, λ_sc∈{0.25,0.5,1} on a class-disjoint validation split carved from *training* classes (15%), then retrain on full train; test never touched).

Schedule: epochs 1–5 warm-up (PA only; queues/buffers/σ/m* initialized during epoch 5); EFML terms active from epoch 6; everything else per §1.1. Train/test operations at deployment are *identical to PA*: one ResNet-50, one 224 center-crop, one 512-D normalized vector, cosine NN.

## 2. Causal zero-shot error mode and degeneracy attack

**Error mode: fiber-law incommensurability.** Model identities as base points μ_y ∈ S^511 with observations Exp_{μ_y}(ξ), ξ ~ F_y in the transported gauge. (A) *Exchangeability lemma:* if F_y ≡ F for all y and identity centers are drawn exchangeably, the joint law of (within-identity distance, cross-identity distances) is identity-independent — so NN ranking statistics tuned on seen identities transfer to disjoint identities at matched center spacing; the "generalization gap across identities" reduces to center-spacing, which the discriminative loss controls. (B) *Margin bound:* if the transported fiber has geodesic support radius R and novel centers are ≥γ apart, NN is exact whenever γ > 4R (triangle inequality both ways). (C) *Heterogeneity ⇒ irreducible error, invisible to the base loss:* take class B with fiber radius Δ and a tight neighbor A at center distance γ; sample-to-proxy margins can all be satisfied (Δ within margin slack) while sample-to-sample within-B distances reach 2Δ > γ−Δ whenever Δ > γ/3 — so B-queries retrieve A-gallery items even at zero training loss. PA's negative-log-likelihood pressure monotonically shrinks fibers, so finite training halts at *heterogeneous, uncontrolled* F_c — exactly the slack (C) exploits. **Pre-registered premise check P0:** under PA-repro, the per-class transported scatter m̂_c must vary across classes by ≥1.5× (90th/10th percentile ratio) on each dataset; if not, the premise — and the method — is falsified before any EFML run.

**Cheapest degeneracies, attacked.** (i) *Collapse* (F=δ₀ satisfies equality): excluded by L_sc with m*>0 frozen at warm-up; collapse would also make novel-class fibers pure extrapolation of an unexercised scatter function — the spectral-decay/over-compression pathology. (ii) *Cosmetic scatter* (fabricate class-symmetric junk): the statistic is computed from *real* within-class displacements of a deterministic encoder; driving L_hom→0 forces the scatter function's class-conditional pushforwards to coincide, i.e., the fiber carries only class-exchangeable image information — and the only image factors available with that property are generic nuisance factors (pose, lighting, background), which is precisely the intended factorization. (iii) *Residual leak I acknowledge:* equality of marginal fiber laws does not forbid class-dependent base–fiber *coupling* (copula); the held-out-class statistic below tests for it empirically rather than pretending the proof closes it. (iv) *Proxy gaming* (proxies drift to fake the statistic): detected by placebo control C5 and by the fact that proxies simultaneously serve L_PA. (v) *Real-world non-exchangeability* (mode-weight heterogeneity — some birds only photographed perched): the pooled MMD target is nonparametric, so *shared* multimodality is permitted; equalizing mode *weights* is the method's deliberate bet, and its failure would surface as vanishing CUB gains (stated as a failure mode, priced into the conservative CUB forecast).

## 3. Adversarial novelty search (primary sources; one-sentence mechanism distinctions)

Inside DML: **NIR** (CVPR 2022) trains a shared normalizing flow for unique proxy-to-sample translatability — a density/structure objective that never compares residual laws *across* classes, pins no dispersion floor, and uses no transport gauge. **SFT** (ECCV 2020) — nearest of all — *augments* by rotating features between classes under an *assumed* similarity of spherical class covariances; EFML synthesizes nothing and instead *enforces* that similarity as a differentiable constraint (making the assumption true rather than exploiting it), with SFT reimplemented as an occupied-alternative control. **DADA** (AAAI 2024) domain-adapts the sample distribution toward the *proxy* distribution (samples-vs-proxies as two domains); EFML aligns class-conditional laws *across classes* — a different alignment index set. **PFML** (CVPR 2025) is a potential-field reformulation preserving sample–sample relations; no cross-class law constraint. **DVML** (ECCV 2018) *assumes* class-invariant additive Gaussian intra-class variance inside a VAE to generate samples; EFML is generation-free, spherical, and enforces rather than assumes, with a pinned non-zero scale. **Density Adaptivity** (Yao et al. 2019) balances per-class compactness/density; no cross-class equality, no gauge, no pinned reference. **Embedding Expansion / Proxy Synthesis / Metrix**: point-synthesis or mixup supervision, no distributional symmetry constraint. **ρ-regularization / coding-rate anti-collapse losses**: global, class-agnostic spectrum shaping, which neither implies nor is implied by equality of per-class conditional laws. Outside DML: **FTL for faces** (Yin et al., CVPR 2019) transfers principal within-class variance from rich to under-represented classes via a decoder (imbalance repair, Euclidean, asymmetric); **VPL** (CVPR 2021) injects memorized variations into prototype logits; **ISDA** augments along *class-specific* covariances for closed-set classification; **Distribution Calibration** (ICLR 2021, few-shot) borrows base-class Gaussians at *test time* under an untested homogeneity assumption — EFML trains the encoder so that assumption becomes true, with nothing at test time. **VICReg**'s variance term is a batch-global per-dimension floor against SSL collapse — not class-conditional, not gauged, not equality-across-conditions. **Deep LDA** pools within-class covariance (homoscedasticity assumed by the estimator, never penalized as a deviation). **Conditional domain alignment** (CDAN et al.) aligns class-conditionals across *domains*, not across *classes* within one domain. I found no primary source that enforces cross-class equality of transported within-class distributions, with a pinned dispersion, as a train-time constraint for zero-shot retrieval; the closest composite ancestors are SFT (assumption, augmentation) and NIR (shared flow, density), each distinct at the mechanism level as stated.

## 4. Decisive matched-compute controls (identical sampler/epochs/recipe; 5 seeds each)

C1 PA-repro (operative baseline). C2 PA+L_sc only — is the anti-collapse floor alone sufficient? C3 PA+L_hom only (no pin) — does equality-without-floor collapse? C4 full method with Euclidean residuals z−p (no Log/transport) — is the spherical gauge causal or cosmetic? C5 **placebo**: each class matched to *its own* EMA buffer (machinery identical, cross-class sharing deleted) — the sharpest test that exchangeability, not regularization-by-MMD, is the mechanism. C6 scalar homoscedasticity (equalize per-class second moments m̂_c only) — does full-law matching add over variance equalization? C7 PA+SFT reimplementation — enforced constraint vs. occupied assumption-exploiting augmentation at matched cost. C8 (upside probe, no claims): EFML+SFT composition. Mechanism-validity metric, pre-registered: E* = median energy distance between transported per-class displacement sets on **held-out training-split classes**; prediction: EFML reduces E* by ≥30% vs C1 and per-dataset gains correlate with E* reduction.

## 5. Frozen forecasts, falsification thresholds, frontier arithmetic (Lane A)

All numbers R@1, 5 seeds, mean±std; *final-epoch* convention (no test-based selection), with best-epoch numbers additionally reported and flagged because the references use best-test selection.

| Dataset | PA-repro (forecast) | **EFML (frozen forecast)** | Δ | Reference (their convention) | Crossing call |
|---|---|---|---|---|---|
| CUB | 69.4±0.4 (official best 69.9) | **71.8±0.5** (best ≈72.2) | +2.4 | PFML 73.4±0.3 | no (P≈0.10) |
| Cars196 | 87.3±0.4 (official best 87.7) | **89.2±0.5** (best ≈89.6) | +1.9 | PFML 92.7±0.3 | no (P≈0.02) |
| SOP | 79.6±0.4 (assumed lit. ≈79.8, uncertain) | **82.2±0.4** (best ≈82.6) | +2.6 | PFML 82.9±0.2 | secondary: P≈0.35 |
| In-Shop | 91.7±0.3 (assumed lit. ≈92.1, uncertain) | **93.2±0.3** (best ≈93.6) | +1.5 | PA+DADA 93.0 (1 run, ~1.06× cost) | **primary: P≈0.60** |

Frontier arithmetic: In-Shop crossing requires Δ ≥ 93.0 − 91.7 = +1.3 over my PA-repro (final-epoch) or +0.9 over the literature PA figure; forecast Δ = +1.5, margin +0.6 at best-epoch convention against a single-run reference of unknown variance and higher training cost (mine ≈1.04×, theirs ≈1.06×). SOP crossing requires Δ ≥ 82.9 − 79.6 = +3.3; forecast Δ = +2.6–3.0, hence deficit −0.3 at point estimate — a genuine coin-toss-minus, stated as such. CUB/Cars are forecast *sub-frontier* against PFML's multimodality-specialized 15-proxy machinery (EFML keeps one proxy/class); I decline to inflate those forecasts. **Falsification (any one kills the corresponding claim):** F0 premise P0 fails. F1 mean Δ over C1 < +1.0 on ≥3 of 4 datasets → method dead. F2 In-Shop < 93.0 *and* SOP < 82.4 (best-epoch) → frontier claim dead. F3 placebo C5 within 0.3 of EFML everywhere → exchangeability not causal → mechanism claim dead. F4 Euclidean C4 within 0.2 everywhere → geometric gauge unnecessary → reduce claim to (still novel) ungauged law-equalization. F5 E* reduction <15% or uncorrelated with gains → causal chain unsupported.

## 6. Costs and risks

**Cost.** Zero new parameters; transport is O(B·d); MMD on 64-D projections with |A_c|≤20, |Q̃|=512 is ~20M multiply-adds/step (<1% of the ResNet forward); total step overhead ≤5%; memory overhead ≤40MB (queues/buffers, fp16); training length unchanged (60 epochs, well under the 200 budget; extending to 200 with pinned fibers is a possible upside, not claimed). Deployment is byte-identical to PA: single R50-512, single view, cosine — 1.00× inference.

**Risks.** (1) Reference-convention asymmetry: cited rows use best-epoch-on-test selection (field-wide contamination); my fixed-epoch protocol is stricter and may understate crossings — both conventions reported. (2) PA-R50 SOP/In-Shop baselines are not officially disclosed; my assumed 79.8/92.1 anchors carry ±0.5 uncertainty that propagates to crossing probabilities. (3) CUB shares bird species with ImageNet-1K pretraining (shared by all references, not differential). (4) Cars196 label noise and In-Shop near-duplicate query/gallery pairs cap headroom. (5) Genuine non-exchangeability of nuisance across classes (mode-weight heterogeneity) would convert L_hom into a bias — the CUB forecast prices this in. (6) Small per-class counts (M=2) on SOP/In-Shop make per-class MMD noisy despite buffers; if C6 (scalar version) matches full EFML there, the distribution-level claim narrows to CUB/Cars. (7) Queue staleness under fast proxy drift early after warm-up; mitigated by short queues, unmodeled residual risk.

**Uncertainties stated plainly:** the honest probability that EFML beats *every* PFML row is low; the defensible claims are the matched-cost mechanism (large, controlled, falsifiable gains over an audited baseline), a primary In-Shop crossing at ~0.6 probability, and a near-coin-flip SOP crossing. The frozen-proposal reviewer should weigh it on that basis.

Sources: [PFML (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [DADA (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29400) · [NIR (CVPR 2022)](https://arxiv.org/abs/2203.08547) · [SFT (ECCV 2020)](https://arxiv.org/abs/2008.01469) · [Proxy-Anchor paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html) · [Proxy-Anchor official repo](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)

