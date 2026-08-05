# FROZEN PROPOSAL — Return-Level Margin learning (RLM)

**Lane declaration: Lane A** (ResNet-50, 512-D normalized global descriptor, ~224-px input, single-view cosine retrieval, ≤200 epochs). All forecasts and comparisons below are Lane A only. Primary references: PFML (CVPR 2025) CUB 0.734±0.003, Cars 0.927±0.003, SOP 0.829±0.002; In-Shop reference PA+DADA (AAAI 2024) 0.930 (no reported seed spread).

**One-line summary.** Deployment R@1 is decided by the *maximum* similarity over thousands of impostors from *classes never seen in training*, yet every standard loss calibrates its repulsive pressure to the batch/seen-class *observed* similarity distribution. RLM fits, inside the loss and differentiably, a peaks-over-threshold (generalized Pareto) model to each anchor's impostor-similarity tail, and enforces the positive margin against the **extrapolated return level** — the estimated maximum similarity among `c·K` *fresh* draws from the class population — with a realized-max guard and a stop-gradient structure that provably prevents the network from gaming the estimator. It adds no parameters, no auxiliary networks, and ~1–2% step-time overhead.

---

## 1. Executable mathematics

### 1.1 Learned objects

- Backbone `f_θ`: ResNet-50, ImageNet-1K initialized, global average pool, linear projection to d=512, L2-normalization: `z_i = h(x_i)/||h(x_i)|| ∈ S^511`.
- Proxies: `M` sub-centers per training class, `p_{c,m} ∈ S^511`, learned, L2-normalized after each step. `M=4` on CUB/Cars, `M=2` on SOP/In-Shop. Class similarity: `s_{i,c} = max_m ⟨z_i, p_{c,m}⟩`.
- No other learned objects. `K` = number of training classes; `B` = batch size.

### 1.2 Base term (scaffolding, not the contribution)

Sub-center normalized-softmax cross-entropy with fixed scale `α = 32`:

```
L_CE,i = −log[ exp(α·s_{i,y_i}) / Σ_c exp(α·s_{i,c}) ]
```

This provides bulk attraction/repulsion and pins the similarity scale (blocks global-deflation degeneracy, §2.3-D2).

### 1.3 The method: differentiable POT return-level margin

For each anchor `i` with label `y_i`, let `S_i = {s_{i,c} : c ≠ y_i}` (K−1 impostor-class similarities; on SOP all 11,317 are computed anyway by the CE term).

**(a) Threshold and exceedances.** With tail fraction `π = 0.1`, `k = ⌈π(K−1)⌉`. Sort `S_i`; the threshold `u_i` is the (k+1)-th largest value, **stop-gradient**. Exceedances in ascending order: `e_{(1)} ≤ … ≤ e_{(k)}`, `e_{(j)} = s_{(j)} − u_i` (gradients flow through the `s` values).

**(b) Probability-weighted-moment GPD fit** (Hosking–Wallis; closed form, differentiable — linear in order statistics):

```
b0_i = (1/k) Σ_j e_{(j)}
b1_i = (1/k) Σ_j ((j−1)/(k−1)) e_{(j)}
λ1_i = b0_i ;   λ2_i = 2·b1_i − b0_i
```

Shape parameter pooled over the batch for statistical stability, **stop-gradient**, clamped:

```
ξ̂ = clamp( 2 − (Σ_i λ1_i)/(Σ_i λ2_i), −0.7, 0.5 )     (ξ̂ := 0 if Σλ2 < 10⁻⁶)
```

Per-anchor scale (gradient flows through `λ1_i` only — i.e., **equal weight 1/k on every tail similarity**):

```
σ̂_i = (1 − ξ̂) · λ1_i
```

**(c) Return level.** The estimated `(1 − 1/N)`-quantile of the per-fresh-class similarity law, with `N = c·(K−1)` and extrapolation factor `c = 4` (the population index — the method's central dial):

```
A = ((Nπ)^ξ̂ − 1)/ξ̂        (= ln(Nπ) at ξ̂ = 0; stop-grad scalar per batch)
m̂_i = u_i + σ̂_i · A
```

This is the level the single most-similar impostor among `N` fresh class draws exceeds with probability ≈ `1 − e^{−1}`; margining at `c·(K−1)` rather than `(K−1)` sets the fresh-gallery breach rate to ≈ `1 − e^{−1/c}` ≈ `1/c`.

**(d) Guarded target and margin loss** (`δ = 0.1`):

```
t_i = min( max( m̂_i , s_i^max ) , 1−10⁻⁴ ),   s_i^max = max_{c≠y_i} s_{i,c}
L_RLM,i = (1/α) · softplus( α·( t_i + δ − s_{i,y_i} ) )
```

Subgradient goes to whichever branch of the max is active.

**(e) Image-level twin** (closes the proxy-parking degeneracy, §2.3-D3): identical construction on in-batch impostor *images* `W_i = {⟨z_i, z_j⟩ : y_j ≠ y_i}` with `π_img = 0.25`, its own pooled shape `ξ̂_img`, and `N_img = c·|X_train|` (extrapolation to a training-set-sized gallery of fresh images). Term `L_img,i`, weight `λ_img = 0.5`.

**(f) Total.**

```
L = (1/B) Σ_i [ L_CE,i + L_RLM,i + 0.5·L_img,i ]
```

### 1.4 Recipe (frozen)

AdamW; backbone lr 1e-4 (wd 1e-4), proxies lr 1e-2 (wd 0); cosine decay, 1-epoch linear warmup; BN layers frozen in eval mode (standard PA practice); batch 128 as 32 classes × 4 images (CUB/Cars) or 64 × 2 (SOP/In-Shop); augmentation: RandomResizedCrop(224, scale 0.16–1.0) + horizontal flip only; eval: resize 256, center crop 224, single view, cosine NN on the 512-D descriptor. Budget ≤ 200 epochs. **Legal model selection**: 4:1 class-disjoint split *of the training classes*; tune only `c ∈ {1,4,16}`, `π ∈ {0.05,0.1,0.25}`, `λ_img ∈ {0,0.5}`, and epoch count on the held-out-class fold; retrain on all training classes with frozen choices; 5 seeds, report mean±std. Defaults above are the frozen priors if tuning is unavailable. Test data, test class counts, and gallery statistics are never touched; `N` is indexed to *training* population size only.

Overhead: one sort of `B×(K−1)` values plus two weighted sums per anchor — <0.5% step time on CUB/Cars, <2% on SOP (the `B×K` similarity matrix already exists for CE). Zero additional parameters or memory.

---

## 2. Causal zero-shot error mode and proof-level degeneracy analysis

### 2.1 The single causal error mode: fresh-impostor max-margin miscalibration

A query succeeds at R@1 iff `s⁺ > max` over ~10³–10⁴ gallery impostors drawn from *classes the loss never saw*. Two quantifiable biases make standard losses calibrate to the wrong level:

- **Finite-sample/scale gap.** Batch or seen-class statistics estimate the bulk; the decisive statistic is an extreme order statistic. For any light-tailed similarity law the max over `N` draws sits `Θ(√log N)`-to-`Θ(log N)` quantile-widths above the batch soft-max; LSE-type losses (softmax, Proxy-Anchor, MS) under-penalize it by construction.
- **Adaptive optimization bias.** SGD actively suppresses the *realized* seen-class argmax each step (whack-a-mole). The realized max is therefore an optimized-against statistic and a downward-biased estimate of where a *fresh* class will land (the adaptive-data-analysis effect, Dwork et al. 2015). Unseen-class impostors systematically breach margins tuned to the memorized seen frontier.

RLM's mechanism: the return level is a smooth functional of the **entire tail ensemble** (`πK` order statistics, equal weights), so no single suppression moves it; lowering the target requires lowering the *population tail*, which is exactly the property that transfers to fresh classes; and the `c`-indexed extrapolation places the margin where the fresh maximum is predicted to land, not where the seen maximum was beaten down to.

**Quantified consequence (informal proposition).** Under (A1) exchangeable class population (train and test classes drawn from the same domain population), (A2) impostor-class similarity tails in the GPD domain of attraction (Pickands 1975; Balkema–de Haan 1974), and (A3) end-of-training margins `s⁺_i ≥ m̂_i + δ` with PWM estimation error `O((πK)^{−1/2})` (Hosking–Wallis 1987 asymptotic normality): for a fresh gallery of `K′ ≈ K` classes, the probability that *any* fresh impostor class exceeds the trained level is ≈ `1 − exp(−K′/(c(K−1)))` ≈ 22% at `c = 4`, versus ≈ 63% for a realized-max margin (`c = 1`) *before* accounting for adaptive bias, which degrades the realized-max rule further but not the ensemble functional. Queries above the level fall back to bulk (CE) geometry rather than failing outright, so this bounds the *mechanism's* target, not R@1 itself — stated as heuristic quantification, not a theorem.

### 2.2 Proof-level anti-gaming guarantee

**Proposition (monotone tail pressure).** With stop-gradients on `u_i`, `ξ̂`, `A`, `k`:
`∂m̂_i/∂s_{i,c} = A(1−ξ̂)/k > 0` if `s_{i,c}` is a tail member, else `0`. Hence `t_i = max(m̂_i, s_i^max)` is coordinatewise non-decreasing in every impostor similarity, so `∂L/∂s_{i,c} ≥ 0` for all impostors and `≤ 0` for the positive. **Corollary 1:** no descent direction of `L` increases any impostor similarity — the tail estimator cannot be gamed by reshaping the empirical tail (the failure that kills a naive fully-differentiable MLE fit, where inflating mid-tail mass can *lower* the fitted quantile through the shape/threshold path — this is why `u` and `ξ̂` carry stop-grads). **Corollary 2:** `t_i ≥ s_i^max` always, so `L_RLM` upper-bounds the realized-max hinge — estimation error can make training more conservative but never softer than hardest-negative max-margin.

### 2.3 Remaining degeneracies, attacked

- **D2 global deflation** (push all similarities down together): blocked by the fixed-scale CE term and the compact sphere; the RLM term is relative (`t_i + δ − s⁺`), CE pins absolute geometry.
- **D3 proxy parking** (keep impostor *proxies* far from anchors while impostor *images* stay close): proxies are pulled onto their own class's images by CE, and the image-level twin `L_img` applies the same tail pressure to real embeddings; residual risk measured by control C7.
- **D4 sub-center collapse**: benign and standard (SoftTriple/sub-center ArcFace behavior); not treated.
- **D5 threshold clumping** (pile mass just below `u`): moving a similarity below `u` *is* a genuine reduction of a top-decile impostor similarity; by monotonicity there is no profitable clumping direction.
- **D6 shape manipulation via batch composition**: batches are drawn by a fixed random class-balanced sampler; `ξ̂` is a pooled running statistic the network cannot select for.
- **Label noise in the tail** (real on SOP): equal 1/k tail weights bound any single false impostor's gradient share to 1/k — strictly more robust than argmax mining, which puts full weight on what is often the noisy pair. Residual risk in §6.

---

## 3. Adversarial novelty search (primary sources; one-sentence mechanism distinctions)

Searches run (queries frozen in the transcript): EVT × metric learning/contrastive/re-ID losses; generalized Pareto / peaks-over-threshold training losses; EVT open-set training; gallery-size/return-level-aware margins. No differentiable train-time EVT margin loss for representation learning was found; EVT in vision is post-hoc calibration.

**Inside DML/retrieval:**
1. **Proxy-Anchor** (Kim et al., CVPR 2020): LSE-weighted pressure on *observed* similarities; RLM replaces the batch aggregate with a parametric quantile target *beyond* the observed sample, indexed to deployment scale.
2. **Multi-Similarity / Circle loss** (Wang 2019; Sun 2020): reweight observed pairs within the bulk; no extrapolation, no population index.
3. **Hard-negative reweighting, HCL** (Robinson et al., ICLR 2021): exponential tilting of the observed negative distribution; RLM fits an explicit tail law and margins at a quantile outside the sample.
4. **XBM** (Wang et al., CVPR 2020) / **Negative Cache** (Lindgren et al., NeurIPS 2021): reach large N by storing stale embeddings; RLM reaches gallery scale *statistically* with fresh gradients and O(1) memory.
5. **Smooth-AP** (Brown et al., ECCV 2020) / **Recall@k surrogate** (Patel et al., CVPR 2022): differentiable *batch-scale* ranking surrogates; RLM explicitly corrects the batch→gallery scale gap in the decisive order statistic.
6. **Sampled-softmax logQ correction** (Bengio & Senécal 2008; Yi et al., RecSys 2019): debiases the *mean* partition function; RLM targets the *maximum* order statistic, which mean-field corrections under-penalize by a `Θ(log N)` quantile shift.
7. **Partial-AUC / DRO ranking** (Zhu et al., ICML 2022; pAUC face losses): CVaR-style tail-*averaged* reweighting of observed pairs; RLM extrapolates parametrically beyond the sample and indexes the target to fresh-class count — a different statistical object (return level vs tail mean).
8. **Proxy Synthesis** (Gu et al., AAAI 2021) / **Virtual Softmax** (Chen et al., NeurIPS 2018) / **Embedding Expansion** (Ko & Gu, CVPR 2020): fabricate phantom classes geometrically to mimic unseen ones; RLM fabricates nothing — it estimates where the unseen-class maximum will land from the population tail.
9. **Threshold-Consistent Margin loss** (Veeramacheneni et al./Amazon, 2023–24): calibrates margins so one distance threshold works across test distributions; a consistency objective on observed score distributions, not an extrapolated fresh-class extreme, and aimed at threshold stability rather than R@1.
10. **Distance-weighted sampling / margin-β loss** (Wu et al., ICCV 2017): learned free-parameter margins and density-based sampling over observed negatives; RLM's target is a closed-form order-statistic functional, not a fitted free parameter.
11. **AdaFace / MagFace / CurricularFace**: per-sample adaptive margins driven by image quality or norm; RLM's margin adapts to the *impostor population tail*, indexed by deployment class count.
12. **PFML** (CVPR 2025): probabilistic multi-proxy class *representation*; orthogonal axis (representation machinery vs margin target) — composable, untouched here.
13. **DADA** (AAAI 2024) / **AdvRF** (ICCV 2025, Lane B): auxiliary adversarial/reconstruction networks; RLM is machinery-free and loss-level.

**Outside DML:**
14. **OpenMax** (Bendale & Boult, CVPR 2016), **EVM** (Rudd et al., TPAMI 2018), **W-SVM** (Scheirer et al.), **GPD/GEV classifiers** (Vignotto & Engelke 2018), **SPOT** (Siffer et al., KDD 2017), **Vocabulary-informed EVL** (Fu et al., ICCV 2017): all fit EVT *post hoc* to a trained scorer for rejection/thresholding; RLM differentiates a PWM/POT fit *inside* the training loss so gradients reshape the tail itself — none trains representations against a return level.
15. **Extreme-event forecasting losses** (Ding et al., KDD 2019): EVT-inspired reweighting of extreme *labels* in time series; RLM applies EVT to the model's own similarity order statistics to set retrieval margins.
16. **Classical POT/return levels** (Pickands 1975; Balkema–de Haan 1974; Hosking & Wallis 1987; Coles 2001): the imported estimator; the invention is the monotone, stop-gradient, guarded embedding of it as a train-time margin target — the anti-gaming construction of §2.2 has no analogue in the source field because nothing there is trained against the fit.

---

## 4. Decisive matched-compute controls

All controls share backbone, recipe, batches, epochs, and seeds; they differ only in the loss line, so compute is matched to <2%.

- **C1** Base CE alone — the floor.
- **C2** CE + realized-max hinge (`t_i = s_i^max`, no EVT) — *the* mechanism control: separates "extrapolated ensemble quantile" from "hardest-negative margin."
- **C3** CE + LSE negative term with hardness swept (`α_neg ∈ {32, 64, 128}`) — rules out "it's just a hotter temperature."
- **C4** Dose–response `c ∈ {1, 4, 16}` — the population-index story predicts systematic movement (rising then saturating margins/gains on SOP/In-Shop); flatness kills it.
- **C5** Per-anchor scale ablation (`σ̂_i → batch-mean σ̂`) — does anchor-specific tail calibration matter, or only global hardness?
- **C6** `ξ̂ ≡ 0` (exponential tail) — does adaptive shape matter?
- **C7** `λ_img = 0` — measures the proxy-parking closure.

---

## 5. Frozen forecasts, frontier arithmetic, falsification thresholds

Lane A, 5 seeds, mean R@1. Own-recipe base (C1) expectations: CUB 0.700±0.004, Cars 0.906±0.004, SOP 0.816±0.003, In-Shop 0.925±0.003 (if C1 lands >1 pt off these, flag recipe drift and re-baseline before interpreting RLM).

**RLM (frozen defaults):**

| Dataset | Median | 90% CI | Reference | P(beat reference mean) |
|---|---|---|---|---|
| SOP | **0.832** | [0.823, 0.839] | PFML 0.829±0.002 | 0.60 |
| In-Shop | **0.934** | [0.926, 0.940] | PA+DADA 0.930 | 0.62 |
| Cars196 | 0.918 | [0.910, 0.926] | PFML 0.927±0.003 | 0.14 |
| CUB | 0.713 | [0.703, 0.722] | PFML 0.734±0.003 | 0.03 |

**Frontier-crossing arithmetic.** SOP: mean-crossing requires >0.829; 2σ-combined crossing requires ≥ 0.829 + 2·√(0.002²+0.003²) = **0.8362** — my median 0.832 crosses in mean (P≈0.60) but reaches the 2σ bar with only P≈0.20. In-Shop: reference has no reported spread; mean-crossing at ≥0.931, P≈0.62; using my seed σ alone, 2σ self-consistent crossing needs ≥0.936 (P≈0.35). **The crossing claim is therefore mean-level on SOP and In-Shop, where the class population (11.3k / 4.0k classes) makes the tail fit statistically rich.** CUB and Cars (K≈100 → only ~10 exceedances per anchor) are declared a priori weak for this mechanism and are *not* crossing claims; PFML's 15-proxy representation machinery does work RLM does not attempt, and the honest composition experiment (RLM margin target inside a PFML-style base) is future work, not forecast.

**Falsification thresholds (frozen).**
- **F1**: RLM − C1 < +0.005 on both SOP and In-Shop → method dead.
- **F2**: RLM − C2 < +0.002 on both SOP and In-Shop → the extrapolation mechanism is dead (it was just max-margin).
- **F3**: C4 dose–response flat (all `c` within 0.002 on SOP) → population-indexing story dead.
- **F4**: best swept C3 matches RLM within 0.002 everywhere → "just temperature" — dead.
- **F5**: RLM < C1 − 0.003 on any dataset → harmful; withdraw.

---

## 6. Costs and risks

**Training cost:** ×1.01–1.02 step time over C1 (a sort plus two weighted sums per anchor; the `B×K` similarity matrix already exists for CE); zero extra parameters, zero extra memory, no auxiliary networks (contrast: DADA ≈ ×1.06 epoch time; AdvRF carries a training-only ResNet-34/U-Net — other lane). **Deployment:** bit-identical to baseline — one ResNet-50, one view, one 512-D normalized descriptor, cosine NN. Fully lane-compliant; no test data, no transduction, no reranking, no text/VLM, no external or generated data.

**Risks, stated plainly.** (i) *Effect-size risk is the dominant one*: hard-mining and hot-LSE already crudely approximate max-pressure, so RLM's calibrated increment may be small — F1/F2 are designed to measure exactly this, and I put ~35–40% probability on failing to mean-cross on SOP. (ii) Exchangeability (A1) is an approximation — benchmark class splits are curated, not iid draws; the constant `c` absorbs some violation and C4 measures sensitivity. (iii) SOP label noise inflates the tail with false impostors (mitigated but not eliminated by uniform 1/k tail weights). (iv) Thin-tail regime: if `ξ̂` trends strongly negative, `A` shrinks and RLM degenerates gracefully toward C2 — observable in logged `ξ̂`, and an honest failure signature. (v) In-Shop reference lacks error bars, so that comparison is mean-only. (vi) Contamination: ImageNet-1K initialization overlaps CUB/Cars imagery — standard, permitted, and shared by every cited reference; benchmark-specific quirks (SOP leave-one-out gallery, In-Shop near-duplicates) affect all methods equally.

**Sources** (novelty-sweep pages consulted): [Threshold-Consistent Margin Loss for Open-World DML](https://arxiv.org/html/2307.04047v2) · [GPD and GEV Classifiers for Open Set](https://www.researchgate.net/publication/327303321_Extreme_Value_Theory_for_Open_Set_Classification_-_GPD_and_GEV_Classifiers) · [Open Set Recognition Survey (EVT usage)](https://arxiv.org/pdf/1811.08581) · [Vocabulary-informed Extreme Value Learning](https://arxiv.org/pdf/1705.09887) · [Open-set face recognition with maximal entropy](https://www.sciencedirect.com/science/article/abs/pii/S0262885623002366) · [Rank-loss optimization survey](https://www.computer.org/csdl/journal/tp/2025/06/10896862/24uGxmYs6ze) · [In Defense of the Triplet Loss](https://arxiv.org/pdf/1703.07737) · [Class Anchor Margin Loss](https://arxiv.org/abs/2306.00630)

**Proposer's closing statement.** The mechanism is a genuine import with a genuine gap: extreme value theory is the correct statistical language for the quantity zero-shot R@1 actually measures, and nobody trains representations against a differentiated return level — the monotone stop-gradient construction that makes this safe (§2.2) is the technical heart and is new. The main uncertainty is not novelty but effect size on top of implicit hard-mining; the controls and falsification thresholds above are built to settle that question decisively at matched compute.
