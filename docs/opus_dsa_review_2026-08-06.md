# Independent Opus review: DSA (Pass 29)

Consultation ID: `97f3aa216ed54605`

I have what I need. Here is the review.

---

# Independent adversarial review — DSA (Pass 29), frozen proposal

**Verdict: DEAD.** Earliest gate failed: **§3, the adversarial-novelty gate** — with §4/C8 as the earliest *arithmetically self-refuting* gate. Both are desk-checkable; no GPU is required to reach either. Single most decisive mechanism-level reason: **the quantity DSA controls (the realized between-identity scatter spectrum λ) is not the quantity its causal argument requires (the encoder gain spectrum g); they are related by λᵢ = gᵢ·sᵢ, and under realistic anisotropy of the training class-contrast spectrum s, flattening λ drives g *away* from flat.**

Details, then preserved subcomponents.

---

## 1. Re-derivation of the mathematics

**What is correct (verify and credit):**

- **Gradient path (§1.5, lines 83–87) is exactly right.** With `f̃ = P_r f/‖P_r f‖`, the Jacobian is `(1/‖P_r f‖)(I − f̃f̃ᵀ)P_r`, and since `f̃ ∈ range(P_r)` we have `P_r f̃ = f̃`, so `(I − f̃f̃ᵀ)P_r = P_r − f̃f̃ᵀ`. Symmetric ⇒ the VJP is as written. The chain to `z` via `‖z‖⁻¹(I − ffᵀ)` is right. No second-order terms; **the single-backbone-backward claim is correct** — both terms are functions of the same `f`, so autograd merges at `f`.
- **Rank claim is right.** `rank(S_B^batch) ≤ |C_B| − 1 = 29`; the EMA is genuinely not optional.
- **Rotation equivariance (D3) is true as stated** for the loss term. `f ↦ Qf ⇒ S ↦ QSQᵀ ⇒ U_r ↦ QU_r ⇒ f̃ ↦ Qf̃`, cosines preserved. Trivially true, and I confirm it.
- **Var[T] = 2σ_g²/(d+2)** is exactly right. Monte Carlo, 200k draws, d=512: measured Var = 7.622e−3 vs formula 7.630e−3. E[T] = ḡ confirmed.
- **The ΔΦ arithmetic reproduces.** Φ⁻¹(0.27) ⇒ z = 0.6128 (stated 0.613); φ(z) = 0.3306 (stated 0.331); φ(z)·z·0.20 = **4.052 pts** (stated 4.1).

**What is wrong:**

**1a. `S` is not the between-identity scatter. It is `Σ_B + Σ_W/m`, and the EMA cannot fix it.**
With `m_c` estimated from `n_c = m = 4` samples,

```
E[S_B^batch] = (|C_B|−1)/|C_B| · (Σ_B + Σ_W/m)
```

This is a **bias, not noise** — averaging over 100 EMA steps reduces its variance and leaves the offset intact. At m=4 the estimator permanently adds **25% of the full within-class covariance**. For a plausible split (tr Σ_B = 0.35, tr Σ_W = 0.65 on the unit sphere), the contamination is **31.7% of tr(S)**. Consequences:

- The object in the §1.1 table ("`S` — EMA between-identity scatter") is mislabelled; the code computes something else.
- `rank(E[S])` is 512, not ≤ C−1, because Σ_W is generically full rank.
- **It falsifies D2's load-bearing premise** (line 123): *"Inflating within-class variance instead does not enter S_B."* It enters at weight 1/m. A pure within-class decoy with **zero class-mean signal** and sd = 0.20 along one axis contributes λ = 0.0100 to `S`, against a top true discriminant of λ = 0.039 in my CUB-like simulation — i.e. it lands in the top-32 easily, and at sd = 0.30 it contributes 0.0225, over half the top true eigenvalue. Its cost to the primary loss is ≈ α·sd²/2 = 1.44 logits at sd=0.30, and it *buys* immunity: it occupies an ablation slot. That is precisely the sacrificial-discriminant / decoy attack D2 declares impossible.
- Worse, it is **progressive**: as DSA flattens Σ_B, the Σ_W/4 term takes over the top eigenslots. At PR(Σ_B)=40 and PR(Σ_W)=10, λ_max(Σ_W)/4 = 0.0163 > λ_max(Σ_B) = 0.0088. The method then silently converts from *"ablate the top discriminants"* into *"ablate the top nuisance directions"* — a different intervention, unclaimed and invisible to C8.

**1b. Singular cases: the formula has no epsilon, and the proposal never bounds the denominator.** `1/‖P_r f‖` and `1/‖P_r p̂‖` are unbounded as the argument enters span(U_r). In practice the shared global-mean component protects samples (I measure E‖P_r f‖ = 0.863 at r=32, amplification 1.16×). **Proxies are the exposed side**: proxies track class means, and the top eigenvectors of S_B are precisely the principal directions of the class-mean cloud. I measure min‖P_r p̂‖ = 0.707 at r=32 ⇒ amplification 1.42× on the worst proxy, and it is unbounded in principle. Combined with **proxy lr ×100 and weight decay 0**, this is a live instability with no guard. It does not change the method — it is a fixable engineering defect (add ε, or skip the term when ‖P_r p̂‖ < τ). But note the amplification is *systematically larger for DSA than for every control*: 1.16 vs 1.03 for a random subspace. **The ablation term therefore carries a ~13% larger effective β than C1/C2/C6 at the same nominal β** — the controls are not weight-matched (see §6).

**1c. Rotation equivariance holds for the loss but not for the trained system.** AdamW's per-coordinate second moments are not equivariant under `W ↦ QW`. So D3's conclusion — *"There is no basis in which the constraint is cheaper"* — is true of the loss surface and false of the reachable optima under the frozen optimizer. Secondary to §3 below, but it removes the "provably" from line 13.

---

## 2. The causal argument does not identify what it claims

**2a. The step that kills it.** §2.1 defines `gᵢ` as *"the encoder's gain along uᵢ"* and then treats DSA's flattening of `S_B`'s spectrum as flattening `g`. These are different objects. Write the identity latent contrast covariance as `diag(sᵢ)` — how much training identities actually differ along each direction — and the encoder gain as `gᵢ`. Then

```
λᵢ (realized between-identity scatter)  =  gᵢ · sᵢ
```

Forcing λ flat forces **gᵢ ∝ 1/sᵢ** — maximally *anti*-aligned with the data's own anisotropy. Meanwhile the Var[T] objective the proposal derives is minimized at **g = const**, which implies **λ ∝ s: a spiky realized scatter.** The method's target and its stated goal point in opposite directions whenever s is anisotropic.

How anisotropic? Fresh calculation with power-law `sᵢ ∝ i^(−a)` over the 99 usable CUB directions:

| s exponent | PR(λ) produced by a **perfectly unbiased, flat-gain** encoder | gain PR forced by DSA's flat-λ target |
|---|---|---|
| a = 0.5 | 66.0 | 88.4 (σ_g/ḡ = 0.35) |
| **a = 1.0** | **16.4** | **74.6 (σ_g/ḡ = 0.57)** |
| a = 1.5 | 4.8 | 63.7 (σ_g/ḡ = 0.74) |

At a = 1 — an unremarkable class-attribute spectrum — an encoder with **zero monopoly at all** yields PR(S_B) = 16.4, sitting inside the proposal's own asserted "diseased" base range of 8–15 (line 171). **The premise of §2.1 is not identifiable from the observable the method optimizes, and C8 cannot separate "encoder monopoly" from "the data is anisotropic."** This is the decisive reason.

**2b. The measured premise does not exist.** Line 105 asserts *"empirically the realized spectrum is far spikier than that bound."* No measurement is given, no citation, no number. PR(S_B) = 8–15 appears first at line 171 as a *forecast*. The entire error mode is a forecast presented as a diagnosis. The one measurable premise in the proposal is therefore absent.

**2c. Exchangeability is self-defeating, and the offsetting cost is tiny.** Under exact exchangeability, "fixed trace" fixes E[T] and only the variance moves — the +4.1. But exchangeability is exactly what ImageNet-init and DML training falsify: novel-identity contrasts *do* align with training-selected directions (the proposal concedes this at line 115). To the degree it fails, flattening `g` at fixed Σgᵢ reduces the *mean* transmitted contrast, and the failure mass responds to the mean with weight φ(z)·1 versus φ(z)·z = φ(z)·0.613 for the variance. Fresh sensitivity:

```
dFail = φ(z)·[ −dḡ/σ_T + z·dσ_T/σ_T ]
variance benefit (φ units) = 0.613 × 0.20 = 0.1226
```

| base PR | σ_g/ḡ | σ_T/ḡ | drop in mean transmitted contrast that erases the entire +4.1 |
|---|---|---|---|
| 8 | 7.94 | 0.495 | **6.07 %** |
| 10 | 7.09 | 0.442 | **5.42 %** |
| 15 | 5.76 | 0.359 | **4.40 %** |

A ~5% misalignment penalty cancels the whole ceiling. The +4.1 is not a ceiling; it is the *variance half of a two-sided ledger whose other half is never computed*. The "~4× discount to +1.1" (line 233) is, as the proposal honestly says, judgement — but it is judgement applied to only one side.

Two further mis-specifications: the threshold `t` (within-class dispersion) is treated as fixed, though it is a quadratic form in the same gains and moves when `g` moves; and R@1 failure is a max-over-~5.9k-gallery extreme-value event, not a single-pair lower-tail event `T < t`. Neither is fatal alone; together they mean the model is uncalibrated in magnitude as well as sign.

**2d. "More transferable evidence" does not follow (see §3, D1).**

---

## 3. D1–D5: explicit counterexamples

**D1 — defeated only for *linear* duplication. Broken by nonlinear re-encoding.** The rank-1 argument is correct: `k` copies of a scalar `s` on orthogonal axes are collinear across class means. But a ReLU backbone produces **information-identical, second-moment-orthogonal** copies for free. Explicit construction, C=100, n=200, channels `(s, ReLU(s), ReLU(−s))` from **one** underlying scalar:

```
S_B eigenvalues = [1.1718, 0.0843, 0.0000]        <- rank 3, not rank 1
between-class energy surviving ablation of the top-1 direction = 6.7 %
direction 2: Fisher ratio S_B/S_W = 0.75          <- genuine class separability
```

Direction 2 is a deterministic function of the *same scalar*. `S_B` is flattened; **zero additional image evidence has been extracted.** This is neither "redistributing existing evidence" nor "extracting new image evidence" — it is a third option §2.1 (line 117) does not admit, and it is by far the cheapest. Because DSA sees only second-order statistics, it cannot distinguish an information-preserving nonlinear recoding from real redundancy. The claim at line 13 — "provably immune to the two cheap ways a network fakes redundancy" — is false; only one way is closed.

This is also where novelty collapses: D1's rotation-invariance argument is the *entire* claimed advantage over axis-aligned erasure (RSC, coordinate dropout, C6), and it does not hold.

**D2 — false four ways.** *"The only cheap satisfaction is a flat spectrum."* Cheaper satisfiers, in ascending cost:

1. **Proxy re-fitting (cheapest, and the proposal built the door).** Both `f` *and* `p̂` are projected, so the residual loss has a gradient path into the proxies — 1,500 proxies × 480 residual dims = **720k free parameters at 100× the backbone lr with zero weight decay**, fitting a batch of 120. And the residual is trivially separable: in my CUB-like simulation, **nearest-class-mean accuracy in the residual is 1.000 at r = 1, 8, 16 and 32** (residual class-mean rank 67 at r=32). L_abl is therefore satisfiable by moving proxies, with the encoder unchanged. The intended target of the constraint is short-circuited by a 100×-faster parameter path.
2. **Nonlinear re-encoding** (D1 above) — free for a ReLU backbone.
3. **Within-class sacrificial decoys** — admitted into `S` by the Σ_W/m bias (§1a).
4. **Train-identity memorization** (D5, admitted). Note the operating point makes this the *dominant* regime, not a tail risk: at PR ≈ 10 and r ~ U{1..32} (mean 16.5), the ablation removes essentially *all* class-mean signal — I measure only **13.5% of between-class energy surviving at r = 32**. At ramp-start the ablated task is near-impossible, and the only available descent direction is nuisance/memorized codes. CUB train is ~5.9k images over 100 identities; a ResNet-50 separating 100 classes in a 480-dim residual is memorization, not discovery.

**D3 — true, and inert.** Equivariance is real but does no causal work (D1 breaks the redundancy claim; AdamW breaks the reachability claim). Staleness is bounded, agreed.

**D4 — correct.** DSA is not a spectral-dispersion penalty; S→0 raises both terms. Exact collapse is structurally excluded. Near-collapse is also unlikely because the descriptor's shared mean component is not in span(U_r) (`S_B` is centered), which is also what keeps ‖P_r f‖ ≈ 0.86 rather than ≈ 0. **This is the one degeneracy argument that survives intact**, and it is a genuine advantage over −log det / entropy formulations.

**D5 — correctly identified, and both guards fail.** Guard (i), augmentation robustness, does not touch memorized instance codes or per-collection statistics that survive crop/flip. Guard (ii), held-out-identity selection, protects *selection* but not *training* — and is underpowered (§6).

**EMA / eigenbasis cycling — the self-refutation the proposal missed.** As the spectrum flattens toward the target, the eigengap `λ_r − λ_{r+1}` → 0 and `U_r` becomes statistically arbitrary. Fresh measurement: mean principal angle between the top-32 subspaces of two independent draws of `S` at N = 3000 effective class-mean samples:

```
spiky base (PR=16):  11.1°
half-flat  (PR=50):  26.3°
DSA target (flat) :  58.9°     <- essentially a random subspace
```

**At its own optimum, DSA *is* control C1.** The DSA−C1 gap must vanish as the mechanism succeeds. That inverts F2's logic (§6).

**Hard multi-proxy assignment.** At K=15, m=4, at most **4 of 15** proxies per present class can receive attraction in a step; the other 11 are pure repulsion targets. SoftTriple uses soft assignment plus an explicit center-merge regularizer to avoid exactly this. The proposal chose the known-pathological variant and did not analyze it.

---

## 4. Multi-proxy loss coherence, and the recipe (primary sources checked)

**Is it an exact PA extension? No — it silently rescales the loss.** PA normalizes the negative term by 1/|P| and the positive by 1/|P⁺|. Under hard assignment, raising K multiplies |P| by K but |P⁺| by at most min(m,K):

| C | K | \|P\| | max \|P⁺\| | \|P\|/\|P⁺\| |
|---|---|---|---|---|
| 100 | 1 | 100 | 30 | 3.3 |
| 100 | 15 | 1,500 | 120 | **12.5** |
| 11,318 | 2 | 22,636 | 60 | 377 |

**Going from K=1 to K=15 changes the positive/negative balance of L_PA by 3.75×.** So C0(K=1) vs C0(K=15) is not "more proxy capacity" — it is a different loss. **C7 ("C0 with K raised to match parameter count") is confounded by the same rescaling**, and the DSA-on-K=1 vs DSA-on-K=15 rows are not comparable. Changing K plus the recipe does confound DSA.

**Empty / unassigned same-class proxies.** Under the stated rule ("all proxies of other classes remain negatives"), an unassigned proxy `p_{c,j}` has `X⁺ = ∅` (excluded from P⁺) but retains a full negative set, so it is repelled every step and never attracted. Its only stable position is the residual region unoccupied by other classes. Not catastrophic, but unanalyzed, and it interacts with the 100× proxy lr.

**A load-bearing ambiguity.** `L_PA(f̃, Π̃)` — is the argmax assignment recomputed on the *projected* similarities or shared with the unablated view? The proposal never says. If recomputed, the two terms can pull one sample toward two different proxies of its own class. If shared, the ablated term is not "the same loss on the residual," which is the entire D4 argument. **The method is not executable as written** without this decision.

**Recipe verification against primary sources.** §6 item 2 flags α=32, δ=0.1, the ×100 proxy lr, the optimizer, and the epoch budget as unverified/undisclosed. Checking [arXiv:2003.13911](https://arxiv.org/abs/2003.13911) directly — **all five are disclosed in the paper**:

- Eq. 4 matches the proposal's §1.2 **verbatim**, including both normalizers. ✓
- α = 32, δ = 0.1. ✓ (proposal's recall correct)
- *"In every experiment, we employ AdamW optimizer."* ✓
- *"The learning rate for proxies is scaled up 100 times for faster convergence."* ✓
- *"Our model is trained for **40 epochs** … on the CUB-200-2011 and Cars-196, and for **60 epochs** … on the SOP and In-shop."*
- *"We assign a single proxy for each semantic class"* — confirms K=1 is PA, and the multi-proxy extension is indeed the proposal's own. ✓ (honestly flagged)
- CUB R@1 @512-D: BN-Inception 68.4, **ResNet-50 69.7**.

Two consequences. (i) **The 200-epoch budget is a 5× departure from the only published recipe DSA claims to reduce to**, and §6 item 4's "the lane's 200-epoch LR schedule is unspecified" mischaracterizes a disclosed 40-epoch budget. (ii) The forecast C0 K=1 = 0.705 sits **+0.8 above the published PA ResNet-50 number at 5× the epochs**, so the headline Δ +1.8 is measured against an unreproduced, already-uplifted baseline. Batch size: the paper's comparison used 150 and Table 5 sweeps {30…180}; I could not retrieve the official README this session (raw URL 404, `gh` unauthenticated), so the "batch 120 for ResNet-50" attribution is unverified but not refuted.

---

## 5. Prior art — judged on supervision object and action

**The disqualifying collision: Representation Self-Challenging (RSC), Huang, Wang, Xing & Huang, ECCV 2020 (oral).** Not cited anywhere in §3.

RSC computes the gradient of the loss w.r.t. the representation, **mutes the top-p% most label-predictive components (channel-wise or spatial), recomputes the same task loss on the muted representation, and backpropagates through the entire network. No extra parameters, no extra branch.** From the paper: *"RSC iteratively challenges (discards) the dominant features activated on the training data, and forces the network to activate remaining features that correlates with labels"*; *"Besides the weights of the original network, no extra parameter needs to be learned."* Default p = 33.3%. The claim is improved generalization to unseen distributions.

Object: the representation's currently most-discriminative component. Action: delete it, re-run the same loss, backprop everywhere, zero parameters. Causal story: force the residual to carry the label ⇒ better out-of-distribution transfer. **That is DSA.** DSA's differences are wrapper-level: eigen-subspace of an EMA between-class scatter instead of per-sample top-gradient channels; rotation-equivariant instead of axis-aligned; proxy-DML instead of softmax; renormalization of the residual view. Under the instruction to judge object and action rather than name, this is a re-wrapping.

Critically, the **only** claimed advantage of the subspace form over the axis-aligned form is D1/D3 — and §3 above shows D1 protects against linear duplication only. So the remaining novelty is "the same intervention, in a rotation-covariant basis, in DML," with the argument for why the basis matters refuted.

**Second collision, inside the field §3 surveys.** [Self-Erasing Network for Person Re-Identification (Sensors, 2021)](https://www.mdpi.com/1424-8220/21/13/4262): a *"maximum activation suppression branch [that] forces the network to find more discriminative cues in the remaining weak information regions by erasing the most activated feature vector."* Feature-vector-level, not spatial — which is exactly the gap §3 item 10 claims to fill ("spatial erasure cannot reach").

**Third: the error mode is a named, published phenomenon with a published remedy.** [Gradient Starvation: A Learning Proclivity in Neural Networks (Pezeshki et al., NeurIPS 2021)](https://arxiv.org/abs/2011.09468) — *"cross-entropy loss is minimized by capturing only a subset of features relevant for the task, despite the presence of other predictive features"* — plus **Spectral Decoupling** as a cheaper (logit-level) intervention. §2.1 presents "leading-direction monopoly" as its own diagnosis.

**Fourth: contrary evidence on the premise.** [Understanding Dimensional Collapse in Contrastive Self-Supervised Learning (Jing et al., ICLR 2022)](https://arxiv.org/abs/2110.09348) establishes that spectral collapse in embeddings has *implicit-regularization and augmentation* causes, and that DirectCLR fixes it by **restricting the loss to a fixed sub-vector** — i.e. axis-aligned, not rotation-covariant, and effective. That is a published counterexample to the claim that basis-freedom is necessary.

The §3 list is well-constructed and the distinctions drawn for BIER / D&C / SoftTriple / MCR² / VICReg / MRL / INLP are individually fair. But the closing claim (line 155) survives only via the conjunction "subspace ∧ deep metric learning." The action is published, at a top venue, in 2020, with the same causal argument.

---

## 6. Controls and falsifiers

**C1 is not difficulty-matched, and the JL justification is misapplied.** JL bounds distortion of a projection *into* k dimensions; here it is invoked for removing 32 of 512. Fresh measurement on a CUB-like descriptor population (PR(S_B) = 15.2):

| r | between-class energy **remaining**, top-r | remaining, random-r | renorm factor 1/‖P_r f‖ top / random |
|---|---|---|---|
| 8 | 44.4 % | 98.6 % | 1.078 / 1.007 |
| 16 | 29.2 % | 96.5 % | 1.109 / 1.015 |
| 32 | **13.5 %** | **93.6 %** | 1.160 / 1.032 |

C1's auxiliary task is *near-vacuous*; DSA's is *near-impossible*. Any DSA−C1 difference is fully explained by auxiliary-task difficulty and by the ~13% larger effective gradient weight, without reference to *which* subspace. C2 (bottom-r) and C6 (coordinate dropout) sit on the same easy side — **every control is on one side of the difficulty axis**. A valid control would be difficulty-matched: ablate a random r-subspace *drawn from within the top-R eigenspace*, or match the ablated loss value / gradient norm. As specified, C1/C2/C6 cannot isolate the mechanism.

C6 additionally does not state whether the mask is applied pre- or post-L2-norm, whether inverted-dropout rescaling is used, or whether proxies are masked (DSA projects them). Not matched.

**F2 is mis-specified.** "C1 Δ ≥ 60% of DSA's Δ ⇒ falsified" — but §3 above shows convergence of DSA toward C1 is what the *successful* mechanism predicts (58.9° subspace scrambling at a flat spectrum). F2 fires on success.

**C3 is legal but under-specified and, on CUB, undefined.** Train-statistics-only whitening is legal. But `S_B` on 100 CUB training identities has rank ≤ 99 in 512 dims — **whitening is undefined on a ≥413-dimensional null space**, and full whitening of the small-λ tail amplifies pure Σ_W/4 noise. Without a stated shrinkage/truncation (`λ^(−α)`, α ≈ 0.25–0.5, as retrieval practice uses), C3 is not executable and will almost certainly *hurt*, making **F3 toothless** — it cannot fire, so it cannot falsify.

**C8 and F4 are arithmetically impossible on the split §4 mandates.** By Cauchy–Schwarz, PR = (Σλ)²/Σλ² ≤ rank ≤ (#identities − 1):

| mandated split | identities | PR ceiling |
|---|---|---|
| **CUB val, ids 81–100** | 20 | **≤ 19** |
| **Cars val, ids 89–98** | 10 | **≤ 9** |

The forecast "base 8–15 → DSA **30–45**" (line 171) is impossible on either. F4 ("PR must rise ≥ 2×", line 205) is **unsatisfiable on Cars** (base 8–15 already exceeds the ceiling of 9) and on CUB requires reaching 84–100% of a hard ceiling. Additionally, the Σ_W/m bias *inflates* measured PR, so any observed rise is confounded with within-class noise. **The central mechanism probe and the central mechanism falsifier are both non-executable as written** — and this is discoverable with a pencil.

**F5 does not isolate rank starvation, and its direction is arguable.** SOP differs from CUB in ≥5 frozen ways: K=2 vs 15, R=64 vs 32, lr 6e-4 vs 1e-4, near-duplicate structure (the proposal notes this itself), and ~5.3 images/identity so m=4 covers most of a class (changing the Σ_W/m bias). Worse, the sign is contestable: on SOP (C ≫ d) a flat 512-dim scatter is *reachable*; on CUB (rank ≤ 99) DSA's own target is unreachable by construction. A mechanism that flattens the gain spectrum should plausibly do *more* on SOP. Either outcome is post-hoc explainable ⇒ F5 is not a falsifier.

**Selection power.** Six new knobs (R, β_max, E₀, E₁, γ, T) on top of K, selected on ~1,200 val query images:

```
CUB val (20 ids, ~1200 queries, R@1≈0.80): SE = 1.15 pts
   winner's-curse bias, best-of-20 configs ≈ 2.16 pts
Cars val (10 ids, ~800 queries, R@1≈0.90): SE = 1.06 pts
   winner's-curse bias ≈ 1.98 pts
```

against a claimed effect of **+1.1 pts**. The protocol is genuinely stricter than the field norm and I credit it, but as sized it is a ~2× under-powered selector for the effect being claimed.

**Does the proposal forecast a defensible crossing? No — and it overstates its own odds.** Recomputing from the frozen numbers:

| | Δ | SE(Δ) | proposal's P(cross) | **recomputed P(cross)** |
|---|---|---|---|---|
| CUB vs PFML | −0.0030 | 0.00224 | 0.15 | **0.090** |
| Cars vs PFML | −0.0040 | 0.00224 | 0.10–0.25 | **0.037** |

The proposal's own conclusion is correct ("reaches PFML's error bar and does not cross it"), and the arithmetic is worse than stated by 1.7–4×. Against DADA's matched-cost rows, +0.2 is inside the ±0.4–0.5 seed noise the proposal itself documents. **There is no forecast crossing of any in-scope method.** The deliverable is a ~+1 pt mechanism claim whose mechanism probe cannot be computed.

---

## 7. Cost and memory, recomputed

**Time — the claim holds, and is conservative.**

- eigh(512), measured: 35.1 ms on this CPU ⇒ **0.70 ms/step** amortized at T=50 (the proposal's 0.2 ms implies ~10 ms; cuSOLVER `syevd` at n=512 is typically a few ms, so ≲0.1 ms/step on GPU). Either way negligible.
- FLOPs/step: `P_r f` 31 MF; proxy projection 512²×1500 ≈ 393 MF fwd (~1.2 GF with backward) on CUB; ~5.9 GF fwd on SOP. Against ResNet-50 fwd+bwd at batch 120 ≈ **1.5 TFLOP/step**, that is **< 0.1%**.
- Backbone backward runs once — verified from the graph structure. ✓
- No forced host-device sync: the EMA update, eigh, and integer slice of `U[:, :r]` are all sync-free.

**1.02–1.03× is defensible.** This is the strongest part of the proposal.

**Memory — the claim is wrong by ~6× to ~64×.** Autograd must retain `P_r p̂` and its normalization for every proxy, plus the extra logit/exp buffers:

| | proxies | buffers (S, U, P_r) | extra proxy activations | extra logit/exp | **total** | claimed |
|---|---|---|---|---|---|---|
| CUB, K=15 | 1,500 | 3.1 MB | 6.1 MB | 2.2 MB | **≈ 11.4 MB** | +2 MB |
| SOP, K=2 | 22,636 | 3.1 MB | 92.7 MB | 32.6 MB | **≈ 128.5 MB** | +2 MB |

Not fatal (both are small against ResNet-50 activations at batch 120), but the stated figure is not what the method costs. Note also that at K=15 on SOP the proxy set would be 169,770 — the K=2 choice there is load-bearing for feasibility, not only for the mechanism story.

---

## 8. Verdict

**DEAD.**

**Earliest protocol gate failed:** the **§3 adversarial-novelty gate**. RSC (ECCV 2020, oral) publishes the same supervision object and the same action — data-dependent removal of the representation's most label-predictive component, the same loss recomputed on the residual, backpropagated through the whole network, zero extra parameters, claimed to improve out-of-distribution transfer. DSA's sole claimed advantage over the axis-aligned form is D1/D3, and D1 closes only *linear* duplication. It never reaches the controls.

*(The earliest self-contained arithmetic failure, if novelty were somehow set aside: §4/C8. PR ≤ rank ≤ #identities − 1 makes the forecast "8–15 → 30–45" impossible on a 20-identity (CUB) or 10-identity (Cars) held-out split, and makes F4 unsatisfiable on Cars. The mechanism probe and the mechanism falsifier are both non-executable, checkable with a pencil.)*

**Single most decisive mechanism-level reason:** λᵢ = gᵢ·sᵢ. DSA controls the realized between-identity scatter; the causal argument requires the encoder gain spectrum. Flattening λ forces g ∝ 1/s — *away* from the Var[T]-minimizing g = const — whenever the training class-contrast spectrum s is anisotropic, which it is. And a perfectly unbiased flat-gain encoder on a mildly anisotropic (i⁻¹) class spectrum already produces PR(S_B) = 16.4, inside the range the proposal calls diseased. **The disease and the cure are the same observable, and it does not identify the mechanism.**

**Not BLOCKED:** every finding above is a mathematical fact, an arithmetic impossibility, a documented publication, or a structural property of the frozen objective. No GPU evidence would change any of them.

### Preserved subcomponents (valid independently of the verdict)

1. **§1.5 gradient derivation** — exactly correct, including `(I − f̃f̃ᵀ)P_r = P_r − f̃f̃ᵀ`. Reusable for any renormalized-projection auxiliary loss.
2. **Single-backbone-backward construction and the cost analysis** — correct and conservatively estimated; < 0.1% FLOP overhead is real. (Fix the memory figure.)
3. **Var[T] = 2σ_g²/(d+2)** for a quadratic form under sphere-uniform contrast — exact, MC-confirmed, and a genuinely useful lemma for any spectral-transfer argument.
4. **D4** — the "same loss on the residual" form structurally excludes collapse and volume inflation, unlike −log det / entropy penalties. This is a correct and non-obvious design argument, worth carrying to any future method.
5. **rank(S_B^batch) ≤ |C_B| − 1 and the necessity of the EMA** — correct. (But use m ≥ 8 or an unbiased estimator; the Σ_W/m bias must be subtracted or the object is mislabelled.)
6. **The held-out-training-identity selection protocol** — genuinely stricter than the literature it compares against; keep it, but size it for the effect (it is ~2× under-powered as drawn).
7. **Refusing to forecast a stacked DSA+PFML number without reproducing PFML**, and refusing an In-Shop crossing claim against an uncertainty-free reference — correct discipline, and rarer than it should be.
8. **Correction to §6 item 2 for future reuse:** arXiv:2003.13911 *does* disclose α=32, δ=0.1, AdamW, the ×100 proxy lr, and 40/60 epochs. The recalled values are right; only the epoch budget conflicts (published 40, proposal 200).

---

**Sources**
- [Proxy Anchor Loss for Deep Metric Learning (Kim et al., CVPR 2020) — arXiv:2003.13911](https://arxiv.org/abs/2003.13911) ([HTML](https://ar5iv.labs.arxiv.org/html/2003.13911))
- [Self-Challenging Improves Cross-Domain Generalization (RSC; Huang, Wang, Xing & Huang, ECCV 2020) — arXiv:2007.02454](https://arxiv.org/abs/2007.02454) · [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf) · [official code](https://github.com/DeLightCMU/RSC)
- [Gradient Starvation: A Learning Proclivity in Neural Networks (Pezeshki et al., NeurIPS 2021) — arXiv:2011.09468](https://arxiv.org/abs/2011.09468) · [NeurIPS page](https://papers.nips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html)
- [Understanding Dimensional Collapse in Contrastive Self-Supervised Learning (Jing, Vincent, LeCun, Ma; ICLR 2022) — arXiv:2110.09348](https://arxiv.org/abs/2110.09348) · [OpenReview](https://openreview.net/pdf?id=YevsQ05DEN7)
- [Self-Erasing Network for Person Re-Identification (Fan et al., Sensors 2021)](https://www.mdpi.com/1424-8220/21/13/4262)
- [Adversarial Erasing Attention for Person Re-Identification (IEEE Access)](https://ieeexplore.ieee.org/document/9043556/)
- [Learning Ordered Representations with Nested Dropout (Rippel, Gelbart & Adams, ICML 2014) — arXiv:1402.0915](https://arxiv.org/pdf/1402.0915)
- [Divide and Conquer the Embedding Space for Metric Learning (Sanakoyeu et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)

All numerical results above are fresh CPU computations (numpy 2.5.0); scripts are at `/tmp/dsa_check.py`, `/tmp/dsa_check2.py`, `/tmp/dsa_check3.py`. I did not edit any repository file, and consulted no other model.

