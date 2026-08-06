# ADJUDICATION — Pass 39 "FCS — Factorial Coordinate Supervision"

**Read:** `docs/opus_fcs_proposal_pass39_2026-08-06.md` only. No other repository file, no git history, no prior review, no ledger. No files edited.

---

## VERDICT: **DEAD**

**Earliest failed gate:** Gate 1 (executable mathematics) — §1.5.5 repair/partial-revert is neither uniquely executable nor feasibility-preserving.

**One decisive mechanism-level reason (Gate 2):** The proposal's sole causal target — *prototype-span confinement* (§2.1) — is manufactured by an asymmetric gradient derivation. §1.3 correctly derives FCS's **pre-normalization** gradient with the projector, `∂ℓ/∂z̃⁽ᵏ⁾ = (s/K r_k)(I − uuᵀ)(p − t)`; §2.1 then indicts the baseline using only its **post-normalization** gradient, `∇_z L ∈ span{p_1…p_C}`. Applying §1.3's own projector to the baseline gives `∇_z̃ ℓ = (1/r)(I − ẑẑᵀ)Σ_c α_c p̂_c ∈ span{p_c} + span{ẑ_i}`, whose span over CUB's 5,864 training samples is generically full 512-D. The claimed 413 "loss-invisible" directions do not exist, and the single rescue clause — "annihilated by weight decay" — is false by ~4 orders of magnitude at the proposal's own frozen `λ=1e-4, lr=1e-4`: `(1 − ηλ)^9773 = (1 − 1e-8)^9773 ≈ 0.99990` on CUB (0.9990 on SOP). **The intervention has no established target.**

---

## Gate-by-gate

### 1. Forward map, loss, gradients, code update — executability

**Verified correct.** `‖φ‖² = (1/K)Σ‖u⁽ᵏ⁾‖² = 1` ✓. `∂ℓ/∂u⁽ᵏ⁾ = (s/K)(p − t)` ✓ (uses `Σt = 1`). `∂ℓ/∂z̃⁽ᵏ⁾ = (s/Kr)(I − uuᵀ)(p−t)` ✓. Train/test forward identity ✓. Deployment envelope is genuinely clean.

**Fails on §1.5.5.** Three defects:

- **"Affinity margin" is undefined** — no formula given for which class in a violating pair moves.
- **A single move cannot preserve (B).** Moving class *c* from symbol *a* to symbol *b* requires *simultaneously* that count(*a*) = ⌈C/d⌉ and count(*b*) = ⌊C/d⌋. "Best alternative symbol having spare capacity" is not that condition. On CUB (C=100, d=16 → 4 symbols at 7, 12 at 6) most moves drop a source symbol to 5 and violate (B). Swaps/cycles are required and are not specified.
- **"Feasible by induction" is false.** The induction hypothesis covers the all-old code `Y^{t−1}` and (hypothetically) the all-new `Y′`. The partial revert produces a **mixture** `{y^{t−1}_c : c∈R} ∪ {y′_c : c∉R}`. Nothing bounds `H(y^{t−1}_c, y′_{c″})` for a reverted *c* against a non-reverted *c″* — (S) can break. And a partial revert changes per-symbol counts arbitrarily — (B) breaks too. Only a **full** revert is guaranteed feasible, and that makes the epoch's code update a no-op.

Everything else in §1.1–§1.4 and §1.6 is uniquely executable.

### 2. The three propositions

**Proposition 1 — FALSE as stated; non-differentiating under the charitable reading.**

- Literal claim `span{φ(codewords)} = ℝ⁵¹²` is false. Every codeword satisfies `⟨v_c, 1⁽ᵏ⁾⟩ = K^{−1/2}` for all *k*, so all codewords lie in a subspace of dimension ≤ `512 − (K−1) = 481`. **On CUB there are C=100 codewords, so their span is at most 100-dimensional — numerically identical to the bound §2.1 attacks in the baseline.**
- `∇_u ℓ = (s/K)(p − t)` is never "dense in ℝᵈ": it sums to zero, so it lies in `1_d^⊥`, giving `K(d−1) = 480`, not 512.
- Charitable reading ("all 512 atoms are softmax targets under (B)"): true, but it holds **identically for K=1, d=512** — the proposal's own control **C4** — and for random frozen codes (**C1**), and for any fixed 512-atom classifier where unused atoms still sit in the denominator. Prop 1 therefore supplies **zero** differential support for K>1, over-subscription, balanced re-assignment, or the min-Hamming constraint. §3.11 already concedes K=1 ≡ Hoffer et al., *Fix your classifier*.

**Proposition 2 — TRUE but vacuous.** `L→0` ⇒ per-block argmax = target; (S) with δ_min≥1 ⇒ injective. This is "if the loss reaches zero, training images are classified correctly" — true of any classification loss, silent on unseen identities. (Attainable: at `u = e_y`, `s=16`, min per-block CE = `−log(1/(1+15e^{−16})) ≈ 1.7e-6`.)

**Proposition 3 — arithmetic correct, headline wrong by 10×, premise nonexistent at deployment.**

The bound `cos(diff) ≤ 1 − (δ_min/K)(1−ξ)` and the condition `η < (δ_min/K)(1−ξ)` are algebraically right. The headline — *"up to 25 % of the block-similarity budget may be destroyed"* — silently sets **ξ = 0**. At the frozen `s=16, ε_ls=0.1`, ξ is not 0.

The per-block minimizer satisfies `(I − uuᵀ)(p − t) = 0` ⇒ `p = t`. With `t_y = 0.90625`, `t_j = 0.00625`: `u_y − u_j = log(145)/16 = 0.311`; solving `16a² + 0.622a − 0.9033 = 0` gives `a = 0.2190`, `u_y = 0.530` (norm checks to 1.000). Then for `y ≠ y′`:

> **ξ = 2u_y a + (d−2)a² = 0.2321 + 0.6716 = 0.9037**

So the true tolerance is `0.25 × (1 − 0.9037) = **0.0241**` — **2.4 %, not 25 %.** At convergence the two most-separated training identities sit at `cos = (24 + 8×0.9037)/32 = 0.9759`; the *minimum possible* cosine between any two training images is 0.9037. The entire deployed descriptor set contracts into a spherical cap of angular radius ≈25°, and the total usable separation budget is 2.4 % of the cosine scale.

**Premise failure:** the hypothesis is stated over "the ≥δ_min blocks that separate the two codewords." At test time both query and gallery are **unseen identities with no codewords**, and *Y* is discarded. Prop 3 makes no statement about the deployed task.

### 3. Proxy-loss gradient audit, weight decay, encoder gain

- **Normalized vs pre-normalization:** the stated subspace limitation holds *only* post-normalization. Pre-normalization — the level at which `W` and `θ` actually update — the projector injects `ẑ_i`, and `∇_W ℓ = [(1/r)(I−ẑẑᵀ)v] gᵀ` has generically full-rank column space over the dataset. Correcting the asymmetry removes the deficit entirely.
- **Weight decay is not selective and is not "annihilation."** Decoupled AdamW shrinks *all* of `W` uniformly (governed directions are merely replenished by gradient). Magnitude at frozen hyperparameters: **0.99990** total over 200 CUB epochs. Even under the `(1−λ)` convention it is `e^{−0.98} = 0.375` — a 2.6× shrink, not deletion, and applied to governed directions too.
- **Encoder gain vs training-class evidence:** not separated anywhere. Neural collapse NC2/NC3 constrains *training-class feature means*; §2.1 transfers it to *unseen-identity geometry* without argument. The prompt's exact conflation is present and load-bearing.

### 4. Degeneracies — constructive attack

**D1 ✓ correct** (infeasible under (B)). **D4 ✓ correct** (constant block ⇒ CE ≥ log 16 = 2.7726). **D2 depends on §1.5.5, which fails above.**

**D3 is not excluded.** (S) constrains the *code*, not the *functions*. A low-rank shared feature with block-specific linear readouts satisfies all K partitions with highly correlated blocks. The K subspaces are orthogonal by *slicing construction*, not by any learned decorrelation; the NMI diagnostic is a measurement with no threshold attached.

**Code chasing (D6) is worse than conceded.** *Y* is re-solved each epoch from the encoder's own affinities `Ŝ`. The system has a self-consistent fixed point in which the code merely ratifies whatever partition the encoder already computes. Hysteresis γ and the epoch-160 freeze **stabilize** that latching rather than damp it.

**Over-subscription is an instruction to destroy information.** On SOP, ρ = 707: each identity shares its block-*k* symbol with ~706 others, and all 707 are pulled to the identical target point `u*` in that block. Over-subscription is framed as "capacity"; operationally it is an explicit per-block collapse of ~707 identities, with residual separation coming only from the ≥8 differing blocks (cos 0.976).

**The prompt's direct question — what constrains two images of a novel identity to share an unused code, and different novel identities to receive separated codes?**

> **Nothing.** No term in `L` references unused codewords, novel identities, or the geometry of the complement. `d^K − C` unoccupied codewords (§3.9) is a counting statement about a **discarded discrete table**, not a constraint on the encoder. At test time the encoder emits continuous `u⁽ᵏ⁾`; there is no force toward unoccupied tuples, no force making two views of a novel identity agree per block, no force separating two novel identities. The only inductive bias is ImageNet init + augmentation invariance — **identical to the baseline.** The zero-shot claim rests entirely on an untested transfer assumption, and Prop 3, its only quantitative statement, is conditioned on objects unseen identities do not possess.

### 5. Balance + separation: existence, feasibility, complexity

- **GV arithmetic ✓ verified.** `16³² = 3.40e38`; `Σ_{i<8} C(32,i)15^i ≈ 5.85e14` (i=7 term = 3,365,856 × 1.7086e8 = 5.751e14); ratio ≈ **5.8e23** ≫ 11,318. Correct.
- **But GV proves neither simultaneity nor reachability.** GV codes need not be balanced. Random balanced codes do satisfy (S) w.h.p. (agreements ~ Bin(32, 1/16), mean 2 ≪ 24), so **epoch-0 initialization is fine**. The failure is per-epoch: **the min-cost flow ignores (S) entirely** and is run independently per block on affinities driven by one shared encoder. Visually similar classes get similar `Ŝ` rows in *every* block ⇒ correlated codewords ⇒ violations concentrate precisely on confusable pairs, and repair moves each to "its best alternative symbol," which is again similar. No potential function is given, so 100-round termination is an assumption; moves can create new violations, so progress is not monotone; and the fallback is the broken partial revert.
- **Pigeonhole check ✓ correct and non-obvious.** 32 blocks into δ_min=8 groups of 4; distance <8 ⇒ ≥1 group fully agrees ⇒ no false negatives. But `O(KC)` *expected* assumes uniform-random bucket occupancy. The code is affinity-correlated; collisions concentrate. Worst case `O(C²)` per group ≈ 1e9/epoch at SOP.
- **Complexity understated ≥32×.** `Cd log₂C = 11318 × 16 × 13.5 = 2.44e6` is the figure quoted — that is **one block**. With K=32 flows it is 7.8e7, yet §6 restates 2.5M as the *per-epoch total for K flows*. Separately, `O(Cd log C)` is not a valid bound for min-cost flow with lower bounds on a C×d transportation problem; SSP is ~`O(C²d)` ≈ 1e12/block at SOP absent a specialized argument, which is not given.
- **Low-count window:** SOP 5.26 img/class, In-Shop 6.5 — both below the n_c<10 trigger, so both run on 5-epoch-stale features while the code is re-solved every epoch. γ=0.05 against an incumbent gap of 0.311 (≈16%) is reasonably calibrated.

### 6. Prior art — **UNRESOLVED** (downstream; immaterial to the verdict)

I did not run new primary-source retrieval. §3 is disciplined and I cannot refute it from memory with primary-source confidence. Three observations that stand on the frozen text alone:

- §3.11 **concedes** K=1 ≡ Hoffer et al. Combined with Prop 1's failure to differentiate K>1, the residual novel content is only (a) over-subscribed d-ary alphabet, (b) per-epoch balanced re-assignment, (c) min-Hamming constraint.
- (b) is the same *action* as SeLa (Asano et al., ICLR 2020) / SwAV: exact balanced optimal-transport assignment alternated with encoder training over a fixed atom set. §3.12 distinguishes on the supervision *object* (classes vs samples, supervised vs not) — a genuine but narrow distinction under the "judge the action" instruction.
- §7.5 concedes DREML and OPQN rest on non-primary sources, and §7.6 concedes the *nearest-neighbour question* (CSQ-style separated centers applied to identity-disjoint DML under a continuous descriptor) was never searched.

### 7. Causal provenance

**§2.1's premise is nowhere measured.** It is derived (incorrectly, see Gate 2) plus a neural-collapse citation. No repository measurement of any existing checkpoint's unseen-class descriptor rank is presented. Likewise `d^K − C` "capacity," F2's implicit assumption that C2's rank is low, and every forecast in §5 are unmeasured.

**Required measurement, zero GPU cost:** on the **existing corrected local Proxy Anchor checkpoint** (In-Shop seed 0, raw 0.9163 / final 0.9137), compute the participation-ratio effective rank of the 512-D descriptor over the *unseen test identities*, plus the fraction of retrieval-relevant variance outside the top-C principal directions. If that rank is already ≫ C — which the corrected pre-normalization gradient argument predicts — §2.1 is dead and the intervention has no target. The proposal instead schedules 10 controls × 4 datasets × 5 seeds × 200 epochs of ResNet-50 **before** ever testing its own premise.

### 8. Controls C1–C10 — decision validity

- **C6 uses forbidden test selection.** "Tuned so its measured **test** effective rank matches FCS's" selects a control model using test-set statistics. This violates the legal envelope *and* the proposal's own §1.6/§6 pledge ("no test identity, image, or gallery statistic is ever touched"). C6 is designated "the decisive one" and F4 is a rejection criterion — so **the decisive control is illegal as written**, and run legally (matching on a held-out training-identity split) F4's 0.3-pt threshold is uncalibrated.
- **C5 cannot isolate over-subscription** — four simultaneous confounds: block dimension *d*, `δ_min = ⌈K/4⌉`, softmax denominator size, and the target geometry ξ(d).
- **C5 and Prop 3 predict opposite directions.** `δ_min/K = 0.25` is **constant** across the entire sweep (16/64, 8/32, 4/16, 2/8, 1/4). Only ξ(d) varies. Recomputing the loss-optimal ξ at the frozen `s, ε_ls`:

| (K,d) | ρ=C/d (CUB) | ξ | Prop-3 margin 0.25(1−ξ) |
|---|---|---|---|
| (64,8) | 12.5 | 0.9284 | 0.0179 |
| (32,16) | 6.25 | 0.9037 | 0.0241 |
| (16,32) | 3.13 | 0.8745 | 0.0314 |
| (4,128) | 0.78 | 0.8057 | 0.0486 |

  C5 *requires* monotone degradation as ρ→1. **Prop 3 predicts monotone improvement as ρ→1 — a 2.7× larger margin at the low-ρ end.** F3 would reject the method for failing a prediction its own theory inverts.
- **C4 has no falsification threshold.** F1–F6 cover C2, rank, C5, C6, C1 and the crossing. C4 is the one control that would expose Prop 1's mechanism as the fixed-512-atom effect, and nothing can be rejected on it.
- **Screening protocol violated.** The envelope requires a same-seed corrected **In-Shop paired control first**, against the local 0.9163/0.9137. The proposal is CUB-first, files In-Shop as "secondary," and references an external single-point PA+DADA 0.930 instead of the corrected local baseline.

### 9. Cost and frontier arithmetic

- **Parameters ✓** — zero added learned parameters is correct.
- **Memory claim is asserted without counting FCS's dominant term.** §6 counts only `CK` integers and omits `Ŝ ∈ ℝ^{K×C×d}`: CUB 51.2k floats (0.2 MB); SOP 5.79M (23 MB), ×5 for the E_w window → 116 MB; In-Shop 2.05M (8.2 MB) → 41 MB windowed. PFML SOP with Adam state = 34.8M floats (139 MB). "Strictly lower" survives numerically but becomes marginal on exactly the two datasets where the window is used, and the dominant FCS term is never accounted.
- **Flow cost 32× understated** (see Gate 5). Repair worst case: 100 rounds × separation check → up to ~1e11 at SOP.
- **Frontier arithmetic ✓ verified.** `SE_diff = √(2×0.004²/5) = 0.00253` ✓; `0.011/0.00253 = 4.35σ` ✓. Welch p at df≈8 is ≈0.0024 two-sided; the quoted "p≈0.005" is conservative, i.e. errs against the proposal. Pooling the asymmetric σ (0.005/0.003) gives t=4.21 — immaterial.
- **Lane separation is maintained honestly, and no general SOTA claim is made.** Correctly so: the proposal's *best-case* forecasts (CUB 0.745, Cars 0.931, In-Shop 0.928) sit below VAPNet (0.762 / 0.948 / 0.939), AdvRF (0.766 / 0.949) and CRT In-Shop 0.9448. Even on full success the maximum available payoff is a matched-Lane-A crossing of +1.1 pt at the author's own P≈0.35.

---

## Correct subcomponents — preserved independently of the verdict

1. **§1.2 forward map / block normalization.** `‖φ‖=1` verified; identical train/test; the deployment envelope is genuinely clean (one encoder, one view, one 512-D descriptor, cosine NN, table discarded, no reranking, no test-time fitting).
2. **§1.3 gradient derivation.** Both the `(s/K)(p−t)` term and the `(I − uuᵀ)/r` projector are correct. Reusable.
3. **§1.3 scale/decay coupling.** `∂ℓ/∂z̃ ∝ 1/r_k` and the resulting λ-vs-effective-angular-LR argument is correct and well posed; the mandatory λ ∈ {0, 1e-4, 5e-4} ablation is right. (It also happens to be the observation that refutes §2.1's own decay clause.)
4. **§1.4 GV computation.** Arithmetic verified end to end. Establishes (S) is satisfiable in isolation.
5. **§1.5.4 pigeonhole separation check.** Correct, non-obvious, no false negatives, and independently reusable for any balanced-code scheme. The strongest technical contribution in the document.
6. **§2.2 D1 and D4.** Both exclusions are sound, including the `CE ≥ log d = 2.7726` bound.
7. **§3 novelty discipline.** Per-item mechanism distinctions, and voluntary disclosure of the practitioner precedent ("embedding as a concatenation of softmax feature groups") rather than omission.
8. **§5's SOP null.** A genuinely risky differential prediction with the correct falsification logic stated ("a uniform gain across all four datasets would be evidence *against* my mechanism"). Exemplary.
9. **§7.3 recipe-gap disclosure.** PFML CUB 0.734 vs the official ProxyAnchor repository's 0.699 at matched ResNet-50/512-D, with the explicit refusal to inherit a published frontier by beating a weaker in-house reproduction. Correct handling of an unmatched published number.
10. **§6 deployment parity and zero-added-parameter accounting.** Correct.

---

## Unresolved uncertainty (stated, not papered over)

- **Gate 6 (novelty) is UNRESOLVED.** I ran no new primary-source retrieval; §3 is neither confirmed nor refuted by me. §7.5 and §7.6 already concede two pivotal distinctions rest on non-primary sources and that the nearest-neighbour question is unsearched. This does not affect the verdict — Gates 1 and 2 fail upstream on the frozen object's own arithmetic.
- **ξ = 0.9037 is the loss-*optimal* value.** The ξ actually achieved at epoch 200 depends on how close optimization gets to the per-block minimizer; a stalled run has smaller ξ and a larger Prop-3 margin. The *direction* of the Prop-3 ↔ C5/F3 contradiction is robust across the whole d ∈ {8…128} sweep and does not depend on convergence.
- **§2.1 is not refuted absolutely, only unestablished.** A skeptic may argue that at convergence `ẑ_i` itself clusters near the proxy span. That is a fixed-point claim about *training* features, not a rank bound on the encoder, and it is exactly the conflation flagged in Gate 3. It is settleable by the zero-GPU measurement in Gate 7 and by nothing else in the proposal.
- I could not independently verify PFML's, DREML's, or PA+DADA's reported numbers.

---

## Note on scope

Every failure above is repairable in principle — a full revert instead of a partial one; a legal C6 matched on a held-out training-identity split; a Prop-3 restatement at the correct ξ; a C5/F3 direction reversed to follow Prop 3; and the Gate-7 measurement run *first*. **Per the adjudication instruction, none of that is applied here.** A substantive repair is a new proposal and must not be used to upgrade this frozen one. Pass 39 as frozen is **DEAD**.
