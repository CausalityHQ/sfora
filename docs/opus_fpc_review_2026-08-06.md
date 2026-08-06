# Cold adjudication — FPC (Pass 46), Lane A

**VERDICT: DEAD.**

**Earliest failed gate:** Audit gate 1 — §1.4's dimensional claim. The proposed proxy set does not have full 512-D span. Its hard-phase **affine** span is at most `m(C−1) = 56`, which is *exactly* the rank of control C1.

**Decisive mechanism-level reason:** FPC's only claimed causal mechanism is a head-capacity bottleneck, but the design *enforces code injectivity by construction* (`L_inj`, `h=2`, deterministic repair, `C^m = 1.7×10⁷ ≫ K`). For an injective code, `I(class; head) = H(class) = log₂100 = 6.64 bits` — identical to a free proxy table. The advertised 24-bit budget is a bound *strictly looser* than the trivial `H(class)` bound, so it constrains nothing. What remains after removing the void bottleneck is a rank-56 label-embedding head — the exact family §3.19 nominates as its hardest-defended distinction.

*Method note: I read only the two named artifacts. Audit item 6 requires primary prior-art search, so I used web search/fetch for primary sources only. No repository files read, no edits, no consultation, no delegation.*

---

## 1. Dimension of the proxy set — **FAILS**

`z^(b) ∈ ℝ^{64}` (m=8), and each block codebook holds only `C=8` unit vectors. Every `μ^(b)_c` — soft (normalized convex combination) or hard (a codeword) — lies in `span{u^(b)_1..u^(b)_8}`, dimension ≤ 8.

- **Linear span** of `{w_c}` ≤ `⊕_b span(U^(b))` = `mC = 64` of 512. (SOP/In-Shop, C=16: 128 of 512.)
- **Affine/difference span**: `w_c − w_{c₀}` has block component in `span{u^(b)_j − u^(b)_{κ_b(c₀)}}`, dim ≤ `C−1 = 7`; over 8 blocks, ≤ **56**.

"Full 512-dimensional span" (line 46) confuses *full support* (dense coordinates) with *full span*. Full span requires `C ≥ d_b`, i.e. `mC ≥ 512`; the largest configuration in the whole proposal reaches 128.

Consequence for C1: PA's loss depends on `z` only through `⟨z, w_p⟩`, and all class discrimination lives in proxy *differences*. C1 is `r = 56`. FPC's difference space is `≤ 56`. **C1 is exactly capacity-matched to FPC in the only subspace that carries discrimination** — arrived at via the wrong stated justification ("soft-phase continuous dof"), but numerically right. F2 should be expected to fire. §3.19's "combinatorial, not a rank, restriction" is refuted by the proposal's own decisive control.

Secondary: `∂L_PA/∂z` lies in `span{w_p}`, so the metric loss supervises ≤64 of 512 deployed dimensions (PA: ≤ K = 100). The remaining 448 receive gradient only from `L_nrm`.

## 2. Proposition 3 — **VOID**

Formalized on one random variable, one unit: let `c ~ Unif{1..K}`, `H(c) = log₂100 = 6.644 bits`.

- FPC hard head: `c ↦ κ(c)`, **injective by construction** ⇒ `I(c; κ(c)) = H(c) = 6.644 bits`.
- PA free head: `c ↦ w_c`, distinct proxies ⇒ `I(c; w_c) = H(c) = 6.644 bits`.

**Identical.** The bound `I ≤ m log₂C = 24` exceeds `H(c) = 6.64`, so it is satisfied by *every* head including a free proxy table. Prop 3 is not weakly true — it is non-binding. "512 continuous dimensions for a free proxy" is a dimension count, not an MI in bits; the comparison the proposition rests on is a unit error.

You cannot simultaneously enforce injectivity and claim a bottleneck. The design deliberately guarantees the code is a *lossless* encoding of identity.

**Assignment logits do alter the claim, against it.** `A ∈ ℝ^{K×m×C}` = 6,400 class-private continuous parameters on CUB, **zero weight decay**, lr `100η`, still receiving straight-through gradient throughout the hard phase. "The head carries exactly `m log₂C` bits per class and nothing more" (line 5) is false as a statement about the head: only the forward composed proxy is quantized, and only after epoch 120 of 200.

## 3. Propositions 1, 2, 4 and Lemma 1 under finite α — **FAIL**

**Prop 1 — invalid algebra.** At the collapse point the negative term is `log(1 + |X⁻_p| e^{α(ρ₀+δ)})`, which vanishes only if `ρ₀ < −δ = −0.1`. For K=100 unit vectors the minimum achievable max-pairwise cosine is `−1/(K−1) = −0.0101 > −0.1`. So `ρ₀+δ ≥ 0.0899` and the negative term **grows** as `0.0899·α`; at α=32 with ~178 in-batch negatives it is `log(1 + 178·e^{2.877}) ≈ 8.06`, not 0. The stated bound `e^{−α(1−ρ₀−2δ)}` applies the positive term's sign to both. `L_PA → 0` is unreachable for *any* configuration at δ=0.1, K=100 — the premise fails landscape-wide, not just at collapse. Separately, "rank ≤ K−1" for a class-constant map is dimensional algebra (Candidate 371's point), not demanded representation rank.

**Prop 2 — non-sequitur.** Set `φ(x) = w_{c(x)}`. The `w_c` are K distinct unit vectors; the head is maximally satisfied. **The constrained head does not forbid the encoder from memorizing the same injective class code** — it only lowers the memorized code's rank from ≤99 to ≤56. Parameter sharing of `u^(b)_j` across classes forces *target coincidence*, not *shared visual evidence*: the encoder may recognize identity first and emit the codeword by lookup. Sharing at the target ≠ sharing at the evidence.

**Prop 4(a) — quantitatively wrong by ~22×.** `ρ·log C = 20.79` nats is `ρ ×` the *maximum* KL over C atoms (total collapse onto one symbol), not the cost of the named shortcut. Isolating one class of K=100 in its own symbol gives column marginal `(0.01, 0.99/7 ×7)` and `KL = 0.0970` nats ⇒ cost `≈ 0.97` nats — purchasable for free by an undecayed logit at lr `100η`. At SOP (K=11,318) the barrier scales as O(ρ/K) ≈ 10⁻³ nats: essentially zero, precisely where balance is needed most. The stated bound also drops the entropy term: correctly, `ρ·KL* ≤ ⟨P*,A⟩−⟨P_bal,A⟩ + τ(H(P*)−H(P_bal))`.

**Prop 4(c) — weak.** A `√(7/8) = 0.935` margin shrink is a 6.5% disincentive. `L_sep` is a hinge at 0, trivially satisfiable by 8 orthogonal vectors in ℝ⁶⁴ — it will sit at ≈0 from initialization and exert no pressure.

**Lemma 1 — premise false, and it is the wrong bound.** "α=32, δ=0.1 drives `s → 1`" is not true: at `s = 0.4` the positive term is `e^{−9.6} = 6.8×10⁻⁵`, already saturated; there is no pressure past ~0.4. Worse, the constraint makes `s → 1` *loss-increasing*. In the hard phase, for classes at the enforced minimum Hamming distance `h = 2`:

```
⟨w_c, w_c'⟩ = (m−h)/m = 0.75   (differing codewords orthogonal; ≥ 0.5 in all cases)
max ⟨z,w_c⟩ − ⟨z,w_c'⟩ ≤ √2·h/m = 0.354
```

So the entire cosine separation budget for the closest code pair is **0.354**, versus ~1.0 for free proxies, and a perfectly-aligned sample contributes `e^{32(0.85)} = e^{27.2}` to the neighbour's negative term. Note `L_inj`'s hinge is at exactly `1 − h/m = 0.75`, so there is **zero** pressure to exceed Hamming distance 2 — the worst case *is* the design point, and §3.18 says the optimizer will place visually similar classes there. FPC caps the margin precisely where discrimination is hardest. The predicted In-Shop/SOP regression (§6) is the mechanism's forecast, not a tail risk. Finally: even at exact equality, the lemma constrains geometry on *training* classes; nothing transports it to disjoint test identities, and the head is discarded at test.

## 4. Executable paths — multiple defects

- **Solver ≠ program.** A KL-relaxed column marginal is unbalanced OT and needs the generalized scaling update with exponent `ρ/(ρ+τ)`; "3 warm-started Sinkhorn iterations" is the balanced alternating projection and does not solve the stated argmax. *(Generalized-scaling requirement asserted from standard unbalanced-OT theory; not verified against primary text in this pass.)*
- **`τ = 0.05` numerics.** `exp(A/τ)` with O(1) logits ⇒ `e^{20}`. Log-domain not specified. Three iterations at that τ are far from converged; the unrolled gradient is a biased estimator of the argmax gradient.
- **Normalization singularity.** No ε guard on `‖Σ_j π_{cj} u^(b)_j‖`; gradient blows up as `1/‖·‖` when `L_sep` drives inner products negative.
- **Initialization pathology.** At `τ=1.0`, π is near-uniform ⇒ all `μ^(b)_c` coincide ⇒ all proxies nearly identical ⇒ maximal negative term for the first ~tens of epochs. The frozen-backbone warm-up does not address head geometry.
- **STE / repair mismatch.** Repair overrides the forward symbol to a runner-up while the surrogate gradient is attached to the un-repaired π. Estimator bias unspecified.
- **Repair has no termination proof.** Substituting a runner-up symbol can create a fresh collision; no potential function, no iteration bound, no guarantee the runner-up is unoccupied. `O(K²m)` is one distance pass, not the loop.
- **Repair destroys balance.** Unconstrained symbol overrides break the column marginals Prop 4 depends on. Unaddressed conflict.
- **Sampled injectivity is too weak.** At K=11,318, a given pair enters the 1024-class subsample with probability `(1024/11318)² = 0.0082` — ~1 step in 122.
- **`C4` is not executable.** At `m=1`: `1 − h/m = −1`, so `L_inj = mean[o+1]²_+ ≥ 1` for probability vectors (`o ≥ 0`) — **unsatisfiable**, at λ=1.0. And `L_dec = mean_{b<b'}` over an empty index set is undefined. The sanity anchor is broken as specified.

## 5. Counts — one headline is inflated ~2×

- Head params **correct**: `4,096 + 6,400 = 10,496` vs PA `51,200`. ✓ (Holds at SOP: 1.46M vs 5.79M.)
- `C^m`, bit budgets, and the 512.4 MB `K×K` figure at K=11,318 are **correct**. ✓
- **§1.8 is wrong.** CUB train = 5,864 images / 100 classes = **58.6/class**, not "~30". The proposal's own 733 = `(K/C)·58.6 = 12.5 × 58.6` — so it computes with 58.6 while stating 30. True ratio is `K/C = 12.5×`, advertised as 24.4×.
- Even 12.5× is not a sample-complexity gain: it is more data for a *coarser, different estimand*. The per-class discriminative burden is unchanged at 58.6 images, and the ≥456 unspanned dimensions now carry no target at all. This is exactly the images-per-class / images-per-shared-parameter conflation the gate forbids.
- **Cost claim fails at scale.** Either repair is global (512 MB + `K²m ≈ 1.0×10⁹` ops per invocation × ~331 steps/epoch at SOP — breaking "1.00× memory, 1.00–1.02× time") or it is subsampled (breaking "Proposition 3 holds by construction"). §1.6 and §1.7 contradict.

## 6. Prior art — the untraced neighbour resolves; novelty fails

**§7.5's neighbour exists.** *Compositional Global Class Embeddings* (Michalkiewicz et al., arXiv:2106.06440; earlier arXiv:2004.06302): "a set of M **codebooks** (embedding tables), with each codebook `C_j` containing m individual **codes**"; "an **attention array of size 21×5×6; i.e., one attention value per (class, codebook, code) triplet**"; `α^j_i = sparsemax(w^j_i)` selects a sparse subset per codebook; "a weighted sum of all codes yields the final embedding." That is FPC's `U^(b)` / `A^(b)_c ∈ ℝ^{K×m×C}` / `μ^(b)_c = normalize(Σ_j π_{cj} u_j)` construction, one-for-one.

Per §7.5's own pre-committed rule, novelty must be re-adjudicated onto three surviving distinctions. **All three are also prior art:**

1. *Balanced learned multi-symbol assignment* — the review-prompt's own record: an earlier FCS proposal already split the descriptor into blocks, assigned each identity a balanced multi-symbol Hamming-separated code, trained block targets, and updated assignments from learned affinities; rejected as learned ECOC.
2. *Soft→hard anneal with an explicit bit budget over shared codebooks* — Shu & Nakayama, ICLR 2018 (arXiv:1711.01068): multi-codebook quantization, codes like `(3,2,1,8)`, learned end-to-end by Gumbel-softmax. Chen, Min & Sun, ICML 2018 (arXiv:1806.09464): "K-way D-dimensional discrete encoding scheme to replace one-hot encoding… the final symbol embedding vector is generated by composing the code embedding vectors," via relaxed discrete optimization.
3. *Injectivity + column decorrelation* — these are Dietterich & Bakiri's two original ECOC code-design criteria (row separation, column independence); N-ary alphabets in arXiv:1603.05850 and Deep N-ary ECOC (arXiv:2009.10465); learned output codes in Crammer & Singer (2002).

Also unlisted and closer than several §3 entries: per-class discrete code targets with enforced Hamming separation over a shared alphabet — CSQ hash centers (Yuan et al., CVPR 2020, arXiv:1908.00347); and attribute-code ZSL (DAP), of which FPC is the version with learned rather than annotated attributes and *without* the unseen-class code that made DAP transfer.

## 7. Supervision-object delta — **none**

PA's supervision object: an injective map from training identity to a learned unit target direction. FPC's: the same injective map to a learned unit target direction, reparameterized onto a rank-≤56 product-of-hulls with OT-computed assignment. Label information identical (6.64 bits), injectivity *strengthened* to a hard constraint, encoder task identical, head discarded at test. The change is confined to parameterization (reachable set, rank) and optimization (grouped gradient flow) — the two levels the gate excludes. The one change that would be supervision-object-level — binding codewords to independently *verified* shared evidence, or extending codes to unseen identities — is absent by construction.

## 8. Controls C0–C10 — not separately identifying

- **C4** broken (§4 above): not an algebraic or operational reduction to PA.
- **C1** is honestly rank-matched (56 = `m(C−1)`) but for the wrong reason; the proposal presents this as the discriminating control while §1.4 asserts the discrimination cannot exist.
- **C10 cannot test the capacity claim.** Bit budgets {12, 24, 32, 32} all exceed `log₂K = 6.64`, so every point is injective-capable — no configuration imposes a bottleneck. The sweep also confounds bits with rank (`mC` = {32, 64, 128, 64}); only the (8,16)/(16,4) pair separates them. F4's "interior optimum a few bits above `log₂K`" is untestable within the specified grid.
- **Missing:** rank-`mC`=64 low-rank control with the same structural penalties; random-rank-64-projection head; **matched hyperparameter-search budget**. FPC introduces ~10 new knobs (m, C, τ schedule, `T_h`, ρ, h, four λ's) tuned on the same benchmark against a single-shot C0.
- **Falsifiers underpowered and adaptive.** F1 (Δ ≥ 0.010, paired, 5 seeds, sd ≈ 0.005) is adequately powered. F2/F3/F5 use a 0.005 point-estimate threshold below the ~0.006 CI half-width — an equivalence claim tested by a difference rule with no test; under a true null it fails to fire ~9% of the time and under a true 0.005 effect fires ~50%. These need a pre-registered TOST margin. "Paired same-seed, **Welch** p" is a mismatched test (Welch is unpaired). No held-out-train-class validation split is declared anywhere, so C7/C10 selection runs on the same test R@1 as the headline.

## 9. Protocol — fails the mandated screen

The envelope requires screening **paired corrected In-Shop first**, reporting raw plus independently selected/final, out-of-sample confirmation, then replication. The proposal makes In-Shop *secondary*; uses ResNet-50/200 epochs against a BN-Inception corrected reference, so the paired screen is not constructible; never distinguishes raw from selected/final; and forecasts In-Shop 0.900 against the corrected final mean 0.915201 (sd 0.001549) — **≈10 sample sd below**, i.e. it predicts it will lose the mandated first screen.

Primary-source recipe check (resolving §7.2): Proxy-Anchor uses AdamW, lr `10⁻⁴` for CUB/Cars but **`6·10⁻⁴` for SOP/In-Shop**, `α=32`, `δ=0.1`, 512-D, and "the learning rate for proxies is **scaled up 100 times**"; batch 150 for Cars-196. §1.7's single `η = 1e-4` across all datasets and batch 180 therefore make C0 an unfaithful PA reproduction on SOP/In-Shop. The ×100 multiplier is confirmed, so §7.2's uncertainty on that point resolves in the proposal's favour. Attribute diagnostics restricted to the training split, post hoc, are legal.

## 10. Standing-objective value — **fails independently**

Every forecast is below its matched frontier: CUB 0.712 vs 0.734 (non-overlap bar 0.740); Cars 0.905 vs 0.927 (bar 0.933); SOP 0.806 vs 0.829; In-Shop 0.900 vs corrected local final 0.915201 and raw 0.918062. Self-assigned crossing probability 7–8%. Under "do not credit an expected gain that does not cross a matched frontier," this is dispositive on its own, before any mechanism audit.

---

## Preserved correct subcomponents

1. **§1.5 reproduces Proxy-Anchor exactly** — `1/|P⁺|` on the positive term, `1/|P|` on the negative, α=32, δ=0.1. Verified against primary text. Reusable verbatim.
2. **Head-parameter arithmetic** (10,496 vs 51,200; holds at SOP and In-Shop scale) is correct.
3. **512.4 MB at K=11,318** and the `C^m` / bit-budget arithmetic are correct.
4. **§5's frontier arithmetic and refusal to inflate.** The non-overlap bars (0.740 / 0.933), the stated 0.028 shortfall, the sensitivity check, and the explicit "does not cross the frontier" declaration are correct and are the right disclosure discipline.
5. **§7's ambiguity ledger**, including §7.5 flagging its own untraced nearest neighbour with a pre-committed re-adjudication rule. That rule is what made this resolution decidable; it should be retained as standard practice.
6. **C10's (8,16) vs (16,4) pair** genuinely disambiguates bit budget (32, 32) from rank (128, 64) — worth keeping in any future rank-vs-structure design.
7. **§6's identification of In-Shop/SOP as where forced code sharing most likely hurts** is correct; the margin analysis in gate 3 shows it is the predicted direction, not a tail risk.
8. **§2's "honest boundary" paragraph** correctly self-classifies the claim as implicit-bias/sample-complexity rather than impossibility.
9. **Train-split-only, post-hoc attribute-coherence diagnostics** are legal under the envelope.
10. **C2, C5, C9** are well-posed controls; C9's sampler arm is genuinely needed.

## Uncertainty

- The unbalanced-Sinkhorn scaling exponent (`ρ/(ρ+τ)`) is asserted from standard theory, not verified against primary text in this pass. It affects the *severity* of the gate-4 solver defect, not the verdict.
- I did not verify PFML's or DADA's reported rows from primary text; I took the audited frontiers from the review envelope as given, as instructed.
- Proxy-Anchor's per-dataset LR-decay step/γ and epoch counts remain unverified (as §7.2 also declares); this does not affect the verdict, since gates 1, 2, 6 and 10 fail independently of the baseline recipe.
- The CGCE identification is a mechanism match on the class-decomposition object, not a claim that CGCE performs deep metric learning. Its domain is few-shot 3D reconstruction. The novelty failure is that §7.5's *own* stated test — "if such a primary work exists… the novelty claim must be re-adjudicated" — is triggered, and all three nominated surviving distinctions independently fail.
- Any substantive repair (removing injectivity to create a real bottleneck, raising `mC` toward 512, redesigning C10 below `log₂K`, or moving to the mandated BN-Inception paired In-Shop screen) is a **new proposal**, not a revision of this one.

**Sources:**
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020 (arXiv:2003.13911)](https://arxiv.org/abs/2003.13911)
- [Learning Compositional Shape Priors for Few-Shot 3D Reconstruction (arXiv:2106.06440)](https://arxiv.org/pdf/2106.06440)
- [Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors (arXiv:2004.06302)](https://arxiv.org/pdf/2004.06302)
- [Compressing Word Embeddings via Deep Compositional Code Learning, ICLR 2018 (arXiv:1711.01068)](https://arxiv.org/abs/1711.01068)
- [Learning K-way D-dimensional Discrete Codes for Compact Embedding Representations, ICML 2018 (arXiv:1806.09464)](https://arxiv.org/abs/1806.09464)
- [Deep N-ary Error Correcting Output Codes (arXiv:2009.10465)](https://arxiv.org/abs/2009.10465)
- [N-ary Error Correcting Coding Scheme (arXiv:1603.05850)](https://arxiv.org/abs/1603.05850)
- [Crammer & Singer, On the Learnability and Design of Output Codes for Multiclass Problems, Machine Learning 47:201–233 (2002)](https://link.springer.com/article/10.1023/A:1013637720281)
- [Central Similarity Quantization for Efficient Image and Video Retrieval, CVPR 2020 (arXiv:1908.00347)](https://arxiv.org/abs/1908.00347)
- [Potential Field Based Deep Metric Learning (PFML), arXiv:2405.18560](https://arxiv.org/html/2405.18560v1)
