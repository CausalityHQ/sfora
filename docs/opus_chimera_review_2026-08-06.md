I've read both files and nothing else. Working the adjudication from the frozen text alone.

---

# VERDICT: **DEAD**

**Earliest failed gate: Gate 1** (executable mathematics — specifically its clause *"whether its constraints have the claimed consequences"*).

**Decisive mechanism-level reason:** The two load-bearing devices — the Sinkhorn **row marginal** (§1.4, declared *"the load-bearing piece"*) and the **calibration law** L_cal (§1.5, declared *"the core"*) — do not block degeneracy D1. They **select for it**. A fully class-private atom code satisfies both Sinkhorn marginals *exactly* whenever K ≥ C, and L_cal's exact solution set is *one private atom family per class*. CHIMERA's stable state is a K-column class-private proxy table — i.e. control C3 — which is the object §0 claims does not exist ("Zero learned per-class parameters exist anywhere in the model").

---

## Gate 1 — Executable mathematics

### 1a. The row marginal does not have its claimed consequence (decisive)

§1.4 asserts: *"the row marginal is the sharing constraint … it makes a class-private atom mathematically inadmissible."*

Counterexample, constructed from the proposal's own CUB numbers (K=256, C=100). Partition the 256 atoms into disjoint per-class groups S_c (56 classes get 3 atoms, 44 get 2; 56·3+44·2 = 256). Set

$$\Gamma_{jc} = \tfrac{0.5}{|S_c|}\mathbf 1[j\in S_c] + \tfrac{0.5}{K}$$

- **Row mass** of any row j (each row lies in exactly one S_c): 0.5/|S_c| + C·0.5/K = 0.195 + 0.195 = 0.39 — **identical for every row**. The matrix is already row-balanced; the row constraint is satisfied with zero residual.
- **Column mass** is 1 for every column; one column scaling gives 1/C. Sinkhorn is a **fixed point** here.
- **Entropy floor**: a(x) = 0.5 on the private atom + 0.5 spread over the remainder has H = −0.5 ln 0.5 − 0.5 ln(510) = 3.464 ≈ ln 32 = 3.466. L4's hinge is **inactive**. Blocker (i) defeated.
- Resulting prototype: $p_c \propto 0.5\,\bar u_{S_c} + 0.5\,\bar u$, where $\bar u = \frac1K\sum_j u_j$ is **identical for every class**. All discriminative content sits on atoms used by no other class. This is multi-proxy DML with ≈2.56 proxies/class plus a constant offset.

Both declared blockers of D1 fail simultaneously, and the surviving object is numerically close to control **C3** ("3 proxies × 100 classes"). **F2 fires by construction.**

§1.4's D1 argument attacks only the strict one-atom-per-class matching, and even there its stated cost is wrong twice: (i) Sinkhorn–Knopp diagonal scaling **preserves the zero pattern exactly** — a zero row makes the balancing problem *infeasible*, not *costly* (Sinkhorn & Knopp, *Pacific J. Math.* 21:343–348, 1967: balancing requires total support); (ii) Γ is a `stop_grad` EMA buffer with **no loss term and no gradient path** (§1.4: *"never to Γ"*), so a "KL cost → ∞" is a cost in a variational problem that is never added to $\mathcal L$ and never differentiated. Nothing learnable pays it.

Worse, on the datasets where the mechanism is forecast to matter (CUB K/C = 2.56, Cars 2.61), K ≥ C makes privacy feasible; on SOP (K/C = 0.09) it is infeasible — but SOP is exactly where the proposal predicts (F5) the mechanism does nothing. **The sharing constraint binds only where the method is predicted not to help.**

### 1b. The calibration law penalizes sharing (decisive, independent)

Fix c, c′ and an atom j with $\Gamma_{jc}=\Gamma_{jc'}=g>0$. Compare mask M (with $M_j=1$) against M′ = M with bit j flipped.

- Coordinate j of the unnormalized chimera: under M it is $1\cdot g + 0 = g$; under M′ it is $0\cdot g + 1\cdot g = g$. All other coordinates are untouched. Therefore $v(M)=v(M')$, $Z(M)=Z(M')$, and **$q(M)=q(M')$ exactly**.
- $\beta(M)-\beta(M') = g/Z > 0$.

So L_cal demands the *same* $\langle \bar f_c-\bar f_{c'},q\rangle$ equal *two different* targets $\kappa(2\beta-1)$. Minimizing $(t-A)^2+(t-B)^2$ over the free variable $t$ leaves

$$\min \mathcal L_{\rm cal} = \tfrac12(A-B)^2 = \frac{2\kappa^2 g^2}{Z^2} > 0$$

**irreducible by any choice of embeddings**, because q is identical in both terms. The only way to drive it to zero is $g \to 0$ — make the atom unshared.

Generally: exact satisfaction for all M requires, per atom, $s_j \equiv \langle \bar f_c - \bar f_{c'}, u_j\rangle = +\kappa\rho$ for every atom in c's support and $-\kappa\rho$ for every atom in c′'s support ($\rho = \lVert U\gamma^\times\rVert > 0$). Any atom in *both* supports demands $+\kappa\rho = -\kappa\rho$, impossible for fixed κ = 0.6. Taking three classes sharing an atom gives the same contradiction. The system is solvable **only** when supports are disjoint, with a private atom family per class giving a uniform off-response to all other classes — the prompt's *lookup-table* degeneracy, realized inside U.

This is the same algebra the proposal uses **as a feature** in D3 (duplicate atoms: same q, different β). D3 and the sharing claim are one mechanism with opposite signs; the proposal cannot have both. §1.5's "core" term is an anti-sharing force.

Corollary on diversity: the claimed $2^K$ virtual identities is really $2^{|A\,\triangle\,B|}$ — flipping a shared-mass bit leaves q unchanged. Chimeric diversity is **maximized by the private degeneracy** and collapses in the shared regime the method wants. Third instance of the same inversion.

### 1c. The Sinkhorn operation is not uniquely executable

"$\Gamma\leftarrow\mathrm{Sinkhorn}_\epsilon(\Gamma)$, 3 iterations, $\epsilon=\tau$" admits at least three readings with different outputs:

| reading | result |
|---|---|
| RAS / matrix balancing of Γ itself | **ε is vacuous**; "ε = τ" carries no meaning, contradicting §1.7's insistence that these scales are operational |
| SwAV-style kernel $\exp(\Gamma/\epsilon)$ then balance | with τ=0.05 and peaked columns, dynamic range up to $e^{20}\!\approx\!4.8\times10^8$; 3 iterations give near-one-hot columns |
| OT-cost reading $\exp(-\Gamma/\epsilon)$ | **semantics inverted** — the atoms a class uses most receive the least mass |

Entropic OT (Cuturi, NeurIPS 2013) defines ε against a *cost* matrix; Γ is already a coupling-like nonnegative table. My analysis above uses the most charitable reading (RAS), under which the method still fails — but the object is under-specified as frozen.

### 1d. Optimizer and initialization inconsistencies

- **U's optimizer is self-contradictory.** §1.2 says "yes, **SGD**"; §1.6 puts everything under **AdamW** and assigns U an AdamW lr of $10^{-2}$. The same §1.2 cell justifies disabling weight decay because *"AdamW decay would fight the unit-norm projection"* — presupposing AdamW in the row that declares SGD.
- **U has no stated initialization anywhere.** §1.6 initializes Γ by "one forward pass with the frozen ImageNet encoder" — but $\tilde\Gamma$ is a mean of $a(x)=\mathrm{softmax}(U^\top f/\tau)$, so Γ's initialization *requires* U. The declared bootstrap is not executable.
- **EMA mass mismatch.** After Sinkhorn a column has mass 1/C; $\tilde\Gamma^{\mathcal B}_{:,c}$ is a mean of simplex points, mass 1. The blend in §1.4 mixes mass 1/C with mass 1, then the next column normalization rescales to 1/C. Effective retained weight on history is $\frac{m_c/C}{m_c/C + (1-m_c)}$. On SOP with $m_c=0.5$, C=11,318 that is **1 : 11,318** — the declared count-debiased momentum is annihilated, and Γ becomes a single-batch (4-image) estimate. Risk 2 names staleness but not this; the mitigation it relies on is the one destroyed.
- **Update order.** The EMA touches only the ~45 batch classes; Sinkhorn rescales all K rows and all C columns every step (~66k steps). A rarely-refreshed SOP column is multiplicatively re-scaled tens of thousands of times between refreshes by factors driven entirely by other classes.
- **Projected gradient + AdamW.** The radial gradient component is discarded by the sphere projection but still accumulates in AdamW's second moment, damping the tangential step. Not addressed.

---

## Gate 2 — The finite-class-rank causal argument: **refuted as stated**

| rank object | status |
|---|---|
| class-mean rank | ≤ C−1 — true but trivial |
| class-prototype rank | ≤ min(C, 512) for a proxy table; **≤ min(C, K, 512) for CHIMERA too**, since only C columns of Γ are ever used |
| full training-feature rank | **not** bounded by C−1; the contraction step is asserted, not derived |
| generated-prototype rank | rank min(K,512) is the *attainable* set, not the *used* set — a K-proxy table has identical attainable rank |
| test-feature rank | outputs of a nonlinear map; nothing constrains them to the training-mean span unless the features are themselves rank-deficient |

The load-bearing step — "neural-collapse terminal dynamics … contracting the *entire* feature cloud toward that affine subspace" — is an asymptotic terminal-phase phenomenon documented for cross-entropy/MSE classifiers trained far past interpolation (Papyan, Han & Donoho, *PNAS* 117(40), 2020). It is not established for 200-epoch proxy-based DML under heavy augmentation, and no measurement is offered. Note also that Galanti, György & Hutter (ICLR 2022, arXiv:2112.15121) — cited by the proposal — argue neural collapse **transfers to new classes**, i.e. is associated with *good* few-shot transfer. The proposal invokes that literature to support a claim its headline result cuts against.

**Critically, the rank argument does not separate CHIMERA from C3.** A K-proxy class-private table has attainable prototype rank min(K,512) independent of C, exactly as claimed for CHIMERA. The proposal's own causal story therefore predicts C3 ≈ CHIMERA, which is F2's rejection condition.

**The 15-vs-2 proxy pattern does not identify the cause.** At least three alternatives are at least as good:
1. **Images per class** — ~60 (CUB) / ~80 (Cars) vs ~5 (SOP). You cannot estimate 15 centers from 5 images; 2 is forced.
2. **Resource** — 11,318 × 15 × 512 ≈ 87M proxy parameters on SOP, 3.4× the backbone.
3. **Intra-class multimodality** — the standard justification for multi-center losses (Qian et al., *SoftTriple*, ICCV 2019, motivates multiple centers by intra-class modes, not rank). SOP classes are near-duplicate product photos; CUB/Cars classes span pose, sex, season, colour, viewpoint.

The pattern is also quantitatively off-target for the rank story: rank saturates at the embedding dimension 512, so CUB needs ⌈512/99⌉ ≈ **6** proxies, not 15; and SOP needs **1**, not 2, since C = 11,318 already exceeds 512 at one proxy per class. The stated "single strongest empirical support" mispredicts both observed numbers.

---

## Gate 3 — Constructive degeneracy attack

| degeneracy | proposal | verdict |
|---|---|---|
| **block-private code** (K/C atoms per class) | not considered | **admissible**: both marginals exact, entropy floor exact, L_cal minimized. Kills D1. |
| **common-plus-private** (private core + uniform tail) | not considered | **admissible**: the ~50% tail is identical across classes, contributes zero discrimination, and supplies the entropy exactly (§1a). Kills blocker (i). |
| **lookup table in U** | claimed impossible ("zero learned per-class parameters") | **this is L_cal's exact solution set** (§1b). $u_j$ for $j\in S_c$ is a learned parameter used by exactly one class — a per-class parameter renamed. |
| **encoder/proxy co-adaptation** | not considered | $\partial\mathcal L_{\rm PA}/\partial u_j$ is a Γ-weighted sum; when Γ is near-private, it is dominated by the one owning class, so $u_j$ trains into that class's proxy. |
| constant codes (D2) | blocked | **sound** — see preserved items |
| duplicate atoms (D3) | blocked | algebra **correct**, but identical to the anti-sharing mechanism |
| nuisance codes (D5) | "blocked by two-view term" | **partial only**: blocks *augmentation-unstable* codes. A view-stable context cue (CUB habitat, Cars background) passes cleanly, and the term covers only 25% of the batch. |
| trivial calibration (D4) | κ fixed + L_sep | κ-fixing is sound; **L_sep is inert** — in the private solution q lies in span{$u_{S_c},u_{S_{c'}}$} and $\bar f_{c''}$ is uniformly off by construction, so the hinge never activates. |

Aggregate balance and entropy are offered as sharing throughout, with no proof — the specific failure the review brief names. §1a is the counterexample.

---

## Gate 4 — New supervision in the chimeric term: **none**

Γ is `stop_grad`, so $\gamma^\times$ and β are **constants**; masks M are random (entropy, not information); q depends only on U. Every quantity is a deterministic function of already-supervised objects. **No chimeric vector has an observed matching input** — L3 introduces zero bits of label information and is a geometric regularizer on U and batch class means.

The annotation *"chimera is a real identity"* on L_sep overstates what L_sep says: it only requires two parents to beat a third class on $\langle\cdot,q\rangle$, which holds by construction.

Versus the nearest work: Proxy Synthesis (AAAI 2021) synthesizes a virtual proxy **and** synthetic embeddings by interpolating real ones, so its virtual class has members. CHIMERA's has none, which makes it *less*, not more, than convex synthesis on this axis. Coordinatewise binary mixing is not novel as an operation — it is uniform crossover (Syswerda 1989), and in representation learning it is the CutMix / binary-mask Manifold-Mixup family. Note also that **control C6 removes the pathology**: under a scalar λ, both columns have equal mass 1/C so β = λ, q varies smoothly with λ, and the same-q/different-β contradiction disappears. The control is better posed than the treatment; F6 is likely to fire for the wrong reason.

---

## Gate 5 — Prior art (mechanism, not conjunction)

Each mechanism has primary prior art; the conjunction's asserted new consequences are the ones refuted above.

- **Soft assignment to a shared learned codebook**, $a(x)=\mathrm{softmax}(U^\top f/\tau)$ on an L2-normalized descriptor, aggregated for retrieval — this is NetVLAD's assignment layer (Arandjelović et al., CVPR 2016, arXiv:1511.07247), and behind it the visual-word/VLAD lineage.
- **Prototype = dictionary × supervised code** — label-consistent discriminative dictionary learning, LC-KSVD (Jiang, Lin & Davis, CVPR 2011). §3 dismisses NMF/sparse coding as unsupervised factorization of the *data* matrix, which does not reach the supervised-dictionary branch.
- **Class as a vector over shared latent dimensions, unseen classes as new codes** — label-embedding classifiers, ALE (Akata et al., CVPR 2013), and the *latent*-attribute branch that requires no annotations or text. §3's dismissal of compositional ZSL ("requires state/object annotations and usually a text encoder") does not cover it.
- **Classes as binary codes over shared learned dichotomies, new classes by new codewords, combined by bit operations** — ECOC (Dietterich & Bakiri, *JAIR* 2:263–286, 1995); attribute-code transfer to unseen classes (Lampert et al., CVPR 2009).
- **Optimal-transport-balanced assignment to a shared prototype set** — SeLa (Asano, Rupprecht & Vedaldi, ICLR 2020, arXiv:1911.05371) and SwAV (Caron et al., NeurIPS 2020, arXiv:2006.09882). Moving the balanced axis from sample×prototype to class×atom is a marginal change, not a mechanism change.

**Judgment:** the exact four-way conjunction is plausibly unclaimed, but per the brief's instruction that is not the test. Every constituent mechanism is prior art, and the two properties that would have made the conjunction novel — sharing enforced by the row marginal, readability enforced by the calibration law — are refuted at Gate 1.

---

## Gate 6 — Causal provenance: **no measured premise**

**Not one repository measurement supports the forecast.** The chain is: a published proxy-count table (15/15/2, itself taken from the brief) → a rank inequality that holds trivially for any C classes → an unmeasured neural-collapse contraction assumption → +0.014 R@1 on CUB. Formula, published hyperparameter choice, cross-dataset difference, plausible analogy — none is a measurement.

**The frozen forecast contradicts the frozen mechanism prediction.** §5 requires gain ordered **CUB ≈ Cars ≫ SOP** because the bottleneck is set by C. But C(CUB) = 100 and C(Cars) = 98 are nearly identical, while the forecast gains are **+0.014 vs +0.006 — a 2.3× gap**, and Cars (+0.006) vs SOP (+0.004) shows no "≫" at all. The stated mechanism cannot produce the stated numbers. The forecast ordering tracks remaining headroom (0.266 / 0.073 / 0.171), not C — and the proposal does not offer that explanation.

---

## Gate 7 — Controls: executability and identifiability

- **C4 (Sinkhorn off) is not matched.** Without Sinkhorn, columns of Γ sum to 1, not 1/C — so L2's target $C\Gamma_{:,c}$ has mass **C**, not 1. The control silently changes the code-loss scale by 100× (CUB) to 11,318× (SOP). The control meant to isolate "the sharing constraint" confounds it with a 2–4-order-of-magnitude loss reweighting.
- **C8 is infeasible at K=1024.** "U frozen at a random orthonormal frame" — no orthonormal set of 1024 vectors exists in $\mathbb R^{512}$. The control cannot run on SOP or In-Shop.
- **C9 is not identifiable.** "ImageNet fc rows projected to 512-D" — projected by what map (W is randomly initialized at that moment), and which 256 of 1000 rows?
- **C6 is not executable as written** — `a` in Beta(a,a) is unspecified; and see Gate 4 on its being better posed than the treatment.
- **C3 anchor matching is loose** (3×100 = 300 vs K = 256, a 17% mismatch) and §7.4 concedes that whether PFML's proxies are freely learned changes what F2 tests.
- **F4 is a test-data diagnostic used as a rejection threshold.** "Participation ratio of the *test*-class-mean covariance" requires embedding test images; listing it among "pre-registered falsification thresholds" that trigger rejection makes test data a **selection** input, which the envelope forbids. F4 also has an unchecked ceiling: on CUB the PR is bounded by C_test − 1 = 99, so "≥1.5× over C1" is unattainable if C1's PR already exceeds 66.
- **Sampler feasibility fails on SOP.** Batch 180 with m=4 requires 45 classes × 4 images. SOP's official training split is 11,318 classes over 59,551 images (mean 5.26/class) with a large mass of classes holding 2–5 images; m=4 without replacement is infeasible for a substantial fraction, and with replacement the duplicated images corrupt $\bar f_c$ — the very quantity L_cal and L_sep constrain. The proposal's own Risk 2 ("~5 images per SOP class") concedes the premise without addressing the sampler. In-Shop (3,997 classes / 25,882 images, ~6.5/class) is tighter but has the same low tail.
- **Envelope precondition unmet.** The brief requires a live candidate to *first* establish a same-seed corrected In-Shop paired control against the local Proxy Anchor seed-0 (raw 0.9163 / final 0.9137). The proposal has **no In-Shop control**; C1–C9 and F1–F6 are CUB-centric, and In-Shop appears only as a reported-not-claimed row. This is an independent protocol failure.
- No control changes capacity, data exposure, identity population, or test-data use except F4 (above) — otherwise clean.

---

## Gate 8 — Arithmetic

**Verified correct:**
- $\sigma_\Delta=\sqrt{(s_1^2+s_2^2)/5}$: CUB 2.608e-3 → 0.0026 ✓, z = 5.37 → 5.4 ✓; Cars 2.236e-3 → 0.0022 ✓, z = 2.68 → 2.7 ✓; SOP 1.612e-3 → 0.0016 ✓, z = 2.48 → 2.5 ✓.
- U = 512×256 = 131,072 = 0.512% of 25.6M ✓. $U^\top f$ = 1.31e5 MACs vs 4.1 GFLOP = 0.003% ✓ (labeled "FLOPs" where it is MACs — 2× convention slip, immaterial).
- Full-table balancing at SOP: 1024×11,318 = 11.6M entries = 46 MB fp32; 3 iterations ≈ 70M element ops ≈ tens of µs on GPU. **"Sub-millisecond" holds.** ✓
- 25% second view → 1.25× epoch; 250 vs 200 epochs = 1.25× ✓ internally consistent.
- $e^{\alpha\delta}=e^{3.2}=24.53$ → "24.5" ✓.

**Defective:**
- **The four "80% intervals" use four different multipliers of σ**: CUB ±0.0125 = **2.50σ**; Cars −0.009/+0.008 ≈ **2.1σ** (and asymmetric); SOP ±0.007 = **2.33σ**; In-Shop ±0.005 = **1.25σ**. Only In-Shop is ≈ the 80% single-draw interval (1.2816σ). No single definition — single-draw or 5-seed-mean (σ/√5 = 1.2816·0.002236 = ±0.0029 on CUB) — makes all four correct.
- **F1 is incoherent with its own interval.** With forecast mean 0.748 and σ_mean = 0.00224, P(5-seed mean < 0.740) ≈ 0.02%; and 0.735–0.740 lies simultaneously *inside* the stated 80% interval and *inside* F1's rejection region.
- **Lane separation is handled correctly.** The forecasts (0.748 CUB, 0.933 Cars) sit below every higher-capacity observation (VAPNet 0.762/0.948, AdvRF 0.766/0.949, CRT In-Shop 0.9448), and §1 restricts every claim to the matched 512-D lane without a general SOTA claim. This is right, and I preserve it.
- **z-scores are computed against a reference §7.1 disclaims.** §7.1 states PFML's numbers are "targets, not a baseline I may build on" and that C2 is "a hard precondition on any crossing claim" — yet §5's entire falsification arithmetic is computed against exactly those numbers.

---

## Preserved as correct (independent of the verdict)

1. **Lane discipline.** §1's refusal to claim a Lane-B crossing on an uncertainty-free row, and §5/§7.3's refusal to claim In-Shop, are correct and should survive into any successor.
2. **All σ_Δ and z arithmetic** (verified above).
3. **All cost accounting** — parameter fraction, FLOP fraction, full-table Sinkhorn timing and memory at K×C = 1024×11,318, and the 1.25×/250-epoch matched-wall-clock construction. C7 (view-budget control) is well designed.
4. **D2** (constant codes blocked by PA) is sound, though the operative force is the attraction term, not the stated repulsion bound.
5. **D3's algebra** is correct — same q under two masks with different β yields an irreducible residual. (It is also what refutes the sharing claim.)
6. **§1.7's observation that (τ, k\*) is one knob**, not two: at $\lVert f\rVert=\lVert u_j\rVert=1$, $H(\mathrm{softmax}(U^\top f/\tau))$ is monotone in τ, so they must be ablated jointly. Correct.
7. **§2.1's train/test operator-mismatch framing** — proxy methods realize training identity as an index lookup and test identity as an encoder output — is a genuinely sharp observation about the proxy family, independent of whether CHIMERA fixes it. Worth keeping.
8. **§7's disclosure discipline** — six named source ambiguities and C2 as a hard precondition — is the right standard.

---

## Unresolved uncertainty

- I could not verify Proxy Anchor's exact hyperparameters (α=32, δ=0.1, proxy-lr ×100, batch 180) against the primary source in this cold pass; §7.2 flags the same gap. My conclusions do not depend on them.
- The PFML 15/15/2 proxy-count pattern is taken from the frozen brief; I could not check it against CVPR 2025. My Gate-2 refutation is robust to the exact counts (it turns on rank saturating at 512), but the "does not identify the cause" argument assumes the pattern is as stated.
- SOP/In-Shop class-size *distributions* (as opposed to totals) are stated from recollection; the sampler-feasibility objection depends on SOP's low tail, which the proposal's own Risk 2 corroborates.
- Whether F4 breaches the envelope depends on whether a test-side diagnostic that can trigger rejection counts as selection. I read it as selection; a narrower reading (post-hoc reporting only, never gating) would clear it.
- Under Sinkhorn reading (b)/(c) rather than RAS, my Gate-1a counterexample would need re-deriving. I used the most charitable reading; the operation remains under-specified as frozen.

---

## Why DEAD rather than UNRESOLVED

Every decisive finding is derivable from the frozen text by algebra alone — no measurement, no GPU, no external recipe. The failures land on the two components the proposal itself designates load-bearing, and they are not gaps but **inversions**: the row marginal and the calibration law each select for the degeneracy they are declared to block, and the resulting equilibrium is numerically ≈ control C3. F2, F3, and F6 would fire. Repairing this requires replacing both the balancing constraint and the calibration law — a new proposal, which per the review brief must not be used to upgrade this frozen one. I have not repaired it and offer no successor here.

**Primary sources cited:** Sinkhorn & Knopp, *Pacific J. Math.* 21:343–348 (1967) · Cuturi, NeurIPS 2013 (arXiv:1306.0895) · Papyan, Han & Donoho, *PNAS* 117(40):24652 (2020) · Galanti, György & Hutter, ICLR 2022 (arXiv:2112.15121) · Caron et al., NeurIPS 2020 (arXiv:2006.09882) · Asano, Rupprecht & Vedaldi, ICLR 2020 (arXiv:1911.05371) · Arandjelović et al., CVPR 2016 (arXiv:1511.07247) · Jiang, Lin & Davis, CVPR 2011 (LC-KSVD) · Akata et al., CVPR 2013 (ALE) · Dietterich & Bakiri, *JAIR* 2:263–286 (1995) · Lampert, Nickisch & Harmeling, CVPR 2009 · Qian et al., ICCV 2019 (arXiv:1909.05235) · Syswerda, ICGA 1989 (uniform crossover).
