Hash verified. Four load-bearing prior-art attributions confirmed against primary sources. Adjudication follows.

---

# VERDICT: **DEAD**

**Earliest failed gate: Gate 1** (diagnosed defect not established, and the enabling structure measured absent).

**Decisive mechanism-level reason:** Proposition 1 certifies only the **rank** of `Cov(zhat)` — a scale-invariant, discontinuous functional — while the deployed object is a cosine-NN descriptor whose usefulness is entirely amplitude/angular. The frozen objective contains **no lower bound on any eigenvalue of `Cov(zhat)` and no bound on decoder gain `||V||`**, so `zhat_delta = normalize(mu_y + delta*q)` with `V ~ 1/delta` satisfies the certificate *exactly* for every `delta > 0` while `L_inv = O(delta^2) → 0`. Shrinking `delta` is a **strict descent direction** on the frozen loss. The certificate excludes a measure-zero point and leaves every neighborhood of class lookup free — it does not make collapse infeasible, it makes collapse *asymptotically approachable at zero cost*.

---

## 1. Gate 1: external literature + a future diagnostic do not satisfy it — **NO**

The premise ("supervised DML attains augmentation invariance by class lookup") is unmeasured in the admissible repository and is **not** transferred by Xue et al. Their result concerns SupCon-style contrastive objectives; PFML is proxy/potential-field. Decisively, PFML's own attractive potential is **constant inside `delta`** (`psi_att = -delta^{-alpha}` for `||r-z_i|| < delta`), so its gradient vanishes there: exact class collapse is *not* a minimizer of the chosen base loss. The proposal reproduces this formula on line 22 and then, twelve pages later, asserts collapse is "globally optimal for `L_base + L_inv`" (line 155). The base loss it selected already contradicts the defect it exists to cure.

D1 (`R > 3`) is a future measurement, and the repository shows no train/test variance ratio above 3. A diagnostic that has not run cannot license a method built on its predicted outcome.

## 2. Locked prospective evidence **contradicts** the register premise

Candidate 225 tested precisely NRQ's load-bearing assumption — that leading within-class subspaces learned on source identities transfer to disjoint identities. Corrected `rho_32` = 0.9312 / 0.9287 / 0.9345, three seeds, **all below 1.15 and all below one**: the source-derived subspace captures *less* target within-class variance than control. There is no transferable within-class nuisance subspace.

Corrected ARCG found augmentation-response compatibility **image-specific on In-Shop**, retaining only 0.3631–0.3640 of same-class pairs. A 64-D linear register asserted to hold shared class-exogenous nuisance faces a fork with no live branch: either it is image-specific (high capacity → instance memorization, the content-dumping degeneracy) or it captures only the shared ~36% (quarantine reaches almost nothing). Note the tightness: on In-Shop — the **mandated first screen** — the rank floor is admitted vacuous, so the register is NRQ's *only* live leg there, and it is the leg measured absent **on that same dataset**.

## 3. Proposition 1 is valid; its corollary is true and inert

The tail bound is **correct**. `V*u_tilde + c` has centered support in a subspace of dimension ≤ `rho`; projection plus Ky Fan gives residual ≥ `sum_{k>rho} lambda_k`. Rank arithmetic (`rho ≤ rank Cov(zhat) + r`; collapse ⇒ rank ≤ `C-1`) is also correct. **Preserve both.**

What breaks is everything downstream:

- **The sphere caps total variance.** `zhat ∈ S^511` ⇒ `tr Cov(zhat) = 1 - ||E zhat||^2 ≤ 1`. The corollary demands ≥ `k*(eps) - r` nonzero eigenvalues sharing a trace of at most 1 — mean eigenvalue ≤ 1/100 at `k*=164`. The certificate therefore *forces* retained directions to be low-amplitude and then declines to bound them from below. It is satisfied identically by eigenvalues at 10^-12.
- **`r = 64` is not derived.** The window `[C-1+r+1, 512+r]` has width `512-C+1 = 413` for **every** `r`; `r` only translates it. With `eps*` defined post hoc from the measured spectrum so that `k* = 400`, the boxed inequality holds for **any `r < 301`**. The constraint's threshold is calibrated from the same object it constrains. C10 then sweeps `r ∈ {0,16,64,256}` — a sweep of a quantity declared underived-by-construction.
- **Normalization breaks the certificate's direction.** `E||h_0 - h_0_bar||^2 = T + ||E h_0 - h_0_bar||^2 ≥ T` under EMA lag, so `L_suf ≤ eps` does **not** imply numerator ≤ `eps*T`. The measured quantity is a lower bound on the certified one.

## 4. The escape is real, and nothing in the frozen text blocks it

Take `q ⊥ span{mu_y}` (a 412-dim subspace of R^512 is available at `C=100`), `A` with `A*mu_y = 0` for all classes, `V = V_0 + A/delta`. Then to leading order `zhat_delta ≈ mu_y + delta*q`, `V*zhat_delta = V_0*mu_y + A*q + O(delta)`: teacher residuals recovered at **O(1) amplitude, uniformly in `delta`**. `rank Cov(zhat_delta)` is high for every `delta > 0`. `L_inv = 1 - cos = O(delta^2)`. `L_base` is flat inside PFML's `delta`-core. So the *only* `delta`-sensitive term strictly decreases as `delta → 0`.

The escape demands **strictly less capacity** than NRQ's intended solution — it *is* the intended solution with amplitude taken to zero.

**Frozen terms that could supply a floor: none.** The non-affine BN gives an amplitude floor to `n` only; `zhat` has no per-direction analogue, only the global `||zhat|| = 1` that `mu_y` already satisfies. No weight decay on `V` is specified, and the base recipe's weight decay is listed as **undisclosed** (§0, ambiguity 3). Line 41 — "I define every NRQ term on scale-free quantities" — states the fatal property as a design virtue. C9 senses a scale problem and files it as a nuisance control rather than as the hole in the certificate. I decline to import a `||V||` penalty; that is a new proposal.

## 5. Rank has no semantic force

Satisfying the certificate requires only high-rank variation, and the cheapest sources are all useless or harmful for unseen identities. The most damaging: `L_inv` compares **two augmentations of the same source image**, so background, pose, articulation, illumination, and instance texture are *view-shared*, hence invisible to `L_inv`, *and* directly reduce `L_suf` (they are what the teacher encodes), *and* fill rank. The certificate thus actively rewards retaining image-specific nuisance in the deployed 512-D descriptor. Instance-private memorization codes are full-rank with zero transfer.

**C12 cannot do its job.** A random-init teacher still poses a generic feature-reconstruction auxiliary — an established regularizer against over-compression. Both C12 arms share unbounded decoder gain and an amplitude-free certificate, so C12 separates "ImageNet semantics" from "auxiliary reconstruction," never "rank floor" from "auxiliary reconstruction." No arm in C0–C12 turns the certificate on or off at fixed auxiliary task.

## 6. Routing is a soft trade, and deletion is not a quotient

"The register is the only remaining route" holds only at `L_inv = 0` exactly. `L_inv` is a finite-weight penalty in expectation against an unbounded decoder gain; the equilibrium therefore places nonzero view-dependence in `zhat`, amplified by `V` — the item-4 escape is the optimum, not an adversarial corner. Both heads read the same trunk `h`, so `W_z` can access anything `W_n` can.

`L_reg` penalizes across-class variance of **per-coordinate batch class means** — first moments only. Identity survives freely in higher moments, cross-coordinate structure, and nonlinear codes with matched coordinate means. With `m = 4`, `E[Var_c(n_bar)] ≈ Var_between + sigma^2_within/4`, and with BN fixing `between + within = 1` this reduces to minimizing `1/4 + (3/4)*Var_between` — a weak, noise-dominated shrinkage, not a class-exogeneity certificate.

**Deletion is not a quotient.** There is no group action and no kernel. `z = W_z h` is computed **independently of `n`**; the deployed descriptor never contained the register. `P: R^576 → R^512` therefore removes nothing at inference — you simply never compute `n`. Proposition 2's "exactly uniform over all inputs, no support assumption" is true and vacuous: it certifies uniform removal of something absent from the pipeline. The non-vacuous claim about `zhat` is exactly the one §2.3 correctly concedes it cannot make.

## 7. Prior art: frozen-feature target + an ineffective rank wrapper is **not substantive**

Judged by supervision object and action, the construction is occupied:

- **Domain Separation Networks** (Bousmalis et al., NeurIPS 2016): shared encoder + private encoder, **shared decoder reconstructs from the concatenation**, orthogonality difference loss, task head on the shared branch. Shared/private split + joint reconstruction + private branch off the task path. **Not cited anywhere in the frozen proposal.**
- **Unsupervised Adversarial Invariance** (Jaiswal, Wu, Abd-Almageed, Natarajan, NeurIPS 2018): split representation `e1`/`e2`, reconstruction pulls nuisance into `e2`, adversarial disentanglers, **prediction from `e1` alone**. **Not cited.**
- **MIC** (Roth, Brattoli, Ommer, ICCV 2019): a **separate encoder for characteristics shared across classes**, trained jointly with the class encoder by reducing mutual information, explicitly to "explain away intra-class variations"; the class embedding is what deploys. This is the DML-native instance of NRQ's exact mechanism. **Not cited** — the proposal cites DiVA and Sharing Matters from the same group but omits the direct ancestor.
- **VICReg** (Bardes, Ponce, LeCun, ICLR 2022): the anti-collapse term is a **hinge on per-dimension standard deviation** toward target γ=1 — an explicit **amplitude floor**. The entire SSL anti-collapse literature converged on amplitude/whitening floors rather than rank constraints. NRQ cites none of it, and VICReg's variance term is precisely the missing object identified in §4 above.

L2-SP/DELTA/LP-FT/InFeR are cited and correctly distinguished (recoverability from a 576-D head is genuinely weaker than weight/feature anchoring). But with the infeasibility construction dead and the rank floor inert, the residual is: a frozen-pretrained-feature reconstruction target (DELTA/InFeR family) inside a shared/private split with the private branch discarded (DSN/UAI/MIC). The §3 "honest residual" rests novelty on three legs — (i) infeasibility, (ii) rank floor, (iii) deployed quotient. (i) and (ii) fail on the mathematics; (iii) is DSN/UAI/MIC. The reviewer who reads this as "AugSelf with a discard" is being *generous*; the closer read is "MIC with a frozen-feature target."

## 8. Degeneracies and C1–C12

- **`L_reg` does not identify nuisance** — it is purely negative (first-moment linear non-separability) and never asserts the register's content *is* nuisance.
- **F5 is mis-set by roughly an order of magnitude.** 40% top-1 on CUB's 100 classes is 40× chance and 2.7× the proposal's own predicted ≤15%. A register at 38% linear identity is not class-exogenous, and deleting it would delete identity. As written F5 cannot fire on a real leak.
- **D3 does not prove routing.** A *linear* `theta`-probe at `R^2 < 0.05` is compatible with nonlinear `theta` encoding, and — decisively — `theta` is a 10-D augmentation parameter that omits the nuisance that governs retrieval (pose, articulation, background), all of which is view-shared and therefore invisible to both `L_inv` and D3. CKA "strictly between low and 1" has no endpoints and no null; almost any value passes. Neither is a test.
- **C7/C8 do not isolate the conjunction.** The claim is {rank floor} ∧ {register} ∧ {deletion}. C3/C4 give floor-without-register; C5 gives register-kept (off-lane). There is **no arm with the register and no floor**, and none varying `eps`. Worse, C4 (`r=0`) has the floor binding *more* tightly (`k*` vs `k*-64`), yet the predicted ordering is NRQ > C4 — the arm with the stronger certificate is predicted to lose. The ordering attributes the gain to quarantine while the headline attributes it to the floor.
- **C11 is not matched compute.** Three arms, three different confounds: 540 epochs stretches a decayed schedule 2.7× into the regime where CUB/Cars DML overfits (biasing *toward* NRQ); 2× batch with `m=4` doubles classes per batch and alters proxy dynamics; "2-view-as-data" is C1 again. NRQ at 200 epochs consumes 400 augmented exposures/image vs. base 200. Cost accounting (2.6–2.8× step) is plausible, but memory 1.3–1.4× is optimistic if both student views are backpropagated in one graph (~2× activations); gradient accumulation over views is unspecified.

## 9. The mandated In-Shop screen is non-diagnostic by construction

The protocol requires corrected paired In-Shop **first**. The proposal supplies **no In-Shop forecast** and declares the rank floor vacuous there (`C-1+r = 4060 >> 2048 ≥ k*`). Consequently: a **positive** In-Shop arm cannot support the rank mechanism — that mechanism is off by the proposal's own arithmetic — and would instead indicate the occupied auxiliary/quarantine alternative; a **null** arm is equally consistent with the proposal. Both outcomes are uninformative for the stated mechanism, and there is no predicted number to screen against. A candidate whose only mandated entry gate cannot discriminate its own mechanism in either direction has not reached the screen.

## 10. Frontier arithmetic and protocol

- **No forecast is quantitatively derived from a measured premise.** +0.012 / +0.006 / +0.004 have no derivation chain; §5's rationale is qualitative and contingent on `k*`, which is unmeasured. Crossing probabilities are stated as subjective.
- **Paired/unpaired confusion.** The design is paired (5 shared seeds) but the quoted SE is the unpaired `0.003*sqrt(2/5) = 0.0019`, applied to all rows including SOP (`sigma = 0.002` ⇒ 0.00126; `0.004/0.00126 = 3.2`, table says 3.0).
- **The SD/SE ambiguity is flagged and then not propagated.** If ±0.003 is an SE over 5 runs, `sigma = 0.0067` and the whole significance column divides by √5: CUB 6.3 → 2.8, Cars 3.2 → 1.4. A factor-2.2 swing on an ambiguity the document itself raised.
- **The ±0.005 reproduction gate is unachievable as stated** with augmentation pipeline, batch size, sampler, weight decay, LR schedule, warm-up, and BN freezing all undisclosed — and the frontier arithmetic is additionally void if PFML's `(alpha,delta)` CV touched test identities, which the proposal cannot resolve.
- **Specification errors, noted without inflation:** the LOOC citation links arXiv 2412.18955, which is not LOOC (Xiao et al., arXiv:2008.05659); DSN, UAI, MIC, VICReg absent from the novelty search. Dataset counts (CUB 100, Cars 98, SOP 11318, In-Shop 3997) and `576 = 512+64` are all correct.

---

## Preserved correct subcomponents (independent of the verdict)

1. **Proposition 1** — a valid affine low-rank regression tail bound with a correct proof. Reusable as a lemma, with the standing caveat that it bounds *regression residual*, never representation quality.
2. Corollary rank arithmetic: `rho ≤ rank Cov(zhat) + r`, collapse ⇒ `rank ≤ C-1`.
3. **Refusal to inherit 0.734/0.927/0.829**, the ±0.005 reproduction gate, and the paired-delta fallback — correct discipline, rare.
4. **Selection protocol**: all λ's fixed once on CUB's last 20 *training* classes, transferred unchanged. Legal and out-of-sample-clean.
5. Explicit disclosure of the five PFML recipe ambiguities and of the `(alpha,delta)` contamination risk.
6. §2.3's concession that no uniform bound exists absent a ResNet-50 Lipschitz constant — correct, and correctly labeled the largest gap.
7. Correct arithmetic that the rank floor is vacuous on In-Shop and SOP.
8. Zero deployment-cost delta — correct; `n`, `V`, teacher, and proxies are all off the inference path.
9. The NAP/WCCN framing ("a post-hoc projection needs the subspace to exist") is a correct statement, and C8's post-hoc arm is well conceived — though Candidate 225 indicates the subspace does not transfer across identities at all.
10. C12 remains a useful control for a *different* question (ImageNet retention vs. generic auxiliary regularization).

## Uncertainty

I did not verify DSN's test-time discard of the private encoder from the paper body — the retrieved material confirms the shared/private split, the shared decoder reconstructing from the concatenation, the orthogonality difference loss, and that the task classifier sits on the shared branch; the discard follows from that architecture but I state it as inference. MIC, UAI, and VICReg's per-dimension std hinge are confirmed directly. My reading of `rho_32 < 1` as "source-derived subspace underperforms control on target identities" follows from the stated threshold semantics; if `rho_32` normalizes differently, item 2's contradiction weakens to "fails to support" — Gate 1 still fails on item 1 alone, and the item-4 mechanism failure is independent of both.

The verdict does not turn on any unmeasured quantity. Even granting `k*(eps) > C-1+r` on measurement, the certificate is amplitude-blind and the decoder gain unbounded, so the construction fails on its own frozen text. Any `||V||` bound, per-dimension variance hinge, or singular-value floor is a **substantive repair and therefore a new proposal**.

**Sources:**
- [Domain Separation Networks, NeurIPS 2016](https://proceedings.neurips.cc/paper/6254-domain-separation-networks.pdf) · [arXiv:1608.06019](https://arxiv.org/pdf/1608.06019)
- [MIC: Mining Interclass Characteristics for Improved Metric Learning, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html) · [arXiv:1909.11574](https://arxiv.org/abs/1909.11574) · [code](https://github.com/Confusezius/ICCV2019_MIC)
- [Unsupervised Adversarial Invariance, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/03e7ef47cee6fa4ae7567394b99912b7-Abstract.html) · [arXiv:1809.10083](https://arxiv.org/pdf/1809.10083)
- [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906)
