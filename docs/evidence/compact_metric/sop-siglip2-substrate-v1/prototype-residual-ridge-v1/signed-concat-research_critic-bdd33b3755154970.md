# research_critic: gpt-6.1-sol

Group 772b1fe7b3fb422e; consultation bdd33b3755154970; exit 0; elapsed 589s. Actual providers: claude, codex. Opus OAuth failure is not a review; research answer is the recorded Sol fallback.

**GO-to-implement one frozen signed-concat trial, with the corrections below. Native execution remains unadmitted.** I found no mathematical reason to KILL this design; its retrieval benefit and cost feasibility remain unproved.

I verified the decision’s receipt/log hashes and independently recomputed all six metrics from the 1,734 per-query rows. The quadratic KILL is supported: versus linear, R1 improves **0.11534 pp** while mAP falls **0.60835 pp**. Preserve both development Pareto heads; neither establishes production success.

1. **Make the common-scale contract explicit.** The authenticated solver file and extracted AST match the supplied hashes. The proposed adapter is sufficient if its only arithmetic change is `scale = reference_scale`, plus the keyword addition. Require a detached, finite, positive, scalar CPU FP32 reference scale derived independently from centered Z in every fit. Preserve `0.1 * (trace(GZ)/32)` and verify the actual penalty bits used in both solves—not merely two recomputed witness values.

   Preserve TRAIN’s outer normalization and the source head’s internal normalization exactly. Direct-FIT inference must retain its existing path. Never call `RidgeStitchModel.transform`: it adds `target_mean`. The residual helper must use only the centered features and fitted weight.

   The nesting claim in the [plan](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-plan-9f581ac1d2164d8b.md:34) is correct in exact arithmetic:
   \[
   \min_W\|Y-\Phi_{\rm concat}W\|_F^2+\lambda\|W\|_F^2
   \leq
   \min_W\|Y-\Phi_ZW\|_F^2+\lambda\|W\|_F^2.
   \]
   This supplies no retrieval guarantee. The frozen stationarity check measures backward residual, not coefficient accuracy or retrieval improvement.

2. **Describe the capacity hypothesis more carefully.** H0 adds regressors and permits a higher-rank correction; the existing linear arm already modifies all 128 output coordinates. Independent added rank has not been established.

   Common λ also does **not** isolate capacity. If an added feature duplicates an existing feature, coefficients can split across the duplicates: for total coefficient \(c=a+b\), the minimum penalty is \(\lambda c^2/2\), versus \(\lambda c^2\) previously. Thus concat can change effective shrinkage even without adding information. A successful trial would validate this specified concatenation procedure, not prove that the latent bottleneck caused the previous result. No additional probe or balancing intervention is needed.

   The target has a useful interpretation. With \(C\) the member-inclusive class-averaging projector,
   \[
   E=(C-I)H_0,\qquad H_{0c}^{T}E=-H_0^{T}(I-C)H_0.
   \]
   The H0 block therefore exposes within-class scatter directly. Frozen prototypes anchor the regression, but the objective has no explicit angular or negative-ranking term. Relevant prior art pairs compactness with discrimination: [center loss uses joint softmax supervision](https://ydwen.github.io/papers/WenECCV16.pdf), and [supervised retrieval whitening uses matching and nonmatching covariance](https://cmp.felk.cvut.cz/~chum/papers/Radenovic-ECCV16.pdf). These support plausibility, not the required gains.

3. **Complete the width-aware helper and admission contract before freezing code.** Existing helpers cannot consume the proposed payload unchanged. [The original readout validators](/home/rb/worktrees/sfora-positive-causality/scripts/quadratic_readout.py:75) require A `[128,32]` and two 32-element `linear/quadratic` means. The fitter also constructs a fixed-width zero witness and calls that old weight validator.

   Reuse authenticated source, feature and finite-value validators; implement procedure-owned weight/means validation for the v2 arms. Route fitting, inference, zero-source witnesses, integrity and reload through that contract. Keep Parameter/leaf/frozen-state requirements.

   Authenticate the new helper through the three-file fitter closure and the evaluator’s training descriptor, including origin/function/global checks and exit authentication. Change prospective arm labels only; historical `ORIGINAL_*` roots, factories and hashes must remain unchanged. Reject cross-version payloads and bind widths, basis order and the penalty rule to the recipe and fitted-state fingerprint.

4. **Keep cost and timeout outcomes decisive, but interpret them correctly.** The 25× Gram, 5× RHS and 125× cubic-factorization figures are operation counts—not runtime multipliers. Common source arithmetic and prototype construction contribute substantial cost.

   The historical sum-core allowance is approximately **0.373367 s**, or **62 ms additional time per fit**, using the old baseline solely as an illustration. Fresh matched endpoints must supply both denominators. Cost passage is unqualified.

   Likewise, the accepted score’s **13.534 s** remaining margin does not qualify the survivor path. The existing interval loop adds eight bootstrap-helper calls, alongside the larger readout and ordinary I/O variation. Any runtime projection remains an estimate. A timeout must close the frozen trial as an engineering failure; partial metrics cannot become an accepted quality decision.

5. **Preserve the production boundary.** Selection GO still requires both gains ≥0.002 fraction, positive paired-product lower bounds, the source floors and all engineering gates. The R1 threshold alone requires at least **four net corrected queries** on this panel. Intervals remain conditional on the frozen source and fixed gallery; previously observed selection is development evidence. Sealed validation provides the next independent confirmation.

   The [public loader](/home/rb/worktrees/sfora-positive-causality/src/sfora/siglip2_compact_serving.py:232) constructs a 1024→128 linear head and cannot serve this composition. The future So400 loader, B1/B32 parity, 10k paired p99 measurements, full-TRAIN refit and official SOP/InShop/transfer confirmation remain necessary. Full-TRAIN fitting needs a separately authenticated payload/authority; parity evidence must cover the final checkpoint.

No files were edited. No native/Torch experiments, SSH, GPU/cloud jobs, children or extra consultations were launched.

