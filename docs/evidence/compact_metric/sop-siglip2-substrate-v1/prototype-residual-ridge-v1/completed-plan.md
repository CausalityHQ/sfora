# Completed bounded plan

Consultation ad39611c41184a95; GPT-6.1 Sol/XHigh; exit0; 494s. Jev route ee0f0e4e-b2cd-4d56-9870-afae368b73e8. Native work and quality unrun.

**Proceed with one fixed prototype-target ridge residual trial.** It is defensible as supervised denoising of the learned source representation: labels supply the contraction target, while the frozen source supplies its coordinates. It does **not** establish an advantage over classification training; raw-output regression can damage angular separation or generalization to unseen identities. The matched retrieval test must decide that.

The residency and serializer cost failures remain closed. This is a new scientific procedure, with no resumed partial states or “1000-update” label. Relevant code is unchanged from `e2a58ef9`; the observed HEAD `efc39bd7` only updates the gate’s status text.

1. **Freeze this exact recipe.**

   Authenticate the accepted warm checkpoint `97db53ab…c7d2`, complete terminal state `b78bd945…35a9`, canonical TRAIN cache `55d37d06…6bdb8`, and partition `702f763e…676c`. Fit only the ordered **6355 TRAIN rows / 1008 identities**. Preserve the complete learned head, classifier, encoder and source buffers. Archived initialized arrays cannot supply features, targets, means or coefficients.

   All fitting arithmetic is **CPU FP32**, autocast disabled, using the source’s qualified numerical flags. For raw canonical cache \(X\):

   ```text
   x = original TRAIN CPU normalization(X)
   H = source_head(x)                         # unchanged source forward
   Z = source_head.down(normalize(x) - source_head.center)

   P[c] = mean(H[i] for TRAIN rows with label c)
   E[i] = P[label[i]] - H[i]

   V_linear = Z
   V_quadratic = Z.square()
   Phi = V - mean(V)
   Y = E - mean(E)

   G = Phi.T @ Phi
   lambda = 0.1 * trace(G) / 32
   A = solve(G + lambda*I, Phi.T @ Y).T        # FP32 [128,32]
   output_raw = source_head(input) + Phi(input) @ A.T
   ```

   Use ordinary image weighting, canonical row order, class means including each member, correction multiplier **1**, and **no fitted intercept**. No prototype normalization, feature standardization, whitening, clipping, class balancing or parameter search.

   Raw prototypes preserve the source output’s scale; unit prototypes would introduce an arbitrary scale mismatch. Trace-scaled regularization matches the existing [ridge implementation](/home/rb/worktrees/sfora-positive-causality/src/sfora/foundation_adapter.py:505). Reuse its `fit_ridge_stitch(..., regularization=0.1)` and retain `weight.T` and `source_mean`; **do not call `model.transform()`**, which adds `target_mean`. Its native solve avoids constructing an inverse. [PyTorch API](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.solve.html)

2. **Implement a separate procedure with these files and interfaces.**

   | Prospective file | Responsibility |
   |---|---|
   | `scripts/fit_siglip2_prototype_residual.py` | `fit_prototype_residual(raw_train, targets, base, arm) -> dict`: coefficients, both fixed means, prototypes/counts and fit witnesses. CLI phases `cpu` and `fit`; arms `linear` and `quadratic`; no seed argument. |
   | `scripts/evaluate_siglip2_prototype_residual.py` | Admit the two new fit terminals, independently reconstruct readouts, replay source wires, and score the authorized panel. CLI phases `cpu` and `score`. |
   | `scripts/test_siglip2_prototype_residual.py` | One bounded check covering solver algebra, discarded intercept, admission rejection and reload integrity. |
   | `docs/inshop_prototype_residual_gate_2026-10-02.md` | Frozen recipe, closure, resources, stop rules and evidence roles. |

   Use a separate `prototype-residual-ridge-v1` evidence directory and new schemas. Reuse the original warm-source admission, encoder-composition predicates, typed fingerprinting, packing and pinned scoring definitions where their predicates apply. Existing `TRAIN1000`, optimizer, schedule and four-endpoint validators **cannot admit these endpoints unchanged**.

   Reuse [quadratic readout arithmetic](/home/rb/worktrees/sfora-positive-causality/scripts/quadratic_readout.py:123), including TRAIN versus direct-FIT normalization. Its API requires a leaf `Parameter` for A: wrap only the solved coefficient for compatibility, with no backward pass or optimizer. Keep means and A explicit.

3. **Run one bounded falsifier after review and source freeze.**

   Execute sequentially:

   - **CPU qualification ≤120s:** actual full-TRAIN fits for both arms, zero-A source parity, finite nonzero coefficients, solver stationarity and unchanged frozen members.
   - **Linear fit ≤300s**, then **quadratic fit ≤300s:** each independently reconstructs the warm head and rereads TRAIN inputs for a second full fit. Require exact coefficients/means and typed state agreement, then complete independent checkpoint reload and raw/unit/int8/FP16-inverse-bit parity.
   - **Evaluator CPU qualification ≤120s**, then **selection scoring ≤300s**.

   Maximum service allowance: **1140 seconds**. Independent refits prove reproducibility; they are not additional seeds or independent training evidence.

   Every unit retains **8GiB host, no swap, zero disallowed events, CUDA allocation <10GB, both lifetime locks, original normal exit and fresh complete exit authentication**. Hide CUDA for this CPU procedure. Retain fresh byte checks at use boundaries, including unchanged-version mutation rejection.

   Engineering KILL for any admission, resource, reload or frozen-state failure; zero feature energy; nonfinite/zero A; or normalized stationarity error above `1e-5`. Prospectively require candidate/control **whole-service and total fit-core cost ratios ≤1.50**. Report measured fitting costs; the old median-update metric is inapplicable.

   Score only **selection1734q/1715g/498**, using the qualified direct-FIT cache. First replay every archived source query exactly against **96.30911188004614% R1 / 80.57229533585297% mAP@R**, with complete wire readback.

   **KILL immediately** if quadratic-minus-linear R1 ≤0, mAP@R <0, or quadratic regresses either source metric. A survivor must additionally achieve **≥0.002 fraction improvement on both metrics** and both paired-product 95% lower bounds >0, using the unchanged shared **5000 draws / 179019** bootstrap; report query intervals too. These intervals are conditional on the single frozen trained source. No `179069` duplicates, retuning or intermediate selection.

4. **Require consequential review before execution.**

   Review must explicitly accept the new objective and cost definitions, verify TRAIN-only construction and the omitted intercept, and establish that admission adapters preserve complete source authentication without rebinding helpers or waiving origin checks. Native checks must demonstrate that corrupted labels/means/source bytes fail and that independent reconstruction cannot reuse fitted sufficient statistics.

   Only selection GO admits unchanged **validation1749q/1730g/498**, with its own source floor and the same quality gates. A survivor then needs the authenticated So400 public readout path, B1/B32 parity, matched image-to-top-k timing and the required 10,000 paired calls before any public-speed claim. Full-TRAIN fitting and official SOP+InShop confirmation remain separate gates.

No files were edited, and no fits, quality reads, Torch/GPU work, jobs, children or consultations were launched.
