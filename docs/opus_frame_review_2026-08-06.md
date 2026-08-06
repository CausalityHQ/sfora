# Independent cold review of FRAME

Consultation ID: `6eec8709c3204dc4`  
Provider/model: Claude Opus  
Caller: `sfora/emafactorial`  
Status: completed, exit 0  
Review prompt: `docs/opus_frame_review_prompt_2026-08-06.txt`  
Review-prompt SHA-256: `92ac4341cb9e2c4725126a4bc9af04d0398d439a186d08f10e943d51def06955`  
Native result truncated: false

Review delivered above; verdict summary also sent to your Telegram.

**DEAD.** The decisive items, in order of weight:

1. **The declared mechanism is refuted against the primary source.** I retrieved arXiv 1809.09307 (the PDF A5 says returned as binary). cw-CR is applied at the ResNet **output/last-hidden layer feeding a learnable linear softmax** — "the best performance is observed when the output layer is regularized." That head is exactly $O(d)$-equivariant under $(z\to Rz, W_{\rm cls}\to W_{\rm cls}R^\top)$, identically to learnable proxies. §3.1's distinction #1 is false. A5 pre-committed that the coordinate-free reading would be the *surviving* leg; it is the leg that died.

2. **Term 1's leading-order effect is conditional whitening.** $\mathbb{E}[\bar\rho]=1-(r_{\rm eff}+2)/d$ exactly. C1 is a global optimum of FRAME's own term, not a rival; F2 is satisfiable with zero frame alignment; and since FRAME never penalizes $\mathrm{tr}\Sigma_c$, it can only *redistribute* nuisance — toward isotropy, hence into the span where unseen-class mean differences live. That contradicts Prop 2 and puts the sign of the forecast in doubt.

3. **D2 escapes by rotation** at cost 0.52 vs 1.62 under the proposal's own $\lambda$'s, on representations C4 proves are identical for retrieval. Term 2 is not rotation-invariant, so the headline equivariance argument does not survive term composition.

Props 3–4 are vacuous (worst-case-over-$\Delta$ discards the signal; "uniquely" has a three-line counterexample; neither mentions more than one class), $\iota_+$'s derivation drops a negative term, the 200-epoch lane is non-executable under PA's decay schedule, and F1 fires with probability 0.003 under the author's own intervals.

What survives is the tetrad U-statistic — Prop 1 verified exactly correct including the full $S_4$ symmetrization, usable SNR, real distinction from cw-CR's plug-in estimator. A component, not a method. Two of the author's three pre-registered withdrawal conditions are, on this analysis, the predicted outcomes: the tripwires were built correctly and then forecast past.

