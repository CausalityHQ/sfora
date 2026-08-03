# Proxy fast-variable audit 236

Date: 2026-08-03. Prior art checked before implementation or GPU method work.

## Proposal

Treat learned class proxies as fast variables and the backbone as a slow variable,
by analogy with adiabatic elimination in control and statistical physics. On each
batch, take a virtual proxy step, recompute the encoder gradient against those
look-ahead proxies, then commit both blocks. The hoped-for benefit was faster
convergence in SOP's 11,318-proxy regime, where proxy learning rate is 100 times
the backbone learning rate.

## Verdict

**DEAD at Gate 2.** The executable operation is ordinary block-coordinate or
extragradient optimization of a joint objective. Gürbüz et al., *ASAP DML: Deep
Metric Learning with Alternating Sets of Alternating Proxies* (withdrawn ICLR
2022 submission), already reformulates proxy DML as alternating projections,
reinitializes proxies from selected embeddings, and regularizes consecutive
proxy problems. Li et al., *Robust Calibrate Proxy Loss for Deep Metric Learning*
(2023), explicitly diagnoses proxies as failing to track class feature
distributions and calibrates them using sample information. Generic lookahead,
extragradient, and block-coordinate methods occupy the remaining update-order
distinction; no DML-specific residual operator remains.

Primary sources:

- https://openreview.net/forum?id=vi9nRayoeaS
- https://arxiv.org/abs/2304.09162
- https://jmlr.org/papers/v23/18-045.html

No implementation or method run follows. The corrected SOP baseline already
running is a protocol repair, not an experiment on this candidate.
