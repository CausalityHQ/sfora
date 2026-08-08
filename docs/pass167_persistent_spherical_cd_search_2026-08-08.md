# Pass 167 — persistent spherical contrastive-divergence bath (NONE before GPU)

The corrected four-seed In-Shop frozen-final baseline is `0.9153889436 ±
0.0013195712`; corrected between:local error ratios are `3.05–3.22`, and the
unseen-minus-seen positive-similarity gap is `-0.04968`. These establish
headroom but do not identify a new training signal.

The proposed mechanism maintains persistent normalized particles `(u_m,c_m)`
and updates them with projected Langevin steps under a Proxy-Anchor-like energy,
then contrasts data energy with detached particle energy. Gate 2 kills it:
persistent contrastive divergence/replay-buffer Langevin phases are established
by Tieleman (ICML 2008), Du et al. (ICML 2021), and Grathwohl et al. (ICLR 2020);
DAML, AdCo, LoOp, Proxy Synthesis, and MemVir cover synthetic/adversarial
embedding negatives. Detached particles have zero encoder gradient; live
particle interactions reduce to synthetic-negative weighting; a train-only
energy head is an ordinary EBM.

Decision: `NONE` before GPU. Forced forecast is `0.9154`, with under 5%
probability of clearing the useful `0.9185` floor after matched controls. No
implementation or GPU run is justified.
