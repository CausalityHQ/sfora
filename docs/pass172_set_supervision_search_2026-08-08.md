# Pass 172 — set-level/coalition supervision search (NONE before GPU)

The set-level search used the In-Shop between:local error ratio `3.05–3.22`
and CUB's near-`1:1` ratio as provenance for a possible class-coalition
training object. CMR/Shapley was already CPU-screened and failed (`ΔR@1 =
-1.01` points).

Primary-art review kills the remaining set-level variants: Ranked List Loss
(CVPR 2019) uses set-based positive/negative galleries and class hyperspheres;
Deep Wasserstein Metric Learning (AAAI 2022), Large-Margin Set-to-Set
Similarity, DeepEMD (CVPR 2020), and DSLL occupy Wasserstein, set-to-set,
Earth-Mover, and distribution-structure supervision. DPP/log-det coverage is
also occupied by DNVP, Ranked List, Reverse Contrastive, and related work.
Auction or submodular variants reduce to selection/reweighting or these
set/OT mechanisms.

Decision: `NONE` before GPU. No implementation or GPU run occurred.
