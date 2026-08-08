# Pass 182 — Coset-Syndrome Metric Learning (CSML)

## Frozen blind proposal

CSML encodes each seen class with a deterministic BCH/LDPC syndrome and trains
the 512-D descriptor only through differentiable sparse parity checks. The
null-space coset is intended to preserve instance variation while distributed
parities carry class identity; the code and syndrome are discarded at
deployment. The proposed CPU test was sign/Hamming and syndrome separation on
saved embeddings.

## Gate 2 audit

Gate 2 is **DEAD**. This is supervised error-correcting output-code/class-code
supervision and deep supervised hashing, already closed by Dietterich & Bakiri
(JAIR 1995), Deep N-ary ECOC, neural/learned ECOC, Central Similarity
Quantization, and minimal-distance-separated hash centers. The BCH/LDPC
syndrome parameterization and a retained null space do not change the
mechanism: fixed algebraic code constraints replace class proxies with a
codebook and do not create data-derived unseen-class supervision. No CPU or
GPU work is authorized.
