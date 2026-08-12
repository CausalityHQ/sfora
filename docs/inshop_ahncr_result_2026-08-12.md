# AHNCR held-out-label result: KILL

The prospectively frozen Asymmetric Hard-Negative Cohort Residual (AHNCR)
mechanism gate did **not** pass. No coefficient, neighborhood size,
normalization, split, or decision threshold was changed after observing the
result, and no second run was made.

## Registered execution

- Design: `67814a5`, amended pre-outcome by review commit `e6e85ad`.
- Implementation at execution: `e49bcc2e36bc391c621e40cfa5ea09ecdad50121`.
- Input: `inshop_corrected_pa_seed0_train_final.npz`, SHA-256
  `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- Result: `reports/generated/inshop_ahncr_train_holdout.json`, SHA-256
  `bc04735d18365575c800e5c2f92c81236bb487bdd0f732099072163da1a36628`.
- CPU only; CUDA hidden; OMP, MKL, and OpenBLAS each pinned to one thread.
- One execution, exit 0. The output is a regular mode-0600 file and no atomic
  temporary remains.

## Outcome

The deterministic split produced 20,908 cohort rows and 797 evaluation
queries over 4,177 gallery rows. Reporting-shard query counts were
`[213, 192, 184, 208]`.

Raw cosine already achieved `797/797 = 1.0` Recall@1. AHNCR and every causal
control also achieved `797/797 = 1.0`:

| Arm | Recall@1 | Gain | W→R | R→W | paired p |
|---|---:|---:|---:|---:|---:|
| raw | 1.0 | 0.0 | 0 | 0 | 1.0 |
| AHNCR | 1.0 | 0.0 | 0 | 0 | 1.0 |
| global mean | 1.0 | 0.0 | 0 | 0 | 1.0 |
| positive expansion | 1.0 | 0.0 | 0 | 0 | 1.0 |
| unary cohort density | 1.0 | 0.0 | 0 | 0 | 1.0 |
| symmetric adaptive normalization | 1.0 | 0.0 | 0 | 0 | 1.0 |

All four AHNCR shard gains were zero. All 20 deranged centroid-assignment null
gains were zero, so their linear 95th percentile was zero. Every direct
AHNCR-versus-control comparison had zero discordant queries and exact paired
`p=1.0`. Consequently all 13 preregistered decision predicates were false.

## Interpretation

This is a valid preregistered **KILL**, but it is not evidence that the score
hurts open-set retrieval. The screen saturated because the held-out identities
came from the encoder's own training split: labels were hidden from cohort
construction, not from representation learning. With raw Recall@1 at 100%, no
post-processing rule could show a positive transition. The result therefore
closes AHNCR under this protocol and forbids a coefficient/`k`/normalization
rescue, while also documenting that this closed-set construction lacked power
to test the intended open-set mechanism.

The candidate is also close to CSLS and adaptive cohort normalization, not a
new broad similarity family. Its narrow distinction was a one-sided vector
residual from a class-disjoint training cohort; that distinction received no
usable evidence here. Any next method must use a genuinely unseen-identity
frozen test and be mechanistically distinct rather than retuning AHNCR.

## Verification

Before execution, 62 affected tests passed along with Ruff, `py_compile`,
`git diff --check`, exact input hashing, and output/temp absence. Afterward the
production strict validator passed. A separate NumPy implementation then
recomputed the hash split, stable top-50 cohort centroids, all six score
matrices, all top-1 vectors, all four shards, the 20 PCG64 derangements, and
the null percentile. It reproduced 797 correct queries for every arm and the
exact persisted counts.

