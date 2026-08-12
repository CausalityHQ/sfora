# LE-IDGP Train-Only Result — 2026-08-12

## Decision

**KILL.** Local-Excess Iso-Density Gradient Projection (LE-IDGP) made the
registered hard-row retrieval margin worse than the unchanged Proxy Anchor
surrogate in every fold and failed five of seven frozen predicates. It does not
authorize training or query/gallery evaluation.

## Frozen execution

- Source HEAD: `9df0080` (`fix LE-IDGP eligible pool validation`)
- Input: `/home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_train_final.npz`
- Input SHA-256: `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`
- Output: `reports/generated/inshop_idgp_train_gate.json`
- Output SHA-256: `c1e114905362e3342a91494475ef0ea0f9d40065e4a19760e7ff43011bde6a6e`
- Environment: CPU only, one OpenMP/MKL/OpenBLAS thread, NumPy 2.5.0

The first process completed the scientific computation but stopped before
publication because the report validator compared the alternate density-pool
count with all archive rows rather than the eligible rows. No report or metric
was published or inspected. The bug was reproduced with an ineligible
singleton identity, fixed in commit `9df0080`, and the same frozen command and
input were rerun. The second process published the result above and exited 1,
the CLI's registered `KILL` exit.

## Evidence

The archive contained 25,882 rows. Exactly 24,602 rows belonged to identities
with at least four examples. The fixed density pool contained 12,612 rows and
the disjoint alternate diagnostic pool contained 11,990 rows. The evaluator
formed 78 complete cohorts (14,040 rows), with 3,510 primary hard/conflicting
rows spanning 1,847 identities and all four folds.

Primary LE-IDGP minus unchanged-PA margin effects were negative in every fold:

```text
fold 0  -0.0001199008778962146
fold 1  -0.00008480567543972492
fold 2  -0.00013674244865388718
fold 3  -0.00006508073540256299
pooled  -0.0001009480951448422
99% LB  -0.0001320963796106263
```

The median absolute unchanged-PA virtual margin change was
`0.003318253744103994`. The nearest-positive similarity contrast was also
negative (`-0.00012198818403574518`, 99% lower bound
`-0.00013663067193447702`). Control lower bounds were negative for shuffled
local (`-0.00006559500314033724`), random tangent
(`-0.00013054001765609456`), and global centering
(`-0.00013212193042419498`). The collective diagnostic was negative
(`-0.000701329537431917`).

All 14,040 evaluated rows were classified as conflicts. Consequently the
one-sided and two-sided projections coincided on every eligible specificity
row and their mean contrast was exactly zero. This passed the non-strict frozen
specificity predicate but supplied no positive specificity evidence.

Predicate results, in frozen order:

```text
coverage                 PASS
raw_advantage            FAIL
fold_consistency         FAIL
control_superiority      FAIL
material_effect          FAIL
positive_similarity      FAIL
one_sided_specificity    PASS (equality only; no safe rows)
```

The local nuisance tangent was not numerically degenerate (mean norm
`0.24400448423300525`), but it was negatively aligned with the global tangent
(mean cosine `-0.5655089645748901`) and strongly stable across the alternate
pool (mean cosine `0.812904881899367`). The result therefore rejects this
particular local-excess projection rule, not merely one unstable pool sample.

## Consequence

LE-IDGP is closed. The established reproducible baseline remains Proxy Anchor
plus fixed gallery local scaling. The next candidate must avoid per-row
half-space projection against this universally conflicting density tangent and
must earn a positive train-only causal gate before any GPU training.
