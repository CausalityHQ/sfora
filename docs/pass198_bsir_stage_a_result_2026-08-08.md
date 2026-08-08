# Pass 198 — BSIR Stage-A result

## Verdict: FAIL at Gate 1; no implementation or GPU

The preregistered artifact-only diagnostic completed on all four corrected In-Shop
Proxy Anchor seeds. It replaced only the 138 batch-shape-sensitive canonical query
tail rows with the corresponding legacy batch-256 reconstruction and kept each
digest-bound canonical gallery fixed.

| seed | max coordinate drift | max descriptor L2 drift | nearest-identity flips | correctness changes | R@1 change (pt) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000708 | 0.004336 | 0 | 0 | 0.000 |
| 1 | 0.000662 | 0.004673 | 0 | 0 | 0.000 |
| 2 | 0.000639 | 0.003911 | 0 | 0 | 0.000 |
| 3 | 0.000845 | 0.006231 | 0 | 0 | 0.000 |

Thus all `4*138=552` paired queries retained the same nearest-gallery identity and
correctness. There were zero correct-to-wrong and zero wrong-to-correct changes. In
each seed, 137 of 138 rows also satisfied the conservative sufficient certificate
`top1-top2 margin > 2*||delta_q||_2`; the one uncertified row still did not change its
nearest identity.

The canonical and tail-replaced R@1 values were exactly equal within every seed:
`0.9137009425`, `0.9167956112`, `0.9151076101`, and `0.9159516106`.
The prefix reconstruction remained bound at `1.19e-7`–`1.42e-7`, confirming that the
diagnostic isolated the known tail-shape perturbation rather than an order or
checkpoint mismatch.

## Mechanism learned

Eval-mode BN-Inception is numerically batch-shape-dependent, but this measured
finite-precision perturbation is too small relative to the local retrieval decisions
to provide the registered quality provenance. Coordinate drift is not a retrieval
failure. A consistency penalty would spend compute learning invariance to an
integrity defect that changed no observed decision, and its symmetric gradient is
likely second-order-small. BSIR therefore fails its necessary retrieval-causality
screen and is closed without implementation.

Remote result:
`/home/riomus/group-learning/reports/generated/pass198_bsir_stage_a_result.json`.
SHA-256: `afba723df1d7c233743c2ffeb128b9879752e3dafcfc6ba09265f7027bc4be16`.
The result-embedded preregistration, manifest, and diagnostic source hashes exactly
match the committed local files.
