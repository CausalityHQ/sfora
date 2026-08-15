# UniCOM imprinted-head seed-0 result

## Decision

The prospectively frozen hardened factorial selected and **PROMOTED
`imprinted_raw`**. This is an internal train-identity holdout result; it is not
an official In-Shop test result or a global-SOTA claim.

The validated artifact is
`reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json`, SHA-256
`c0666a68e70990115d80e8dc06a9f94efe83156a3fddd50f36bdbf2b3b8cd217`.
It was produced by evaluator commit
`88604a4689cf8a3ef162b16cefbbcd1e787711d2` after the prospective
fresh-evidence binding amendment `d5bb44f`. The replacement process ran once
from `2026-08-14T23:57:18Z` to `2026-08-15T01:26:30Z`, exited `0`, left no
temporary file, and the artifact passed the production validator both remotely
and locally.

## Hardened seed-0 evidence

At epoch 16:

| Cell | mAP@R | Recall@1 |
|---|---:|---:|
| `random_raw` | 0.8940674841991119 | 0.9774153074027604 |
| `random_ema` | 0.8863667322435276 | 0.9799247176913425 |
| `imprinted_raw` | 0.9177888490361535 | 0.9912170639899623 |
| `imprinted_ema` | 0.9080498732305643 | 0.9861982434127980 |

The selected-minus-control mAP@R delta is `0.02372136483704168`; the Recall@1
delta is `0.013801756587201952`. The paired 10,000-replicate query-bootstrap
95% interval is
`[0.015788336747824056, 0.031729734392229356]`. All registered promotion
conditions pass.

`imprinted_raw` first exceeds the random epoch-16 mAP@R target at registered
epoch 4 (`0.8975288707513048`), yielding the frozen `4.0×` time-to-quality
ratio. Its registered mAP@R trajectory is:

- epoch 4: `0.8975288707513048`;
- epoch 8: `0.9082495977401236`;
- epoch 12: `0.9165701881508569`;
- epoch 16: `0.9177888490361535`.

Training wall time was `15479` seconds for random initialization and `14272`
seconds for imprinting; peak GPU memory was respectively `87187` and `87167`
MiB. The architecture-level inference latency was
`11.814534576551523` ms/image and is identical across cells by construction.
These single-seed cost observations are descriptive until paired confirmation.

## Mechanism-prediction accounting

The four predictions frozen at `d4c6cae` are resolved without rewriting them:

1. **Pass.** Hardened epoch-16 `imprinted_raw` gain is
   `0.02372136483704168`, above `0.003`.
2. **Fail under the strict wording.** The hardened imprinted-minus-random raw
   deltas at epochs 4/8/12/16 are approximately
   `0.05653377`, `0.02365010`, `0.02345879`, and `0.02372136`. They remain
   strongly positive and shrink sharply overall, but the final value increases
   slightly from epoch 12, so the sequence is not monotone.
3. **Fail.** Epoch-16 EMA-minus-raw mAP@R is `-0.007700751955584284` for the
   random run and `-0.009738975805589245` for the imprinted run. EMA is not
   neutral and is worse after imprinting; the frozen evaluator closes EMA.
4. **Pending direct leakage control.** The confirmation phase must still prove
   that the imprinted epoch-0 backbone bytes equal the initial pretrained
   backbone and report the zero-shot/epoch-0 checks.

## Frozen confirmation choice

Only `imprinted_raw` advances. Seeds 1 through 6 compare the unchanged random
raw recipe with the unchanged imprinted raw recipe. EMA, its decay, and all
other cells are closed for this candidate. No seed, epoch, threshold,
initialization norm, optimizer, objective, split, or evaluator constant changes
after observing seed 0.

The confirmation claim requires all six mAP@R deltas positive, positive
nonzero paired sample variance, a paired Student-t 95% lower bound above zero,
the exact two-sided sign-test value `0.03125`, and every Recall@1 delta at least
`-0.00125`. Official-protocol evaluation and cost/Pareto comparison remain
separate required gates.
