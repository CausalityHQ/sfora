# UniCOM ProxyMuon F0 result

The preregistered cached-feature screen closes the exact pinned ProxyMuon
candidate. The reviewed source was
`1d7fcaa5635fe425a748f6e46045954be83672ca`; the direct-child run handoff was
`64da998cdb65e9a52c8b60061e68bcf146a6870f`. The immutable result is
`reports/generated/unicom-proxy-muon-f0-1d7fcaa.json`, SHA-256
`f36f6fa4669038c6d126edcdc2b9c6316a96f9a7ca69150f020b55cda5b643eb`
(94,478 bytes). Production strict validation and an independent Claude review
both passed with no Critical or Important findings.

## Decision

The registered status is `CLOSE_PROXY_MUON`. Both optimizers selected the
strict interior learning rate `0.0002`; AdamW's mean phase-1 final loss was
`0.8416255402068297`, versus `0.9216068778187037` for ProxyMuon. In phase 2,
neither BF16 nor FP32 ProxyMuon reached the matched AdamW step-512 loss by step
512 on any seed:

| seed | AdamW step-512 loss | AdamW accuracy | BF16 reach | BF16 accuracy delta at 512 | FP32 reach | FP32 accuracy delta at 512 |
|---:|---:|---:|:---:|---:|:---:|---:|
| 3 | 0.3695081323 | 0.8687656838 | `>512` | +0.0019702792 | `>512` | +0.0019947851 |
| 4 | 0.4544190075 | 0.8692214947 | `>512` | -0.0013282230 | `>512` | -0.0013331242 |
| 5 | 0.3793792073 | 0.8686137469 | `>512` | -0.0017497255 | `>512` | -0.0017448243 |

All step-512 accuracy deltas satisfy the registered -0.002 noninferiority
floor. Thus this is an optimization-speed failure, not an accuracy failure.
`all_reach_noninferior=false` is vacuous because no ProxyMuon arm reached the
loss target; it must not be read as evidence of accuracy degradation. BF16 and
FP32 trajectories were nearly identical, so update precision does not explain
the convergence gap.

The exact predicate vector was:

- `adamw_lr_interior=true`
- `proxy_muon_lr_interior=true`
- `all_reach_by_307=false`
- `all_reach_noninferior=false`
- `all_step512_noninferior=true`
- `any_bf16_accuracy_loss=false`
- `fp32_sensitivity_supported=false`

## Cost and execution

The sole scientific process completed all 42 registered cells in
69.8708137181 seconds after parent reconstruction. Peak CUDA memory was
363,026,944 bytes allocated and 436,207,616 bytes reserved on an NVIDIA GB10.
It published one mode-0600 result, no failure receipt, and no temporary output.

The result contains no query/gallery retrieval metrics and therefore makes no
SOTA claim. The previously confirmed official In-Shop class-proxy imprinting
result remains separate: mean R@1 95.15%, best 95.20%, unchanged inference
cost, and a one-time initialization pass.

## Routing

Do not retune Muon. The registered automatic continuation is the already
designed full-width objective, with cross-dataset imprinting replication as
the alternate route. Hyperbolic geometry, CAP, and custom kernels remain
closed unless new measured evidence defeats their specific prior kill
conditions. This result closes only the exact pinned PyTorch 2.12.1 BF16
ProxyMuon candidate; it is not a claim about every possible orthogonalized
optimizer.
