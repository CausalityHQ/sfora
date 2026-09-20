# Same-teacher zero-training ladder result

## Decision

The frozen four-arm ladder is terminal.  None of the three candidates survives
the preregistered five-dataset rule against PCA128-int8: a point mAP@R effect
of at least `0.005` and a paired per-query 95% bootstrap lower endpoint above
zero on every dataset.

- `pca256_int4` is killed by CUB and In-Shop.
- `learned128_int8` is killed by CUB.
- `learned256_int4` is killed by CUB.

The label-free width selector is also closed.  It selected the outer winner on
prospective Food-101 but selected width 128 on the unseen Flowers-102 check,
where width 256 won (`0.983412` versus `0.981970` mAP@R).  Because the frozen
contract required both datasets, it cannot rescue the non-universal width arm.

## Verified quality

Every value below is from the corrected exact packed-int8 receipt.  `CI` is the
paired per-query 10,000-draw PCG64(20260920) interval for learned256-int4 minus
learned128-int8.

| Dataset and split | PCA128-int8 | PCA256-int4 | Learned128-int8 | Learned256-int4 | Learned width delta (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cars, 8,054 fit / 8,131 class-disjoint eval | 0.767861 / 0.970975 | 0.813610 / 0.972205 | 0.833710 / 0.972820 | **0.851807 / 0.975157** | +0.018098 `[+0.017006,+0.019216]` |
| CUB, 2,997 fit / 2,857 class-disjoint eval | 0.719120 / **0.912146** | 0.719034 / 0.911796 | **0.720486** / 0.905495 | 0.720194 / 0.908645 | -0.000292 `[-0.001432,+0.000832]` |
| SOP, 59,551 official train / 60,502 official test | 0.457572 / 0.728935 | 0.468577 / 0.739314 | 0.485659 / 0.751083 | **0.503619 / 0.764950** | +0.017960 `[+0.017088,+0.018824]` |
| In-Shop, 25,870 fit / 14,218 query / 12,612 gallery | 0.777802 / 0.945773 | 0.778088 / 0.946125 | 0.794141 / **0.952806** | **0.794839** / 0.952033 | +0.000698 `[-0.000659,+0.002030]` |
| Food-101, 20,250 fit / 5,000 class-disjoint eval | 0.810694 / 0.969600 | **0.833903** / 0.971000 | 0.817521 / 0.969400 | 0.827972 / **0.971200** | +0.010451 `[+0.009445,+0.011477]` |

Cells show `mAP@R / Recall@1`.  The receipt also contains all per-query AP and
Recall@1 vectors and each arm's paired contrast against PCA128-int8.  In
particular, learned128 and learned256 improve strongly on four domains, but
their CUB deltas against PCA128 are only `+0.001367` and `+0.001075`; both CIs
include zero and both miss the `+0.005` effect gate.

## Performance and storage status

This gate measured representation quality, not serving speed.  The full run
took about four minutes on the existing NVIDIA GB10 DGX and incurred no new
cloud spend.  Per-dataset end-to-end experiment times were 42.77 s (Cars),
36.88 s (CUB), 54.45 s (SOP), 41.31 s (In-Shop), and 59.01 s (Food-101).

The library's exact int8 wire is **130 bytes/item**, not 128: 128 signed code
bytes plus one f16 inverse norm.  The int4 wire is 128 bytes/item plus 1,024
shared scale bytes per index.  No 1M latency, p50/p99, throughput, RSS, or
kernel profile is claimed.  CUDA/cuTile work remains blocked because no
quality candidate survived and no serving profile has identified a >30% hot
kernel.

## Authorities

- Source commit: `e87d25da3831495b902a90d8ac897fc6d051b1ac`
- Runner SHA-256:
  `e12b5dc760e70efaa8891b8b41a2c7830bd81e161c8a1d1f14ac08cc7a40b514`
- Preregistration SHA-256:
  `57f4492d17e8abd22f12d7c3aa40867cc5cb7ee5f080fdc0cdbd049e0ae48e1c`
- Full receipt SHA-256:
  `149313594c2e5da36c3a1a8d7137bb0afb163da9e0209038039bed98bf69a188`
- Compact checked-in summary:
  `docs/evidence/same_teacher_zero_training_ladder_e87d25d_summary.json`

The first receipt (`36a844af...`) is invalid because its int8 scorer restored
and renormalized codes rather than using the library's packed integer-dot plus
f16 inverse-norm path.  It is retained in the negative ledger and contributes
no decision.

## Consequence

There is no evidence winner to augment or benchmark at 1M, so top-2 database
augmentation, matched-SOTA performance claims, and custom kernels do not
advance from this gate.  The observed problem is cross-domain mechanism
instability, not codec fidelity: learned width is valuable on Cars/SOP/Food,
but has no measurable mAP@R benefit on CUB/In-Shop.  The next research family
must therefore target domain-robust neighborhood information and pass
provenance/mechanism controls before another multi-domain compute run.
