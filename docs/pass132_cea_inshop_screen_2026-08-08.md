# Pass 132 — CEA In-Shop screen (2026-08-08)

CEA passed its CPU operating-point gate and was then run once on In-Shop with
the frozen `pa_cea` recipe.  The run was stopped after the mechanism produced
an immediate, unrecovering collapse; no distance-control arm was launched.

## Preregistered decision

The preregistration required a frozen checkpoint R@1 of at least `0.9175` and
an improvement over both Proxy Anchor and a matched distance-only graph.  The
first post-gate checkpoint was already far below the floor, so the candidate
fails before any ablation or second dataset.

## Observed curve

* pre-gate best / frozen epoch 10: **raw R@1 = 0.8738**
* selection diagnostic (local-neighbour trend around the selected epoch,
  descriptive only): **0.7366**, giving a **+13.718 pt** gap.  This value is
  dominated by the abrupt intervention at epoch 10 and is not an unbiased
  correction; it must not be presented as a corrected benchmark score.
* after enabling the CEA graph: epoch 11 **0.6156**, epoch 12 **0.5939**;
  training loss fell to `0.0007` while retrieval collapsed.

The raw and frozen values are therefore both `0.8738`, far below the registered
`0.9175` floor.  The complete original log is preserved at
`docs/evidence/pass131_pa_cea.seed0.log`.

## Mechanism judgement

This is a **Gate-4 failure**, not evidence of a novel improvement.  The
evidence graph was selective at the operating point (density `0.2556`,
multi-component classes `0.8538`, close rejection `0.6102`, far acceptance
`0.1460`), but adding its detached positive targets to the unchanged Proxy
Anchor negative term drove the embedding into a low-loss retrieval collapse.
The candidate is closed; no CEA ablations, extra seed, or second-dataset run
are authorized.  The implementation and the failed run remain in the tree so
the failure mode is reproducible rather than silently discarded.
