# Pass 120 invalidation: derived objective was not executed

## Defect

The first Pass-120 controller invoked a derived coalition recipe with
`--objectives proxy_anchor`. The CLI correctly resolved the recipe and copied its
coalition fields, but then overwrote the recipe's declared objective with the base
selector. The report therefore contained the contradictory state:

```text
recipe_delta.objectives = ["proxy_anchor_coalition"]
config.objectives       = ["proxy_anchor"]
methods                 = ["proxy_anchor_end_to_end:bn_inception"]
```

The `proxy_anchor` loss handler does not read `coalition_weight` or
`coalition_mode`; those fields are consumed only by the separate
`proxy_anchor_coalition` handler. The completed `pa_coalition_single` artifact is
therefore an ordinary Proxy Anchor rerun with a derived-recipe label, not a CIS
control. Its raw best `0.917288` and final `0.915037` are invalid for every coalition
claim.

Activating the correct objective then exposed a second contract defect: the coalition
recipe selects an indexed training dataset, but the training loop unpacked an indexed
three-field batch only for the older graph objectives. The corrected objective stopped
before its first step with `too many values to unpack (expected 2)`.

## Repair and regression evidence

Commit `8252862` makes the resolved recipe authoritative for the objective that
executes. Commit `1e63c68` factors the indexed-batch condition into one shared helper
used by both dataset construction and batch unpacking. Regression coverage now proves:

1. `--objectives proxy_anchor --recipe pa_coalition` resolves to and executes
   `proxy_anchor_coalition`;
2. coalition recipes require indexed training batches while plain Proxy Anchor does
   not; and
3. a two-step end-to-end coalition run with duplicate-class rows completes with finite
   loss.

On the DGX, 72 focused CLI/objective/recipe tests passed after the first repair, then
11 focused contract tests and the dedicated two-step end-to-end integration test
passed after the second repair. The corrected live log identifies
`inshop proxy_anchor_coalition` at step 100 and epoch 1.

## Quarantine and scope audit

The invalid report, checkpoint, completed log, partial complementary log, controller
log, and pre-repair source snapshots are preserved under:

`reports/quarantine-pass120-noop-20260808T2046Z/`

The invalid report SHA-256 is
`0692eb62fdef66a8c01f4f65ed35b6a41b1d83a0d71da69cee06fd82fcfa67ea`;
the checkpoint SHA-256 is
`fe820623016ceaf8b68b7d578916842f3a377697ac4cbb97067a4c703d8ecf0c`.
Neither may enter an empirical table.

A metadata scan of every JSON report on the DGX found exactly three declared-versus-
executed objective mismatches. Two older HIST/local-NCA artifacts were already in
`reports/quarantine-wrong-objective/`; the third was this Pass-120 artifact. No
additional unquarantined report is implicated by this exact signature. This audit does
not prove that every historical result is bug-free; it bounds this specific CLI defect.
