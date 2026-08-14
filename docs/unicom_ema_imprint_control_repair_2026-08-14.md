# UniCOM EMA × imprint control repair

## Scope

This repair changes only the instrument-continuation gate. It does not change training,
the four factorial cells, registered epochs, candidate selection, or promotion criteria.

The promotion criteria remain:

- selected epoch-16 mAP@R gain over the current-lineage `random_raw` arm: at least `0.003`;
- Recall@1 delta: at least `-0.00125`;
- paired-query bootstrap 95% lower bound: strictly greater than zero.

No imprinted-arm or candidate metric was observed before this repair.

## Observed failure and diagnosis

The new seed-0 `random_raw` run produced mAP@R `0.8854017757500122` and Recall@1
`0.9786700125470514`. The original v1 control report is retained as `INVALID` because
its symmetric ±`0.002` gate compared those values with a checkpoint from an older
trainer lineage: mAP@R `0.8716329439260202`, Recall@1 `0.972396486825596`.

The old epoch-16 checkpoint was evaluated again with the current hardened evaluator and
exactly reproduced the archived values. Therefore the discrepancy is in the training
trajectory, not the evaluator. The GPU environment is not deterministic, and the current
trainer additionally maintains an FP32 EMA shadow after each optimizer step; this can
change memory and scheduling even though the raw-arm update rule is intended to be the
same.

Registered evidence:

- original v1 report SHA-256: `9af6f811eb162a4054d9c86bc117c2e261951750da207670841e058521c077fa`;
- current epoch-16 checkpoint SHA-256: `df53194c44e4131a6d89832639cedcd10b78d27b18b3c2a802997e12abd85e55`;
- current history-file SHA-256: `20afa8ac4404d86c2af6b07d390013af3fddf6433e9c6d51c4bf272ee2feb84f`;
- current row history-evidence SHA-256: `de62f014367fdc492f84dc96d05f02d6c29bd711d46813e642181c476c37ef96`;
- archived report SHA-256: `c8bb65dcff33b09c40f602f1d243336d83a8d29f93c3c8c2e75236fed375140a`;
- archived raw checkpoint SHA-256: `210d0113b40d2a5ef3bb836f818ed2d632d046f3e818b1bb5049e25ba845f0a5`;
- archived trainer SHA-256: `b2cfdaed33d46ec445141bb40b1a3f28aed0d3ca859101843ddf825866640bb1`.
- archived initial checkpoint SHA-256: `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`;
- archived partition SHA-256: `cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`;
- recomputation evidence: `reports/unicom_archived_raw_current_evaluator_2026-08-14.json`,
  SHA-256 `b64bd3df5d9abe4ee7d1a1f4332148e1cc8adfc40ba2421e5fb5d2b2be6cdfe7`;
- recomputation evaluator revision: `d4f5f3e5029b2f29fe11d725545a56a3cf904b63`,
  source SHA-256 `2adf842db58574fd6c8487743f2dab341881aad1d5ca5224e647022a4f168a4d`.

The row history-evidence digest was independently recomputed from the real
epoch-16 checkpoint's canonical JSON history on the training host; it is not a
value inferred from the test fixture.

## Repaired gate

The v2 control report embeds the unchanged v1 `INVALID` gate and its evidence. It permits
continuation only when the current-lineage random control is not inferior to the archived
reference by more than `0.002` in either mAP@R or Recall@1. Improvement above the archived
reference is not an invalidation condition.

The v2 validator also requires the current control's initial checkpoint and
partition to equal the archived run, and requires its metrics and complete
per-query evidence to equal the factorial's fresh `random_raw` evaluation.
The archived-checkpoint recomputation is a separate committed JSON artifact;
the v2 report binds its bytes and the evaluator revision instead of asserting
the archived metric constants as self-evidence.

The factorial still compares every candidate with the current-lineage `random_raw` row.
Consequently the higher current baseline makes promotion harder, not easier. The v2 report
also records `candidate_values_observed=false` and `promotion_thresholds_unchanged=true`.

The v2 report is derived without GPU work:

```bash
python -I -B scripts/evaluate_unicom_ema_imprint_factorial.py \
  --mode repair-control \
  --legacy-control-report /path/to/original.control.json \
  --output /path/to/repaired.control-v2.json
```

The original report is never overwritten. Factorial execution requires the strict v2
report and rejects the v1 report directly.
