# Current connected-method decision

This table records the latest accepted paired connected-MLP selection result, not an official benchmark or a new experiment. Full production quality and matched image-to-top-k speed remain unmet on both SOP and In-Shop.

| Dataset / split | Control | Candidate | Uncertainty / decision | Next gate |
|---|---|---|---|---|
| DeepFashion In-Shop exposed TRAIN selection; 1,734 queries, 1,715 gallery, 498 products; paired seeds 179061/179069 | R@1 96.74163783%; mAP@R 82.76548249% | R@1 97.08765859%; mAP@R 85.06567149% | Recorded product R@1 delta interval crosses zero; original KILL retained | No official read or deployment promotion from this arm |

All numbers above are verified historical measurements derived from the original receipt, not forecasts. The original two-seed average R@1 change is +0.34602076 percentage points, with recorded 95% product interval [-0.08484163, 0.82816729]. mAP@R changes +2.30018899 percentage points, interval [1.77101602, 2.85225920]. The parent did not recompute these bootstrap intervals.

| Seed | Control whole TRAIN service (s) | Candidate whole TRAIN service (s) | Candidate / control core | Candidate / control whole |
|---|---:|---:|---:|---:|
| 179061 | 2207.053 | 2295.117 | 1.032114 | 1.039901 |
| 179069 | 2188.712 | 2345.673 | 1.059007 | 1.071714 |

These costs passed the historical <=1.50 admission limit; the candidate is slower, and they prove no serving-speed win. No images/s, p95, p99, official quality or independent confirmation is inferred.

Source: [receipt](connected-mlp-evaluation-full-selection-score-v1/receipt.json), [parent verification](connected-mlp-evaluation-full-selection-score-v1/verification.json), [normal terminal](connected-mlp-evaluation-full-selection-score-v1/unit.json).

The selected rank-routing intervention remains unqualified after its 8 GiB CPU admission rejection. The pending [CPU-only envelope decision](rank-routed-cpu-envelope-decision/proposal.md) must be resolved before another qualification; no unchanged retry is authorized. The new installed-artifact work is a parallel production prerequisite, not a substitute quality result: original environment correspondence now covers all 1,754 wheel files, 1,505 fresh source hashes, 32 distribution owners and 13 separately anchored system libraries. Native relocation, complete typed payload parity, identical-batch output/wire/top-k/tie parity, lifetime/exit and resource acceptance remain unrun.
