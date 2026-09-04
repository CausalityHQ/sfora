# UniCOM rank-finish screen result

The preregistered seed-0 rank-finish screen returned `PROMOTE`. The exact
reviewed source was `0d4c12bdb25ee2ed46add08d4e5731164267d1f6`. The canonical
result is retained at
`/tmp/unicom-rank-finish-0d4c12bdb25ee2ed46add08d4e5731164267d1f6.json`,
SHA-256 `3a8cf818e66248fa124cbfd6231a17298cf9fb1734ce9dd1b9a47d4274d8b111`,
1,947 bytes. It is valid JSON with exactly one trailing newline.

The screen resumed the authenticated imprinted epoch-4 control checkpoint,
SHA-256 `8f1cda1b61583ac678447c1f22463b64cd69cf5b4a0a47074bb7353c0a8dbcbb`,
and used the registered optimization partition, SHA-256
`cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`.
It trained epochs 5--8 with the frozen identity-balanced SmoothAP finish and
did not consume the standard-test split or the earlier query-expansion result.

## Result

| metric | epoch-4 control | epoch-6 | epoch-8 | epoch-8 delta |
|---|---:|---:|---:|---:|
| mAP@R | 0.8975116742 | 0.9055260305 | 0.9106583524 | +0.0131466781 |
| Recall@1 | 0.9861982434 | 0.9874529486 | 0.9899623588 | +0.0037641154 |
| Recall@10 | 0.9974905897 | 1.0000000000 | 1.0000000000 | +0.0025094103 |

Mean training loss declined monotonically across the four finish epochs:
0.1818005, 0.1722153, 0.1575132, and 0.1520944. The run completed 161 steps per
epoch in 3,548.20 seconds. Peak CUDA allocation was 88,041,814,528 bytes and
peak reservation was 93,610,573,824 bytes. Memory PSI remained zero at the
observed checkpoints and no resource stop fired.

The epoch-6 delta mAP@R of +0.0080143562 cleared the early stop. The epoch-8
result cleared all frozen promotion predicates: mAP@R improved by more than
+0.010, Recall@1 did not decline by 0.001, and Recall@10 did not decline by
0.001. This is the first positive evidence that directly optimizing late
positive ordering repairs the measured UniCOM quality deficit without changing
the deployed descriptor geometry.

## Interpretation and next boundary

This result is deliberately `claim_eligible=false`. It is one seed on the
identity-disjoint development holdout, so it is evidence to promote the method,
not a final model or publication claim. The earlier standard-test query
expansion result remains closed because it traded recall for mAP and is not
combined with this method.

The next step is a separately frozen multi-seed confirmation of the identical
rank-finish method. Confirmation must preserve the optimizer/checkpoint lineage,
use fixed seed-specific schedules, report paired per-query evidence, and require
both central improvement and seed robustness before a final untouched-test
readout. No threshold, temperature, batch composition, descriptor prefix, or
epoch count is tuned from this result.
