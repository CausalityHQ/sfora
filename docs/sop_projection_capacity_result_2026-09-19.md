# SOP Projection-Capacity Result

## Scope

This resolves the preregistered question in
`docs/sop_projection_parameterization_preregistration_2026-09-12.md`: does adapting only
inside the deployed 128-dimensional subspace discard useful directions that remain
available in the frozen 768-dimensional teacher representation?

It answers yes. Under this optimizer and schedule, training the full 768-to-128 affine
projection beats training a 128-by-128 adapter over the same frozen base codes, and a
parameter-matched control shows the gain comes from the directions available, not from
the extra parameters.

Diagnostic on the SOP official-train class-disjoint validation split: 11,847 rows across
2,264 classes. `official_test_touched` is `false` in every receipt. Receipts are
`claim_eligible=false`. This is not a publication claim and not an absolute retrieval
claim.

## Arms

All three arms start from the same deployed affine map, emit one normalized
128-dimensional code, and are scored through the established symmetric-int8 packed
evaluator on each arm's actual folded deployment head.

| Arm | Trainable parameters | What it may change |
| --- | ---: | --- |
| `restricted_adapter` | 16,384 | a 128-by-128 map inside the deployed subspace |
| `factorized_adapter` | 98,432 | same budget as `direct`, but routed through the 128-dimensional bottleneck |
| `direct_projection` | 98,432 | the 768-to-128 affine head itself |

`factorized_adapter` is the control that separates capacity from information: it spends
exactly as many trainable parameters as `direct_projection` while still seeing only what
the 128-dimensional bottleneck preserves.

## Result, five fixed seeds

Packed int8 128-dimensional codes, mean over seeds 0--4:

| Arm | packed mAP@R | packed Recall@1 |
| --- | ---: | ---: |
| base (frozen deployed head) | 0.555601 | 0.811176 |
| `restricted_adapter` | 0.581347 | 0.826403 |
| `factorized_adapter` | 0.583457 | - |
| `direct_projection` | **0.591528** | **0.832295** |

Every preregistered gate passes on every seed; `decision.passed` is `true` in all five
receipts. Worst case across the five seeds, against the registered threshold:

| Gate | Threshold | Worst observed |
| --- | --- | ---: |
| packed mAP@R gain, direct over restricted | >= 0.003 | 0.010111 |
| packed Recall@1 gain, direct over restricted | >= 0 | 0.005149 |
| one-sided paired class-cluster bootstrap lower bound | > 0 | 0.008897 |
| direct minus factorized packed mAP@R | >= 0.005 | 0.007813 |
| closure fraction, information-leading if below | < 0.60 | 0.2342 |

`direct_projection` beats `restricted_adapter` on 5 of 5 seeds, by 0.010180 mAP@R on
average, and beats the parameter-matched `factorized_adapter` by 0.008070. Against the
frozen base the direct head gains 0.035927 mAP@R and 0.021119 Recall@1.

## Reading

The parameter-matched control closes only 18.1% to 23.4% of the gap between the
restricted adapter and the direct projection. Capacity is therefore not the explanation.
What carries the effect is access to directions in the frozen 768-dimensional teacher
representation that the deployed 128-dimensional subspace has already discarded; once
they are gone, no amount of adaptation inside that subspace recovers them.

The practical consequence for this library is that the projection into the compact code
should be trained, not treated as a fixed frame with a learned adapter bolted on top.

Limits. One dataset, one split, one optimizer, one schedule, and a diagnostic split
rather than a held-out test. The gain is measured against this specific base head and
does not establish an absolute retrieval result or generality to other domains. A
parameter-matched control rules out capacity under *this* factorization; it does not
prove that no higher-capacity head could close the gap.

## Next boundary

Per the preregistration, a pass promotes direct projection to a fresh-dataset replication
and systems measurement. It explicitly does not authorize reuse of the already observed
SOP official test.

## Authorities

- Source revision: `61c95178a04ddcf73387283977e4924b48fa9254`
- Driver SHA-256: `cb5c165089a4af20fd3e530ea1cbb1435fedf8604b3046126329f6517116aae9`
- Source snapshot SHA-256: `6e315945ba4b005baabaa81f7051de23fef2ec8658db09298694ac92438f3471`
- Teacher snapshot SHA-256: `b0da9f6097646ffad78c21c84970751ae9a7da003e0eb2979e28f72009785264`
- Base checkpoint SHA-256: `9365e6135e6e44f826f99cba2734bbaccd973c8a0a211dc703291a6d3db426ac`
- Base parent receipt SHA-256: `df3064bbef5b53a0de15cd45dd03f8c9bc32ae5a82af8512bd5d39a5b933f5a5`
- Expected packed base mAP@R / Recall@1: `0.5556011035874895` / `0.8111758251034017`, matched.

Seed receipts, SHA-256:

```text
d635a3f708aced91a0639ebe0c0139179ab883f38dd639ff153e3a69ab3c1d25  seed0-complete.json
2ce6ed8faaf4d75bd20733fbd25b1a99a834957282a3710fdd6c246c4fdbc581  seed1-complete.json
5d58f04e3ddc3f8b5a42072691e4c1f85bd32321f620c70342fb5582d55a2b6f  seed2-complete.json
bf83811ad62e7ad1a5e2e82ba77d0ed59b3300e0bcabaa94d43138605d857a22  seed3-complete.json
7a5ca2ec4519940c2191854790e52c57536dbe902ed0573f2fcec1187e8718c4  seed4-complete.json
```

Retained on the GPU host at `~/runs/sop-projection-capacity-61c95178/`, with each arm's
checkpoint alongside its receipt. The runs executed on 2026-09-12 and were analysed here;
no training was repeated to produce this document.
