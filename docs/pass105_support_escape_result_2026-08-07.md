# Pass 105 — activation-support escape diagnostic

## Measurement

To test the remaining off-support activation hypothesis, the trained official
Proxy Anchor In-Shop seed-0 checkpoint was run on the corrected official-pixel
train, query, and gallery partitions. For each of the 1,024 pre-head channels,
the train split's min/max envelope was fitted first; query/gallery feature-map
pooled values were then scored against that frozen envelope. No test labels were
used in the diagnostic.

| split | values outside train envelope | rows with any outside channel |
|---|---:|---:|
| query | 1.7652e-05 | 0.01779 |
| gallery | 1.6183e-05 | 0.01625 |

The result is effectively on-support. Only about 1.6–1.8% of rows touch an
outside channel, and only roughly 17 values per million leave the train
extrema.

## Decision

The proposed off-support activation/architecture escape is **killed at Gate 1**.
Changing an activation outside the realized train support cannot explain a
zero-shot retrieval gain under this operating point. The result is a useful
negative measurement, not evidence that all activation functions are
impossible. Artifact: `reports/generated/inshop_support_escape_seed0.json`.
No candidate training run was authorized.
