# Relational linear embedding compaction

Sfora's relational linear compressor learns a single shared, bias-free projection
from paired source and teacher embeddings. It matches teacher neighborhood
distributions on training items, then discards the teacher. It consumes no class
labels and applies the same projection to queries and gallery items.

The deployed representation is an `int8` vector plus a little-endian `float16`
inverse norm. A 64-dimensional embedding therefore occupies exactly 66 persistent
bytes. This is a storage format: implementations may restore rows to float32 in
memory for dense matrix multiplication, or score the packed codes directly. The
reference packed scorer converts codes to float32 for matrix multiplication; it
is an exact implementation of this wire contract, not a specialized integer ANN
kernel.

## Frozen evaluation

The preregistered DeepFashion In-Shop evaluation used three fixed seeds, a
train-only 64-dimensional basis, an identity-cluster bootstrap, and ordinary
cosine retrieval without test fitting or reranking. Against the 64-dimensional
PCA int8 control (`MAP@R=0.397063`, `R@1=0.665494`), all three relational-linear
arms passed:

The compressed source was `UNICOM-ViT-B/16` (archive SHA-256
`11e26db36b21aab995591c3fd83202982a148e93f3bf237440812414e9975711`),
and the training-only teacher was `UNICOM-ViT-L/14@336px` (archive SHA-256
`6eae13715e18d7eb99450bade5056538f8f08f1e9b550d0f24ee09e52bb25d0e`).
The sealed evaluation ran on CUDA on an aarch64 DGX host.

| Seed | int8 MAP@R | int8 R@1 | MAP improvement | familywise MAP lower bound |
|---:|---:|---:|---:|---:|
| 17 | 0.446343 | 0.714868 | +0.049280 | +0.045493 |
| 1729 | 0.446034 | 0.712829 | +0.048971 | +0.045215 |
| 65537 | 0.445771 | 0.714235 | +0.048708 | +0.044986 |

The 64-dimensional method retained 99.5–99.6% of the 128-dimensional PCA
control's MAP@R (`0.448061`) at approximately half its persistent item bytes. An
An earlier exploratory receipt reports improvement over its named baseline on an
identity-disjoint SOP validation split (`MAP@R=0.329449–0.334026` versus
`0.309024`). Its producer is not recoverable in this repository, its baseline is
not explicitly identified as PCA64, and its status is `cross-domain-stopped`.
It therefore motivates cross-domain replication but is not release evidence for
the exact 66-byte scorer.

The target-CPU paired benchmark used one thread, 12,612 gallery rows, 1,000 warmup
pairs, and 10,000 measured pairs. It timed source normalization, projection,
packing, packed cosine scoring, and top-256 selection. Relational linear p95 was
1,082,069 ns versus 1,079,477 ns for PCA64, a ratio of 1.00240 against the fixed
1.10 gate. Both arms used the same exact packed scorer.
The two pipelines are computationally identical apart from projection weights;
this benchmark detects accidental implementation regressions rather than a
method-specific speed advantage.

## Evidence authority and limits

- Quality receipt SHA-256:
  [`88c82a82dfac2e685e01505fd2c4e7b72961b9af29292b40930b3cf10b7f68e0`](evidence/relational_linear_compaction/relational-linear-inshop-v22.json).
- Latency receipt SHA-256:
  [`d16d8833cd39d77ab47ba204167029ec055765f6e908595c873c5a817eb2a9fd`](evidence/relational_linear_compaction/relational-linear-latency-v22.json).
- Deployment model SHA-256:
  [`c46d5c7eff99b4962b9491688ac9a1ad5d345ea1e2bc9c4bd7ae3bfc0b186521`](evidence/relational_linear_compaction/relational-linear-v22.sfora-rl1)
  (`SFORA-RL1`, 196,625 bytes).
- Cross-domain SOP receipt SHA-256:
  [`a8daaf5fe9585c9d74b67ea0a3f250c043f717d41005f6cf1063b271c0ce098d`](evidence/relational_linear_compaction/joint-relational-sop-v13.json).
- The quality, latency, and SOP receipts are explicitly claim-ineligible. The
  latency receipt does not report an
  end-to-end service percentile under concurrent load.
- The authenticated release evidence establishes the method on one
  image-retrieval domain. The stopped SOP receipt is exploratory cross-domain
  evidence only. This is not yet evidence of state-of-the-art end-to-end
  retrieval training, text retrieval, or universal teacher transfer.
- The nonlinear residual extension did not clear its fixed incremental gate and
  is not part of the public method.

## Recommended use

Fit on paired training embeddings from the source encoder that will be deployed
and a stronger teacher. Keep validation and retrieval-test identities disjoint
from fitting. Start with 64 dimensions and the frozen optimizer defaults; treat
dimension, temperature, and teacher choice as training-set-selected parameters,
not test-set tuning knobs. Each epoch shuffles training rows and uses complete
mini-batches; a final incomplete batch is intentionally omitted.
