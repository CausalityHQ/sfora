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
control's MAP@R (`0.448061`) at approximately half its persistent item bytes.

The independently sealed Stanford Online Products evaluation reused the same
recipe and seeds without SOP-result-driven tuning. Against PCA64 int8
(`MAP@R=0.357638`, `R@1=0.635946`), all three learned 64-dimensional arms passed:

| Seed | int8 MAP@R | int8 R@1 | MAP improvement | familywise MAP lower bound | familywise R@1 lower bound |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.372987 | 0.654177 | +0.015350 | +0.013618 | +0.015400 |
| 1729 | 0.372852 | 0.654160 | +0.015214 | +0.013528 | +0.015429 |
| 65537 | 0.372878 | 0.653945 | +0.015240 | +0.013569 | +0.015234 |

SOP used the official 59,551-image training split and 60,502-image symmetric
leave-self-out test split. Labels were evaluation-only. The PCA128 int8 control
reached `MAP@R=0.393314`, and the full source and teacher float controls reached
`0.420759` and `0.476360`, respectively. These ceilings show that 64-dimensional
compression still leaves quality headroom; the result establishes repeatable
compression gains rather than absolute retrieval state of the art.

The target-CPU paired benchmark used one thread, 12,612 gallery rows, 1,000 warmup
pairs, and 10,000 measured pairs. It timed source normalization, projection,
packing, packed cosine scoring, and top-256 selection. Relational linear p95 was
1,086,693 ns versus 1,083,492 ns for PCA64, a ratio of 1.00295 against the fixed
1.10 gate. Both arms used the same exact packed scorer.
The two pipelines are computationally identical apart from projection weights;
this benchmark detects accidental implementation regressions rather than a
method-specific speed advantage.

On SOP's 60,502-row gallery, the same one-thread paired operation measured
3,231,472 ns p95 for the learned map versus 3,224,272 ns for PCA64, a ratio of
1.00223 across 10,000 measured pairs. This is again a controlled kernel
comparison, not concurrent end-to-end service latency.

## Evidence authority and limits

- Quality receipt SHA-256:
  [`d2d4aab49482c53a53de1344d38efd4cc95e48359e9ad64ccb87be392b4fd4eb`](evidence/relational_linear_compaction/relational-linear-inshop-v24.json).
- Latency receipt SHA-256:
  [`a6c26242b408b98b6f7d24e8d5ffe6ca191d4cc4189943281c5679eb2e2ba477`](evidence/relational_linear_compaction/relational-linear-latency-v24.json).
- Deployment model SHA-256:
  [`c46d5c7eff99b4962b9491688ac9a1ad5d345ea1e2bc9c4bd7ae3bfc0b186521`](evidence/relational_linear_compaction/relational-linear-v24.sfora-rl1)
  (`SFORA-RL1`, 196,625 bytes).
- Cross-domain SOP receipt SHA-256:
  [`a8daaf5fe9585c9d74b67ea0a3f250c043f717d41005f6cf1063b271c0ce098d`](evidence/relational_linear_compaction/joint-relational-sop-v13.json).
- Sealed official SOP quality receipt SHA-256:
  [`5125a8e0bfe242345257ca172ea62c92289d3585e7185e5b11d31bf92da2110c`](evidence/relational_linear_compaction/sop/sop-relational-linear-evaluation-v1.json).
- Sealed official SOP latency receipt SHA-256:
  [`d18e6dedf55746f97271a320e1b7c9a87e498e147ea81c549b4a5229ee5a498a`](evidence/relational_linear_compaction/sop/sop-relational-linear-latency-v1.json).
- Sealed official SOP deployment model SHA-256:
  [`dad73cd0dd4662f3098899d862d5ef42dde0a2f2c81302cae3c08740f72ec100`](evidence/relational_linear_compaction/sop/sop-relational-linear-seed17.sfora-rl1).
- The quality, latency, and SOP receipts are explicitly claim-ineligible. The
  latency receipt does not report an
  end-to-end service percentile under concurrent load.
- The authenticated release evidence establishes repeatable gains on two
  image-retrieval domains. This is not yet evidence of state-of-the-art
  end-to-end retrieval training, text retrieval, or universal teacher transfer.
- The nonlinear residual extension did not clear its fixed incremental gate and
  is not part of the public method.

## Recommended use

Fit on paired training embeddings from the source encoder that will be deployed
and a stronger teacher. Keep validation and retrieval-test identities disjoint
from fitting. Start with 64 dimensions and the frozen optimizer defaults; treat
dimension, temperature, and teacher choice as training-set-selected parameters,
not test-set tuning knobs. Each epoch shuffles training rows and uses complete
mini-batches; a final incomplete batch is intentionally omitted.
