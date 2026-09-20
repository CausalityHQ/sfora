# Fixed PCA–whitening fusion diagnostic

The preregistered equal-RMS fusion is rejected. It does not combine the useful
geometry of PCA and within-class whitening into a better generic 128-byte code.

| Dataset | best constituent int8-128 mAP@R | fusion float256 | fusion float128 | fusion int8-128 | int8 delta vs best |
|---|---:|---:|---:|---:|---:|
| Cars | 0.846998 whitening | 0.824869 | 0.813100 | 0.812997 | -0.034001 |
| CUB | 0.719066 PCA | 0.720673 | 0.718668 | 0.718862 | -0.000204 |
| Pet | 0.870446 whitening | 0.868155 | 0.867843 | 0.867730 | -0.002716 |

The failure is already present in the uncompressed float256 representation on
Cars and Pet, so int8 quantization is not causal. Compression is additionally
harmful on Cars (−0.011769 mAP@R from float256 to float128), but tuning a fusion
weight after observing these datasets would violate the frozen no-sweep
protocol. This branch stops here.

The result is diagnostic and `claim_eligible=false`. It used only cached,
class-disjoint feature archives and completed in 3.15 seconds on the existing
GB10. No custom CUDA kernel or iterative training was used.

- Preregistration SHA-256:
  `14afc698d00597c17b58aeb17ff254ff24af126fbe6f3cfa00c7f437cfaf79dd`
- Executed script SHA-256:
  `353bdf21f0d6ba5aa6ead0c564cac60b9532106aec44c2cd371edde67986562c`
- Canonical result SHA-256:
  `c3fa28b4486aaf32ee86181209f1c1a3ed5559249a91e834b0fc8a84a8d2e0be`
- Canonical result:
  `docs/evidence/compact_metric/pca-whitening-fusion-v1/result.json`
