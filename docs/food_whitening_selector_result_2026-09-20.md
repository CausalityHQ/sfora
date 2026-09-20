# Food101 whitening-initialized compact selector

## Result

A selector frozen before external scoring chose a within-class-whitened
initialization for the existing compact affine learner on a class-disjoint
Food101 protocol. The persistent representation remains the same exact signed
int8 128-dimensional code (128 bytes/item); whitening changes fitting only.

The authenticated UNICOM ViT-L/14@336 official-test feature archive contains
25,250 rows from 101 classes. Classes were ordered by
`SHA256("food-whitening-selector-v1:" + decimal_label)`: the first 50 classes
(12,500 rows) were used for fitting and fit-only cross-validation, and the
remaining 51 classes (12,750 rows) were scored once.

Fit-only three-fold results were:

| Candidate | mAP@R | Recall@1 |
| --- | ---: | ---: |
| PCA int8-128 | 0.798497 | 0.970800 |
| PCA-initialized learned int8-128 | 0.770086 | 0.966880 |
| **whitening-initialized learned int8-128** | **0.808696** | **0.969440** |

The frozen rule first selected PCA as the incumbent, because the existing
learned arm lost to PCA. It then promoted whitening-initialized learning because
its mAP@R exceeded the incumbent by at least 0.003 while its Recall@1 loss was
less than the registered 0.003 tolerance.

One-shot external results were:

| Representation | Persistent bytes/item | mAP@R | Recall@1 |
| --- | ---: | ---: | ---: |
| PCA int8-128 | 128 | 0.723156 | **0.942118** |
| **whitening-initialized learned int8-128** | **128** | **0.744477** | 0.938902 |
| frozen teacher float32-768 | 3,072 | 0.757123 | 0.945804 |

The selected arm gained **0.021320 mAP@R** over the same-byte incumbent. A
paired, query-weighted 10,000-draw evaluation-class bootstrap interval was
`[0.010563, 0.032178]`. Recall@1 fell by **0.003216** (41 fewer correct
top-one queries), exceeding the selection tolerance externally by 0.000216.
The preregistration did not define an external pass gate, so this is not a
protocol failure, but it prevents any claim that the Recall@1 tolerance
generalizes to unseen classes.

## Verification

The original deterministic run finished in 159.09 seconds on an NVIDIA GB10.
A separate deterministic refit reproduced the selected encoder SHA-256
`22842f0a4877898c80dfdc1550091e6e6eee435aaeb36791d41dc1b2e39f9096`.
An independent NumPy full-ranking replay reproduced Recall@1 exactly and mAP@R
to `1.1e-16` absolute difference from the GPU scorer.

After the one-shot result was sealed, a labelled post-hoc mechanism audit
scored the arms that the preregistration had not exposed externally:

| Post-hoc arm | mAP@R | Recall@1 |
| --- | ---: | ---: |
| PCA int8-128 | 0.723156 | **0.942118** |
| PCA-initialized learned int8-128 | 0.710098 | 0.939373 |
| whitening-initialized learned int8-128 | 0.744477 | 0.938902 |
| **within-class whitening int8-128** | **0.761514** | 0.939922 |

Pure whitening is therefore the best mAP@R arm on this split and slightly
exceeds the float-768 teacher's 0.757123 mAP@R, while retaining the same
128-byte code. This comparison is post-hoc and cannot be promoted to the
preregistered result. A separate fit-only fold audit nevertheless found
whitening-only at 0.828520 mAP@R / 0.970960 Recall@1 versus PCA at
0.798497 / 0.970800. Had this arm been in the frozen candidate set, the same
+0.003/no-R1-loss rule would have selected it before external scoring.

The post-hoc selected-versus-PCA Recall@1 class-bootstrap interval was
`[-0.006118, -0.000392]`, confirming a small but real top-one tradeoff rather
than sampling noise under this conditional analysis. The canonical per-query
audit arrays remain on the evidence host with SHA-256
`1b79c31b121c90dfef3986f2c3b740c6e7958c764debbb8fc6d1a6e709d16a0a`.

Authorities:

- feature archive SHA-256:
  `277c192d91ae05ec5389f5cb70e43e5b88d8898140a2a06c64f6d8961e9893b5`;
- preregistration SHA-256:
  `13fa79be9d7fe78a32937a1113a9493b64c85805a73705091c2d43568e02de32`;
- executed probe SHA-256:
  `3b1a6836339556bfdae554845dd372d817e4998a3c6f3defd72f66473db886a8`;
- whitening core SHA-256:
  `245967014642b76aff74625ac5b40de48690d7dbe4d51bb6084fa559f177cea2`;
- canonical result SHA-256:
  `6a325ae2295a729373bc6e899ae21a03c072af20896b6c0c0c1df165f9d7b49c`;
- independent replay SHA-256:
  `06d8282f51661f95909ea19fd00cfaee780cd1faa9ebe64a527f1448d9ecf4ec`.
- post-hoc mechanism audit SHA-256:
  `c6e05f0cdceb2a327186871a8b9aaa3511150b8c5068b724babec0b034349017`;
- post-hoc fit-only CV audit SHA-256:
  `9f54a5c341614143aa2570f3679c2c6702a7da7ccaac8fa3909afb1f24d7fb31`.

The retained files are under
`docs/evidence/compact_metric/food-whitening-selector-v1/`.

## Interpretation and decision

Food101 had already appeared in a frozen cross-domain transfer screen, so this
is not a previously unseen domain or an official Food101 benchmark result. It
is a prospective evaluation of a newly frozen selector on class-disjoint data
within an already observed domain. The fit partition also comes from the
official test feature archive. The receipt therefore remains
`claim_eligible=false`.

The combined panel supports an **experimental opt-in** four-family policy:
PCA, PCA-initialized learning, within-class shrinkage whitening, and
whitening-initialized learning. The Food whitening-only arm was added after the
one-shot result, so this policy still needs a genuinely prospective gate.
It does not justify changing the generic default, claiming universal
whitening, or claiming Pareto dominance. Production integration must use an
explicit initializer argument rather than the probe's module-global
monkeypatch, expose the selected family and fit-fold evidence, and preserve the
existing affine/int8 serving format. Because initialization also changes the
frozen hard-negative mining geometry and anchor targets, the method should be
described as a whitening-initialized recipe rather than a pure optimizer
initialization ablation.

No custom CUDA kernel is justified: fitting and evaluation completed in under
three minutes, while serving remains the unchanged single affine projection,
normalization, and int8 rounding path already measured separately.
