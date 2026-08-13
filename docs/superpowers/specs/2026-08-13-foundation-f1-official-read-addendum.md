# Foundation F1 In-Shop Official-Read Addendum

**Status:** prospective. No official In-Shop query/gallery pixel has been read
under this addendum.

## 1. Authority and chronology

This addendum is the separately reviewed authority required by Task 8 of
`docs/superpowers/plans/2026-08-13-foundation-identity-disjoint-comparator.md`.
It binds the immutable train-only report committed as
`e39705a7579252a0cf0919689451e3d7265a2275` at
`docs/foundation_f1_identity_disjoint_inshop_train_only_report.json`:

- report SHA-256:
  `791cd1499327bd95abb5093d993a68c7192d44965af8669073bf35ad5b6ae066`;
- reviewed train-only source commit:
  `7c898354d0f897d5226098d06ee19349e21701a8`;
- report status: `CONTINUE`;
- decision SHA-256:
  `a3400169c6b94dbde2a2ecbb329a839ffba9edd43679587877133c5f3c83a9c8`;
- candidate/comparator R@1 points: `96.78731191541277` and
  `97.19398129320862`;
- exact quality gap: `-0.406669378` points;
- continuation basis: `quality_margin` only; `quality_within_point_four=false`
  and `cost_pareto_dominant=false`;
- official-test reads and published-metric audits: exact empty arrays.

The train-only result is not a SOTA result and is not a quality win. It only
authorizes the narrow official comparison below.

## 2. Exact one-read capability

Populate `docs/foundation_test_read_register.json` with exactly two records in
this order and retain its existing absolute receipt root:

The top-level key order remains exactly `schema_version`, `status`,
`receipt_root`, `records`; values remain
`foundation-test-read-register-v3`, `frozen`, and
`/home/riomus/group-learning/reports/generated/foundation_test_read_receipts`.
Each record key order is exactly `dataset`, `arm`, `model_revision`,
`checkpoint_sha256`, `metrics`, `purpose`, `permitted_evaluations`.

1. dataset `inshop`, arm `siglip2-base-patch16-256`, model revision
   `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`, checkpoint/weight SHA-256
   `6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863`;
2. dataset `inshop`, arm `inshop-pa-bninception-disjoint-seed2`, model revision
   and checkpoint SHA-256 both
   `2eb588846cde6846fbd1ca7f9894a60eb1491239f23e099dd2136bfe739fe08b`.

Each record has metrics in this exact order:
`recall_at_1`, `recall_at_10`, `recall_at_20`, `recall_at_30`,
`recall_at_100`, `map_at_r`; purpose
`registered_f1_quality_evaluation`; and `permitted_evaluations=1` as a builtin
integer. The contaminated control `inshop-pa-bninception-seed2` is absent and
must remain incapable of an official read.

Populate `docs/foundation_published_metric_register.json` with the exact
cross-product of those two arms and six metrics, arm-major and metric-minor in
the orders above. Every record has `native_value=null`, `tolerance=null`,
`provenance=repository_only`, and source text
`No independent published value exists for this exact checkpoint and protocol; repository evaluation only.`
No official value is inserted prospectively.
Its top-level key order remains exactly `schema_version`, `status`, `records`,
with values `foundation-published-metrics-v1` and `frozen`; every record key
order is exactly `arm`, `metric`, `native_value`, `tolerance`, `source`,
`provenance`.

## 3. Reviewed handoff

The authority implementation commit `H_OFFICIAL` must descend from this
addendum and may change only:

- `docs/foundation_test_read_register.json`;
- `docs/foundation_published_metric_register.json`; and
- `tests/test_foundation_pareto.py` for exact order, identity, report-binding,
  mutation, and contaminated-control rejection coverage.

Production source bytes must remain identical to train-only source
`7c898354d0f897d5226098d06ee19349e21701a8`. Before execution, an independent
review must find no Critical or Important defect and authenticate the committed
train-only report SHA/status/decision, the two exact register products, the
model-spec identities, the empty receipt root, and the absence of official
output and cache destinations.

## 4. One official process

From a fresh detached checkout at reviewed `H_OFFICIAL`, run exactly one
In-Shop `foundation-screen` process with `--allow-registered-test-read`. Use
validation seed `0`, validation fraction `0.2`, the two populated registers,
the same model/fixture/tolerance authorities, and a new cache directory and
report path named by `H_OFFICIAL`. Do not run SOP. Do not evaluate the
contaminated control officially. Do not rerun on a scientific result.

The exact command is the following literal array after replacing each
`${H_OFFICIAL}` with the independently reviewed 40-hex commit and no other
substitution:

```text
[
  "env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "CUDA_VISIBLE_DEVICES=0",
  "/home/riomus/foundation-f1-official-${H_OFFICIAL}/.venv/bin/python",
  "-I", "-B", "-m", "sfora.cli", "foundation-screen",
  "--dataset", "inshop",
  "--dataset-root", "/home/riomus/datasets/inshop_official_standard",
  "--model-specs", "/home/riomus/foundation-f1-official-${H_OFFICIAL}/docs/foundation_model_specs.json",
  "--fixture-authority", "/home/riomus/foundation-f1-official-${H_OFFICIAL}/docs/foundation_native_fixtures.json",
  "--tolerance-authority", "/home/riomus/foundation-f1-official-${H_OFFICIAL}/docs/foundation_metric_tolerances.json",
  "--published-register", "/home/riomus/foundation-f1-official-${H_OFFICIAL}/docs/foundation_published_metric_register.json",
  "--test-read-register", "/home/riomus/foundation-f1-official-${H_OFFICIAL}/docs/foundation_test_read_register.json",
  "--cache-dir", "/home/riomus/group-learning/reports/generated/foundation_f1_official_${H_OFFICIAL}_cache",
  "--report", "/home/riomus/group-learning/reports/generated/foundation_f1_official_${H_OFFICIAL}_inshop.json",
  "--validation-seed", "0", "--validation-fraction", "0.2",
  "--allow-registered-test-read"
]
```

Before launch, require:

- strict reload of the committed train-only report and exact SHA/status/decision;
- source and authority Git-blob/worktree equality;
- candidate and comparator checkpoint/model hashes;
- a real non-symlink mode-`0700` receipt root containing no receipt;
- official report, cache directory, and owned temporary paths absent;
- no competing foundation or GPU process; and
- the exact reviewed Python/Torch/NumPy/Transformers/CUDA environment.

The process must publish one no-clobber receipt per arm before loading official
pixels for that arm. Offline validation must require exactly two official read
rows and twelve repository-only metric audits in registered order, receipt
bytes and decision hashes matching, no contaminated-control row, and strict
recomputation of all retrieval metrics and geometry-selection relations.

## 5. Interpretation and next action

The official report is evidence for choosing the next research candidate, not
permission to claim SOTA. Preserve all candidate and comparator metrics and
costs. A candidate quality loss closes direct frozen-feature transfer and sends
the research goal to the already authorized cached-feature adapter lane. A
quality tie or win also proceeds to that lane because the current candidate is
slower and larger; a Pareto claim still requires the preregistered multi-seed
adapter experiment, statistical support, and measured training/inference/storage
costs. Any structural failure before official pixel loading may be repaired
prospectively; any completed official read is one-shot and immutable.
