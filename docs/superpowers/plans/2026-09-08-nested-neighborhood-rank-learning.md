# Nested Neighborhood Rank Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a generic end-to-end nested similarity encoder that combines Proxy-Anchor separation, asymmetric teacher-neighborhood preservation, and deployment-rank finishing in a 128-byte int8 representation.

**Architecture:** Reusable PyTorch components live in one focused library module and know nothing about SOP or filesystem layouts. A strict SOP runner adapts authenticated images and a train-only teacher snapshot into those components; a separate evaluator recomputes class-disjoint quality and promotion gates; a serving benchmark scans the first 32 bytes of the single int8-128 database code and reranks with that same code.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, pytest, Ruff, mypy, existing Sfora atomic-publication and retrieval-metric utilities.

**Spec:** `docs/superpowers/specs/2026-09-08-nested-neighborhood-rank-learning-design.md`

## Global Constraints

- Keep dataset parsing, filesystem authority, CLI handling, and evidence publication out of the reusable method module.
- Use only official-training identities for optimization and development selection; the official test split is unavailable until confirmation passes.
- Use exactly widths `(32, 128)` for primary arms, batch size 128, 32 identities by four images, and the frozen loss weights and temperatures from the spec. The PA-768 and nested 768-to-128 S2SD controls are explicit seed-17 exceptions defined by the spec.
- Every artifact is no-clobber, canonical, source-bound, and `claim_eligible=false`.
- Do not add or promote an int4 path.
- Preserve configured Git identity and add no AI attribution.

---

### Task 1: Generic nested head and losses

**Files:**
- Create: `src/sfora/nested_neighborhood_rank.py`
- Create: `tests/test_nested_neighborhood_rank.py`

**Interfaces:**
- Produces: `NestedRankConfig`, `NestedRankHead`, `nested_proxy_anchor_loss`, `asymmetric_neighborhood_loss`, and `smooth_ap_loss`.
- Consumes: finite FP32 encoder features, integer labels, and aligned frozen teacher features.

- [ ] **Step 1: Write failing configuration and head tests**

```python
def test_nested_head_emits_independently_normalized_prefixes() -> None:
    config = NestedRankConfig(
        input_dim=8, hidden_dim=16, output_dim=8, widths=(4, 8), class_count=4
    )
    head = NestedRankHead(config)
    result = head(torch.randn(6, 8, dtype=torch.float32))
    assert tuple(result) == (4, 8)
    for values in result.values():
        torch.testing.assert_close(values.norm(dim=1), torch.ones(6))
```

Use a test-specific config with widths `(4, 8)` so the fixture is coherent.
Mutation cases must reject bool-as-int dimensions, unordered widths, nonfinite
inputs, wrong input width, and zero-norm rows.

- [ ] **Step 2: Run the focused RED**

Run: `.venv/bin/pytest -q tests/test_nested_neighborhood_rank.py -k 'config or head'`

Expected: import failure for `sfora.nested_neighborhood_rank`.

- [ ] **Step 3: Implement the configuration and head**

```python
@dataclass(frozen=True, slots=True)
class NestedRankConfig:
    input_dim: int
    hidden_dim: int = 1024
    output_dim: int = 128
    widths: tuple[int, ...] = (32, 128)
    class_count: int = 1

class NestedRankHead(nn.Module):
    def forward(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        dense = self.skip(features) + self.residual_scale * self.output(
            F.gelu(self.norm(self.hidden(features)))
        )
        return {width: F.normalize(dense[:, :width], dim=1) for width in self.config.widths}
```

Initialize `residual_scale` to exact zero and validate concrete types, shapes,
dtypes, devices, and finiteness before computation.

- [ ] **Step 4: Write and run loss REDs**

Test exact formula agreement against small scalar references, exclusion of the
self diagonal, stop-gradient teacher behavior, label/proxy shape rejection,
finite gradients, exact width weighting, tie behavior, and Smooth-AP positive
inventory rejection.

Mutation-lock the normative spec constants and geometry: Proxy-Anchor scale
32/margin 0.1/mean reductions/shared raw proxy prefixes, proxy initialization,
teacher and student-key stop-gradient, immutable-sample masks, cosine scores,
and Smooth-AP competitor/positive masks. Reject any query with no valid key and
require finite forward/backward behavior under diagonal and same-label masks.
No loss default may remain implicit.

Run: `.venv/bin/pytest -q tests/test_nested_neighborhood_rank.py -k 'loss'`

Expected: failures at missing loss functions.

- [ ] **Step 5: Implement minimal loss functions and verify**

```python
teacher_prob = F.softmax(teacher_similarity.masked_fill(~off_class_mask, -torch.inf) / temperature, dim=1)
student_log_prob = F.log_softmax(student_similarity.masked_fill(~off_class_mask, -torch.inf) / temperature, dim=1)
safe_student_log_prob = torch.where(off_class_mask, student_log_prob, 0.0)
loss = -(teacher_prob.detach() * safe_student_log_prob).sum(dim=1).mean()
```

Run: `.venv/bin/pytest -q tests/test_nested_neighborhood_rank.py`

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/sfora/nested_neighborhood_rank.py tests/test_nested_neighborhood_rank.py
git commit -m "Add generic nested neighborhood rank losses"
```

### Task 2: Deterministic class-disjoint protocol

**Files:**
- Create: `src/sfora/nested_rank_protocol.py`
- Create: `tests/test_nested_rank_protocol.py`

**Interfaces:**
- Produces: `ClassDisjointFold`, `class_disjoint_fold`, `shared_optimization_rows`, `identity_balanced_schedule`, `cluster_bootstrap_lower_bound`.
- Consumes: immutable sample IDs, class IDs, seed, batch size, and images per identity.

- [ ] **Step 1: Write protocol REDs**

```python
fold = class_disjoint_fold(sample_ids, labels, seed=17)
assert not set(labels[fold.optimization]) & set(labels[fold.validation])
assert len(fold.optimization) + len(fold.validation) == len(labels)
assert tuple(fold.validation_queries) == tuple(fold.validation)
```

Mutation-lock exact equality with the committed 80/20 class partition for all
three seeds, all-validation
leave-one-out query/gallery membership, teacher-centroid nearest-identity batch
construction, repeated sparse-identity draws, and deterministic replay.

Run: `.venv/bin/pytest -q tests/test_nested_rank_protocol.py`

Expected: import failure for `sfora.nested_rank_protocol`.

- [ ] **Step 2: Implement the protocol**

Call `deterministic_class_partition(..., fit_fraction=0.8)` directly. Sampling
must reserve eight unique anchors, process them in sampled order,
and select each anchor's three nearest globally unused identities with class-ID
ties. Compute neighbors with bounded matrix-multiplication blocks and retain
only the exact nearest 31 classes per source; mutation-lock this against a full
scalar sort. Compute the temperature-preflight membership as the intersection
of optimization rows across all three registered folds. Then cycle a
deterministic per-class permutation only after exhausting
physical members. Mutation-test colliding neighbor lists for exactly 32 unique
labels and four rows per label. Matched arms consume identical schedules.
Use domain-separated PCG64 streams for the anchor permutation and each
class-image permutation. Consume anchors in order; at permutation rollover,
skip identities already selected for the current batch until eight unique
anchors are collected. Mutation-test a non-multiple-of-eight class count with a
deliberate rollover collision.
Bootstrap exactly 10,000 original-class draws with PCG64 seed 17, recomputing
each replicate as sampled class-difference sums divided by sampled query counts,
and use the fixed nearest-rank 5th percentile.

- [ ] **Step 3: Verify and commit Task 2**

```bash
.venv/bin/pytest -q tests/test_nested_rank_protocol.py
git add src/sfora/nested_rank_protocol.py tests/test_nested_rank_protocol.py
git commit -m "Add deterministic nested-rank protocol"
```

### Task 3: Authenticated SOP end-to-end runner

**Files:**
- Create: `scripts/build_sop_nnrl_train_snapshot.py`
- Create: `tests/test_build_sop_nnrl_train_snapshot.py`
- Create: `scripts/train_sop_nested_neighborhood_rank.py`
- Create: `tests/test_train_sop_nested_neighborhood_rank.py`
- Reuse: `scripts/export_unicom_sop_embeddings.py`

**Interfaces:**
- Consumes: official SOP training root, exact UNICOM B/16 checkpoint and checkout, exact train-only L/14 snapshot, source commit, arm, split seed, and output directory.
- Produces: immutable checkpoint, run receipt, and canonical terminal result; it does not evaluate validation images.

- [ ] **Step 0: Preflight the neighborhood hypothesis without GPU training**

Replay exactly 1,000 seed-17-scheduled batches over classes in the intersection
of all three folds' optimization memberships from the authenticated train-only
teacher snapshot. For temperatures `(0.05, 0.10, 0.20)`, mask repeated sample IDs and
same-label rows, then recompute target entropy/effective support. Select the
smallest temperature meeting the spec's median/5th-percentile gates. If none
passes, publish a redundant-objective receipt and run only Proxy-Anchor-128 and
Proxy-Anchor-768; record neighborhood, combined, and S2SD as not run. Also bind
the sealed source/teacher validation controls used by the later premise check.

- [ ] **Step 1: Write CLI and authority REDs**

Require explicit
`--arm {proxy-anchor,neighborhood,combined,proxy-anchor-768,s2sd-768-to-128}`,
`--split-seed {17,1729,65537}`, exact input hashes, `--output`, and
`--execute-nnrl`. Reject unknown, duplicate, missing, test-data, class-name,
network, and overwrite flags. Mutate every archive identity and source hash.

Run: `.venv/bin/pytest -q tests/test_train_sop_nested_neighborhood_rank.py -k 'args or authority'`

Expected: import or missing-interface failure.

- [ ] **Step 2: Implement strict loading and source closure**

Build the official-training-only snapshot in a separate process by
authenticating the frozen source archive, extracting every official-training
ID/label/path and teacher row but no official-test row, and publishing a
canonical descriptor plus array. The trainer uses a train-only record parser,
indexes only its seed's optimization rows, and has no parameter or code path for
source-archive test arrays. The evaluator alone may index validation rows.
Require exact ordered identity equality and mutation-test that test members are
never accessed. Hash every loaded Sfora source file.

- [ ] **Step 3: Write training-loop REDs**

Use tiny injected encoder/dataset fixtures to prove exact optimizer groups,
ten epochs, the frozen update count and augmentation recipe, 32x4 batches,
FP16 boundary with dynamic gradient scaling, the preflight-bound temperature,
arm-specific terms, frozen encoder BatchNorm statistics, constant learning
rates, no warm-up/decay/clipping, exact AdamW betas/epsilon and
weight-decay exclusions, remaining-module train/eval modes, drop-path zero, no
early stop, deterministic replay, nonfinite failure receipts, the exact UNICOM
feature tap, and one model forward per batch.

For `proxy-anchor-768`, require a one-width 768 head and only `L_PA768`. For
`s2sd-768-to-128`, retain the primary 32/128 head and require the exact spec
objective with the normalized 768-dimensional encoder output as stop-gradient
target. Detach target probabilities and student keys, but require nonzero
query-path gradients through the 128 head into the encoder. Both are mandatory in
the five-arm seed-17 execution manifest but do not enter primary promotion.

- [ ] **Step 4: Implement the training loop**

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    dense = F.normalize(encoder(images), dim=1)
student = head(dense.float())
loss = arm_loss(student, labels, teacher_rows)
```

Use AdamW with backbone LR `1e-5`, head/proxy LR `1e-3`, betas
`(0.9,0.999)`, epsilon `1e-8`, and constant rates. Apply weight decay `1e-4`
only to encoder/head weights, not proxies, biases, or normalization parameters;
use no warm-up, decay, gradient clipping, or drop-path.
Publish via existing descriptor-backed no-clobber helpers and fsync the output
directory.

- [ ] **Step 5: Verify and commit Task 3**

```bash
.venv/bin/pytest -q tests/test_train_sop_nested_neighborhood_rank.py
git add scripts/train_sop_nested_neighborhood_rank.py tests/test_train_sop_nested_neighborhood_rank.py
git commit -m "Add authenticated SOP nested-rank trainer"
```

### Task 4: Evaluation, promotion, and finish

**Files:**
- Create: `scripts/evaluate_sop_nested_neighborhood_rank.py`
- Create: `tests/test_evaluate_sop_nested_neighborhood_rank.py`
- Modify: `scripts/train_sop_nested_neighborhood_rank.py`
- Modify: `tests/test_train_sop_nested_neighborhood_rank.py`

**Interfaces:**
- Consumes: authenticated arm results and checkpoints plus official-training validation image IDs, labels, and paths.
- Produces: recomputed per-query metrics, bootstrap evidence, promotion decision, and an optional two-epoch finish result.

- [ ] **Step 1: Write evaluator REDs**

Construct coherent ranked-ID fixtures and mutate query membership, gallery
membership, self-exclusion, ranks, AP@R, Recall@1, bootstrap rows, arm identity,
and decision. Require all validation rows as leave-one-out queries and the exact
`+0.005`, positive lower-bound, and `-0.001` Recall@1 gates. Bind the frozen
teacher/source premise controls and the matched 768-coordinate Proxy-Anchor and
S2SD-style self-distillation controls on seed 17.

- [ ] **Step 2: Implement independent recomputation**

The evaluator must derive AP@R and recalls from ranked sample IDs and labels,
then derive all aggregates, deltas, bootstrap bounds, and decisions. It must not
trust copied scalar summaries. If Proxy-Anchor-128 exceeds the external teacher,
the receipt must label the combined comparison as weaker-teacher regularization
rather than stronger-teacher transfer.

For epoch-10 and finished checkpoints, run a no-gradient, eval-mode inference
pass over validation images, without loading validation teacher rows, and emit
the immutable ranked IDs that metric recomputation consumes. Mutation-test that
the optimization loop never samples validation images and that this evaluator
never opens teacher-validation or official-test arrays.

- [ ] **Step 3: Add the fixed finish phase under tests**

Resume both a promoted combined checkpoint and its matched Proxy-Anchor
checkpoint for two additional epochs with fresh optimizer state and
`0.25 * ProxyAnchor128 + SmoothAP128(temperature=0.01)`. Reject changed parent,
seed, schedule, weights, widths, or test access. Recompute paired mAP@R,
bootstrap, and Recall@1 gates before and after the identical finish.

- [ ] **Step 4: Verify and commit Task 4**

```bash
.venv/bin/pytest -q tests/test_train_sop_nested_neighborhood_rank.py tests/test_evaluate_sop_nested_neighborhood_rank.py
git add scripts/train_sop_nested_neighborhood_rank.py scripts/evaluate_sop_nested_neighborhood_rank.py tests/test_train_sop_nested_neighborhood_rank.py tests/test_evaluate_sop_nested_neighborhood_rank.py
git commit -m "Add nested-rank promotion and finish"
```

### Task 5: Int8 serving and two-stage benchmark

**Files:**
- Create: `src/sfora/nested_rank_serving.py`
- Create: `tests/test_nested_rank_serving.py`
- Create: `scripts/benchmark_nested_rank_serving.py`
- Create: `tests/test_benchmark_nested_rank_serving.py`

**Interfaces:**
- Produces: symmetric per-row int8 encoding, float-query/int8-database scores, same-code int8-prefix candidate scan, exact rerank, and canonical benchmark receipt.

- [ ] **Step 1: Write codec and top-k differential REDs**

Compare scalar reference against optimized scoring for random values, ties,
subnormals, saturation, zero rows, nonfinite inputs, and chunk boundaries.
Require exact top-k IDs and deterministic `(score desc, row ordinal asc)` ties.

- [ ] **Step 2: Implement codec and scorer**

Store 128 signed bytes plus one explicit finite positive float16 scale per row.
The prefix scan reads the first 32 bytes, derives the database prefix norm while
scanning, retains exactly `min(4096, gallery_size)` candidates, and reranks with
all 128 bytes. Do not store a duplicate prefix or decode the complete database
into a temporary float matrix.

- [ ] **Step 3: Implement benchmark authority and gates**

Require 1,000 warm-ups, at least 10,000 raw latency samples, single-query
concurrency, a one-million-row gallery, top-100 output, named hardware,
p50/p95/p99 recomputation, accounting for every resident representation and
scratch buffer, top-k implementation equality, at least 3.5x memory reduction,
at least 1.5x throughput, and no p95 regression. Independently compare actual
two-stage rankings with exhaustive int8 and float32 on the full SOP gallery at
both 4,096 candidates and the fraction-matched 248 candidates; gate
approximate-search loss at 0.003 mAP@R and 0.001 Recall@1. Construct the
one-million-row performance gallery only with the spec's deterministic
hash-order cycling and counter-based saturated integer perturbation, and bind
its digest. A failed two-stage gate emits a valid rejection result.

- [ ] **Step 4: Verify and commit Task 5**

```bash
.venv/bin/pytest -q tests/test_nested_rank_serving.py tests/test_benchmark_nested_rank_serving.py
git add src/sfora/nested_rank_serving.py scripts/benchmark_nested_rank_serving.py tests/test_nested_rank_serving.py tests/test_benchmark_nested_rank_serving.py
git commit -m "Add nested-rank serving benchmark"
```

### Task 6: Repository assurance and monitored DGX screen

**Files:**
- Modify: `docs/superpowers/plans/2026-09-08-nested-neighborhood-rank-learning.md`
- Create after science: `docs/evidence/nested_neighborhood_rank/README.md`
- Create after science: `docs/evidence/nested_neighborhood_rank/sop-nnrl-screen-v1.json`

**Interfaces:**
- Consumes: Tasks 1-5 and exact authenticated SOP/UNICOM inputs.
- Produces: a verified source commit and one bounded seed-17 five-arm screen.

- [ ] **Step 1: Run final repository assurance once**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy src scripts
python3 -m compileall -q src scripts tests
git diff --check
```

- [ ] **Step 2: Obtain independent release review**

Start one context-complete Astra review and one context-complete Fable review of
the exact diff, spec, plan, gates, and failure semantics. Resolve demonstrated
Critical or Important findings with focused RED/GREEN tests, then rerun only the
affected layer and one final assurance gate.

- [ ] **Step 3: Commit and push the verified implementation**

```bash
git add -f docs/superpowers/specs/2026-09-08-nested-neighborhood-rank-learning-design.md docs/superpowers/plans/2026-09-08-nested-neighborhood-rank-learning.md
git add src scripts tests
git commit -m "Add nested neighborhood rank learning"
git push origin HEAD:master
test "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/master | cut -f1)"
```

- [ ] **Step 4: Run the bounded seed-17 screen**

Deploy the exact clean commit to DGX. Run all five seed-17 arms serially with
one original process each, a two-hour arm cap, ten-minute progress stop, available host
memory floor of 8 GiB, swap-growth cap of 256 MiB, and the established sustained
PSI stop. Preserve terminal evidence and never replace a scientific result.

- [ ] **Step 5: Apply the frozen decision and final-readout rule**

If the combined arm passes, run confirmation seeds 1729 and 65537 and finish
both combined and matched controls. Otherwise stop this objective and publish
its exact failure classification. Confirmation means use seeds 1729 and 65537
only; seed 17 remains the separately reported screen. After confirmation, train
combined and Proxy-Anchor once each on the complete official training partition
using seed 17. Use 10+2 epochs only if the matched post-finish gates pass and
the finished combined checkpoint versus its own epoch-10 checkpoint has a
nonnegative one-sided mAP@R lower bound and Recall@1 delta at least -0.001 on
every seed; otherwise retain the passing epoch-10 anchored schedule. One
evaluator opens the official test
partition once and scores both frozen checkpoints with standard self-exclusion;
there is no checkpoint selection or post-readout refit. Only a confirmed method
receives this readout and the serving benchmark.

- [ ] **Step 6: Freeze transfer adapters before generic claims**

Add In-Shop and CUB protocol adapters, matched controls, and the numerical gates
from the spec before running either dataset. The SOP-only milestone remains
claim-ineligible. Text and semantic class-name proxies are explicitly outside
this release.

- [ ] **Step 7: Verify and publish evidence**

Independently recompute canonical bytes, source/input hashes, per-query metrics,
bootstrap bounds, decision, runtime, and resource observations. Add the exact
result plus a concise README, rerun documentation/diff checks, commit, push, and
verify remote equality.
