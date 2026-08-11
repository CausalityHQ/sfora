# In-Shop Reciprocal Re-ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and falsify a deterministic, training-free reciprocal-neighborhood re-ranker on frozen In-Shop Proxy Anchor embeddings.

**Architecture:** A small reusable NumPy module computes exact blockwise cosine top-k neighbors, query-specific reciprocal sets, Jaccard structural agreement, and a blended score. A CLI loads frozen `.npz` embeddings, selects hyperparameters only on a deterministic subset of training identities, evaluates untouched query/gallery splits, and writes a compact JSON report. Raw cosine is always reported beside re-ranked recall.

**Tech Stack:** Python 3.12+, NumPy, pytest, standard-library JSON/argparse/hashlib.

## Global Constraints

- No model training and no dependency on CUDA or nondeterministic GPU kernels.
- Never use test labels to choose `k`, candidate depth, or blend weight.
- Preserve query/gallery embeddings and labels byte-for-byte; outputs go to a new JSON file.
- The method is an engineering re-ranking baseline, not a novelty claim: k-reciprocal re-ranking and In-Shop re-ranking have prior art.
- Stop the research branch if the published Proxy Anchor checkpoint gains less than 0.15 percentage points in Recall@1.
- Compare re-ranked methods only with their own unre-ranked scores; never compare re-ranked ours against plain published numbers.

---

### Task 1: Exact reciprocal-neighborhood scorer

**Files:**
- Create: `src/sfora/reciprocal_reranking.py`
- Create: `tests/test_reciprocal_reranking.py`

**Interfaces:**
- Produces: `cosine_topk(queries, gallery, k, block_size) -> tuple[np.ndarray, np.ndarray]`
- Produces: `gallery_reciprocal_sets(gallery, k, block_size) -> tuple[np.ndarray, ...]`
- Produces: `rerank_queries(queries, gallery, *, k, candidate_depth, blend, block_size) -> RerankResult`
- `RerankResult` contains raw and re-ranked indices/scores without labels.

- [ ] **Step 1: Write failing synthetic tests**

Create tests with normalized 2-D vectors that verify exact top-k order, self-exclusion in the gallery graph, reciprocal-set membership, Jaccard values, deterministic tie-breaking by gallery index, blend boundaries (`blend=0` equals cosine), and rejection of non-finite/non-float/rank-mismatched inputs.

- [ ] **Step 2: Run the scorer tests and observe RED**

Run: `pytest -q tests/test_reciprocal_reranking.py`

Expected: collection failure because `sfora.reciprocal_reranking` does not exist.

- [ ] **Step 3: Implement the minimal exact scorer**

Use blockwise `queries @ gallery.T`, `np.argpartition`, and a final lexicographic sort by `(-score, gallery_index)`. For a query `q`, a top-k gallery candidate `g` is reciprocal when `cos(q,g)` is at least the k-th non-self gallery similarity threshold of `g`. Build gallery reciprocal sets by applying the same threshold relation to gallery top-k edges. Re-rank only the original top `candidate_depth` candidates using

```python
score(q, g) = (1.0 - blend) * cosine(q, g) + blend * jaccard(R_k(q), R_k(g))
```

The result contains only this top candidate window, which is sufficient for the
registered Recall@1 falsifier and avoids materializing a 14,218 x 12,612 full
ranking matrix.

- [ ] **Step 4: Run scorer tests and observe GREEN**

Run: `pytest -q tests/test_reciprocal_reranking.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the scorer**

```bash
git add src/sfora/reciprocal_reranking.py tests/test_reciprocal_reranking.py
git commit -m "add exact reciprocal retrieval reranking"
```

### Task 2: Train-only tuner and In-Shop evaluator

**Files:**
- Create: `scripts/evaluate_inshop_reciprocal_reranking.py`
- Create: `tests/test_evaluate_inshop_reciprocal_reranking.py`

**Interfaces:**
- Consumes: the Task 1 scorer and `.npz` files containing `embeddings`, `labels`, and `example_ids`.
- Produces: `select_train_split(...)`, `recall_at_one(...)`, `tune_parameters(...)`, and an atomic JSON report.

- [ ] **Step 1: Write failing evaluator tests**

Create tiny `.npz` fixtures and assert: exact required keys/types; unit-normalized finite embeddings; query/gallery dimensional agreement; deterministic selection of one pseudo-query per class from the first 1,024 labels ordered by SHA-256 of the exact label bytes; no test-label access inside `tune_parameters`; exact grid iteration order; correct Recall@1; JSON no-clobber publication; and a report containing input SHA-256 values, raw/tuned metrics, selected parameters, grid results, and the falsification decision.

- [ ] **Step 2: Run evaluator tests and observe RED**

Run: `pytest -q tests/test_evaluate_inshop_reciprocal_reranking.py`

Expected: collection failure because the evaluator script does not exist.

- [ ] **Step 3: Implement the evaluator**

Use grid `k=(3,5,10,20)`, `candidate_depth=(20,50,100)`, and `blend=(0.10,0.25,0.50,0.75)`. Maximize train pseudo-split Recall@1, breaking ties lexicographically by smaller `k`, smaller candidate depth, then smaller blend. Apply the selected tuple once to the test query/gallery embeddings. Mark the candidate `passes_falsifier` only when published-checkpoint Recall@1 gain is at least `0.0015` absolute.

- [ ] **Step 4: Run evaluator tests and observe GREEN**

Run: `pytest -q tests/test_evaluate_inshop_reciprocal_reranking.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the evaluator**

```bash
git add scripts/evaluate_inshop_reciprocal_reranking.py tests/test_evaluate_inshop_reciprocal_reranking.py
git commit -m "add train-only In-Shop reranking falsifier"
```

### Task 3: Frozen-embedding falsifier

**Files:**
- Create after execution: `reports/generated/inshop_reciprocal_reranking.json`

**Interfaces:**
- Consumes remote frozen `.npz` files for training, published-checkpoint query/gallery, and reproduced seed-0 query/gallery embeddings.
- Produces one immutable local JSON report; does not alter checkpoints or embeddings.

- [ ] **Step 1: Copy exact frozen inputs to an ignored external scratch directory**

Use `scp` into `/home/rb/reranking-inputs-2026-08-11/`, compute SHA-256 immediately, and do not copy them into Git.

- [ ] **Step 2: Run the one CPU-only evaluation**

Run the evaluator once with `CUDA_VISIBLE_DEVICES=''`, the frozen train embeddings for tuning, and both published/reproduced test pairs for evaluation.

- [ ] **Step 3: Verify the result independently**

Reload the JSON, recompute raw Recall@1 from the frozen arrays using a separate blockwise cosine implementation, confirm selected parameters occur in the train grid, and confirm the output hashes match the copied inputs.

- [ ] **Step 4: Apply the hard decision**

If published-checkpoint gain is below `0.0015`, record `CLOSE` and stop. If it passes, retain the result as a deterministic engineering baseline and request an Opus/Sol code-and-method critique before any broader benchmark or GPU work.

### Task 4: Final assurance and research note

**Files:**
- Modify: `docs/superpowers/plans/2026-08-11-inshop-reciprocal-reranking.md`
- Create: `docs/inshop_reciprocal_reranking_result_2026-08-11.md`

**Interfaces:**
- Consumes all test and result evidence.
- Produces a concise distinction between reproducible baselines, extra-data headline SOTA, known re-ranking prior art, and the measured result.

- [ ] **Step 1: Run focused and full assurance**

Run the two new test files, Ruff on the two new Python files, `py_compile`, `git diff --check`, then the repository's full pytest suite once after the diff is stable.

- [ ] **Step 2: Request cross-provider review**

Use a single review consultation with exact models `['opus', 'gpt-5.6-sol']`; ask it to inspect the Git diff, synthetic oracles, train/test separation, prior-art framing, and real JSON evidence without editing.

- [ ] **Step 3: Write the result note and commit**

State exact commits and checkpoint/input hashes, raw and re-ranked Recall@1, the falsification decision, and limitations. Commit only source, tests, plan, result note, and compact JSON; never commit large embedding files.
