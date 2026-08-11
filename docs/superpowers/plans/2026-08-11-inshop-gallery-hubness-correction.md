# In-Shop Gallery Hubness Correction Plan

**Goal:** Falsify a deterministic, training-free gallery-density correction on frozen Proxy Anchor In-Shop embeddings.

**Method:** For unit embeddings, score each query/gallery pair with
`2 * cosine(q, g) - density_k(g)`, where `density_k(g)` is the mean cosine
between gallery item `g` and its `k` nearest non-self gallery neighbors.
Select `k` only on a deterministic train-identity pseudo split, then evaluate
the untouched published-checkpoint and corrected seed-0 query/gallery pairs.

**Constraints:** CPU/NumPy only; bounded-memory blocks; exact deterministic
tie-breaking; no test-label tuning; raw cosine and negative controls reported;
close the idea unless the published checkpoint gains at least 0.001 Recall@1.

## Task 1: Hubness and corrected retrieval primitives

- [x] Add failing tests for non-self local density, all-gallery density,
  deterministic top-1 correction, hubness counts/skewness, and input rejection.
- [x] Observe RED because `sfora.gallery_hubness` is absent.
- [x] Implement the minimal blockwise NumPy primitives.
- [x] Observe GREEN and commit source/tests.

## Task 2: Train-only evaluator and controls

- [x] Add failing tests for strict embedding archives, deterministic train split,
  `k=(1,5,10,50,all)` tuning, raw/corrected/permuted/global/query-invariant
  controls, input hashes, and atomic no-clobber JSON.
- [x] Observe RED because the evaluator is absent.
- [x] Implement the minimal CLI and report schema.
- [x] Observe GREEN and commit source/tests.

## Task 3: Frozen CPU falsifier

- [x] Run once on the exact frozen train, published, and reproduced embeddings.
- [x] Independently recompute raw Recall@1 and validate input hashes/report.
- [x] Record PASS only if published absolute gain is at least 0.001; otherwise
  record CLOSE. The hubness diagnostic kill condition is skewness < 1 and
  maximum incoming top-1 count < 20.

## Task 4: Assurance and review

- [x] Run focused tests, Ruff, py_compile, diff-check, then one full suite.
  The full suite reached 650 passing tests before an unrelated existing CLI
  test escaped into an unauthenticated Hugging Face download and was stopped.
- [x] Request one read-only review using models `opus`, then `gpt-5.6-sol`.
- [x] Commit the compact JSON and result note; never commit embedding archives.
