# In-Shop Gallery Local-Scaling Result

## Outcome

This run is **exploratory and not a confirmatory pass**. The fixed score raised
Recall@1 on both frozen Proxy Anchor embedding pairs, but the train pseudo split
was saturated and supplied no evidence for choosing `k=50`: it tied `k="all"`
at zero train gain and won only by grid order. Commit `a73952a` now aborts such
future tuning and keeps every non-query train row as a distractor.

The evaluated score was:

```text
score(q, g) = 2 * cosine(q, g) - mean_top50_nonself_gallery_cosine(g)
```

This belongs to established local-scaling / hubness-correction prior art and is
closely related to CSLS (Conneau et al., ICLR 2018). It is not a new learning
method. For a disjoint query/gallery protocol it is not exactly standard
cross-domain CSLS: the gallery term here uses gallery→gallery neighbors, while
standard CSLS estimates the target/gallery term from source/query neighbors.
The per-query CSLS term would be rank-invariant and is therefore omitted.

Repository history already contains a CUB diagnostic (`bf6057c`) where CSLS
reduced N10 hubness but lowered mean Recall@1 by 0.65 percentage points across
17 models. The opposite In-Shop sign here is dataset/checkpoint-specific and
requires external validation.

## Frozen exploratory results

| Embeddings | Raw R@1 | Corrected R@1 | Gain | Wrong→right | Right→wrong | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Official published Proxy Anchor | 0.9176396118 | 0.9196792798 | +0.0020396680 | 89 | 60 | 0.02148335 |
| Corrected reproduced seed 0 | 0.9137009425 | 0.9161626108 | +0.0024616683 | 110 | 75 | 0.01222064 |

The fixed-seed density permutation degraded to 0.9091292728 (published) and
0.9101139401 (reproduced). One permutation is not a calibrated null, so these
values are descriptive only. The v1 JSON also contains global-density and
query-invariant “control” fields; both are mathematical ranking identities,
not empirical controls. They were removed from future evaluator output in
`a73952a` and must not be cited as evidence.

The preregistered `passes_falsifier=false` remains unchanged. Its N1
query→gallery diagnostic (maximum incoming count 7; skew 0.7383 / 0.7257) rejects
only a severe top-1-hub story. It does not close the gallery-local-density
mechanism because that mechanism lives on a gallery→gallery neighborhood graph.

## Reproducibility evidence

- Exact scorer: `71f02cf`.
- Original evaluator: `f387cec`.
- Review hardening: `a73952a`.
- v1 JSON SHA-256:
  `a0b876cdd5f5a7226c6d7d0699ce85a190ff4e7a32e6fc6ba62232792c3f59ff`.
- Input hashes are embedded in the JSON and match all five frozen archives.
- A separate one-off blockwise full-sort calculation reproduced both raw and
  corrected metrics exactly; it was not retained as an executable artifact, so
  this is supporting execution evidence rather than a reusable verifier.
- Execution was CPU NumPy with `CUDA_VISIBLE_DEVICES=''`; no training or
  checkpoint mutation occurred.

## Verification and boundary

Seventeen affected tests pass; Ruff, py_compile, and diff-check pass. A full
repository run reached 650 passing tests before an unrelated existing CLI test
escaped its fake boundary into an unauthenticated Hugging Face download. The
run was stopped rather than repeat the known cgroup reclaim thrash; its one
failure is not attributed to this branch.

The reproducible learning baseline remains Proxy Anchor. This local-scaling
result is better only under the same frozen In-Shop embeddings and transductive
retrieval protocol; it is neither a SOTA claim nor evidence of a better trained
representation. A defensible continuation needs a prospectively fixed local
scale on another dataset/checkpoint pair, with a non-saturated selector or no
tuning, paired uncertainty, and a calibrated null.

