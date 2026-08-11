# In-Shop reciprocal re-ranking result (2026-08-11)

## Outcome

**CLOSE.** Deterministic reciprocal-neighborhood re-ranking reduced Recall@1 on
both frozen Proxy Anchor embedding pairs. No GPU training or further benchmark
expansion is justified.

| Frozen embedding pair | Raw R@1 | Re-ranked R@1 | Absolute change |
|---|---:|---:|---:|
| Published Proxy Anchor checkpoint | 0.9176396118 | 0.9148262766 | -0.0028133352 |
| Corrected Proxy Anchor seed 0 | 0.9137009425 | 0.9108876073 | -0.0028133352 |

Train-identity-only selection chose `k=3`, candidate depth `20`, and structural
blend `0.1`. The registered survival threshold was an absolute improvement of
`0.0015`; the observed effect had the opposite sign.

## Reproducibility evidence

- Implementation commits: `2d692fc` (exact reciprocal scorer) and `65d75f8`
  (train-only In-Shop evaluator).
- Result JSON SHA-256:
  `2d5ac1df0679e190dc53dedf1c4be18371d80723095fd211d2cb6c51e536e40e`.
- Published query embedding SHA-256:
  `55d1633e80a5ea036910a3c0bb360fdcf7f232e989adfa042239dc5d663a3289`.
- Published gallery embedding SHA-256:
  `db0b835788627cb437ecdcc856c4822966c6a45f392dbe52d975caa02362e532`.
- Corrected seed-0 query embedding SHA-256:
  `ef5278fd9aae7a6398a6c74133e6acc0ded05e39647087bdf78459223b9eb761`.
- Corrected seed-0 gallery embedding SHA-256:
  `6eb89ff57e7a6002f2ba71f9659e04dabd0cafdb1996be3d85f5211731ba861a`.
- Train embedding SHA-256:
  `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- Remote and local hashes matched before evaluation.
- The raw published-checkpoint score independently recomputed to
  `0.9176396117597412`, matching the established reproduction.
- Focused scorer/evaluator suite: 19 passed; Ruff, `py_compile`, and diff check
  passed before the real run.

An earlier run selected a similarly named but incorrect legacy embedding pair
whose raw R@1 was only `0.87347024898`. It was rejected before interpretation;
its JSON SHA-256 is
`0063955c12c8f0be62af7bc75371ea484b89150b588291012c3dcffae9913544`.

## Baseline and prior-art audit

The strongest independently runnable, checkpoint-backed, In-Shop-documented
anchor remains Proxy Anchor at upstream commit
`51db57031e38f75c03f69bbdfad1a3233afd9787`. Higher headline results are not
currently comparable reproductions:

- [UNICOM](https://github.com/deepglint/unicom) commit
  `d71992ed969e6c271436ac0a0ee1f3ca61474ac0` reports supervised In-Shop R@1 up
  to 96.7, but uses LAION-400M pretraining and does not provide a matching
  In-Shop fine-tuning checkpoint/recipe in the inspected repository state.
- [Hyp-ViT](https://github.com/htdt/hyp_metric) commit
  `c89de0490691bacbd7332171c5455651fe49f25e` provides an In-Shop training
  command but no released checkpoint, requires 400 epochs, and pins a
  Torch 1.7.1/CUDA 11.0 stack that is not directly runnable on the current
  hardware.
- [Contextual similarity optimization](https://github.com/Chris210634/metric-learning-using-contextual-similarity)
  commit `8433dcb67c2205c0e30ec07ed1e5b2fb92da016d` has an exact In-Shop command
  and release checkpoint, but its own preserved sample output reports R@1
  `0.9072302715`, below the Proxy Anchor anchor.
- [Recall@k Surrogate](https://github.com/yashvarpatel/RecallatK_surrogate)
  commit `ed052029d258555df2f94dd82d6f7df60ef7cc6f` does not support In-Shop in
  its official datasets or training recipe, so it cannot serve as an In-Shop
  baseline despite strong results elsewhere.

Reciprocal re-ranking itself is known prior art, including
[k-reciprocal encoding](https://arxiv.org/abs/1701.08398), intra-batch metric
learning that uses k-reciprocal test batches, and modern In-Shop re-rankers such
as [LOCORE](https://arxiv.org/abs/2503.21772). This experiment was therefore an
engineering falsifier, never a novelty claim.

