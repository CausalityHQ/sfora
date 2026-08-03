# In-Shop wrong-pixel-corpus retraction 297

Date: 2026-08-03.

## Finding

Every In-Shop experiment in this project through diagnostic 296 used the wrong
DeepFashion pixel corpus. The split metadata was official and all 52,712 paths,
identities and partition counts were present, but `Img/img` was a symlink to
`/home/riomus/datasets/inshop_kaggle/img_highres`. The acquisition log identifies
the source as Kaggle dataset `hserdaraltan/deepfashion-inshop-clothes-retrieval`.

This is not an interchangeable resolution choice. DeepFashion distributes both
`Img/img.zip` and `Img/img_highres.zip` under identical relative filenames.
OpenMMLab's primary dataset documentation says explicitly: use `img.zip` for
fashion retrieval and `img_highres.zip` for fashion parsing and segmentation;
the retrieval images are centered/resized to 256 pixels with aspect ratio
preserved. The local images are the high-resolution 750x1101-scale parsing
corpus. Checking only split rows, labels, paths and within-corpus content hashes
could not detect this error.

Sources:

- DeepFashion In-Shop dataset specification: https://mmfashion.readthedocs.io/en/latest/dataset/IN_SHOP_DATASET/
- MMPretrain In-Shop loader instructions: https://onedl-mmpretrain.readthedocs.io/en/latest/_modules/mmpretrain/datasets/inshop.html
- local acquisition log: `/home/riomus/experiment-logs/inshop-acquire.log`

## Decisive experiment

Diagnostic 296 was committed before downloading or evaluating the Proxy Anchor
authors' linked In-Shop checkpoint. Its SHA-256 is
`925cc1a1a5207f8f50ea6fa55189a2d8aed2523feca648132fe5cc74299f705a`.
The vendored and exact upstream BN-Inception implementations produce bit-identical
outputs (`max_abs = 0`) after loading that state. The local and upstream
deterministic evaluation transforms also produce bit-identical tensors on a real
image. Nevertheless the authors' checkpoint scores only:

- upstream strict-negative-rank R@1: **0.8735405823603882**;
- canonical float64 Euclidean R@1: **0.8735405823603882**;
- float64 cosine R@1: **0.8734702489801660**;
- exact-tie-expected R@1: **0.8734350822900548**.

The registered prediction was `[0.917, 0.921]`, centered on the source's reported
`0.919`. This is a decisive failure and localizes the defect outside model,
transform and scorer code. The high-resolution substitution is the identified
dataset mismatch.

## Retraction scope

All historical In-Shop absolute scores, paired effects, variance estimates,
fragmentation/acquisition measurements and candidate deaths are observations on
the wrong pixel corpus, not benchmark-matched In-Shop evidence. They must not be
used to validate or falsify a method. In particular:

- the three-seed averaging non-replication is retracted;
- the dual-EMA and RSPG In-Shop screening verdicts are retracted as benchmark
  evidence (their independent prior-art/mechanism objections remain separate);
- `sigma = 0.12 pt`, the one-seed decisiveness rule, and the 0.9035/0.9038
  thresholds are retracted;
- every acquisition-series and fragmentation statistic is corpus-specific and
  cannot motivate a benchmark method until remeasured on `img.zip`;
- corrected seed-0 PA raw `0.9039246` and frozen-final `0.9020959` are invalid as
  official references, despite internally exact artifact/scorer audits.

This invalidates negatives as well as positives. No candidate is revived directly:
each must restart from Gate 0 on the correct corpus.

## Repair and prevention

The loader now refuses an `Img/img` tree whose resolved path contains
`img_highres`, with a regression test for the exact symlink failure. The official
830,546,396-byte `Img/img.zip` is being acquired independently from the archived
DeepFashion torrent (info hash
`05e9118c9ccf175f8aaeabbc4fa3fcaebbb4ece7`). Before any training it must pass:

1. archive integrity and exact 52,712-path coverage;
2. a frozen content/dimension manifest;
3. the already registered published-checkpoint diagnostic `[0.917, 0.921]`.

Only then may a new In-Shop baseline or candidate screen run.
