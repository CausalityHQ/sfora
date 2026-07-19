# DeepFashion In-Shop and iNaturalist Retrieval Design

## Goal

Extend SFORA's image-retrieval research from CUB, Cars196, and SOP to:

1. the canonical DeepFashion In-Shop Clothes Retrieval benchmark; and
2. a reproducible, explicitly project-defined iNaturalist 2018 zero-shot species
   retrieval protocol.

The implementation must support frozen-backbone and end-to-end runners, preserve the
existing three datasets, and never present a derived split or an unexecuted experiment
as canonical evidence.

## Research findings that constrain the design

DeepFashion In-Shop defines 52,712 images from 7,982 clothing identities and an
official three-way partition: 25,882 training images, 14,218 queries, and 12,612
gallery images. Retrieval is query-to-gallery by clothing identity. The convenient
`Marqo/deepfashion-inshop` Hugging Face mirror contains 52,591 resized images and an
`item_ID` field, but it has one undifferentiated `data` split and does not contain the
official train/query/gallery partition. Therefore that mirror cannot be treated as a
canonical In-Shop source.

iNaturalist 2018 is an imbalanced fine-grained classification dataset, not a standard
deep-metric-learning retrieval benchmark. Its official release has 8,142 species,
437,513 labeled training images, and 24,426 labeled validation images. SFORA will use a
named project protocol instead of implying that a canonical retrieval protocol exists.

## Approaches considered

### 1. Pretend the Hugging Face In-Shop mirror has the canonical split

This is the smallest code change, but it would fabricate benchmark semantics from
filename order or identity order. Rejected because its numbers would not be comparable
to published In-Shop results.

### 2. Add one-off scripts for In-Shop and iNaturalist

This avoids touching the core runners, but duplicates training, checkpoint evaluation,
embedding export, and report logic. Rejected because the duplicate paths would drift
from the verified CUB/Cars/SOP harness.

### 3. Generalize the dataset/evaluation boundary incrementally

Add first-class split/protocol metadata, local canonical loaders, and optional gallery
inputs to the existing runners. This keeps one trainer and one reporting schema while
allowing self-retrieval and query/gallery retrieval. Chosen because it is the smallest
change that remains scientifically honest and reusable.

## Dataset contracts

### DeepFashion In-Shop

The CLI requires `--dataset-root` for `inshop`. The root must contain:

```text
<root>/Eval/list_eval_partition.txt
<root>/Img/img/<paths named by the partition file>
```

The partition parser reads `<image_name> <item_id> <evaluation_status>`, accepts only
`train`, `query`, and `gallery`, assigns stable integer labels from sorted item IDs,
and validates that every selected image exists. The loader supports existing
development caps without changing the underlying official partition.

Evaluation uses query embeddings against gallery embeddings. It reports the project's
uniform R@1/2/4/8 and MAP@R metrics plus canonical In-Shop R@10/20/30. Training
identities must be disjoint from query/gallery identities; queries without a gallery
match are an error for the canonical full split.

### iNaturalist 2018 zero-shot species v1

The CLI requires `--dataset-root` for `inat2018`. The root must contain official
COCO-style `train2018.json` and `val2018.json` annotations and the image paths named by
those files.

The protocol is named `inat2018-zero-shot-species-v1` and is project-defined:

- Find species present in both official training and validation annotations.
- Sort their integer category IDs and split them deterministically in half.
- Optimize the model only on official training images from the first half.
- Use official validation images from the second half as queries.
- Use official training images from the second half as the gallery, without using them
  for optimization.

This creates disjoint training and evaluation species while retaining a query/gallery
match for each eligible evaluation species. Reports and documentation must label these
results as SFORA's protocol, not an iNaturalist standard or SOTA comparison.

## Core architecture

`sfora.data` owns dataset names, split semantics, canonical/local parsing, deterministic
selection, and lazy image paths. A small dataset-protocol helper tells callers whether
a dataset uses held-out self-retrieval (`cub`, `cars`, `sop`) or separate
query/gallery retrieval (`inshop`, `inat2018`). Existing Hugging Face loading remains
unchanged for the original datasets.

Both image runners accept an optional `gallery_examples` collection. If absent, they
retain current self-retrieval behavior. If present, `test_examples` means queries and
the runner encodes both collections, scores query-to-gallery, and records query and
gallery counts. Checkpoint selection remains train-only. Best-over-training test
selection uses the same evaluation protocol as final evaluation.

Filesystem-backed examples carry `pathlib.Path` values. Image materialization happens
at the encoder/dataset boundary so loaders remain cheap and do not keep tens of
thousands of open PIL images in memory.

## CLI and runners

Both `sfora image-benchmark` and `sfora image-end-to-end` accept:

- `--dataset-name inshop|inat2018` in addition to existing names;
- `--dataset-root PATH`, required for local datasets and rejected when missing; and
- the same class/image caps used by existing debug runs.

Dataset loading returns train, query/test, and optional gallery examples through one
shared helper so the frozen and end-to-end CLIs cannot disagree on splits.

The multi-seed benchmark API exposes `Dataset.INSHOP` and `Dataset.INAT2018`. Because
its default runner has no CLI parameter, local dataset roots are supplied through a
new config field rather than process-global state.

## Full-headline experiment orchestration

A checked-in shell runner executes three seeds for the same honest matrix used on the
existing datasets:

- Proxy Anchor;
- HIST;
- best-base + relational distillation (including HERD where HIST is the selected
  base).

The script accepts dataset roots through environment variables, runs sequentially by
default to avoid GPU-memory contention, writes per-seed JSON artifacts, and skips no
method silently. A preflight command validates dataset topology and split counts before
starting GPU work.

No result number is added to `docs/results.md`, the site, or headline tables until the
corresponding artifacts exist and pass the split/protocol validation.

## Error handling

Fail early with actionable messages for missing roots, missing annotation files,
unknown partition statuses, nonexistent image paths, fewer than two selected training
labels, query identities with no gallery match, overlapping In-Shop train/evaluation
identities, and overlapping iNaturalist optimization/evaluation species.

Development caps must remain deterministic by seed and must select compatible query
and gallery identities together. They may reduce counts, but may not create orphaned
queries or relabel the protocol as canonical full-scale evidence.

## Testing

Tests use tiny temporary filesystem fixtures; no external dataset download is required.
Coverage includes:

- exact In-Shop partition parsing and stable identity labels;
- missing/invalid In-Shop topology failures;
- iNaturalist COCO annotation parsing and zero-shot species disjointness;
- paired query/gallery selection under caps;
- lazy path materialization;
- R@10/20/30 query/gallery metrics;
- frozen and end-to-end runner dispatch to query/gallery scoring;
- CLI validation and argument propagation;
- catalog/benchmark API expansion; and
- unchanged behavior for CUB, Cars, and SOP.

The full verification gate is `uv run pytest`, `uv run ruff check .`, and
`uv run mypy src`.

## Non-goals

- Inventing a partition for the partitionless Hugging Face In-Shop mirror.
- Downloading or redistributing licensed DeepFashion images in the repository.
- Calling iNaturalist results a canonical metric-learning benchmark.
- Publishing placeholder, estimated, partial, or best-guessed result numbers.
- Refactoring unrelated objective/trainer internals.
