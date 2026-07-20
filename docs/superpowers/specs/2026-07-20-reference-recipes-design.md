# Publication-Backed Dataset Recipe Design

## Goal

Replace the global `proxy-anchor-resnet50-512` training preset with an explicit
method-by-dataset recipe system. A run must use the authors' published or official-code
recipe when one exists. When no recipe was published for a method/dataset pair, SFORA
must select the best available published recipe using training-only evidence and label
the result as an extension rather than a reproduction.

## Problem

The extended-dataset runner currently applies the CUB/Cars ResNet-50 settings to every
dataset and objective. In particular, it runs DeepFashion In-Shop with ResNet-50,
batch size 120, learning rate `1e-4`, frozen BatchNorm, five warm-up epochs, and a
10-epoch decay interval. The official Proxy Anchor In-Shop command instead uses
BN-Inception, batch size 180, learning rate `6e-4`, unfrozen BatchNorm, one warm-up
epoch, a 20-epoch decay interval, and decay gamma 0.25. The same global runner also
overrides HIST's method- and dataset-specific settings.

This makes the completed In-Shop seed-0 artifacts valid measurements of a modified
common protocol, but not reference reproductions. They must not be used in headline
reference comparisons.

## Primary Sources

Recipe values must be transcribed from primary sources, in this precedence order:

1. the authors' official executable training command and its code defaults;
2. the publication and supplementary material;
3. a tagged model release or configuration published by the authors.

The initial sources are:

- Proxy Anchor paper: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf>
- Proxy Anchor official code: <https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020>
- HIST paper: <https://openaccess.thecvf.com/content/CVPR2022/papers/Lim_Hypergraph-Induced_Semantic_Tuplet_Loss_for_Deep_Metric_Learning_CVPR_2022_paper.pdf>
- HIST official code: <https://github.com/ljin0429/HIST>

The source repository revision and the exact source location used for each recipe must
be recorded alongside the recipe. Repository commands take precedence over prose when
the repository explicitly states that its released settings improve the paper result.
Conflicts must be represented in metadata, not silently resolved.

## Recipe Model

Add an immutable recipe registry keyed by:

```text
(base_method, dataset, recipe_track)
```

`base_method` initially supports `proxy_anchor` and `hist`. `dataset` supports `cub`,
`cars`, `sop`, `inshop`, and `inat2018`. `recipe_track` is one of:

- `reference`: an exact author recipe for that method/dataset pair;
- `selected_extension`: a training-only selection among published source recipes when
  no same-dataset reference exists;
- `modified`: an explicit user override of a resolved recipe.

Each recipe contains all behavior that can affect the result, including:

- backbone implementation and pretrained-weight identity;
- embedding dimension, pooling, head structure, initialization, and normalization;
- input and evaluation transforms;
- optimizer and parameter-group behavior;
- batch construction and samples per class;
- learning rates for backbone, head, proxies, class distributions, and HGNN;
- weight decay exclusions;
- epoch count, warm-up, scheduler, decay interval, and gamma;
- BatchNorm training/freeze policy;
- loss hyperparameters;
- checkpoint/evaluation convention;
- provenance URL, source revision, source method, source dataset, and a concise note
  explaining any source conflict.

Recipe resolution must return a complete validated configuration. Runners must not add
dataset-independent hyperparameter overrides after resolution.

## Published Recipe Coverage

### Proxy Anchor

- CUB and Cars use the official ResNet-50 commands.
- SOP and In-Shop use the official BN-Inception commands.
- iNaturalist 2018 has no published Proxy Anchor retrieval recipe and therefore uses
  `selected_extension`.

The BN-Inception path must be the architecture used by the authors, with the same
pretrained-weight family and embedding head. A generic ResNet substitute is not an
exact reference implementation. The implementation or dependency revision must be
pinned and recorded.

### HIST

- CUB, Cars, and SOP use their separate official ResNet-50 commands.
- In-Shop and iNaturalist 2018 were not evaluated in the HIST publication and therefore
  use `selected_extension`.

The official HIST values differ by dataset. The registry must preserve, at minimum,
the published differences in epochs, backbone learning rate, distribution learning
rate, HGNN multiplier, weight decay, scheduler interval, BatchNorm policy, `tau`, and
`alpha`. HIST's official batch size is 32 unless a primary source for a specific recipe
states otherwise.

## Derived SFORA Methods

`pa_distill` inherits the complete resolved Proxy Anchor recipe for the target dataset
and changes only the registered distillation fields. `herd` inherits the complete
resolved HIST recipe and changes only the registered LayerNorm and distillation fields.

The artifact must include both the base recipe ID and a machine-readable delta. This
makes the comparison paired: the base and derived method use the same architecture,
data transforms, optimizer, schedule, and batch construction unless the method
definition explicitly requires a difference.

HERD and PA-distill are SFORA methods and are never labeled as published reference
recipes. Their base recipe may be `reference` or `selected_extension`.

## Best-Available Selection for Unpublished Pairs

Selection may choose only among complete official recipes already registered for the
same base method. It must not create an undocumented hybrid or tune on the official
evaluation split.

For each unpublished pair:

1. Hold out a deterministic, class-disjoint subset of the official training identities
   or species. Optimization and recipe-selection labels must not overlap.
2. Retain only held-out labels with at least two images. Deterministically divide each
   held-out label into a selection query and selection gallery, ensuring every query
   has a gallery match.
3. Run each eligible published recipe with selection seed 0 on the same optimization
   and selection data. Architecture-specific recipes remain intact.
4. Rank candidates by selection MAP@R, then Recall@1, then the stable recipe ID.
5. Persist the complete selection table and chosen source recipe before any official
   query/gallery/test run begins.
6. Freeze the winner and run the final three seeds. Final evaluation data cannot alter
   the chosen recipe.

For DeepFashion In-Shop, selection uses only identities from the official `train`
partition. For iNaturalist, it uses only species and images assigned to the protocol's
optimization side; the existing evaluation species remain untouched.

The selected artifact records `selected_extension`, the candidate set, the selection
protocol version, selection seed, scores, and winning source recipe. A selected recipe
is "best available" evidence for this project, not a claimed publication recipe.

## Checkpoint and Score Semantics

Exact reference reproduction and strict evaluation are separate reported fields:

- `reference_compatible`: follows the checkpoint/evaluation convention in the authors'
  released code when that convention is known, including best-over-training test
  reporting if the reference code does so;
- `strict`: uses a fixed final epoch or a checkpoint selected only from training-side
  validation data.

The report must never display a test-selected value without its selection label. Recipe
selection itself always uses the training-only procedure above, regardless of the
reference checkpoint convention.

## CLI and Orchestration

The end-to-end CLI accepts a recipe selector rather than relying on a misleading global
protocol name:

```text
--recipe auto|REFERENCE_ID|SELECTED_EXTENSION_ID
```

`auto` resolves the exact reference recipe when available. If none is available, it
requires a persisted selection result; it does not silently invent defaults. Explicit
training overrides remain available for ablations, but applying any behavior-changing
override changes provenance to `modified` and records the field-level diff.

The remote matrix runner must:

- resolve and print the recipe before launching a GPU process;
- run any required training-only selection stage once;
- run paired base/derived methods from the same resolved base recipe;
- skip only artifacts whose recipe ID and recipe digest match the requested run;
- reject stale artifacts with the same output name but a different digest;
- write recipe IDs into filenames or manifests so incompatible results cannot be
  aggregated accidentally.

## Existing Artifacts and Active Runs

Existing global-preset In-Shop artifacts are preserved as negative/common-protocol
evidence. They receive a manifest classification of `modified_legacy` and are excluded
from the reference headline aggregator. They are not deleted or overwritten.

Before relaunching the DGX matrix, the controller must stop launching additional jobs
from the legacy recipe. A partially completed legacy process may be terminated only
after its artifact/checkpoint location is captured. Reference and selected-extension
runs use new output paths, leaving legacy outputs recoverable.

## Validation and Failure Behavior

Tests must prove that:

- every published method/dataset pair resolves to the exact source values;
- Proxy Anchor In-Shop resolves to BN-Inception and its official optimization recipe;
- HIST CUB, Cars, and SOP resolve to distinct official configurations;
- unpublished pairs cannot resolve as `reference`;
- `auto` for an unpublished pair fails until a persisted training-only selection exists;
- derived methods differ from their base only by declared method deltas;
- recipe selection is class-disjoint and never receives official evaluation examples;
- explicit overrides produce `modified` provenance and a field-level diff;
- artifact reuse requires a matching recipe digest;
- legacy artifacts cannot enter reference aggregation.

Failures must name the unresolved method/dataset pair and the missing selection action.
Source conflicts, unavailable pretrained weights, unsupported backbones, and incomplete
recipe fields fail before GPU allocation.

## Documentation and Reporting

Documentation must provide a method-by-dataset table containing recipe ID, provenance
class, architecture, primary source, expected published score where available, and the
status of local reproduction. In-Shop and iNaturalist HIST/HERD results must state that
the pair was not evaluated by the HIST publication. iNaturalist results must continue
to be labeled as the project-defined `inat2018-zero-shot-species-v1` protocol.

## Out of Scope

- Claiming a canonical iNaturalist retrieval benchmark where none exists.
- Retrofitting old common-protocol results into reference reproductions.
- Unbounded hyperparameter search on official test/query/gallery data.
- Changing the mathematical definition of Proxy Anchor, HIST, HERD, or PA-distill.

## Acceptance Criteria

The work is complete when exact published recipes resolve with pinned provenance,
unpublished pairs have a persisted training-only selection path, reference and derived
runs are paired correctly, legacy artifacts cannot contaminate headline aggregation,
the relevant unit/integration tests pass, and the DGX controller launches only recipe-
identified jobs.
