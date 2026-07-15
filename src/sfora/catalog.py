"""Type-safe named constants for the benchmark API — no raw strings at call sites.

Each constant *is* its ``Literal`` type, so it is fully interchangeable with the
underlying string in every typed signature while giving autocomplete and rejecting
typos:

    from sfora.catalog import Dataset, Protocol
    from sfora.benchmark import benchmark
    from sfora.method import herd

    benchmark(herd(), dataset=Dataset.CUB, protocol=Protocol.PROXY_ANCHOR_R50_512)
    grid(methods, datasets=Dataset.ALL)          # (Dataset.CUB, Dataset.CARS, Dataset.SOP)
"""

from __future__ import annotations

from typing import Final


class Dataset:
    """Image retrieval benchmark datasets (each value is its ``ImageDatasetName`` literal)."""

    CUB: Final = "cub"
    CARS: Final = "cars"
    SOP: Final = "sop"
    ALL: Final = ("cub", "cars", "sop")


class Protocol:
    """Backbone/embedding protocol families (each value is its ``EndToEndProtocol`` literal)."""

    HPL_R50_512: Final = "hpl-resnet50-512"
    PFML_R50_512: Final = "pfml-resnet50-512"
    PROXY_ANCHOR_R50_512: Final = "proxy-anchor-resnet50-512"
    SOTA_R50_512: Final = "sota-resnet50-512"


class Combine:
    """How ``sfora.compose.Join`` merges branch embeddings (each is its ``JoinKind``)."""

    CONCAT: Final = "concat"
    MEAN: Final = "mean"
    ALIGNED_MEAN: Final = "aligned_mean"


class RankBy:
    """Retrieval metric to rank/select by (each is the metric's field name literal)."""

    RECALL_AT_1: Final = "recall_at_1"
    RECALL_AT_2: Final = "recall_at_2"
    RECALL_AT_4: Final = "recall_at_4"
    RECALL_AT_8: Final = "recall_at_8"
    MAP_AT_R: Final = "map_at_r"
