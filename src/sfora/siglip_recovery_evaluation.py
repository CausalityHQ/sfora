"""Post-sealing recovery evaluation inputs and fixed exploratory advancement gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from decimal import Decimal
from time import perf_counter_ns
from typing import Any

import torch
from PIL import Image

from sfora.data import ImageExample
from sfora.siglip_recovery_inputs import CARS_REVISION, RECOVERY_MANIFEST_SHA256


def rank_recovery_block(vectors: torch.Tensor, start: int) -> torch.Tensor:
    """Complete resident128-query cosine ranking, stable ties and self exclusion."""
    if (
        vectors.device.type != "cpu"
        or vectors.dtype != torch.float32
        or vectors.ndim != 2
        or type(start) is not int
        or start < 0
        or start + 128 > len(vectors)
    ):
        raise ValueError("recovery search block differs")
    scores = vectors[start : start + 128] @ vectors.T
    scores[torch.arange(128), torch.arange(start, start + 128)] = -torch.inf
    return torch.argsort(scores, dim=1, descending=True, stable=True)


def profile_recovery_search(descriptors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Alternate complete searches; keep100raw post-warmup samples per arm."""
    if not descriptors or len({len(v) for v in descriptors.values()}) != 1:
        raise ValueError("recovery search galleries differ")
    vectors = {}
    for name, value in descriptors.items():
        cpu = value.detach().to(device="cpu", dtype=torch.float32)
        if (
            cpu.ndim != 2
            or len(cpu) < 128
            or cpu.shape[1] < 1
            or not bool(torch.isfinite(cpu).all())
            or not bool((torch.linalg.vector_norm(cpu, dim=1) > 0).all())
        ):
            raise ValueError("recovery search vectors differ")
        vectors[name] = torch.nn.functional.normalize(cpu, dim=1)
    names = tuple(vectors)
    samples: dict[str, list[int]] = {name: [] for name in names}
    starts = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index in range(110):
            start = (index * 128) % (len(vectors[names[0]]) - 127)
            if index >= 10:
                starts.append(start)
            for name in names if index % 2 == 0 else tuple(reversed(names)):
                began = perf_counter_ns()
                output = rank_recovery_block(vectors[name], start)
                elapsed = perf_counter_ns() - began
                del output
                if index >= 10:
                    if elapsed <= 0:
                        raise ValueError("recovery search clock failed")
                    samples[name].append(elapsed)
    finally:
        torch.set_num_threads(threads)
    return {
        "configuration_order": list(names),
        "reverse_on_odd_round": True,
        "query_starts": starts,
        "samples_ns": samples,
        "threads": 1,
        "batch_size": 128,
        "warmups": 10,
    }


def load_recovery_evaluation_images(
    *,
    dataset_loader: Callable[..., Any] | None = None,
    expected_manifest: str = RECOVERY_MANIFEST_SHA256,
) -> tuple[ImageExample, ...]:
    """Authenticate complete metadata, then decode only the exposed49..81 band.

    The evaluator caller must authenticate both final checkpoints first. This
    separate loader never changes the immutable optimization-only training API.
    """
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    records: list[tuple[str, int, Any, int]] = []
    for split in ("train", "test"):
        dataset = dataset_loader("tanganke/stanford_cars", split=split, revision=CARS_REVISION)
        labels = dataset["label"]
        offset = len(records)
        for index, label in enumerate(labels):
            if type(label) is not int or not 0 <= label < 196:
                raise ValueError("evaluation metadata labels differ")
            records.append((f"cars-train-{label}-{offset + index}", label, dataset, index))
    if len(records) != 16185 or {r[1] for r in records} != set(range(196)):
        raise ValueError("evaluation source metadata cardinality differs")
    ordered = sorted((r for r in records if r[1] < 98), key=lambda r: r[0])
    counts = (
        sum(r[1] < 49 for r in ordered),
        sum(49 <= r[1] < 82 for r in ordered),
        sum(82 <= r[1] < 98 for r in ordered),
        sum(r[1] >= 98 for r in records),
    )
    if counts != (3963, 2746, 1345, 8131):
        raise ValueError("evaluation metadata class-band counts differ")
    raw = (
        json.dumps(
            {"examples": [{"example_id": r[0], "label": r[1]} for r in ordered]},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if hashlib.sha256(raw).hexdigest() != expected_manifest:
        raise ValueError("evaluation metadata digest differs before pixel access")
    examples = []
    for example_id, label, dataset, index in ordered:
        if not 49 <= label < 82:
            continue
        row = dataset[index]
        if (
            type(row["label"]) is not int
            or row["label"] != label
            or not isinstance(row["image"], Image.Image)
        ):
            raise ValueError("evaluation row differs from metadata")
        examples.append(ImageExample(example_id, row["image"].convert("RGB"), label))
    return tuple(examples)


def recovery_decision(
    teacher: dict[str, Any],
    students: dict[str, dict[str, Any]],
    search_samples: dict[str, list[int]],
) -> dict[str, Any]:
    """Apply literal quality/storage/search gates; prefer PA without tuning."""
    if set(students) != {"pa", "relational"} or set(search_samples) != {
        "teacher",
        "pa",
        "relational",
    }:
        raise ValueError("recovery evaluation arms differ")
    for item in (teacher, *students.values()):
        if (
            type(item["queries"]) is not int
            or item["queries"] != 2746
            or type(item["correct"]) is not int
            or not 0 <= item["correct"] <= 2746
            or type(item["map_at_r"]) is not float
            or not math.isfinite(item["map_at_r"])
            or not 0 <= item["map_at_r"] <= 1
        ):
            raise ValueError("recovery quality authority differs")
    if teacher["correct"] != 2596:
        raise ValueError("teacher hit-count reproduction failed")
    for samples in search_samples.values():
        if (
            type(samples) is not list
            or len(samples) != 100
            or any(type(t) is not int or t <= 0 for t in samples)
        ):
            raise ValueError("search requires100positive integer timing samples")
    baseline_p95 = sorted(search_samples["teacher"])[94]
    arms = {}
    for name, item in students.items():
        if type(item["descriptor_bytes"]) is not int or item["descriptor_bytes"] <= 0:
            raise ValueError("descriptor storage evidence invalid")
        p95 = sorted(search_samples[name])[94]
        map_loss = Decimal(str(teacher["map_at_r"])) - Decimal(str(item["map_at_r"]))
        gates = {
            "recall": item["correct"] >= 2591,
            "map": map_loss <= Decimal("0.002"),
            "storage": item["descriptor_bytes"] == 5623808,
            "search": 20 * p95 <= 21 * baseline_p95,
        }
        arms[name] = {
            "gates": gates,
            "passed": all(gates.values()),
            "search_p95_ns": p95,
            "map_loss": float(map_loss),
        }
    selected = next((name for name in ("pa", "relational") if arms[name]["passed"]), None)
    return {
        "claim_eligible": False,
        "surface": "exploratory-reuse-49..81",
        "selected_arm": selected,
        "arms": arms,
    }
