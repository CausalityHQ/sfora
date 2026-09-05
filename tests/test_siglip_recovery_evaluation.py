"""Fixed post-sealing quality gates and evaluation-only metadata/pixel boundary."""

import hashlib
import json
from typing import Any

import pytest
from PIL import Image

from sfora.siglip_recovery_evaluation import load_recovery_evaluation_images, recovery_decision


def test_literal_quality_search_and_storage_boundaries_and_pa_preference() -> None:
    teacher = {"correct": 2596, "queries": 2746, "map_at_r": 0.7913744556922272}
    students = {
        name: {
            "correct": 2591,
            "queries": 2746,
            "map_at_r": 0.7893744556922272,
            "descriptor_bytes": 5623808,
        }
        for name in ("pa", "relational")
    }
    samples = {"teacher": [100] * 100, "pa": [105] * 100, "relational": [105] * 100}
    result = recovery_decision(teacher, students, samples)
    assert result["selected_arm"] == "pa" and result["claim_eligible"] is False
    students["pa"]["correct"] = 2590
    assert recovery_decision(teacher, students, samples)["selected_arm"] == "relational"
    samples["relational"] = [106] * 100
    assert recovery_decision(teacher, students, samples)["selected_arm"] is None
    students["pa"]["correct"] = 2591
    students["pa"]["map_at_r"] -= 1e-10
    assert recovery_decision(teacher, students, samples)["selected_arm"] is None
    students["pa"]["map_at_r"] = teacher["map_at_r"]
    students["pa"]["descriptor_bytes"] += 4
    assert recovery_decision(teacher, students, samples)["selected_arm"] is None


@pytest.mark.parametrize(
    "mutation", ["teacher-hits", "nan-map", "bool-hits", "query-count", "short-latency"]
)
def test_invalid_quality_authority_is_not_a_failed_method(mutation: str) -> None:
    teacher = {"correct": 2596, "queries": 2746, "map_at_r": 0.7913744556922272}
    students = {
        name: {"correct": 2591, "queries": 2746, "map_at_r": 0.79, "descriptor_bytes": 5623808}
        for name in ("pa", "relational")
    }
    samples = {name: [100] * 100 for name in ("teacher", "pa", "relational")}
    if mutation == "teacher-hits":
        teacher["correct"] = 2595
    elif mutation == "nan-map":
        students["pa"]["map_at_r"] = float("nan")
    elif mutation == "bool-hits":
        students["pa"]["correct"] = True
    elif mutation == "query-count":
        students["pa"]["queries"] = 2745
    else:
        samples["pa"].pop()
    with pytest.raises(ValueError):
        recovery_decision(teacher, students, samples)


def test_evaluation_loader_authenticates_metadata_before_reading_only49to81() -> None:
    labels = (
        [i % 49 for i in range(3963)]
        + [49 + i % 33 for i in range(2746)]
        + [82 + i % 16 for i in range(1345)]
        + [98 + i % 98 for i in range(8131)]
    )
    rows = sorted(
        ({"example_id": f"cars-train-{y}-{i}", "label": y} for i, y in enumerate(labels) if y < 98),
        key=lambda r: str(r["example_id"]),
    )
    raw = (json.dumps({"examples": rows}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    expected = [r["example_id"] for r in rows if 49 <= int(str(r["label"])) < 82]
    accesses = []

    class Dataset:
        def __init__(self, values: list[int], offset: int) -> None:
            self.values, self.offset = values, offset

        def __getitem__(self, key: Any) -> Any:
            if key == "label":
                return self.values
            y = self.values[key]
            assert 49 <= y < 82, "outside evaluation pixel access"
            accesses.append(self.offset + key)
            return {"label": y, "image": Image.new("RGB", (2, 2))}

    data = {"train": Dataset(labels[:5000], 0), "test": Dataset(labels[5000:], 5000)}

    def loader(name: str, split: str, revision: str) -> Dataset:
        assert (
            name == "tanganke/stanford_cars"
            and revision == "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
        )
        return data[split]

    with pytest.raises(ValueError):
        load_recovery_evaluation_images(dataset_loader=loader, expected_manifest="0" * 64)
    assert accesses == []
    actual = load_recovery_evaluation_images(dataset_loader=loader, expected_manifest=digest)
    assert [e.example_id for e in actual] == expected
    assert len(accesses) == 2746 and set(accesses) == set(range(3963, 6709))


def test_resident_search_is_stable_self_excluding_and_restores_threads() -> None:
    import torch

    from sfora.siglip_recovery_evaluation import profile_recovery_search, rank_recovery_block

    vectors = torch.ones(130, 3, dtype=torch.float32)
    ranked = rank_recovery_block(vectors, 1)
    assert ranked.shape == (128, 130)
    assert ranked[0, :4].tolist() == [0, 2, 3, 4]
    assert ranked[0, -1].item() == 1
    before = torch.get_num_threads()
    profile = profile_recovery_search({"teacher": vectors, "pa": vectors, "relational": vectors})
    assert torch.get_num_threads() == before
    assert profile["threads"] == 1 and profile["warmups"] == 10
    assert profile["query_starts"] == [(i * 128) % 3 for i in range(10, 110)]
    assert all(
        len(x) == 100 and all(type(t) is int and t > 0 for t in x)
        for x in profile["samples_ns"].values()
    )
    with pytest.raises(ValueError):
        rank_recovery_block(vectors, 3)
    with pytest.raises(ValueError):
        profile_recovery_search({"teacher": vectors, "pa": vectors[:129]})
