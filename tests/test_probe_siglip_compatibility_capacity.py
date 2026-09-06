from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_siglip_compatibility_capacity.py"
_SPEC = importlib.util.spec_from_file_location("probe_siglip_compatibility_capacity", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

parse_args = _MODULE.parse_args
write_capacity_descriptor_artifact = _MODULE.write_capacity_descriptor_artifact
load_capacity_descriptor_artifact = _MODULE.load_capacity_descriptor_artifact
capacity_image_manifest_sha256 = _MODULE.capacity_image_manifest_sha256
analyze_capacity_descriptors = _MODULE.analyze_capacity_descriptors
CapacityDescriptorArtifact = _MODULE.CapacityDescriptorArtifact

_DESCRIPTOR_AUTHORITY = {
    "checkpoint_sha256": "1" * 64,
    "control_binding_sha256": "2" * 64,
    "optimization_manifest_sha256": "3" * 64,
    "spatial_artifact_sha256": "4" * 64,
    "image_manifest_sha256": "5" * 64,
    "preprocessing": "siglip-evaluation-transform-v1",
}


def _args(tmp_path: Path) -> list[str]:
    return [
        "--control-binding",
        str(tmp_path / "binding.json"),
        "--control-binding-sha256",
        "1" * 64,
        "--checkpoint-seed17",
        str(tmp_path / "checkpoint.pt"),
        "--optimization-manifest",
        str(tmp_path / "optimization.json"),
        "--optimization-manifest-sha256",
        "2" * 64,
        "--optimization-image-root",
        str(tmp_path / "images"),
        "--spatial-artifact",
        str(tmp_path / "tail.safetensors"),
        "--spatial-artifact-sha256",
        "3" * 64,
        "--descriptor-artifact",
        str(tmp_path / "descriptors.safetensors"),
        "--result",
        str(tmp_path / "result.json"),
        "--execute-capacity-diagnostic",
    ]


def test_capacity_cli_is_local_burned_data_only_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    assert parse_args(_args(tmp_path)).execute_capacity_diagnostic is True
    for mutation in (
        _args(tmp_path)[:-1],
        _args(tmp_path) + ["--spatial-artifact", str(tmp_path / "duplicate")],
        _args(tmp_path) + ["--evaluation-manifest", str(tmp_path / "forbidden")],
        _args(tmp_path) + ["--class-names", str(tmp_path / "forbidden")],
        _args(tmp_path) + ["--aws-profile", "forbidden"],
    ):
        with pytest.raises(SystemExit):
            parse_args(mutation)


def test_capacity_descriptor_artifact_roundtrips_identity_and_partition(
    tmp_path: Path,
) -> None:
    generator = torch.Generator().manual_seed(91)
    student = torch.nn.functional.normalize(torch.randn(98, 8, generator=generator), dim=1)
    teacher = torch.nn.functional.normalize(torch.randn(98, 8, generator=generator), dim=1)
    ids = tuple(f"cars-train-{index // 2}-{index}" for index in range(98))
    labels = tuple(index // 2 for index in range(98))
    path = tmp_path / "descriptors.safetensors"

    digest = write_capacity_descriptor_artifact(
        path, student, teacher, ids, labels, **_DESCRIPTOR_AUTHORITY
    )
    restored = load_capacity_descriptor_artifact(
        path, ids=ids, labels=labels, **_DESCRIPTOR_AUTHORITY
    )

    assert len(digest) == 64
    assert torch.equal(restored.student, student)
    assert torch.equal(restored.teacher, teacher)
    assert restored.ids == ids
    assert restored.labels == labels
    assert set(restored.fit_labels) | set(restored.development_labels) == set(range(49))
    assert set(restored.fit_labels).isdisjoint(restored.development_labels)
    with pytest.raises(FileExistsError):
        write_capacity_descriptor_artifact(
            path, student, teacher, ids, labels, **_DESCRIPTOR_AUTHORITY
        )


def test_capacity_analysis_rejects_insufficient_anchor_coverage_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = tuple(label for label in range(49) for _ in range(2))
    ids = tuple(f"sparse-{index}" for index in range(len(labels)))
    descriptors = torch.nn.functional.normalize(
        torch.randn(len(labels), 8, generator=torch.Generator().manual_seed(18)), dim=1
    )

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("fitting must not start before anchor coverage validation")

    monkeypatch.setattr(_MODULE, "fit_regularized_affine", forbidden_fit)
    fitting, development = _MODULE.spatial_tail_class_split(tuple(range(49)))
    with pytest.raises(ValueError, match="capacity anchor coverage"):
        analyze_capacity_descriptors(
            CapacityDescriptorArtifact(
                student=descriptors,
                teacher=descriptors,
                ids=ids,
                labels=labels,
                fit_labels=tuple(sorted(fitting)),
                development_labels=tuple(sorted(development)),
            ),
            checkpoint_sha256="1" * 64,
            descriptor_artifact_sha256="2" * 64,
            control_binding_sha256="3" * 64,
            optimization_manifest_sha256="4" * 64,
            spatial_artifact_sha256="5" * 64,
            image_manifest_sha256="6" * 64,
            preprocessing="siglip-evaluation-transform-v1",
        )


def test_capacity_descriptor_artifact_rejects_identity_and_tensor_drift(
    tmp_path: Path,
) -> None:
    student = torch.nn.functional.normalize(
        torch.arange(1, 98 * 4 + 1, dtype=torch.float32).reshape(98, 4), dim=1
    )
    teacher = student.clone()
    ids = tuple(f"id-{index}" for index in range(98))
    labels = tuple(index // 2 for index in range(98))
    path = tmp_path / "descriptors.safetensors"
    write_capacity_descriptor_artifact(path, student, teacher, ids, labels, **_DESCRIPTOR_AUTHORITY)

    with pytest.raises(ValueError, match="capacity descriptor artifact"):
        load_capacity_descriptor_artifact(
            path,
            ids=ids[:-1] + ("changed",),
            labels=labels,
            **_DESCRIPTOR_AUTHORITY,
        )
    with pytest.raises(ValueError, match="capacity descriptor artifact"):
        load_capacity_descriptor_artifact(
            path,
            ids=ids,
            labels=labels,
            **(_DESCRIPTOR_AUTHORITY | {"spatial_artifact_sha256": "9" * 64}),
        )
    with pytest.raises(ValueError, match="capacity descriptor artifact"):
        write_capacity_descriptor_artifact(
            tmp_path / "bad.safetensors",
            torch.full_like(student, float("nan")),
            teacher,
            ids,
            labels,
            **_DESCRIPTOR_AUTHORITY,
        )


def test_capacity_image_manifest_binds_ordered_ids_and_exact_image_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    digest = capacity_image_manifest_sha256(("a", "b"), (first, second))
    assert len(digest) == 64
    assert digest != capacity_image_manifest_sha256(("b", "a"), (second, first))
    second.write_bytes(b"changed-image")
    assert digest != capacity_image_manifest_sha256(("a", "b"), (first, second))


def test_cross_fitted_oracle_scores_only_inside_each_held_out_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = tuple(label for label in (4, 5, 12, 15, 19, 24, 26, 32, 40, 45) for _ in range(2))
    ids = tuple(f"development-{index}" for index in range(20))
    descriptors = torch.eye(10, dtype=torch.float32).repeat_interleave(2, dim=0)
    training_id_sets: list[set[str]] = []

    class IdentityFit:
        def __init__(self, training_ids: tuple[str, ...], seed: int) -> None:
            self.relational = True
            self.seed = seed
            self.anchor_ids = training_ids
            self.losses = {
                name: (0.0,) * 2_000 for name in ("paired", "forward", "reverse", "self")
            }
            self.state_dict = {
                "down.weight": torch.zeros(32, 10),
                "up.weight": torch.zeros(10, 32),
                "up.bias": torch.zeros(10),
            }
            self.device = "cpu"
            self.torch_version = str(torch.__version__)

        def apply(self, values: torch.Tensor) -> torch.Tensor:
            return values

    def fake_fit(
        student: torch.Tensor,
        teacher: torch.Tensor,
        training_ids: tuple[str, ...],
        *,
        relational: bool,
        seed: int,
    ) -> IdentityFit:
        assert student.shape == teacher.shape == (10, 10)
        assert relational is True and seed in {20260909, 20260910}
        training_id_sets.append(set(training_ids))
        return IdentityFit(training_ids, seed)

    monkeypatch.setattr(_MODULE, "fit_teacher_anchored_residual", fake_fit)
    forward, reverse, self_cell, optimization = _MODULE.cross_fitted_oracle_evidence(
        descriptors,
        descriptors,
        ids,
        labels,
    )

    assert len(training_id_sets) == 2
    assert training_id_sets[0].isdisjoint(training_id_sets[1])
    assert training_id_sets[0] | training_id_sets[1] == set(ids)
    assert forward.ids == reverse.ids == self_cell.ids == ids
    assert forward.hits == reverse.hits == self_cell.hits == (True,) * 20
    assert max(forward.hub_counts) == 9
    assert [record["seed"] for record in optimization] == [20260909, 20260910]


def test_capacity_analysis_executes_registered_controls_and_seals_fold_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitting, development = _MODULE.spatial_tail_class_split(tuple(range(49)))
    labels = tuple(
        label for label in range(49) for _ in range(52 if label in set(development) else 10)
    )
    ids = tuple(f"example-{index}" for index in range(len(labels)))
    descriptors = torch.eye(49, dtype=torch.float32)[torch.tensor(labels)]
    calls: list[tuple[bool, int, int]] = []

    class IdentityFit:
        def __init__(self, training_ids: tuple[str, ...], relational: bool, seed: int) -> None:
            self.relational = relational
            self.seed = seed
            self.anchor_ids = tuple(
                sorted(
                    training_ids,
                    key=lambda identity: (
                        __import__("hashlib")
                        .sha256(b"sfora-compatibility-anchor-v1\0" + identity.encode())
                        .digest()
                    ),
                )[:256]
            )
            self.losses = {
                name: (0.0,) * 2_000 for name in ("paired", "forward", "reverse", "self")
            }
            self.state_dict = {
                "down.weight": torch.zeros(32, 49),
                "up.weight": torch.zeros(49, 32),
                "up.bias": torch.zeros(49),
            }
            self.device = "cpu"
            self.torch_version = str(torch.__version__)

        def apply(self, values: torch.Tensor) -> torch.Tensor:
            return values

    def fake_fit(
        student: torch.Tensor,
        teacher: torch.Tensor,
        training_ids: tuple[str, ...],
        *,
        relational: bool,
        seed: int,
    ) -> IdentityFit:
        assert student.shape == teacher.shape
        calls.append((relational, seed, len(training_ids)))
        return IdentityFit(training_ids, relational, seed)

    monkeypatch.setattr(_MODULE, "fit_teacher_anchored_residual", fake_fit)
    raw = analyze_capacity_descriptors(
        CapacityDescriptorArtifact(
            student=descriptors,
            teacher=descriptors,
            ids=ids,
            labels=labels,
            fit_labels=tuple(sorted(fitting)),
            development_labels=tuple(sorted(development)),
        ),
        checkpoint_sha256="1" * 64,
        descriptor_artifact_sha256="2" * 64,
        control_binding_sha256="3" * 64,
        optimization_manifest_sha256="4" * 64,
        spatial_artifact_sha256="5" * 64,
        image_manifest_sha256="6" * 64,
        preprocessing="siglip-evaluation-transform-v1",
    )
    result = __import__("json").loads(raw)

    assert set(result["control_fold_results"]) == {
        "centered-similarity",
        "paired-only-residual",
    }
    assert len(result["fold_evidence"]["centered-similarity"]) == 3
    assert sum(not relational and rows == 260 for relational, _, rows in calls) == 3
    assert (
        sum(
            relational and seed in {20260905, 20260906, 20260907} and rows == 260
            for relational, seed, rows in calls
        )
        == 3
    )
    assert sum(relational and rows == 260 for relational, _, rows in calls) == 5
