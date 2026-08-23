from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_probe import ProbeFit, ProbeGradientMetrics, ProbeMetrics, ProbeSplit

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_unicom_spherical_probe.py"
SPEC = importlib.util.spec_from_file_location("screen_unicom_spherical_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _validation(loss: float, *, correct: int = 160_000) -> dict[str, object]:
    observations = 3_188 * 64
    return {
        "mean_loss": loss,
        "accuracy": correct / observations,
        "correct_count": correct,
        "observation_count": observations,
        "per_mask_mean_losses": [loss] * 64,
        "per_mask_represented_mean_losses": [loss] * 64,
        "per_mask_unrepresented_mean_losses": [loss] * 64,
        "per_image_mean_losses": [loss] * 3_188,
        "represented_mean_loss": loss,
        "unrepresented_mean_loss": loss,
    }


def _gradient() -> dict[str, float]:
    return {
        "class_mean_l2": 1.0,
        "spherical_probe_l2": 1.0,
        "relative_l2_difference": 0.1,
        "spherical_to_class_mean_l2_ratio": 1.0,
        "cosine_min": 0.98,
        "cosine_p05": 0.985,
        "cosine_median": 0.99,
        "cosine_mean": 0.99,
        "zero_gradient_row_count": 0,
    }


def _probe(seed: int) -> dict[str, object]:
    target = 0.01 * math.sqrt(768.0)
    return {
        "fit_seed": seed,
        "sha256": f"{seed + 1:x}" * 64,
        "row_norm_min": target,
        "row_norm_max": target,
        "fit": {"initial_loss": 2.0, "final_loss": 1.9, "steps": 512},
        "head_cosine": {"minimum": 0.8, "p05": 0.9, "median": 0.97, "mean": 0.95},
        "validation": _validation(0.9),
        "gradient": _gradient(),
    }


def _seed_decision(seed: int) -> dict[str, object]:
    return {
        "fit_seed": seed,
        "statistics": {
            "mean_paired_loss_delta": 0.09999999999999998,
            "paired_95_lower_bound": 0.09999999999999998,
            "identity_95_lower_bound": 0.09999999999999999,
            "positive_mask_count": 64,
            "unrepresented_paired_95_lower_bound": 0.09999999999999998,
            "unrepresented_positive_mask_count": 64,
            "accuracy_delta": 0.0,
            "represented_loss_delta": 0.09999999999999998,
            "unrepresented_loss_delta": 0.09999999999999998,
        },
        "predicates": {
            "fit_loss_decreased": True,
            "head_cosine_minimum": True,
            "head_cosine_mean": True,
            "validation_loss_positive": True,
            "paired_95_lower_positive": True,
            "identity_95_lower_positive": True,
            "positive_mask_majority": True,
            "validation_accuracy_noninferior": True,
            "represented_loss_positive": True,
            "unrepresented_loss_positive": True,
            "unrepresented_paired_95_lower_positive": True,
            "unrepresented_positive_mask_majority": True,
            "gradient_median_cosine": True,
        },
    }


def _valid_result() -> dict[str, object]:
    target = 0.01 * math.sqrt(768.0)
    return {
        "schema_version": "unicom-spherical-probe-causal-screen-v3",
        "source": {
            "git_revision": "a" * 40,
            "script_sha256": "b" * 64,
            "module_sha256": "c" * 64,
            "training_sha256": "f" * 64,
            "inshop_sha256": "1" * 64,
        },
        "model": {
            "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
            "checkpoint_sha256": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
        },
        "dataset": {
            "partition_sha256": "d" * 64,
            "optimization_identity_count": 3200,
            "optimization_image_count": 20_650,
            "fitting_image_count": 14_330,
            "validation_image_count": 3_188,
            "validation_class_count": 3_188,
            "singleton_class_count": 12,
            "excluded_same_series_count": 3_132,
            "represented_validation_count": 2_162,
            "unrepresented_validation_count": 1_026,
        },
        "protocol": {
            "holdout_fraction": 0.2,
            "holdout_seed": 0,
            "split_seed": 23_000,
            "fit_seeds": [0, 1, 2],
            "fit_steps": 512,
            "batch_size": 128,
            "batch_seed": 23_001,
            "mask_seed": 23_002,
            "evaluation_mask_seed": 23_003,
            "diagnostic_seed": 23_004,
            "gradient_seed": 23_005,
            "evaluation_mask_sets": 64,
            "shards": 8,
            "selected_features": 512,
            "margin": 0.25,
            "accuracy_margin": 0.0,
            "scale": 32.0,
            "optimizer": "AdamW(lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)",
            "row_norm": target,
            "paired_t_critical_df63": 1.998340542520741,
            "paired_t_critical_df3187": 1.9607086212236648,
            "positive_mask_minimum": 48,
            "head_cosine_minimum": 0.8,
            "head_cosine_mean_minimum": 0.95,
            "gradient_median_cosine_maximum": 0.995,
        },
        "class_mean": {
            "sha256": "e" * 64,
            "row_norm_min": target,
            "row_norm_max": target,
            "validation": _validation(1.0),
        },
        "probes": [_probe(seed) for seed in range(3)],
        "decision": {
            "status": "PROMOTE",
            "per_seed": [_seed_decision(seed) for seed in range(3)],
        },
        "runtime": {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "numpy": "2.5.0",
            "cuda": "13.0",
            "elapsed_seconds": 123.0,
            "peak_gpu_mib": 8192,
        },
    }


def test_parse_args_freezes_exact_screen_paths() -> None:
    args = MODULE.parse_args(
        [
            "--unicom-checkout",
            "/unicom",
            "--checkpoint",
            "/weights.pt",
            "--dataset-root",
            "/inshop",
            "--output",
            "/result.json",
        ]
    )

    assert args.unicom_checkout == Path("/unicom")
    assert args.checkpoint == Path("/weights.pt")
    assert args.dataset_root == Path("/inshop")
    assert args.output == Path("/result.json")
    assert args.batch_size == 128
    assert args.workers == 4


def test_validate_result_accepts_exact_recomputed_three_seed_decision() -> None:
    MODULE.validate_result(_valid_result())


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("decision", "status"), "CLOSE_DIRECTION"),
        (("decision", "per_seed", 1, "statistics", "paired_95_lower_bound"), 0.2),
        (("probes", 1, "validation", "mean_loss"), 1.1),
        (("probes", 1, "validation", "per_mask_mean_losses", 0), 1.1),
        (("probes", 1, "gradient", "cosine_median"), 0.999),
        (("probes", 1, "gradient", "class_mean_l2"), -1.0),
        (("probes", 1, "gradient", "cosine_min"), -1.1),
        (("probes", 1, "head_cosine", "mean"), 0.4),
        (("probes", 1, "row_norm_max"), 1.0),
        (("class_mean", "validation", "observation_count"), 1),
        (("dataset", "singleton_class_count"), 13),
        (("protocol", "fit_steps"), 511),
    ),
)
def test_validate_result_rejects_relational_or_protocol_drift(
    path: tuple[str | int, ...], replacement: object
) -> None:
    value = deepcopy(_valid_result())
    target: object = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        MODULE.validate_result(value)


def test_atomic_writer_strict_reloads_and_refuses_clobber(tmp_path: Path) -> None:
    output = tmp_path / "screen.json"
    value = _valid_result()

    MODULE.write_result_atomic(value, output)

    assert json.loads(output.read_text(encoding="utf-8")) == value


def test_atomic_writer_completes_partial_os_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "screen.json"
    value = _valid_result()
    original_write = MODULE.os.write
    calls = 0

    def partial_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        return original_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr(MODULE.os, "write", partial_write)

    MODULE.write_result_atomic(value, output)

    assert calls > 1
    assert MODULE.strict_json_object(output.read_bytes()) == value
    assert output.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        MODULE.write_result_atomic(value, output)
    assert json.loads(output.read_text(encoding="utf-8")) == value


def test_atomic_writer_rolls_back_output_after_post_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "screen.json"
    value = _valid_result()
    original_fsync = MODULE.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        MODULE.write_result_atomic(value, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_preserves_foreign_output_that_wins_link_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "screen.json"
    value = _valid_result()
    foreign = b"foreign publisher\n"
    original_link = MODULE.os.link

    def race_link(source: Path, destination: Path) -> None:
        destination.write_bytes(foreign)
        original_link(source, destination)

    monkeypatch.setattr(MODULE.os, "link", race_link)

    with pytest.raises(FileExistsError):
        MODULE.write_result_atomic(value, output)

    assert output.read_bytes() == foreign
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("nested_key", ("entry", "statistics", "predicates"))
def test_validate_result_rejects_reordered_nested_decision_schema(
    nested_key: str,
) -> None:
    value = deepcopy(_valid_result())
    entry = value["decision"]["per_seed"][0]
    if nested_key == "entry":
        value["decision"]["per_seed"][0] = dict(reversed(tuple(entry.items())))
    else:
        entry[nested_key] = dict(reversed(tuple(entry[nested_key].items())))

    with pytest.raises(ValueError, match="decision schema"):
        MODULE.validate_result(value)


def test_source_binding_authenticates_all_scientific_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path
    paths = (
        "scripts/screen_unicom_spherical_probe.py",
        "src/sfora/unicom_probe.py",
        "src/sfora/unicom_training.py",
        "src/sfora/unicom_inshop.py",
    )
    for index, relative in enumerate(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}".encode())
    monkeypatch.setattr(MODULE, "__file__", str(root / paths[0]))
    monkeypatch.setattr(MODULE, "_git_revision", lambda _root: "a" * 40)

    def git_show(command: list[str], **_kwargs: object) -> object:
        relative = command[-1].split(":", 1)[1]
        return type("Completed", (), {"stdout": (root / relative).read_bytes()})()

    monkeypatch.setattr(MODULE.subprocess, "run", git_show)

    revision, digests = MODULE._source_binding()

    assert revision == "a" * 40
    assert tuple(digests) == paths
    assert all(len(value) == 64 for value in digests.values())


def test_official_model_loader_imports_only_authenticated_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def load(name, *, download_root):\n"
        "    return (('model', name), ('transform', download_root))\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"weights")
    stale = tuple(name for name in sys.modules if name == "unicom" or name.startswith("unicom."))
    for name in stale:
        sys.modules.pop(name)
    try:
        model, transform = MODULE._load_official_model(checkout, checkpoint)
    finally:
        for name in tuple(sys.modules):
            if name == "unicom" or name.startswith("unicom."):
                sys.modules.pop(name)

    assert model == ("model", "ViT-L/14@336px")
    assert transform == ("transform", str(tmp_path))


def test_main_structural_failure_returns_two_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "screen.json"
    monkeypatch.setattr(
        MODULE, "run", lambda _args: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    status = MODULE.main(
        [
            "--unicom-checkout",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "weights.pt"),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert not output.exists()


def test_run_restores_all_rng_states_around_authenticated_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = MODULE.parse_args(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "weights.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    inventory = object()
    calls: list[object] = []
    monkeypatch.setattr(MODULE, "_authenticate_run", lambda actual: inventory, raising=False)

    def execute(actual: object, authenticated: object) -> dict[str, object]:
        calls.extend((actual, authenticated))
        random.random()
        np.random.random()
        torch.rand(3)
        return _valid_result()

    monkeypatch.setattr(MODULE, "_execute_screen", execute, raising=False)
    random.seed(9)
    np.random.seed(10)
    torch.manual_seed(11)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    assert MODULE.run(args) == _valid_result()

    assert calls == [args, inventory]
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_run_refuses_existing_destination_before_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    output.write_text("foreign\n", encoding="utf-8")
    args = MODULE.parse_args(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "weights.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output",
            str(output),
        ]
    )
    monkeypatch.setattr(
        MODULE,
        "_authenticate_run",
        lambda _args: pytest.fail("authentication should not run"),
        raising=False,
    )

    with pytest.raises(FileExistsError):
        MODULE.run(args)

    assert output.read_text(encoding="utf-8") == "foreign\n"


def test_authenticate_run_binds_model_partition_source_and_probe_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "unicom"
    dataset = tmp_path / "inshop"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkout.mkdir()
    partition.parent.mkdir(parents=True)
    partition.write_text("partition\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    output_parent = tmp_path / "results"
    output_parent.mkdir()
    base_records = tuple(
        InshopRecord("train", dataset / f"{index}_1_front.jpg", f"id{index}")
        for index in range(8)
    )
    optimization = (base_records * 2_582)[:20_650]
    labels = {f"id{index}": index for index in range(3200)}
    split = ProbeSplit(
        (base_records * 1_792)[:14_330],
        (base_records * 399)[:3_188],
        (True,) * 2_162 + (False,) * 1_026,
        3_188,
        12,
        3_132,
    )
    calls: list[object] = []
    monkeypatch.setattr(
        MODULE,
        "_git_revision",
        lambda path: calls.append(path) or MODULE.UNICOM_REVISION,
        raising=False,
    )
    monkeypatch.setattr(
        MODULE,
        "_sha256_file",
        lambda path: MODULE.CHECKPOINT_SHA256 if path == checkpoint else "d" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        MODULE, "parse_inshop_partition", lambda root: optimization, raising=False
    )
    monkeypatch.setattr(
        MODULE,
        "identity_holdout",
        lambda actual, *, fraction, seed: (
                calls.extend((actual, fraction, seed))
                or (optimization, (), (), labels)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        MODULE,
        "split_probe_records",
        lambda actual, actual_labels: (
            calls.extend((actual, actual_labels)) or split
        ),
        raising=False,
    )
    monkeypatch.setattr(
        MODULE,
        "_source_binding",
        lambda: (
            "a" * 40,
            {
                "scripts/screen_unicom_spherical_probe.py": "b" * 64,
                "src/sfora/unicom_probe.py": "c" * 64,
                "src/sfora/unicom_training.py": "f" * 64,
                "src/sfora/unicom_inshop.py": "1" * 64,
            },
        ),
        raising=False,
    )
    args = MODULE.parse_args(
        [
            "--unicom-checkout",
            str(checkout),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output_parent / "result.json"),
        ]
    )

    actual = MODULE._authenticate_run(args)

    assert actual.partition_sha256 == "d" * 64
    assert actual.optimization == optimization
    assert actual.labels == labels
    assert actual.split == split
    assert actual.source_revision == "a" * 40
    assert actual.script_sha256 == "b" * 64
    assert actual.module_sha256 == "c" * 64
    assert actual.training_sha256 == "f" * 64
    assert actual.inshop_sha256 == "1" * 64
    assert calls[0] == checkout


@pytest.mark.parametrize("mutation", ("revision", "checkpoint", "output_parent", "batch"))
def test_authenticate_run_rejects_invalid_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    checkout = tmp_path / "unicom"
    dataset = tmp_path / "inshop"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkout.mkdir()
    partition.parent.mkdir(parents=True)
    partition.write_text("partition\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    output_parent = tmp_path / "results"
    if mutation != "output_parent":
        output_parent.mkdir()
    monkeypatch.setattr(
        MODULE,
        "_git_revision",
        lambda _path: "0" * 40 if mutation == "revision" else MODULE.UNICOM_REVISION,
        raising=False,
    )
    monkeypatch.setattr(
        MODULE,
        "_sha256_file",
        lambda path: (
            "0" * 64
            if mutation == "checkpoint" and path == checkpoint
            else MODULE.CHECKPOINT_SHA256
            if path == checkpoint
            else "d" * 64
        ),
        raising=False,
    )
    monkeypatch.setattr(
        MODULE,
        "_source_binding",
        lambda: (
            "a" * 40,
            {
                "scripts/screen_unicom_spherical_probe.py": "b" * 64,
                "src/sfora/unicom_probe.py": "c" * 64,
                "src/sfora/unicom_training.py": "f" * 64,
                "src/sfora/unicom_inshop.py": "1" * 64,
            },
        ),
        raising=False,
    )
    args = MODULE.parse_args(
        [
            "--unicom-checkout",
            str(checkout),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output_parent / "result.json"),
            "--batch-size",
            "0" if mutation == "batch" else "128",
        ]
    )

    with pytest.raises((TypeError, ValueError, FileNotFoundError)):
        MODULE._authenticate_run(args)


def test_validate_inventory_rejects_wrong_registered_counts(tmp_path: Path) -> None:
    row = InshopRecord("train", tmp_path / "01_1_front.jpg", "id0")
    wrong = MODULE.ScreenInventory(
        partition_sha256="d" * 64,
        optimization=(row,) * 20_650,
        labels={f"id{index}": index for index in range(3199)},
        split=ProbeSplit(
            (row,) * 14_330,
            (row,) * 3_188,
            (True,) * 2_162 + (False,) * 1_026,
            3_188,
            12,
            3_132,
        ),
        source_revision="a" * 40,
        script_sha256="b" * 64,
        module_sha256="c" * 64,
        training_sha256="f" * 64,
        inshop_sha256="1" * 64,
    )

    with pytest.raises(ValueError):
        MODULE._validate_inventory(wrong)


def test_execute_screen_builds_and_validates_complete_three_seed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {f"id{index}": index for index in range(3200)}
    base_fitting = tuple(
        InshopRecord("train", tmp_path / f"{index}_1_front.jpg", label)
        for index, label in enumerate(labels)
    )
    fitting = (base_fitting * 5)[:14_330]
    validation = tuple(
        InshopRecord("train", tmp_path / f"{index}_2_side.jpg", f"id{index}")
        for index in range(3188)
    )
    inventory = MODULE.ScreenInventory(
        partition_sha256="d" * 64,
        optimization=fitting + validation + (fitting[0],) * 3_132,
        labels=labels,
        split=ProbeSplit(
            fitting=fitting,
            validation=validation,
            validation_group_represented=(True,) * 2162 + (False,) * 1026,
            validation_class_count=3188,
            singleton_class_count=12,
            excluded_same_series_count=3132,
        ),
        source_revision="a" * 40,
        script_sha256="b" * 64,
        module_sha256="c" * 64,
        training_sha256="f" * 64,
        inshop_sha256="1" * 64,
    )
    fitting_features = torch.ones(1, 768, dtype=torch.float32)
    validation_features = torch.ones(3188, 768, dtype=torch.float32)
    mean_head = torch.ones(3200, 768, dtype=torch.float32)
    mean_head *= (0.01 * math.sqrt(768.0)) / torch.linalg.vector_norm(mean_head, dim=1)[:, None]
    fit = ProbeFit(mean_head.clone(), 2.0, 1.9, 512, 0.8, 0.9, 0.97, 0.95)
    class_metrics = ProbeMetrics(
        1.0,
        160_000 / (3188 * 64),
        160_000,
        3188 * 64,
        (1.0,) * 64,
        (1.0,) * 64,
        (1.0,) * 64,
        (1.0,) * 3188,
        1.0,
        1.0,
    )
    probe_metrics = ProbeMetrics(
        0.9,
        160_000 / (3188 * 64),
        160_000,
        3188 * 64,
        (0.9,) * 64,
        (0.9,) * 64,
        (0.9,) * 64,
        (0.9,) * 3188,
        0.9,
        0.9,
    )
    gradient = ProbeGradientMetrics(
        1.0, 1.0, 0.1, 1.0, 0.98, 0.985, 0.99, 0.99, 0
    )
    encoded: list[tuple[tuple[InshopRecord, ...], tuple[InshopRecord, ...]]] = []
    monkeypatch.setattr(
        MODULE,
        "_encode_feature_sets",
        lambda _args, fit_rows, validation_rows: (
            encoded.append((fit_rows, validation_rows))
            or (fitting_features, validation_features)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        MODULE, "class_mean_head", lambda *_args, **_kwargs: mean_head, raising=False
    )
    monkeypatch.setattr(MODULE, "fit_spherical_probe", lambda *_args, **_kwargs: fit, raising=False)
    monkeypatch.setattr(
        MODULE,
        "evaluate_probe_heads",
        lambda *_args, **_kwargs: {
            "class_mean": class_metrics,
            "spherical_probe": probe_metrics,
        },
        raising=False,
    )
    monkeypatch.setattr(
        MODULE, "compare_probe_gradients", lambda *_args, **_kwargs: gradient, raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 8192 * 1024**2)
    args = MODULE.parse_args(
        [
            "--unicom-checkout",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "FP16-ViT-L-14-336px.pt"),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    actual = MODULE._execute_screen(args, inventory)

    MODULE.validate_result(actual)
    assert encoded == [(fitting, validation)]
    assert actual["decision"]["status"] == "PROMOTE"  # type: ignore[index]
    assert [probe["fit_seed"] for probe in actual["probes"]] == [0, 1, 2]  # type: ignore[index]
