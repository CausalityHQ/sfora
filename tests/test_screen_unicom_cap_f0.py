from __future__ import annotations

import base64
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
import weakref
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_probe import ProbeSplit

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_unicom_cap_f0.py"
SPEC = importlib.util.spec_from_file_location("screen_unicom_cap_f0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _accept(_candidate: object) -> None:
    return None


def _metric(loss: float, correct: int) -> dict[str, object]:
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


def _valid_inventory() -> dict[str, object]:
    return {
        "authority": {
            "spec_path": "docs/spec.md",
            "spec_sha256": "1" * 64,
            "spec_commit": "1" * 40,
            "parent_path": "reports/parent.json",
            "parent_sha256": "2" * 64,
            "parent_source_commit": "2" * 40,
            "source_commit": "3" * 40,
            "handoff_commit": "4" * 40,
            "unicom_revision": "5" * 40,
            "checkpoint_sha256": "6" * 64,
            "partition_sha256": "7" * 64,
        },
        "runtime": {
            "python": "3.13.9",
            "torch": "2.12.1+cu130",
            "numpy": "2.5.0",
            "sklearn": "1.9.0",
            "cuda": "13.0",
            "device": "NVIDIA GB10",
            "model_dtype": "float32",
            "reduction_dtype": "float64",
        },
        "dataset": {
            "partition_sha256": "7" * 64,
            "optimization_identity_count": 2,
            "optimization_image_count": 4,
            "fitting_image_count": 4,
            "validation_image_count": 3_188,
            "validation_class_count": 3_188,
            "singleton_class_count": 1,
            "excluded_same_series_count": 1,
            "represented_validation_count": 2_162,
            "unrepresented_validation_count": 1_026,
        },
        "protocol": {
            "fit_seeds": [0, 1, 2],
            "snapshot_steps": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "evaluation_mask_sets": 64,
            "covariance_mask_sets": 2,
            "shards": 2,
            "feature_count": 2,
            "row_norm": 0.25,
            "paired_t_critical_df63": 1.998340542520741,
            "paired_t_critical_df3187": 1.9607086212236648,
            "loss_delta_minimum": 0.0501203852609845,
            "accuracy_delta_minimum": 0.006380126646800488,
            "non_worse_mask_minimum": 60,
            "head_cosine_mean_minimum": 0.95,
            "step_equivalence_minimum": 64,
        },
    }


def _summary(value: float) -> dict[str, float]:
    return {"minimum": value, "p05": value, "median": value, "mean": value}


def _statistics(loss: float, accuracy: float) -> dict[str, object]:
    return {
        "loss_delta": loss,
        "accuracy_delta": accuracy,
        "non_worse_mask_count": 64,
        "unrepresented_loss_delta": loss,
        "mask_paired_mean_delta": loss,
        "mask_paired_95_lower_bound": loss,
        "identity_paired_mean_delta": loss,
        "identity_paired_95_lower_bound": loss,
    }


def _static_predicates() -> dict[str, bool]:
    return {
        "loss_delta_at_least_0_0501203852609845": True,
        "accuracy_delta_at_least_0_006380126646800488": True,
        "mask_and_stratum_consistent": True,
        "paired_95_lower_bound_positive": True,
        "identity_95_lower_bound_positive": True,
    }


def _valid_result() -> tuple[dict[str, object], dict[str, object]]:
    inventory = _valid_inventory()
    matrix_bytes = struct.pack("<4d", 1.0, 0.0, 0.0, 2.0)
    probabilities = (1.0 / 3.0, 2.0 / 3.0)
    effective_rank = math.exp(
        -sum(value * math.log(value) for value in probabilities)
    )
    class_mean = _metric(1.0, 150_000)
    centered = _metric(0.75, 180_000)
    uncentered = _metric(0.875, 180_000)
    target = _metric(0.625, 185_000)
    losses = {
        0: 1.0,
        1: 0.984375,
        2: 0.96875,
        4: 0.953125,
        8: 0.9375,
        16: 0.921875,
        32: 0.90625,
        64: 0.875,
        128: 0.8125,
        256: 0.75,
        512: 0.625,
    }
    cap_metrics = {
        "cap_centered": {
            "validation": centered,
            "statistics": _statistics(
                0.25, centered["accuracy"] - class_mean["accuracy"]
            ),
            "predicates": _static_predicates(),
        },
        "cap_uncentered": {
            "validation": uncentered,
            "statistics": _statistics(
                0.125,
                uncentered["accuracy"] - class_mean["accuracy"],
            ),
            "predicates": _static_predicates(),
        },
    }
    seeds = []
    for seed in (0, 1, 2):
        seeds.append(
            {
                "fit_seed": seed,
                "fitted_target": {
                    "sha256": f"{seed + 8:x}" * 64,
                    "row_norm_min": 0.25,
                    "row_norm_max": 0.25,
                    "validation": deepcopy(target),
                },
                "trajectory": [
                    {
                        "step": step,
                        "sha256": (
                            "e" * 64
                            if step == 0
                            else (f"{seed + 8:x}" * 64 if step == 512 else f"{index + 1:x}" * 64)
                        ),
                        "validation": (
                            deepcopy(target)
                            if step == 512
                            else _metric(loss, 150_000)
                        ),
                    }
                    for index, (step, loss) in enumerate(losses.items())
                ],
                "cap_to_target": {
                    "cap_centered": {
                        "row_cosines": [0.98, 0.98],
                        "summary": _summary(0.98),
                    },
                    "cap_uncentered": {
                        "row_cosines": [0.97, 0.97],
                        "summary": _summary(0.97),
                    },
                },
                "step_equivalence": {"cap_centered": 256, "cap_uncentered": 64},
                "predicates": {
                    "cap_centered": {
                        "head_cosine_at_least_0_95": True,
                        "step_equivalence_at_least_64": True,
                    },
                    "cap_uncentered": {
                        "head_cosine_at_least_0_95": True,
                        "step_equivalence_at_least_64": True,
                    },
                },
            }
        )
    decision_variants = {
        "cap_centered": {
            "statistics": deepcopy(cap_metrics["cap_centered"]["statistics"]),
            "predicates": deepcopy(cap_metrics["cap_centered"]["predicates"]),
            "passes_static": True,
            "passes_all": True,
            "decision_level": 2,
            "min_step_equivalence": 256,
        },
        "cap_uncentered": {
            "statistics": deepcopy(cap_metrics["cap_uncentered"]["statistics"]),
            "predicates": deepcopy(cap_metrics["cap_uncentered"]["predicates"]),
            "passes_static": True,
            "passes_all": True,
            "decision_level": 2,
            "min_step_equivalence": 64,
        },
    }
    result = {
        "schema_version": "unicom-cap-f0-v1",
        "authority": deepcopy(inventory["authority"]),
        "runtime": {
            **inventory["runtime"],
            "elapsed_seconds": 1.0,
            "peak_gpu_mib": 1024,
        },
        "dataset": deepcopy(inventory["dataset"]),
        "protocol": deepcopy(inventory["protocol"]),
        "covariance": {
            "sample_count": 4,
            "feature_count": 2,
            "shrinkage": 0.5,
            "matrix_fp64_le_base64": base64.b64encode(matrix_bytes).decode("ascii"),
            "trace": 3.0,
            "cholesky_diagonal_min": 1.0,
            "cholesky_diagonal_max": math.sqrt(2.0),
            "sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "condition_number": 2.0,
            "effective_rank": effective_rank,
            "construction_mask_sha256": [["a" * 64, "b" * 64]] * 2,
            "mismatch": {
                "cap_centered": {
                    "row_cosines": [0.9] * 4,
                    "summary": _summary(0.9),
                },
                "cap_uncentered": {
                    "row_cosines": [0.8] * 4,
                    "summary": _summary(0.8),
                },
            },
        },
        "class_mean": {
            "sha256": "e" * 64,
            "row_norms": [0.25, 0.25],
            "row_norm_min": 0.25,
            "row_norm_max": 0.25,
            "validation": class_mean,
        },
        "cap_metrics": cap_metrics,
        "seeds": seeds,
        "decision": {
            "per_variant": decision_variants,
            "selected_variant": "cap_centered",
            "status": "PROCEED_STAGE_A",
        },
        "candidate_values_computed": True,
    }
    return result, inventory


def test_validate_result_accepts_exact_recursive_cap_fixture() -> None:
    value, inventory = _valid_result()

    MODULE.validate_result(value, inventory=inventory)


def test_validate_result_rejects_fitted_target_row_norm_drift() -> None:
    value, inventory = _valid_result()
    value["seeds"][1]["fitted_target"]["row_norm_min"] = 0.5
    value["seeds"][1]["fitted_target"]["row_norm_max"] = 0.5

    with pytest.raises(ValueError, match="fitted target row norm differs"):
        MODULE.validate_result(value, inventory=inventory)


def test_validate_result_rejects_class_mean_row_norm_drift() -> None:
    value, inventory = _valid_result()
    value["class_mean"]["row_norms"] = [0.5, 0.5]
    value["class_mean"]["row_norm_min"] = 0.5
    value["class_mean"]["row_norm_max"] = 0.5

    with pytest.raises(ValueError, match="class mean row norm differs"):
        MODULE.validate_result(value, inventory=inventory)


def test_validate_result_accepts_registered_fp32_row_norm_roundoff() -> None:
    value, inventory = _valid_result()
    value["class_mean"]["row_norms"] = [
        0.2499999701976776,
        0.2500000298023224,
    ]
    value["class_mean"]["row_norm_min"] = 0.2499999701976776
    value["class_mean"]["row_norm_max"] = 0.2500000298023224
    for seed in value["seeds"]:
        seed["fitted_target"]["row_norm_min"] = 0.24999994039535522
        seed["fitted_target"]["row_norm_max"] = 0.2500000596046448

    MODULE.validate_result(value, inventory=inventory)


def test_validate_result_accepts_cpu_peak_gpu_zero() -> None:
    value, inventory = _valid_result()
    value["runtime"]["peak_gpu_mib"] = 0

    MODULE.validate_result(value, inventory=inventory)


def _replace_nested(
    value: object, path: tuple[str | int, ...], replacement: object
) -> None:
    current = value
    for key in path[:-1]:
        if type(key) is int:
            assert type(current) is list
            current = current[key]
        else:
            assert type(current) is dict
            current = current[key]
    final = path[-1]
    if type(final) is int:
        assert type(current) is list
        current[final] = replacement
    else:
        assert type(current) is dict
        current[final] = replacement


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("runtime", "elapsed_seconds"), 0.0),
        (("runtime", "peak_gpu_mib"), -1),
        (("covariance", "sample_count"), 5),
        (("covariance", "feature_count"), 3),
        (("covariance", "trace"), 4.0),
        (("covariance", "sha256"), "f" * 64),
        (("covariance", "condition_number"), 3.0),
        (("covariance", "effective_rank"), 1.0),
        (("covariance", "mismatch", "cap_centered", "summary", "mean"), 0.75),
        (("class_mean", "row_norm_min"), 0.125),
        (("class_mean", "validation", "observation_count"), 1),
        (
            ("cap_metrics", "cap_centered", "validation", "per_mask_mean_losses"),
            [0.75] * 63,
        ),
        (("cap_metrics", "cap_centered", "statistics", "loss_delta"), 0.0),
        (("seeds", 0, "fit_seed"), True),
        (("seeds", 0, "trajectory", 0, "step"), False),
        (("seeds", 0, "trajectory", 0, "sha256"), "f" * 64),
        (
            ("seeds", 0, "cap_to_target", "cap_centered", "summary", "mean"),
            0.5,
        ),
        (("seeds", 0, "step_equivalence", "cap_centered"), True),
        (
            (
                "seeds",
                0,
                "predicates",
                "cap_centered",
                "head_cosine_at_least_0_95",
            ),
            False,
        ),
        (("decision", "selected_variant"), None),
        (("decision", "status"), "CLOSE_CAP"),
        (("candidate_values_computed",), False),
    ),
    ids=(
        "elapsed-runtime",
        "negative-peak-gpu-memory",
        "covariance-sample-count",
        "covariance-feature-count",
        "covariance-trace",
        "covariance-digest",
        "covariance-condition-number",
        "covariance-effective-rank",
        "mismatch-summary",
        "class-mean-row-extrema",
        "metric-observation-count",
        "mask-evidence-count",
        "cap-statistics",
        "seed-concrete-type",
        "trajectory-step-concrete-type",
        "trajectory-initial-head",
        "target-cosine-summary",
        "step-equivalence-concrete-type",
        "seed-predicate",
        "selected-variant",
        "decision-status",
        "candidate-computation-flag",
    ),
)
def test_validate_result_rejects_recursive_evidence_drift(
    path: tuple[str | int, ...], replacement: object
) -> None:
    value, inventory = _valid_result()
    _replace_nested(value, path, replacement)

    with pytest.raises((TypeError, ValueError)):
        MODULE.validate_result(value, inventory=inventory)


@pytest.mark.parametrize(
    "path",
    (
        (),
        ("covariance",),
        ("class_mean", "validation"),
        ("cap_metrics", "cap_centered", "statistics"),
        ("seeds", 0),
        ("seeds", 0, "trajectory", 0),
        ("decision",),
    ),
    ids=(
        "top-level",
        "covariance",
        "metric",
        "statistics",
        "seed",
        "trajectory",
        "decision",
    ),
)
def test_validate_result_rejects_nested_key_reordering(
    path: tuple[str | int, ...],
) -> None:
    value, inventory = _valid_result()
    current: object = value
    for key in path:
        if type(key) is int:
            assert type(current) is list
            current = current[key]
        else:
            assert type(current) is dict
            current = current[key]
    assert type(current) is dict
    reordered = dict(reversed(tuple(current.items())))
    if path:
        _replace_nested(value, path, reordered)
    else:
        value = reordered

    with pytest.raises(ValueError, match="schema differs"):
        MODULE.validate_result(value, inventory=inventory)


def test_parse_args_freezes_exact_cap_paths_and_replay_mode() -> None:
    args = MODULE.parse_args(
        [
            "--config",
            "/repo/docs/run.json",
            "--unicom-checkout",
            "/models/unicom",
            "--checkpoint",
            "/models/checkpoint.pt",
            "--dataset-root",
            "/data/inshop",
            "--parent-result",
            "/repo/reports/parent.json",
            "--output",
            "/repo/reports/cap.json",
            "--parent-replay-only",
        ]
    )

    assert vars(args) == {
        "config": Path("/repo/docs/run.json"),
        "unicom_checkout": Path("/models/unicom"),
        "checkpoint": Path("/models/checkpoint.pt"),
        "dataset_root": Path("/data/inshop"),
        "parent_result": Path("/repo/reports/parent.json"),
        "output": Path("/repo/reports/cap.json"),
        "parent_replay_only": True,
    }


@pytest.mark.parametrize(
    "payload",
    (
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b"[]",
    ),
)
def test_strict_json_object_rejects_duplicate_nonfinite_or_nonobject(
    payload: bytes,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MODULE.strict_json_object(payload)


def test_atomic_writer_completes_partial_writes_and_refuses_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap.json"
    value = {"status": "synthetic"}
    validator = _accept
    original_write = MODULE.os.write
    calls = 0

    def partial_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        return original_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr(MODULE.os, "write", partial_write)

    MODULE.write_result_atomic(value, output, validator=validator)

    assert calls > 1
    assert MODULE.strict_json_object(output.read_bytes()) == value
    assert output.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        MODULE.write_result_atomic(value, output, validator=validator)


def test_atomic_writer_rolls_back_only_its_owned_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap.json"
    value = {"status": "synthetic"}
    validator = _accept
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
        MODULE.write_result_atomic(value, output, validator=validator)

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_preserves_foreign_link_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap.json"
    value = {"status": "synthetic"}
    foreign = b"foreign publisher\n"
    validator = _accept
    original_link = MODULE.os.link

    def race_link(source: Path, destination: Path) -> None:
        destination.write_bytes(foreign)
        original_link(source, destination)

    monkeypatch.setattr(MODULE.os, "link", race_link)

    with pytest.raises(FileExistsError):
        MODULE.write_result_atomic(value, output, validator=validator)

    assert output.read_bytes() == foreign
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_validates_before_write_and_after_strict_reload(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cap.json"
    calls: list[object] = []

    def validator(candidate: object) -> None:
        calls.append(candidate)

    MODULE.write_result_atomic({"status": "synthetic"}, output, validator=validator)

    assert calls == [{"status": "synthetic"}, {"status": "synthetic"}]


def test_atomic_writer_rolls_back_when_reloaded_validation_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cap.json"
    calls = 0

    def validator(_candidate: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("reloaded result invalid")

    with pytest.raises(ValueError, match="reloaded result invalid"):
        MODULE.write_result_atomic({"status": "synthetic"}, output, validator=validator)

    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_rejects_nonfinite_before_creating_temp(tmp_path: Path) -> None:
    output = tmp_path / "cap.json"

    with pytest.raises(ValueError):
        MODULE.write_result_atomic(
            {"value": float("nan")}, output, validator=_accept
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_main_refuses_existing_output_before_candidate_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap.json"
    output.write_bytes(b"existing\n")
    called = False

    def candidate_run(_args: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "must not run"}

    monkeypatch.setattr(MODULE, "run", candidate_run)
    exit_code = MODULE.main(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert called is False
    assert output.read_bytes() == b"existing\n"


def test_main_binds_run_inventory_to_both_publication_validations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap.json"
    value, inventory = _valid_result()
    execution_inventory = MODULE.CapExecutionInventory(
        result=inventory,
        fitting=(),
        validation=(),
        validation_group_represented=(),
        labels={},
        class_mean_sha256="a" * 64,
        target_sha256_by_seed={0: "b" * 64, 1: "c" * 64, 2: "d" * 64},
        fit_steps=512,
        batch_size=128,
        peak_gpu_mib=17,
    )
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        MODULE, "run", lambda _args: (value, execution_inventory)
    )

    def inventory_bound_validator(candidate: object, *, inventory: object) -> None:
        calls.append((candidate, inventory))

    monkeypatch.setattr(MODULE, "validate_result", inventory_bound_validator)

    exit_code = MODULE.main(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls == [(value, inventory), (value, inventory)]
    assert MODULE.strict_json_object(output.read_bytes()) == value


def test_parent_replay_mode_never_calls_candidate_or_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cap.json"
    replay = {
        "class_mean_sha256": "a" * 64,
        "target_sha256_by_seed": {
            "0": "b" * 64,
            "1": "c" * 64,
            "2": "d" * 64,
        },
        "candidate_values_computed": False,
    }

    def forbidden_candidate(_args: object) -> dict[str, object]:
        raise AssertionError("candidate path reached")

    monkeypatch.setattr(MODULE, "run", forbidden_candidate)
    monkeypatch.setattr(MODULE, "run_parent_replay_preflight", lambda _args: replay)
    arguments = [
        "--config",
        str(tmp_path / "config.json"),
        "--unicom-checkout",
        str(tmp_path / "unicom"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--parent-result",
        str(tmp_path / "parent.json"),
        "--output",
        str(output),
        "--parent-replay-only",
    ]

    assert MODULE.main(arguments) == 0
    first = capsys.readouterr().out
    assert MODULE.main(arguments) == 0
    second = capsys.readouterr().out

    expected = json.dumps(replay, separators=(",", ":")) + "\n"
    assert first == expected
    assert second == expected
    assert not output.exists()


def test_parent_replay_mode_is_byte_identical_in_two_fresh_candidate_free_processes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cap.json"
    child = tmp_path / "replay_child.py"
    child.write_text(
        """
import importlib.util
import sys
from pathlib import Path

script = Path(sys.argv[1])
output = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("cap_replay_child", script)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
replay = {
    "class_mean_sha256": "a" * 64,
    "target_sha256_by_seed": {"0": "b" * 64, "1": "c" * 64, "2": "d" * 64},
    "candidate_values_computed": False,
}
module.run_parent_replay_preflight = lambda _args: replay
module.run = lambda _args: (_ for _ in ()).throw(
    AssertionError("candidate path reached")
)
arguments = [
    "--config", str(output.parent / "config.json"),
    "--unicom-checkout", str(output.parent / "unicom"),
    "--checkpoint", str(output.parent / "checkpoint.pt"),
    "--dataset-root", str(output.parent / "dataset"),
    "--parent-result", str(output.parent / "parent.json"),
    "--output", str(output),
    "--parent-replay-only",
]
code = module.main(arguments)
for forbidden in ("torch", "sklearn", "sfora.unicom_cap", "sfora.unicom_probe"):
    if forbidden in sys.modules:
        raise AssertionError(f"candidate import reached: {forbidden}")
if output.exists() or output.is_symlink():
    raise AssertionError("replay mode published a result")
sys.stderr.write("candidate-free\\n")
raise SystemExit(code)
""".lstrip(),
        encoding="utf-8",
    )
    command = [sys.executable, "-I", "-B", str(child), str(SCRIPT), str(output)]

    first = subprocess.run(command, check=False, capture_output=True)
    second = subprocess.run(command, check=False, capture_output=True)

    expected = (
        json.dumps(
            {
                "class_mean_sha256": "a" * 64,
                "target_sha256_by_seed": {
                    "0": "b" * 64,
                    "1": "c" * 64,
                    "2": "d" * 64,
                },
                "candidate_values_computed": False,
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    assert (first.returncode, first.stdout, first.stderr) == (
        0,
        expected,
        b"candidate-free\n",
    )
    assert (second.returncode, second.stdout, second.stderr) == (
        0,
        expected,
        b"candidate-free\n",
    )
    assert not output.exists()


def test_main_returns_structural_two_without_output_on_ordinary_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap.json"
    monkeypatch.setattr(MODULE, "run", lambda _args: (_ for _ in ()).throw(ValueError("bad")))

    exit_code = MODULE.main(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()
    assert not output.with_name(f".{output.name}.{os.getpid()}.tmp").exists()


@pytest.mark.cap_real_cpu
def test_execute_screen_runs_complete_real_math_tiny_cpu_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_inventory = _valid_inventory()
    result_inventory["dataset"] = {
        "partition_sha256": "7" * 64,
        "optimization_identity_count": 8,
        "optimization_image_count": 24,
        "fitting_image_count": 16,
        "validation_image_count": 8,
        "validation_class_count": 8,
        "singleton_class_count": 0,
        "excluded_same_series_count": 0,
        "represented_validation_count": 4,
        "unrepresented_validation_count": 4,
    }
    result_inventory["protocol"] = {
        **result_inventory["protocol"],
        "covariance_mask_sets": 8,
        "shards": 8,
        "feature_count": 768,
        "row_norm": 0.27712812921102037,
    }
    labels = {f"id{index}": index for index in range(8)}
    fitting = tuple(
        InshopRecord(
            "train",
            tmp_path / f"fit-{row}.jpg",
            f"id{row // 2}",
        )
        for row in range(16)
    )
    validation = tuple(
        InshopRecord("train", tmp_path / f"validation-{row}.jpg", f"id{row}")
        for row in range(8)
    )
    inventory = MODULE.CapExecutionInventory(
        result=result_inventory,
        fitting=fitting,
        validation=validation,
        validation_group_represented=(True,) * 4 + (False,) * 4,
        labels=labels,
        class_mean_sha256="56b015cdf07f02d7608dd3642bbb913307f30128a3f524aa609c99acf8ba7133",
        target_sha256_by_seed={
            0: "6db88dd23bb4f4fb7d836e01d75df418f898318a766e76223f748f83c0b1f084",
            1: "5f8f5bcbb0670e62e983c9c3409a39ba121445b2b1acab7ff82c55eb27855122",
            2: "8f3293fb7aca735b167d80cb16f4bfd457a19ee084325fbf0ca251446e731a4e",
        },
        fit_steps=512,
        batch_size=8,
        peak_gpu_mib=0,
    )
    feature_references: list[weakref.ReferenceType[torch.Tensor]] = []

    def encode(
        _args: object,
        fitting_rows: tuple[InshopRecord, ...],
        validation_rows: tuple[InshopRecord, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert fitting_rows == fitting
        assert validation_rows == validation
        fitting_features = torch.zeros(16, 768, dtype=torch.float32)
        fitting_labels = torch.arange(8, dtype=torch.int64).repeat_interleave(2)
        for row in range(16):
            label = int(fitting_labels[row])
            fitting_features[row, label] = 1.0
            fitting_features[row, 8 + label] = 0.2 if row % 2 == 0 else -0.2
            fitting_features[row, 16 + row] = 0.05
        validation_features = torch.zeros(8, 768, dtype=torch.float32)
        for row in range(8):
            validation_features[row, row] = 1.0
            validation_features[row, 8 + row] = 0.1
            validation_features[row, 64 + row] = 0.025
        fitting_features = fitting_features.contiguous()
        validation_features = validation_features.contiguous()
        feature_references.extend(
            (weakref.ref(fitting_features), weakref.ref(validation_features))
        )
        return fitting_features, validation_features

    monkeypatch.setattr(MODULE, "_encode_feature_sets", encode, raising=False)
    args = MODULE.parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(tmp_path / "cap.json"),
        ]
    )

    started = time.monotonic()
    result = MODULE.execute_screen(args, inventory)
    elapsed = time.monotonic() - started

    MODULE.validate_result(result, inventory=result_inventory)
    assert result["class_mean"]["sha256"] == inventory.class_mean_sha256
    assert [seed["fitted_target"]["sha256"] for seed in result["seeds"]] == [
        inventory.target_sha256_by_seed[seed] for seed in (0, 1, 2)
    ]
    assert [snapshot["step"] for snapshot in result["seeds"][0]["trajectory"]] == [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    assert result["candidate_values_computed"] is True
    MODULE.write_result_atomic(
        result,
        args.output,
        validator=lambda candidate: MODULE.validate_result(
            candidate, inventory=result_inventory
        ),
    )
    assert MODULE.strict_json_object(args.output.read_bytes()) == result
    gc.collect()
    assert feature_references and all(reference() is None for reference in feature_references)
    assert elapsed < 90.0


@pytest.mark.parametrize("mismatch_seed", (None, 0, 1, 2))
def test_execute_screen_rejects_parent_primitive_mismatch_before_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch_seed: int | None,
) -> None:
    result_inventory = _valid_inventory()
    result_inventory["dataset"] = {
        "partition_sha256": "7" * 64,
        "optimization_identity_count": 2,
        "optimization_image_count": 4,
        "fitting_image_count": 4,
        "validation_image_count": 2,
        "validation_class_count": 2,
        "singleton_class_count": 0,
        "excluded_same_series_count": 0,
        "represented_validation_count": 1,
        "unrepresented_validation_count": 1,
    }
    result_inventory["protocol"] = {
        **result_inventory["protocol"],
        "evaluation_mask_sets": 2,
        "covariance_mask_sets": 2,
        "shards": 2,
        "feature_count": 2,
    }
    fitting = tuple(
        InshopRecord("train", tmp_path / f"fit-{index}.jpg", f"id{index // 2}")
        for index in range(4)
    )
    validation = tuple(
        InshopRecord("train", tmp_path / f"validation-{index}.jpg", f"id{index}")
        for index in range(2)
    )
    class_head = torch.tensor([[0.25, 0.0], [0.0, 0.25]], dtype=torch.float32)
    target_heads = {
        seed: torch.tensor(
            [[0.25, seed * 0.001], [seed * 0.001, 0.25]], dtype=torch.float32
        )
        for seed in (0, 1, 2)
    }
    metric = MODULE._metric_from_json(
        {
            "mean_loss": 1.0,
            "accuracy": 0.5,
            "correct_count": 2,
            "observation_count": 4,
            "per_mask_mean_losses": [1.0, 1.0],
            "per_mask_represented_mean_losses": [1.0, 1.0],
            "per_mask_unrepresented_mean_losses": [1.0, 1.0],
            "per_image_mean_losses": [1.0, 1.0],
            "represented_mean_loss": 1.0,
            "unrepresented_mean_loss": 1.0,
        },
        mask_count=2,
        image_count=2,
        name="parent fixture metric",
    )
    class_metric_sha = MODULE._sha256_bytes(
        MODULE._canonical_bytes(MODULE._metric_payload(metric))
    )
    target_metric_shas = {seed: class_metric_sha for seed in (0, 1, 2)}
    if mismatch_seed is None:
        class_metric_sha = "0" * 64
    else:
        target_metric_shas[mismatch_seed] = "0" * 64
    inventory = MODULE.CapExecutionInventory(
        result=result_inventory,
        fitting=fitting,
        validation=validation,
        validation_group_represented=(True, False),
        labels={"id0": 0, "id1": 1},
        class_mean_sha256=MODULE._tensor_sha256(class_head),
        target_sha256_by_seed={
            seed: MODULE._tensor_sha256(head) for seed, head in target_heads.items()
        },
        parent_class_mean_metric_sha256=class_metric_sha,
        parent_target_metric_sha256_by_seed=target_metric_shas,
        fit_steps=1,
        batch_size=2,
        peak_gpu_mib=0,
    )
    monkeypatch.setattr(
        MODULE,
        "_encode_feature_sets",
        lambda *_args: (
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            torch.eye(2),
        ),
    )
    monkeypatch.setattr(
        "sfora.unicom_probe.class_mean_head", lambda *_args, **_kwargs: class_head
    )
    monkeypatch.setattr(
        "sfora.unicom_probe.fit_spherical_probe_trajectory",
        lambda *_args, fit_seed, **_kwargs: (
            SimpleNamespace(head=target_heads[fit_seed]),
            {
                step: class_head if step == 0 else target_heads[fit_seed]
                for step in result_inventory["protocol"]["snapshot_steps"]
            },
        ),
    )
    monkeypatch.setattr(
        "sfora.unicom_probe.evaluate_probe_head", lambda *_args, **_kwargs: metric
    )
    cap_calls: list[bool] = []

    def forbidden_cap(*_args: object, **_kwargs: object) -> object:
        cap_calls.append(True)
        raise AssertionError("CAP was constructed before parent reproduction")

    monkeypatch.setattr("sfora.unicom_cap.build_cap_heads", forbidden_cap)
    args = MODULE.parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(tmp_path / "cap.json"),
        ]
    )

    with pytest.raises(ValueError, match="parent .* metric differs"):
        MODULE.execute_screen(args, inventory)

    assert cap_calls == []


def test_run_authenticates_and_builds_inventory_before_scientific_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = MODULE.parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(tmp_path / "cap.json"),
        ]
    )
    authenticated = {"config": {"environment": _valid_inventory()["runtime"]}}
    inventory = object()
    value = {"candidate_values_computed": True}
    calls: list[str] = []

    def authenticate(_args: object) -> object:
        calls.append("authenticate")
        return authenticated

    def build(_args: object, authority: object) -> object:
        assert authority is authenticated
        calls.append("parent_inventory")
        return inventory

    def execute(_args: object, execution_inventory: object) -> tuple[object, object]:
        assert execution_inventory is inventory
        calls.append("scientific")
        return value, inventory

    monkeypatch.setattr(MODULE, "authenticate_run", authenticate)
    monkeypatch.setattr(MODULE, "_build_execution_inventory", build, raising=False)
    monkeypatch.setattr(
        MODULE,
        "_validate_runtime",
        lambda *_args: calls.append("runtime"),
        raising=False,
    )
    monkeypatch.setattr(MODULE, "_execute_with_runtime_observation", execute)

    actual, actual_inventory = MODULE.run(args)

    assert (actual, actual_inventory) == (value, inventory)
    assert calls == ["authenticate", "runtime", "parent_inventory", "scientific"]


def test_build_execution_inventory_excludes_query_gallery_and_binds_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_path = Path(__file__).parents[1] / MODULE._FROZEN_PARENT["path"]
    parent = MODULE.strict_json_object(parent_path.read_bytes())
    train_row = InshopRecord("train", tmp_path / "train.jpg", "id0")
    query_row = InshopRecord("query", tmp_path / "query.jpg", "heldout")
    gallery_row = InshopRecord("gallery", tmp_path / "gallery.jpg", "heldout")
    optimization = (train_row,) * 20_650
    labels = {f"id{index}": index for index in range(3_200)}
    fitting = (train_row,) * 14_330
    validation = tuple(
        InshopRecord("train", tmp_path / f"validation-{index}.jpg", f"id{index}")
        for index in range(3_188)
    )
    split = ProbeSplit(
        fitting=fitting,
        validation=validation,
        validation_group_represented=(True,) * 2_162 + (False,) * 1_026,
        validation_class_count=3_188,
        singleton_class_count=12,
        excluded_same_series_count=3_132,
    )
    calls: list[str] = []

    def parse(_root: object) -> tuple[InshopRecord, ...]:
        calls.append("partition")
        return (train_row, query_row, gallery_row)

    def holdout(
        rows: tuple[InshopRecord, ...], *, fraction: float, seed: int
    ) -> tuple[object, ...]:
        assert rows == (train_row,)
        assert (fraction, seed) == (0.2, 0)
        calls.append("train_only")
        return optimization, (query_row,), (gallery_row,), labels

    def split_records(
        rows: tuple[InshopRecord, ...], actual_labels: object, *, seed: int
    ) -> ProbeSplit:
        assert rows is optimization
        assert actual_labels is labels
        assert seed == 23_000
        calls.append("split")
        return split

    monkeypatch.setattr("sfora.unicom_inshop.parse_inshop_partition", parse)
    monkeypatch.setattr("sfora.unicom_training.identity_holdout", holdout)
    monkeypatch.setattr("sfora.unicom_probe.split_probe_records", split_records)
    config = {
        "spec": MODULE._FROZEN_SPEC,
        "parent": MODULE._FROZEN_PARENT,
        "environment": MODULE._FROZEN_ENVIRONMENT,
        "inputs": MODULE._FROZEN_INPUTS,
        "protocol": MODULE._FROZEN_PROTOCOL,
        "source": {"commit": "a" * 40},
    }
    args = MODULE.parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(parent_path),
            "--output",
            str(tmp_path / "cap.json"),
        ]
    )

    inventory = MODULE._build_execution_inventory(
        args,
        {
            "config": config,
            "repo_root": Path(__file__).parents[1],
            "source_commit": "a" * 40,
            "head": "b" * 40,
        },
    )

    assert calls == ["partition", "train_only", "split"]
    assert inventory.fitting is fitting
    assert inventory.validation is validation
    assert inventory.class_mean_sha256 == parent["class_mean"]["sha256"]
    assert inventory.target_sha256_by_seed == {
        row["fit_seed"]: row["sha256"] for row in parent["probes"]
    }
    assert inventory.parent_class_mean_metric_sha256 == MODULE._sha256_bytes(
        MODULE._canonical_bytes(parent["class_mean"]["validation"])
    )
    assert inventory.parent_target_metric_sha256_by_seed == {
        row["fit_seed"]: MODULE._sha256_bytes(
            MODULE._canonical_bytes(row["validation"])
        )
        for row in parent["probes"]
    }
    assert query_row not in inventory.fitting + inventory.validation
    assert gallery_row not in inventory.fitting + inventory.validation


def test_run_restores_python_numpy_and_torch_rng_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = MODULE.parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(tmp_path / "cap.json"),
        ]
    )
    authenticated = {"config": {"environment": _valid_inventory()["runtime"]}}
    inventory = object()
    monkeypatch.setattr(MODULE, "authenticate_run", lambda _args: authenticated)
    monkeypatch.setattr(MODULE, "_validate_runtime", lambda _expected: None)
    monkeypatch.setattr(MODULE, "_build_execution_inventory", lambda *_args: inventory)
    observed_torch_threads: list[int] = []

    def mutate_rng(_args: object, _inventory: object) -> dict[str, object]:
        observed_torch_threads.append(torch.get_num_threads())
        random.seed(90_001)
        np.random.seed(90_002)
        torch.manual_seed(90_003)
        return {"candidate_values_computed": True}

    monkeypatch.setattr(
        MODULE,
        "_execute_with_runtime_observation",
        lambda actual_args, actual_inventory: (
            mutate_rng(actual_args, actual_inventory),
            actual_inventory,
        ),
    )
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    torch_threads_before = torch.get_num_threads()

    MODULE.run(args)

    numpy_after = np.random.get_state()
    assert random.getstate() == python_before
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert observed_torch_threads == [1]
    assert torch.get_num_threads() == torch_threads_before


def test_encode_feature_sets_uses_only_fitting_then_validation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows: list[InshopRecord] = []
    for index, value in enumerate((10, 20, 30)):
        path = tmp_path / f"row-{index}.png"
        Image.new("RGB", (2, 2), (value, 0, 0)).save(path)
        rows.append(InshopRecord("train", path, f"id{index}"))
    seen_batches: list[list[float]] = []

    def transform(image: Image.Image) -> torch.Tensor:
        return torch.tensor([float(image.getpixel((0, 0))[0])], dtype=torch.float32)

    class Model:
        def cuda(self) -> Model:
            return self

        def eval(self) -> Model:
            return self

        def __call__(self, batch: torch.Tensor) -> torch.Tensor:
            seen_batches.append(batch[:, 0].tolist())
            return batch.repeat(1, 768).contiguous()

    monkeypatch.setattr(
        MODULE,
        "_load_official_model",
        lambda *_args: (Model(), transform),
        raising=False,
    )
    monkeypatch.setattr(
        torch.Tensor, "cuda", lambda self, non_blocking=False: self, raising=False
    )
    args = SimpleNamespace(
        unicom_checkout=tmp_path / "unicom", checkpoint=tmp_path / "checkpoint.pt"
    )

    fitting, validation = MODULE._encode_feature_sets(
        args, tuple(rows[:2]), tuple(rows[2:]), loader_workers=0
    )

    assert seen_batches == [[10.0, 20.0, 30.0]]
    assert fitting.shape == (2, 768)
    assert validation.shape == (1, 768)
    assert fitting[:, 0].tolist() == [10.0, 20.0]
    assert validation[:, 0].tolist() == [30.0]


def test_validate_runtime_requires_exact_observed_cuda_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    expected = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "sklearn": str(sklearn.__version__),
        "cuda": str(torch.version.cuda),
        "device": "Test GPU",
        "model_dtype": "float32",
        "reduction_dtype": "float64",
    }

    MODULE._validate_runtime(expected)
    for key in expected:
        mutated = dict(expected)
        mutated[key] = "wrong"
        with pytest.raises(ValueError, match="CAP runtime differs"):
            MODULE._validate_runtime(mutated)


def test_parent_replay_preflight_authenticates_without_candidate_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = MODULE.parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--parent-result",
            str(tmp_path / "parent.json"),
            "--output",
            str(tmp_path / "cap.json"),
            "--parent-replay-only",
        ]
    )
    authenticated = {"config": {"environment": _valid_inventory()["runtime"]}}
    inventory = object()
    replay = {
        "class_mean_sha256": "a" * 64,
        "target_sha256_by_seed": {"0": "b" * 64, "1": "c" * 64, "2": "d" * 64},
        "candidate_values_computed": False,
    }
    calls: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "authenticate_run",
        lambda _args: calls.append("authenticate") or authenticated,
    )
    monkeypatch.setattr(
        MODULE, "_validate_runtime", lambda _expected: calls.append("runtime")
    )
    monkeypatch.setattr(
        MODULE,
        "_build_execution_inventory",
        lambda *_args: calls.append("parent_inventory") or inventory,
    )
    monkeypatch.setattr(
        MODULE,
        "_compute_parent_replay",
        lambda _args, actual: (
            calls.append("parent_replay") or replay
            if actual is inventory
            else (_ for _ in ()).throw(AssertionError("inventory differs"))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        MODULE,
        "execute_screen",
        lambda *_args: (_ for _ in ()).throw(AssertionError("candidate reached")),
    )

    assert MODULE.run_parent_replay_preflight(args) == replay
    assert calls == ["authenticate", "runtime", "parent_inventory", "parent_replay"]


def test_execute_with_runtime_observation_records_synchronized_peak_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, result_inventory = _valid_result()
    inventory = MODULE.CapExecutionInventory(
        result=result_inventory,
        fitting=(),
        validation=(),
        validation_group_represented=(),
        labels={},
        class_mean_sha256="a" * 64,
        target_sha256_by_seed={0: "b" * 64, 1: "c" * 64, 2: "d" * 64},
        fit_steps=512,
        batch_size=128,
        peak_gpu_mib=0,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "execute_screen",
        lambda _args, actual: calls.append("execute") or value
        if actual is inventory
        else (_ for _ in ()).throw(AssertionError("inventory differs")),
    )
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda: int(16.25 * 1024**2)
    )

    actual, observed_inventory = MODULE._execute_with_runtime_observation(
        SimpleNamespace(output=tmp_path / "cap.json"), inventory
    )

    assert calls == ["reset", "execute", "synchronize"]
    assert actual["runtime"]["peak_gpu_mib"] == 17
    assert observed_inventory.peak_gpu_mib == 17
    MODULE.validate_result(actual, inventory=result_inventory)
