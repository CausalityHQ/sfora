from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_sop_positive_coverage_official.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_sop_positive_coverage_official", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBJECT
SPEC.loader.exec_module(SUBJECT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_sha256(*values: torch.Tensor) -> str:
    return SUBJECT.parameter_sha256(*values)


def _write_panel(tmp_path: Path) -> tuple[Path, tuple[object, ...]]:
    base = tmp_path / "base.pt"
    base_state = {
        "weight": torch.nn.functional.pad(torch.eye(128), (0, 640)).contiguous(),
        "bias": torch.linspace(-0.1, 0.1, 128, dtype=torch.float32).contiguous(),
    }
    torch.save(base_state, base)
    base_parameter = _parameter_sha256(base_state["weight"], base_state["bias"])
    artifacts = []
    for seed in range(5):
        paths = {}
        arms = {}
        for name, scale in (("pooled", 1.0), ("coverage", 1.01)):
            path = tmp_path / f"seed{seed}-{name}.pt"
            state = {"weight": (torch.eye(128) * scale).float().contiguous()}
            torch.save(state, path)
            paths[name] = path
            arms[name] = {
                "checkpoint": {"bytes": path.stat().st_size, "sha256": _sha256(path)},
                "parameter_sha256": _parameter_sha256(state["weight"]),
                "score": {"packed_map_at_r": 0.5, "packed_per_query_ap": [0.5]},
            }
        receipt = {
            "arms": arms,
            "base": {
                "checkpoint_sha256": _sha256(base),
                "parameter_sha256": base_parameter,
                "score": {"packed_map_at_r": 0.4},
            },
            "claim_eligible": False,
            "dataset": "sop-official-train-class-disjoint-validation",
            "inputs": {
                "source_snapshot_sha256": "1" * 64,
                "teacher_snapshot_sha256": "2" * 64,
            },
            "official_test_touched": False,
            "passes": True,
            "schema": "sfora-positive-coverage-metric-screen-v2",
            "seed": seed,
            "source": {"driver_sha256": "3" * 64, "source_revision": "4" * 40},
        }
        receipt_path = tmp_path / f"seed{seed}.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        artifacts.append(
            SUBJECT.SopOfficialSeedArtifact(
                seed=seed,
                receipt=receipt_path,
                receipt_sha256=_sha256(receipt_path),
                pooled_checkpoint=paths["pooled"],
                coverage_checkpoint=paths["coverage"],
            )
        )
    return base, tuple(artifacts)


def _official_arguments(tmp_path: Path) -> list[str]:
    source = tmp_path / "source.npz"
    teacher = tmp_path / "teacher.npz"
    source.write_bytes(b"source")
    teacher.write_bytes(b"teacher")
    base, artifacts = _write_panel(tmp_path)
    arguments = [
        "--source-snapshot",
        str(source.resolve()),
        "--source-sha256",
        _sha256(source),
        "--teacher-snapshot",
        str(teacher.resolve()),
        "--teacher-sha256",
        _sha256(teacher),
        "--base-checkpoint",
        str(base.resolve()),
        "--base-checkpoint-sha256",
        _sha256(base),
        "--source-revision",
        "6" * 40,
        "--training-source-revision",
        "4" * 40,
        "--driver-sha256",
        "5" * 64,
        "--output",
        str((tmp_path / "official.json").resolve()),
    ]
    for artifact in artifacts:
        arguments.extend(
            (
                "--seed-artifact",
                str(artifact.seed),
                str(artifact.receipt.resolve()),
                artifact.receipt_sha256,
                str(artifact.pooled_checkpoint.resolve()),
                str(artifact.coverage_checkpoint.resolve()),
            )
        )
    arguments.append("--execute-official-evaluation")
    return arguments


def test_official_cli_is_strict_local_inference_only(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_official_args(_official_arguments(tmp_path))

    assert parsed.output == (tmp_path / "official.json").resolve()
    assert parsed.source_revision == "6" * 40
    assert parsed.training_source_revision == "4" * 40
    assert tuple(artifact.seed for artifact in parsed.seed_artifacts) == (0, 1, 2, 3, 4)
    for forbidden in ("--train", "--learning-rate", "--bucket", "--s3-uri"):
        with pytest.raises(ValueError, match="official argument authority"):
            SUBJECT.parse_official_args([*_official_arguments(tmp_path), forbidden, "x"])


def test_official_output_is_reserved_before_work_and_published_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "official.json"
    with SUBJECT.reserved_official_output(output) as reservation:
        assert reservation.path == tmp_path / "official.json.partial"
        assert reservation.path.exists()
        with (
            pytest.raises(FileExistsError),
            SUBJECT.reserved_official_output(output),
        ):
            pass
        SUBJECT.publish_reserved_official_output(reservation, b'{"complete":true}\n')

    assert output.read_bytes() == b'{"complete":true}\n'
    assert not (tmp_path / "official.json.partial").exists()


def test_official_output_preserves_interrupted_run_marker(tmp_path: Path) -> None:
    output = tmp_path / "official.json"
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        SUBJECT.reserved_official_output(output),
    ):
        raise RuntimeError("interrupted")

    assert (tmp_path / "official.json.partial").exists()
    with pytest.raises(FileExistsError), SUBJECT.reserved_official_output(output):
        pass


def test_frozen_panel_authenticates_all_five_seed_checkpoints(tmp_path: Path) -> None:
    base, artifacts = _write_panel(tmp_path)

    panel = SUBJECT.load_frozen_positive_coverage_panel(
        artifacts,
        base_checkpoint=base,
        base_checkpoint_sha256=_sha256(base),
        source_snapshot_sha256="1" * 64,
        teacher_snapshot_sha256="2" * 64,
        source_revision="4" * 40,
    )

    assert panel.seeds == (0, 1, 2, 3, 4)
    assert panel.base_weight.shape == (128, 768)
    assert len(panel.pooled_weights) == len(panel.coverage_weights) == 5
    torch.testing.assert_close(panel.coverage_weights[0], torch.eye(128) * 1.01)


@pytest.mark.parametrize(
    "mutation", ("receipt", "checkpoint", "seed", "source", "base", "coverage-shape")
)
def test_frozen_panel_rejects_identity_and_schema_drift(tmp_path: Path, mutation: str) -> None:
    base, artifacts_value = _write_panel(tmp_path)
    artifacts = list(artifacts_value)
    if mutation == "receipt":
        artifacts[0] = artifacts[0]._replace(receipt_sha256="0" * 64)
    elif mutation == "checkpoint":
        artifacts[0].coverage_checkpoint.write_bytes(b"drift")
    elif mutation == "seed":
        artifacts[0] = artifacts[0]._replace(seed=1)
    elif mutation == "source":
        receipt = json.loads(artifacts[0].receipt.read_bytes())
        receipt["source"]["source_revision"] = "5" * 40
        artifacts[0].receipt.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        artifacts[0] = artifacts[0]._replace(receipt_sha256=_sha256(artifacts[0].receipt))
    elif mutation == "coverage-shape":
        changed_weight = torch.eye(64)
        torch.save({"weight": changed_weight}, artifacts[0].coverage_checkpoint)
        receipt = json.loads(artifacts[0].receipt.read_bytes())
        receipt["arms"]["coverage"]["checkpoint"] = {
            "bytes": artifacts[0].coverage_checkpoint.stat().st_size,
            "sha256": _sha256(artifacts[0].coverage_checkpoint),
        }
        receipt["arms"]["coverage"]["parameter_sha256"] = _parameter_sha256(changed_weight)
        artifacts[0].receipt.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        artifacts[0] = artifacts[0]._replace(receipt_sha256=_sha256(artifacts[0].receipt))
    else:
        base.write_bytes(b"drift")

    with pytest.raises(ValueError, match="official panel authority"):
        SUBJECT.load_frozen_positive_coverage_panel(
            tuple(artifacts),
            base_checkpoint=base,
            base_checkpoint_sha256=_sha256(base) if mutation != "base" else "0" * 64,
            source_snapshot_sha256="1" * 64,
            teacher_snapshot_sha256="2" * 64,
            source_revision="4" * 40,
        )


def test_official_codes_apply_frozen_base_then_adapter(tmp_path: Path) -> None:
    base, artifacts = _write_panel(tmp_path)
    panel = SUBJECT.load_frozen_positive_coverage_panel(
        artifacts,
        base_checkpoint=base,
        base_checkpoint_sha256=_sha256(base),
        source_snapshot_sha256="1" * 64,
        teacher_snapshot_sha256="2" * 64,
        source_revision="4" * 40,
    )
    rows = torch.arange(2 * 768, dtype=torch.float32).reshape(2, 768).add_(1.0)

    base_codes, pooled_codes, coverage_codes = SUBJECT.official_panel_codes(rows, panel)

    expected = torch.nn.functional.normalize(
        torch.nn.functional.linear(
            torch.nn.functional.normalize(rows, dim=1),
            panel.base_weight,
            panel.base_bias,
        ),
        dim=1,
    )
    torch.testing.assert_close(base_codes, expected)
    torch.testing.assert_close(pooled_codes[0], expected)
    torch.testing.assert_close(coverage_codes[0], expected)


def test_official_result_recomputes_gates_and_is_canonical() -> None:
    def score(map_value: float, r1: float) -> dict[str, float]:
        return {
            "float_map_at_r": map_value + 0.001,
            "float_r1": r1 + 0.001,
            "packed_map_at_r": map_value,
            "packed_r1": r1,
        }

    result = SUBJECT.official_result(
        seed_rows=(
            {
                "seed": seed,
                "pooled": score(0.5, 0.8),
                "coverage": score(0.51, 0.801),
            }
            for seed in range(5)
        ),
        pooled_lower_bound=0.001,
        pooled_r1_lower_bound=-0.001,
        base_score=score(0.48, 0.78),
        teacher_score=score(0.52, 0.82),
        input_authority={"source_snapshot_sha256": "1" * 64},
        training_source_revision="3" * 40,
        source_revision="4" * 40,
    )

    assert result["passes"] is True
    assert result["official_test_touched"] is True
    assert result["coverage_minus_pooled_packed_map"] == pytest.approx(0.01)
    assert result["training_source_revision"] == "3" * 40
    assert result["pooled_class_cluster_r1_lower_bound"] == -0.001
    wire = SUBJECT.canonical_official_result_bytes(result)
    assert (
        wire
        == json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )

    drift = deepcopy(result)
    drift["passes"] = False
    with pytest.raises(ValueError, match="official result authority"):
        SUBJECT.canonical_official_result_bytes(drift)
