from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_unicom_cap_f0.py"
SPEC = importlib.util.spec_from_file_location("screen_unicom_cap_f0_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _valid_config() -> dict[str, object]:
    source_paths = (
        "pyproject.toml",
        "scripts/screen_unicom_cap_f0.py",
        "src/sfora/unicom_cap.py",
        "src/sfora/unicom_probe.py",
        "src/sfora/unicom_training.py",
        "src/sfora/unicom_inshop.py",
        "tests/test_screen_unicom_cap_f0.py",
        "tests/test_unicom_cap.py",
        "tests/test_unicom_probe.py",
        "tests/test_unicom_cap_f0_run_config.py",
    )
    return {
        "schema_version": "unicom-cap-f0-run-v1",
        "spec": {
            "path": "docs/superpowers/specs/2026-08-25-unicom-cap-f0-design.md",
            "sha256": "cf6994c9bda0677a714cd0a12dcca459af0fe610d28a9869091992724a4e880a",
            "commit": "cfd2ebf18b4d3a2c80c3b96957d777e23224a4cc",
        },
        "parent": {
            "path": "reports/generated/unicom-spherical-probe-ed2e789.json",
            "sha256": "d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf",
            "source_commit": "ed2e7893b05d3b5105ff992691efccc5b13ad5a0",
            "artifact_commit": "d07ff819eccafdfd048f19ff8b4d22c7ea1ed20f",
        },
        "environment": {
            "python": "3.13.9",
            "torch": "2.12.1+cu130",
            "numpy": "2.5.0",
            "sklearn": "1.9.0",
            "cuda": "13.0",
            "device": "NVIDIA GB10",
            "model_dtype": "float32",
            "reduction_dtype": "float64",
        },
        "inputs": {
            "unicom_checkout": "/home/riomus/UniCOM",
            "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
            "checkpoint": "/home/riomus/checkpoints/FP16-ViT-L-14-336px.pt",
            "checkpoint_sha256": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
            "dataset_root": "/home/riomus/datasets/In-shop Clothes Retrieval Benchmark",
            "partition": (
                "/home/riomus/datasets/In-shop Clothes Retrieval Benchmark/"
                "Eval/list_eval_partition.txt"
            ),
            "partition_sha256": "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c",
        },
        "protocol": {
            "holdout_fraction": 0.2,
            "holdout_seed": 0,
            "split_seed": 23_000,
            "fit_seeds": [0, 1, 2],
            "fit_steps": 512,
            "snapshot_steps": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "batch_size": 128,
            "batch_seed": 23_001,
            "mask_seed": 23_002,
            "evaluation_mask_seed": 23_003,
            "diagnostic_seed": 23_004,
            "gradient_seed": 23_005,
            "covariance_mask_seed": 23_006,
            "evaluation_mask_sets": 64,
            "covariance_mask_sets": 8,
            "shards": 8,
            "selected_features": 512,
            "feature_count": 768,
            "margin": 0.25,
            "accuracy_margin": 0.0,
            "scale": 32.0,
            "optimizer": "AdamW(lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)",
            "row_norm": 0.27712812921102037,
            "paired_t_critical_df63": 1.998340542520741,
            "paired_t_critical_df3187": 1.9607086212236648,
            "loss_delta_minimum": 0.0501203852609845,
            "accuracy_delta_minimum": 0.006380126646800488,
            "non_worse_mask_minimum": 60,
            "head_cosine_mean_minimum": 0.95,
            "step_equivalence_minimum": 64,
        },
        "source": {
            "commit": "a" * 40,
            "files": [{"path": path, "sha256": "b" * 64} for path in source_paths],
        },
        "handoff": {
            "parent_commit": "a" * 40,
            "sole_path": "docs/unicom_cap_f0_run_config.json",
            "detached_clean": True,
        },
        "result": {
            "relative_path": "reports/generated/unicom-cap-f0-aaaaaaa.json",
            "schema_version": "unicom-cap-f0-v1",
        },
    }


def test_validate_run_config_accepts_exact_future_schema() -> None:
    MODULE.validate_run_config(_valid_config())


@pytest.mark.parametrize("mutation", ("top_order", "nested_order", "extra", "source_order"))
def test_validate_run_config_rejects_schema_or_source_order_drift(mutation: str) -> None:
    config = _valid_config()
    if mutation == "top_order":
        config = {
            "spec": config["spec"],
            "schema_version": config["schema_version"],
            **dict(list(config.items())[2:]),
        }
    elif mutation == "nested_order":
        spec = config["spec"]
        config["spec"] = {"sha256": spec["sha256"], "path": spec["path"], "commit": spec["commit"]}
    elif mutation == "extra":
        config["result"]["extra"] = False
    else:
        config["source"]["files"] = list(reversed(config["source"]["files"]))

    with pytest.raises((TypeError, ValueError)):
        MODULE.validate_run_config(config)


def test_validate_run_config_rejects_wrong_output_binding() -> None:
    config = deepcopy(_valid_config())
    config["result"]["relative_path"] = "reports/generated/other.json"

    with pytest.raises(ValueError):
        MODULE.validate_run_config(config)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    (
        ("spec", "path", "docs/other.md"),
        ("spec", "sha256", "f" * 64),
        ("spec", "commit", "f" * 40),
        ("parent", "sha256", "f" * 64),
        ("parent", "source_commit", "f" * 40),
        ("parent", "artifact_commit", "f" * 40),
        ("environment", "python", "2.7"),
        ("environment", "device", "cpu"),
        ("inputs", "unicom_revision", "f" * 40),
        ("inputs", "checkpoint_sha256", "f" * 64),
        ("inputs", "partition_sha256", "f" * 64),
        ("protocol", "fit_steps", 8),
        ("protocol", "margin", 0.5),
        ("protocol", "non_worse_mask_minimum", 1),
        ("protocol", "fit_seeds", [False, True, 2]),
    ),
)
def test_validate_run_config_rejects_frozen_authority_or_protocol_drift(
    section: str, key: str, replacement: object
) -> None:
    config = _valid_config()
    config[section][key] = replacement

    with pytest.raises((TypeError, ValueError)):
        MODULE.validate_run_config(config)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _authority_fixture(tmp_path: Path) -> tuple[Namespace, dict[str, object]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "CAP Test")
    _git(repo, "config", "user.email", "cap@example.test")

    config = _valid_config()
    spec = repo / config["spec"]["path"]
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_bytes(b"frozen CAP specification\n")
    config["spec"]["sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
    _git(repo, "add", str(spec.relative_to(repo)))
    _git(repo, "commit", "-qm", "frozen specification")
    config["spec"]["commit"] = _git(repo, "rev-parse", "HEAD")

    parent_producer = repo / "scripts" / "parent_producer.py"
    parent_producer.parent.mkdir(parents=True, exist_ok=True)
    parent_producer.write_bytes(b"# frozen parent producer\n")
    _git(repo, "add", str(parent_producer.relative_to(repo)))
    _git(repo, "commit", "-qm", "reviewed parent producer")
    config["parent"]["source_commit"] = _git(repo, "rev-parse", "HEAD")
    parent_result = repo / config["parent"]["path"]
    parent_result.parent.mkdir(parents=True, exist_ok=True)
    parent_result.write_bytes(b'{"parent":"frozen"}\n')
    config["parent"]["sha256"] = hashlib.sha256(parent_result.read_bytes()).hexdigest()
    _git(repo, "add", "-f", str(parent_result.relative_to(repo)))
    _git(repo, "commit", "-qm", "registered parent result")
    config["parent"]["artifact_commit"] = _git(repo, "rev-parse", "HEAD")

    source_files = config["source"]["files"]
    for index, row in enumerate(source_files):
        path = repo / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}\n".encode())
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"partition")
    config["inputs"].update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "dataset_root": str(dataset),
            "partition": str(partition),
            "partition_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        }
    )

    unicom = tmp_path / "unicom"
    unicom.mkdir()
    _git(unicom, "init", "-q")
    _git(unicom, "config", "user.name", "CAP Test")
    _git(unicom, "config", "user.email", "cap@example.test")
    (unicom / "model.py").write_bytes(b"model\n")
    _git(unicom, "add", "model.py")
    _git(unicom, "commit", "-qm", "model")
    config["inputs"]["unicom_checkout"] = str(unicom)
    config["inputs"]["unicom_revision"] = _git(unicom, "rev-parse", "HEAD")

    _git(repo, "add", *(row["path"] for row in source_files))
    _git(repo, "commit", "-qm", "reviewed source")
    source_commit = _git(repo, "rev-parse", "HEAD")
    config["source"]["commit"] = source_commit
    config["handoff"]["parent_commit"] = source_commit
    config["result"]["relative_path"] = (
        f"reports/generated/unicom-cap-f0-{source_commit[:7]}.json"
    )

    config_path = repo / config["handoff"]["sole_path"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    _git(repo, "add", str(config_path.relative_to(repo)))
    _git(repo, "commit", "-qm", "config-only handoff")
    _git(repo, "checkout", "-q", "--detach", "HEAD")

    output = repo / config["result"]["relative_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    args = Namespace(
        config=config_path,
        unicom_checkout=unicom,
        checkpoint=checkpoint,
        dataset_root=dataset,
        parent_result=parent_result,
        output=output,
        parent_replay_only=False,
    )
    return args, config


def _patch_frozen_authorities(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object]
) -> None:
    for section in ("spec", "parent", "environment", "inputs", "protocol"):
        monkeypatch.setattr(MODULE, f"_FROZEN_{section.upper()}", config[section])


def test_authenticate_run_accepts_exact_linear_config_only_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, expected = _authority_fixture(tmp_path)
    _patch_frozen_authorities(monkeypatch, expected)
    monkeypatch.setattr(MODULE, "__file__", str(args.config.parents[1] / "scripts" / SCRIPT.name))

    authenticated = MODULE.authenticate_run(args)

    assert authenticated["config"] == expected
    assert authenticated["repo_root"] == args.config.parents[1]
    assert len(
        {
            expected["spec"]["commit"],
            expected["parent"]["source_commit"],
            expected["parent"]["artifact_commit"],
            expected["source"]["commit"],
        }
    ) == 4


def test_checked_in_run_config_binds_parent_source_and_artifact_commits() -> None:
    repo = SCRIPT.parents[1]
    config_path = repo / "docs" / "unicom_cap_f0_run_config.json"
    config = MODULE.strict_json_object(config_path.read_bytes())
    MODULE.validate_run_config(config)
    parent = config["parent"]
    parent_bytes = (repo / parent["path"]).read_bytes()

    assert hashlib.sha256(parent_bytes).hexdigest() == parent["sha256"]
    assert (
        subprocess.run(
            ["git", "show", f"{parent['artifact_commit']}:{parent['path']}"],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        == parent_bytes
    )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            parent["source_commit"],
            parent["artifact_commit"],
        ],
        cwd=repo,
        check=True,
    )


@pytest.mark.parametrize("mutation", ("wrong_flag", "dirty_source", "existing_output"))
def test_authenticate_run_rejects_path_source_or_destination_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    args, config = _authority_fixture(tmp_path)
    _patch_frozen_authorities(monkeypatch, config)
    monkeypatch.setattr(MODULE, "__file__", str(args.config.parents[1] / "scripts" / SCRIPT.name))
    if mutation == "wrong_flag":
        args.checkpoint = tmp_path / "other.pt"
        args.checkpoint.write_bytes(b"checkpoint")
    elif mutation == "dirty_source":
        (args.config.parents[1] / "src/sfora/unicom_cap.py").write_bytes(b"dirty\n")
    else:
        args.output.write_bytes(b"existing")

    with pytest.raises((TypeError, ValueError, FileExistsError)):
        MODULE.authenticate_run(args)


@pytest.mark.parametrize(
    "mutation", ("attached_head", "noisy_handoff", "intervening_commit", "config_blob_drift")
)
def test_authenticate_run_rejects_git_topology_or_blob_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    args, config = _authority_fixture(tmp_path)
    _patch_frozen_authorities(monkeypatch, config)
    repo = args.config.parents[1]
    monkeypatch.setattr(MODULE, "__file__", str(repo / "scripts" / SCRIPT.name))
    config_bytes = args.config.read_bytes()
    if mutation == "attached_head":
        _git(repo, "switch", "-qc", "attached")
    elif mutation == "config_blob_drift":
        args.config.write_bytes(config_bytes + b"\n")
    else:
        _git(repo, "checkout", "-q", config["source"]["commit"])
        extra = repo / "unregistered.txt"
        extra.write_bytes(b"unregistered\n")
        if mutation == "intervening_commit":
            _git(repo, "add", str(extra.relative_to(repo)))
            _git(repo, "commit", "-qm", "intervening commit")
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_bytes(config_bytes)
        _git(repo, "add", str(args.config.relative_to(repo)))
        if mutation == "noisy_handoff":
            _git(repo, "add", str(extra.relative_to(repo)))
        _git(repo, "commit", "-qm", "mutated handoff")
        _git(repo, "checkout", "-q", "--detach", "HEAD")

    with pytest.raises((subprocess.CalledProcessError, TypeError, ValueError)):
        MODULE.authenticate_run(args)
