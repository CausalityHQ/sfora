#!/usr/bin/env python3
"""Run the authenticated UniCOM FEPF campaign one retained process at a time."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sfora.atomic_publication import BudgetedPublisher, publish_bytes_noreplace

RUNTIME_ORDER = ("current", "composed", "composed", "current") * 2
CONFIRMATION_PAIRS = (
    (7, 20_260_828), (8, 271_828), (9, 314_159),
    (10, 1_618_033), (11, 57_721),
)
QUALITY_PROFILE_ORDER = ("control", "candidate", "candidate", "control")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_registered_command_vectors(config: object, *, checkout_root: Path) -> None:
    value = _required_config(config)
    commands = value["commands"]
    trainer = _load_module(checkout_root / "scripts/train_unicom_inshop.py", "fepf_trainer_cli")
    profiler = _load_module(
        checkout_root / "scripts/profile_unicom_training_step.py", "fepf_profiler_cli"
    )
    evaluator = _load_module(
        checkout_root / "scripts/evaluate_unicom_fepf.py", "fepf_evaluator_cli"
    )
    runtime = ["/tmp/out.json" if item == "{output}" else item for item in commands["runtime"][0]]
    profiler.parse_args(runtime[4:])
    trainer.parse_args(
        _train_command(list(commands["train"]), mode="imprinted", training_seed=0,
                       holdout_seed=0, stop=4, output=Path("/tmp/run"))[4:]
    )
    evaluator.parse_args([
        "--phase", "epoch4", "--sources", "/tmp/sources.json",
        "--sources-sha256", "a" * 64, "--sources-bytes", "1",
        "--evidence-root", "/tmp/evidence", "--output", "/tmp/result.json",
        "--temporary", "/tmp/result.tmp", "--config", "/tmp/config.json",
    ])
    quality = [
        *commands["profile_quality"], "--runtime-mode", "current",
        "--run-checkpoint", "/tmp/checkpoint.pt", "--run-receipt",
        "/tmp/run-receipt.json", "--output", "/tmp/profile.json",
        "--config", "/tmp/config.json",
    ]
    profiler.parse_args(quality[4:])
    for phase in ("epoch4", "exploratory", "confirmation"):
        parsed = evaluator.parse_args([
            "--phase", phase, "--sources", "/tmp/sources.json",
            "--sources-sha256", "a" * 64, "--sources-bytes", "1",
            "--evidence-root", "/tmp/evidence", "--output", "/tmp/result.json",
            "--temporary", "/tmp/result.tmp", "--config", "/tmp/config.json",
        ])
        if parsed.phase != phase:
            raise ValueError("registered evaluator phase differs")


def select_runtime_from_receipts(receipts: object, *, checkout_root: Path) -> str:
    profiler = _load_module(
        checkout_root / "scripts/profile_unicom_training_step.py", "fepf_profiler_decision"
    )
    decision = profiler.compare_runtime_smoke(tuple(receipts))
    if decision not in {"PASS_CURRENT", "PASS_COMPOSED"}:
        raise ValueError("runtime smoke is structurally invalid")
    return decision


def apply_runtime_selection(command: list[str], decision: str, *, profile: bool) -> list[str]:
    if decision not in {"PASS_CURRENT", "PASS_COMPOSED"}:
        raise ValueError("runtime selection differs")
    result = list(command)
    if profile:
        index = result.index("--runtime-mode") + 1
        result[index] = "composed" if decision == "PASS_COMPOSED" else "current"
    elif decision == "PASS_COMPOSED":
        result.extend(("--compile", "--fused", "--no-ema"))
    return result


def resolve_canary_environment_commands(
    config: dict[str, object], environment_sha256: str
) -> None:
    """Resolve the sole post-canary digest placeholder in registered commands."""

    if not isinstance(environment_sha256, str) or len(environment_sha256) != 64:
        raise ValueError("CUDA environment digest differs")

    def resolve(value: object) -> object:
        if type(value) is dict:
            return {key: resolve(item) for key, item in value.items()}
        if type(value) is list:
            return [resolve(item) for item in value]
        if value == "{cuda_environment_sha256}":
            return environment_sha256
        if type(value) is str and "{" in value and value != "{output}":
            raise ValueError("unregistered command placeholder differs")
        return value

    commands = config.get("commands")
    if type(commands) is not dict:
        raise ValueError("registered commands differ")
    resolved = resolve(commands)
    commands.clear()
    commands.update(resolved)


def validate_profile_environment(terminal: object, expected: object) -> None:
    if (
        type(terminal) is not dict
        or type(expected) is not dict
        or terminal.get("environment") != expected
    ):
        raise ValueError("profile environment differs from CUDA canary authority")


def _command_argument(command: object, option: str) -> str:
    if type(command) is not list or option not in command:
        raise ValueError("runtime command authority differs")
    index = command.index(option)
    if index + 1 >= len(command) or type(command[index + 1]) is not str:
        raise ValueError("runtime command authority differs")
    return command[index + 1]


def validate_runtime_terminal(
    stage: dict[str, object], terminal: object, *, profiler: object,
    expected_environment: object,
) -> None:
    command = stage["command"]
    validator = getattr(profiler, "validate_runtime_profile", None)
    if not callable(validator):
        raise ValueError("public runtime validator differs")
    validator(
        terminal,
        expected_mode=_command_argument(command, "--runtime-mode"),
        checkpoint=Path(_command_argument(command, "--run-checkpoint")),
        run_receipt=Path(_command_argument(command, "--run-receipt")),
        config=Path(_command_argument(command, "--config")),
        expected_environment=expected_environment,
    )


class RegisteredTerminalValidator:
    def __init__(self, *, checkout_root: Path, config: dict[str, object]) -> None:
        self.checkout_root = checkout_root
        self.config = config
        self.trainer = _load_module(checkout_root / "scripts/train_unicom_inshop.py", "fepf_tv")
        self.profiler = _load_module(
            checkout_root / "scripts/profile_unicom_training_step.py", "fepf_pv"
        )
        self.evaluator = _load_module(
            checkout_root / "scripts/evaluate_unicom_fepf.py", "fepf_ev"
        )
        self.canary = _load_module(
            checkout_root / "scripts/run_unicom_fepf_cuda_canary.py", "fepf_cv"
        )
        self.profile_environment: dict[str, object] | None = None
        self._validated_canary_digests: set[str] = set()

    def __call__(self, stage: dict[str, object], terminal: object) -> None:
        name = str(stage["name"])
        if name == "cuda-canary":
            authority = self.config["cuda_canary_authority"]
            if authority == {}:
                environment_path = Path(
                    self.config["cuda_canary_environment"]["path"]
                )
                environment_payload = environment_path.read_bytes()
                environment = json.loads(environment_payload)
                authority = {
                    "device_uuid": environment["device_uuid"],
                    "environment_sha256": _sha256(environment_payload),
                }
            self.canary.validate_cuda_canary_receipt(
                terminal,
                self.config,
                expected_device_uuid=authority["device_uuid"],
                expected_environment_sha256=authority["environment_sha256"],
            )
            manifest_path = (
                Path(self.config["artifact_root"])
                / "preflight/canary-evidence/manifest.json"
            )
            manifest_payload = manifest_path.read_bytes()
            manifest_authority = {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_payload),
                "bytes": len(manifest_payload),
            }
            if (
                type(terminal) is not dict
                or terminal.get("evidence_manifest_sha256")
                != manifest_authority["sha256"]
            ):
                raise ValueError("canary evidence manifest authority differs")
            self.canary.validate_canary_evidence_manifest(
                manifest_authority, evidence_root=manifest_path.parent
            )
            environment = terminal.get("environment") if type(terminal) is dict else None
            if type(environment) is not dict:
                raise ValueError("canary profile environment authority differs")
            validation_digest = _sha256(
                manifest_payload + b"\0" + _canonical_json(terminal)
            )
            if validation_digest not in self._validated_canary_digests:
                if stage.get("fresh_execution") is True:
                    # The just-completed child verified fitted authority before
                    # publishing its terminal; the controller stays CUDA-free.
                    self.canary.reconstruct_canary_authority(
                        self.config, manifest_path, terminal=terminal
                    )
                else:
                    deterministic = environment.get("deterministic_execution")
                    if (
                        type(deterministic) is not dict
                        or deterministic.get("cublas_workspace_config") != ":4096:8"
                    ):
                        raise ValueError("registered canary child environment differs")
                    child_environment = dict(os.environ)
                    inherited = child_environment.get("CUBLAS_WORKSPACE_CONFIG")
                    if inherited not in (None, ":4096:8"):
                        raise ValueError("registered canary child environment differs")
                    child_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    command = list(stage.get("command", ()))
                    if not command or not all(type(item) is str for item in command):
                        raise ValueError("registered canary resume command differs")
                    root = Path(self.config["artifact_root"])
                    if "--publication-stage" not in command:
                        command.extend(("--publication-stage", "cuda-canary"))
                    if "--campaign-root" not in command:
                        command.extend(("--campaign-root", str(root)))
                    completed = subprocess.run(
                        command,
                        cwd=self.checkout_root,
                        env=child_environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if completed.returncode != 0:
                        raise ValueError("registered canary resume child failed")
                    self.canary.reconstruct_canary_authority(
                        self.config, manifest_path, terminal=terminal
                    )
                self._validated_canary_digests.add(validation_digest)
            self.profile_environment = environment
        elif "profile" in name or name.startswith("runtime-"):
            if self.profile_environment is None:
                raise ValueError("canary environment must precede profiling")
            if "profile" in name:
                self.profiler.validate_quality_profile(terminal)
                validate_profile_environment(terminal, self.profile_environment)
            else:
                validate_runtime_terminal(
                    stage, terminal, profiler=self.profiler,
                    expected_environment=self.profile_environment,
                )
        elif "decision" in name:
            self.evaluator.validate_fepf_result(
                terminal,
                Path(stage["evidence_root"]),
                sources_authority=stage["sources_authority"],
            )
        else:
            self.trainer.validate_training_run_receipt_v2(
                terminal, evidence_root=Path(stage["destination"])
            )


def prepare_campaign_storage(
    config: dict[str, object], *, physical_admission: bool = True
) -> Path:
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"), "fepf_builder_storage"
    )
    root = Path(config["artifact_root"])
    if not os.path.lexists(root):
        if physical_admission:
            builder.prepare_artifact_root(
                root, required_bytes=config["artifact_budget_bytes"],
                required_inodes=config["artifact_budget_inodes"],
            )
        else:
            if (
                not root.is_absolute()
                or root.parent.is_symlink()
                or not root.parent.is_dir()
            ):
                raise ValueError("artifact parent differs")
            root.mkdir(mode=0o700)
    else:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("artifact root differs")
        if physical_admission:
            require_campaign_remaining_capacity(config, root)
    preflight = root / "preflight"
    if not os.path.lexists(preflight):
        preflight.mkdir(mode=0o700)
    budget_path = root / config["publication_budget_path"]
    budget_payload = _canonical_json(config["publication_budget"])
    if _sha256(budget_payload) != config["publication_budget_sha256"]:
        raise ValueError("embedded publication budget differs")
    self_rows = [
        row for row in config["publication_budget"]["publications"]
        if row.get("name") == "campaign:publication-budget"
        and row.get("path") == config["publication_budget_path"]
    ]
    if len(self_rows) != 1 or len(budget_payload) > self_rows[0]["persistent_bytes"]:
        raise ValueError("embedded publication budget self-row differs")
    if os.path.lexists(budget_path):
        if (
            budget_path.is_symlink()
            or budget_path.read_bytes() != budget_payload
        ):
            raise FileExistsError(budget_path)
    else:
        published = publish_bytes_noreplace(
            budget_path,
            budget_payload,
            validator=lambda payload: (
                None
                if payload == budget_payload
                else (_ for _ in ()).throw(ValueError("publication budget differs"))
            ),
        )
        published.close()
    orphaned_status = root / ".controller-status.json.tmp"
    if os.path.lexists(orphaned_status):
        if orphaned_status.is_symlink() or not orphaned_status.is_file():
            raise ValueError("campaign status temporary path differs")
        orphaned_status.unlink()
        root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
    return root


def require_campaign_remaining_capacity(config: dict[str, object], root: Path) -> None:
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"), "fepf_builder_capacity"
    )
    consumed_bytes = 0
    consumed_inodes = 1
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("campaign artifact symlink differs")
        consumed_inodes += 1
        if path.is_file():
            consumed_bytes += path.stat().st_size
    builder.require_remaining_capacity(
        root, total_budget_bytes=config["artifact_budget_bytes"],
        total_budget_inodes=config["artifact_budget_inodes"],
        consumed_bytes=consumed_bytes, consumed_inodes=consumed_inodes,
    )


def load_campaign_resume(config: dict[str, object]) -> dict[str, object]:
    root = Path(config["artifact_root"])
    if not root.exists():
        return {}
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"),
        "fepf_builder_resume_validation",
    )
    builder.validate_campaign_resume(
        config, root, terminal_validator=lambda path: json.loads(path.read_bytes())
    )
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"), "fepf_builder_resume"
    )
    inventory = set(builder.registered_stage_inventory(config))
    result: dict[str, object] = {}
    canary = root / config["cuda_canary_receipt"]
    if os.path.lexists(canary):
        if canary.is_symlink() or not canary.is_file():
            raise ValueError("campaign resume canary differs")
        result["cuda-canary"] = json.loads(canary.read_bytes())
    for child in root.iterdir():
        if child.name in {
            "preflight", "controller-status.json", ".controller-status.json.tmp"
        }:
            continue
        if child.name.endswith("-sources.json") or child.name.endswith("-result.json"):
            continue
        if child.name not in inventory or child.is_symlink() or not child.is_dir():
            raise ValueError("campaign resume stage differs")
        candidates = tuple(
            path for path in (
                child / "terminal.json", child / "run-receipt.json"
            ) if path.is_file() and not path.is_symlink()
        )
        if len(candidates) != 1:
            raise ValueError("campaign resume terminal differs")
        result[child.name] = json.loads(candidates[0].read_bytes())
    for name in inventory:
        result_path = root / f"{name}-result.json"
        if result_path.is_file() and not result_path.is_symlink():
            result[name] = json.loads(result_path.read_bytes())
    return result


def _resume_stage(config: dict[str, object], name: str) -> dict[str, object]:
    root = Path(config["artifact_root"])
    if name == "cuda-canary":
        return _stage(
            name, list(config["cuda_canary_command"]), root,
            terminal_path=root / config["cuda_canary_receipt"],
        )
    if name.startswith("runtime-"):
        index = int(name.rsplit("-", 1)[1])
        output = root / name / "terminal.json"
        command = [
            str(output) if item == "{output}" else item
            for item in config["commands"]["runtime"][index]
        ]
        return _stage(name, command, root)
    if name.endswith("-decision"):
        sources = root / f"{name}-sources.json"
        payload = sources.read_bytes()
        stage = _stage(
            name, [], root, terminal_path=root / f"{name}-result.json"
        )
        stage["sources_authority"] = {
            "path": str(sources.resolve()), "sha256": _sha256(payload),
            "bytes": len(payload),
        }
        stage["evidence_root"] = root
        return stage
    terminal = (
        root / name / "terminal.json"
        if "profile" in name
        else root / name / "run-receipt.json"
    )
    return _stage(name, [], root, terminal_path=terminal)


def prevalidate_campaign_resume(
    config: dict[str, object], prior: Mapping[str, object],
    *, terminal_validator: Callable[[dict[str, object], object], None],
    checkout_root: Path,
) -> None:
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"),
        "fepf_builder_resume_order",
    )
    order = tuple(builder.registered_stage_inventory(config))
    unknown = set(prior) - set(order)
    if unknown:
        raise ValueError("campaign resume stage differs")
    indices = [index for index, name in enumerate(order) if name in prior]
    if indices and set(indices) != set(range(max(indices) + 1)):
        raise ValueError("campaign resume chain is incomplete")
    for name in order:
        if name in prior:
            terminal_validator(_resume_stage(config, name), prior[name])
    runtime_names = tuple(f"runtime-{index:02d}" for index in range(8))
    resumed_runtime = tuple(prior[name] for name in runtime_names if name in prior)
    if resumed_runtime and len(resumed_runtime) != 8:
        raise ValueError("runtime resume chain is incomplete")
    if resumed_runtime:
        select_runtime_from_receipts(resumed_runtime, checkout_root=checkout_root)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish_evaluation_sources(
    root: Path,
    name: str,
    sources: object,
    *,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    path = root / f"{name}-sources.json"
    payload = _canonical_json(sources)
    publisher = None
    if config is not None:
        publisher = BudgetedPublisher(
            campaign_root=root,
            budget_path=root / config["publication_budget_path"],
            budget_sha256=config["publication_budget_sha256"],
            exact_budget=config["publication_budget"],
        )
        publisher.validate_payload(
            name=f"{name}:sources", destination=path, payload=payload
        )
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(path)
    else:
        def validator(persisted: bytes) -> None:
            if persisted != payload:
                raise ValueError("evaluation sources differ")

        published = (
            publish_bytes_noreplace(path, payload, validator=validator)
            if publisher is None
            else publisher.publish_bytes(
                name=f"{name}:sources",
                destination=path,
                payload=payload,
                validator=validator,
            )
        )
        published.close()
    return {"path": str(path.resolve()), "sha256": _sha256(payload), "bytes": len(payload)}


def validate_recoverable_publication_prefix(root: Path) -> None:
    for source in root.glob("*-sources.json"):
        if source.is_symlink() or not source.is_file():
            raise ValueError("recoverable publication prefix differs")
        json.loads(source.read_bytes())
    for result in root.glob("*-result.json"):
        source = root / result.name.replace("-result.json", "-sources.json")
        if not source.is_file() or source.is_symlink():
            raise ValueError("publication result lacks source authority")


def run_fresh_process_contract_preflight(
    *, checkout_root: Path, config_path: Path, artifact_root: Path
) -> None:
    config = json.loads(config_path.read_bytes())
    environment = config["cuda_canary_environment"]
    environment_path = Path(environment["path"])
    environment_payload = environment_path.read_bytes()
    environment_object = json.loads(environment_payload)
    if (
        environment_path.is_symlink()
        or _canonical_json(environment_object) != environment_payload
    ):
        raise ValueError("post-canary environment authority differs")
    environment_sha256 = _sha256(environment_payload)
    budget = artifact_root / config["publication_budget_path"]
    stage = "exploratory-control-stage4"
    subprocess.run(
        [
            sys.executable, "-I", "-B", "scripts/train_unicom_inshop.py",
            "--unicom-checkout", str(checkout_root), "--checkpoint", str(config_path),
            "--dataset-root", str(checkout_root), "--output-dir",
            str(artifact_root / stage), "--run-config", str(config_path),
            "--environment-authority", str(environment_path),
            "--environment-sha256", environment_sha256,
            "--publication-budget", str(budget), "--publication-budget-sha256",
            config["publication_budget_sha256"], "--run-arm", stage,
            "--classifier-init", "imprinted", "--stop-after-epoch", "4",
            "--authority-preflight-only",
        ],
        cwd=checkout_root,
        check=True,
    )
    resolve_canary_environment_commands(config, environment_sha256)
    runtime_output = artifact_root / "runtime-00/terminal.json"
    runtime_command = [
        str(runtime_output) if item == "{output}" else item
        for item in config["commands"]["runtime"][0]
    ]
    runtime_command.extend((
        "--publication-stage", "runtime-00", "--campaign-root", str(artifact_root),
        "--authority-preflight-only",
    ))
    subprocess.run(
        runtime_command,
        cwd=checkout_root,
        check=True,
    )
    decision = "exploratory-epoch4-decision"
    subprocess.run(
        [
            sys.executable, "-I", "-B", "scripts/evaluate_unicom_fepf.py",
            "--phase", "epoch4", "--sources", str(config_path),
            "--sources-sha256", _sha256(config_path.read_bytes()), "--sources-bytes",
            str(config_path.stat().st_size), "--evidence-root", str(artifact_root),
            "--output", str(artifact_root / f"{decision}-result.json"),
            "--temporary", str(artifact_root / ".evaluation.tmp"),
            "--config", str(config_path), "--publication-stage", decision,
            "--campaign-root", str(artifact_root), "--authority-preflight-only",
        ],
        cwd=checkout_root,
        check=True,
    )


def run_non_authentic_partial_cpu_preflight(
    *, source_root: Path, workspace: Path, stop_before_cuda: bool
) -> dict[str, object]:
    """Exercise post-builder CLI seams with synthesized, non-authentic inputs.

    The real four-argument builder is invoked and must stop at the absent
    target-only authorities.  The remainder uses a synthesized config and is
    parser/preflight evidence only, never a committed-crossing claim.
    """

    if stop_before_cuda is not True:
        raise ValueError("contract preflight must stop before CUDA")
    checkout = workspace / "committed-checkout"
    checkout.mkdir()
    registered = (
        "scripts/build_unicom_fepf_run_config.py",
        "scripts/run_unicom_fepf_campaign.py",
        "scripts/run_unicom_fepf_cuda_canary.py",
        "scripts/train_unicom_inshop.py",
        "scripts/profile_unicom_training_step.py",
        "scripts/evaluate_unicom_fepf.py",
        "src/sfora/unicom_retrieval_audit.py",
        "src/sfora/unicom_inshop.py",
        "src/sfora/unicom_fepf.py",
        "src/sfora/atomic_publication.py",
        "pyproject.toml",
        "uv.lock",
    )
    for relative in registered:
        source = source_root / relative
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Contract Fixture"], cwd=checkout,
                   check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=checkout, check=True,
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "contract source"], cwd=checkout, check=True)
    partition = {
        "query_rows": 2, "gallery_rows": 4,
        "maximum_relevant_count": 2, "maximum_path_bytes": 64,
    }
    environment = {
        "python_vv": "fixture", "torch": "fixture", "torchvision": "fixture",
        "timm": "fixture", "numpy": "fixture", "cuda": "fixture",
        "cudnn": "fixture", "compile": {"available": "False", "inductor": "none"},
        "device_uuid": "GPU-contract", "gpu_inventory": ["fixture"],
        "pyproject_sha256": "1" * 64, "uv_lock_sha256": "2" * 64,
        "deterministic_execution": {
            "deterministic_algorithms": True,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
    }
    fixture_root = workspace / "external-authorities"
    fixture_root.mkdir()
    builder = _load_module(
        checkout / "scripts/build_unicom_fepf_run_config.py", "contract_builder"
    )
    trainer = _load_module(
        checkout / "scripts/train_unicom_inshop.py", "contract_legacy_trainer"
    )
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    signature = trainer.build_inference_signature(
        model, descriptor=torch.zeros((1, 512), dtype=torch.float32)
    )
    structure = {
        "schema": "unicom-fepf-structure-v1",
        "tensors": [
            {key: value for key, value in row.items() if key != "sha256"}
            for row in signature["tensors"]
        ],
        "classifier": {
            "shape": [4, 2], "dtype": "torch.float32", "numel": 8,
            "element_size": 4, "bytes": 32,
        },
        "operations": list(signature["operations"]),
    }
    canonical = builder.canonical_json_bytes
    structure_path = fixture_root / "structure.json"
    signature_path = fixture_root / "signature.json"
    partition_path = fixture_root / "partition.json"
    environment_path = fixture_root / "environment.json"
    for path, value in (
        (structure_path, structure), (signature_path, signature),
        (partition_path, partition),
    ):
        path.write_bytes(canonical(value))
    environment_payload = canonical(environment)
    non_authentic_root = fixture_root / "non-authentic"
    non_authentic_root.mkdir()
    legacy_config = non_authentic_root / "legacy-config.json"
    legacy_config.write_text(
        json.dumps(
            {
                "parent_trainer_commit": builder.PARENT_TRAINER_COMMIT,
                "parent_trainer_path": builder.PARENT_TRAINER_PATH,
                "parent_trainer_sha256": builder.PARENT_TRAINER_SHA256,
                "runtime_inference_signature": signature,
            },
            indent=2,
        )
        + "\n"
    )
    legacy_history = non_authentic_root / "legacy-history.json"
    legacy_history.write_text("[]\n")
    optimizer = torch.optim.AdamW(model.parameters())
    legacy_checkpoints = []
    for epoch in (4, 8, 12, 16):
        checkpoint_path = non_authentic_root / f"legacy-epoch-{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": dict(model.state_dict()),
                "classifier": torch.zeros((4, 2)),
                "ema": {},
                "optimizer": optimizer.state_dict(),
                "scheduler": {},
                "scaler": None,
                "mask_generator": torch.Generator().get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": [torch.Generator().get_state()],
                "selection_holdout": {"seed": 0, "fraction": 0.2},
                "training_protocol": {
                    "trainer_sha256": builder.PARENT_TRAINER_SHA256
                },
                "history": [],
            },
            checkpoint_path,
        )
        legacy_checkpoints.append(checkpoint_path)
    legacy_receipt = trainer.training_run_receipt(
        source_commit="1" * 40,
        config_path=str(legacy_config),
        config_sha256=_sha256(legacy_config.read_bytes()),
        seed=2,
        arm="sampled_512",
        objective="official-eight-mask",
        selected_features=512,
        evaluation_features=768,
        command=["python", "trainer.py", "--classifier-init", "imprinted"],
        started_unix_ns=1,
        finished_unix_ns=2,
        elapsed_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
        exit_status=0,
        history_path=legacy_history,
        checkpoint_paths=tuple(legacy_checkpoints),
        runtime={"python": "3.12", "torch": "2.6", "cuda": "12.4"},
    )
    legacy_receipt_path = non_authentic_root / "legacy-run-receipt.json"
    legacy_receipt_path.write_text(json.dumps(legacy_receipt, indent=2) + "\n")

    def binding(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(payload),
            "bytes": len(payload),
        }

    legacy_authority = {
        "run_receipt": binding(legacy_receipt_path),
        "config": binding(legacy_config),
        "history": binding(legacy_history),
        "checkpoints": [
            {"epoch": epoch, **binding(path)}
            for epoch, path in zip((4, 8, 12, 16), legacy_checkpoints, strict=True)
        ],
    }
    artifact_root = workspace / "artifacts"
    checkout_template = str(workspace / "execution-{config_commit}")
    output = checkout / "docs/unicom_fepf_run_config.json"
    normal_builder_argv = [
        sys.executable, "-I", "-B", "scripts/build_unicom_fepf_run_config.py",
        "--repo", str(checkout),
        "--checkout-root-template", checkout_template,
        "--artifact-root", str(artifact_root),
        "--output", str(output),
    ]
    normal_builder = subprocess.run(
        normal_builder_argv,
        cwd=checkout,
        check=False,
        capture_output=True,
    )
    if normal_builder.returncode == 0:
        raise RuntimeError(
            "non-authentic CPU contract unexpectedly crossed the target-only builder"
        )
    provisional = builder.build_run_config(
        repo=checkout, checkout_root_template=checkout_template,
        artifact_root=artifact_root, inference_structure=structure,
        partition_inventory=partition,
        cuda_canary_authority={},
        cuda_canary_environment={"path": str(environment_path.resolve())},
        publication_budget={
            "path": str((fixture_root / "budget.json").resolve()),
            "sha256": "0" * 64, "bytes": 1,
        },
        runtime_inference_signature=signature,
        legacy_runtime_authority=legacy_authority,
    )
    # Loading the builder as a module may create interpreter cache files; they
    # are not source authority and must not contaminate the sole config commit.
    for cache in checkout.rglob("__pycache__"):
        for entry in cache.iterdir():
            entry.unlink()
        cache.rmdir()
    budget_path = artifact_root / provisional["publication_budget_path"]
    budget_payload = canonical(builder.exact_publication_budget(provisional))
    output.parent.mkdir()
    output.write_bytes(canonical(provisional))
    subprocess.run(["git", "add", "docs/unicom_fepf_run_config.json"], cwd=checkout,
                   check=True)
    subprocess.run(["git", "commit", "-qm", "config handoff"], cwd=checkout, check=True)
    config_commit = str(
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
            capture_output=True, text=True,
        ).stdout
    ).strip()
    execution_checkout = Path(
        checkout_template.replace("{config_commit}", config_commit)
    )
    builder.validate_transfer_handoff(provisional, execution_checkout)
    subprocess.run(
        ["git", "clone", "-q", "--no-hardlinks", str(checkout),
         str(execution_checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "-q", config_commit],
        cwd=execution_checkout,
        check=True,
    )
    execution_output = execution_checkout / output.relative_to(checkout)
    builder.validate_non_authentic_synthesized_membership(
        execution_output, execution_checkout
    )
    environment_was_absent_before_canary = not os.path.lexists(environment_path)
    observed_public_runs: list[dict[str, object]] = []

    def run_public(command: list[str]) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            command, cwd=execution_checkout, check=False, capture_output=True
        )
        observed_public_runs.append(
            {"argv": list(command), "returncode": completed.returncode}
        )
        completed.check_returncode()
        return completed

    run_public(
        [sys.executable, "-I", "-B", "scripts/run_unicom_fepf_campaign.py",
         "--config", str(execution_output), "--through-stage", "runtime",
         "--authority-preflight-only", "--non-authentic-synthesized-authorities"]
    )
    run_public(
        [sys.executable, "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
         "--config", str(execution_output), "--publication-stage", "cuda-canary",
         "--campaign-root", str(artifact_root), "--authority-preflight-only",
         "--non-authentic-synthesized-authorities"]
    )
    published_environment = publish_bytes_noreplace(
        environment_path,
        environment_payload,
        validator=lambda payload: (
            None
            if payload == environment_payload
            else (_ for _ in ()).throw(ValueError("environment authority differs"))
        ),
    )
    published_environment.close()
    if budget_path.read_bytes() != budget_payload:
        raise RuntimeError("controller-materialized budget differs")
    environment_sha256 = _sha256(environment_payload)
    run_public(
        [sys.executable, "-I", "-B", "scripts/train_unicom_inshop.py",
         "--unicom-checkout", str(fixture_root), "--checkpoint", str(signature_path),
         "--dataset-root", str(fixture_root), "--output-dir",
         str(artifact_root / "exploratory-control-stage4"), "--run-config",
         str(execution_output),
         "--environment-authority", str(environment_path), "--environment-sha256",
         environment_sha256, "--publication-budget", str(budget_path),
         "--publication-budget-sha256", provisional["publication_budget_sha256"],
         "--run-arm", "exploratory-control-stage4", "--classifier-init", "imprinted",
         "--stop-after-epoch", "4", "--authority-preflight-only"]
    )
    run_public(
        [sys.executable, "-I", "-B", "scripts/profile_unicom_training_step.py",
         "--run-checkpoint", str(legacy_checkpoints[-1]), "--run-receipt",
         str(legacy_receipt_path),
         "--config", str(execution_output), "--unicom-checkout", str(fixture_root),
         "--initial-checkpoint", str(legacy_checkpoints[-1]), "--dataset-root",
         str(fixture_root),
         "--runtime-mode", "current", "--profile-kind", "runtime",
         "--parent-trainer-source", "registered", "--output",
         str(artifact_root / "runtime-00/terminal.json"), "--environment-authority",
         str(environment_path), "--environment-sha256", environment_sha256,
         "--publication-stage", "runtime-00", "--campaign-root", str(artifact_root),
         "--authority-preflight-only"]
    )
    run_public(
        [sys.executable, "-I", "-B", "scripts/evaluate_unicom_fepf.py",
         "--phase", "epoch4", "--sources", str(partition_path), "--sources-sha256",
         _sha256(partition_path.read_bytes()), "--sources-bytes",
         str(len(partition_path.read_bytes())), "--evidence-root", str(artifact_root),
         "--output", str(artifact_root / "exploratory-epoch4-decision-result.json"),
         "--temporary", str(artifact_root / ".evaluation.tmp"), "--config",
         str(execution_output),
         "--publication-stage", "exploratory-epoch4-decision", "--campaign-root",
         str(artifact_root), "--authority-preflight-only"]
    )
    run_public(
        [sys.executable, "-I", "-B", "scripts/run_unicom_fepf_campaign.py",
         "--config", str(execution_output), "--through-stage", "runtime",
         "--authority-preflight-only", "--non-authentic-synthesized-authorities"]
    )
    return {
        "normal_builder_argv": normal_builder_argv,
        "normal_builder_returncode": normal_builder.returncode,
        "source_checkout": str(checkout),
        "checkout_root_template": checkout_template,
        "artifact_root": str(artifact_root),
        "source_config_path": str(output),
        "config_commit": config_commit,
        "execution_checkout": str(execution_checkout),
        "execution_config_path": str(execution_output),
        "public_runs": observed_public_runs,
        "pre_canary_absent_paths": (
            [str(environment_path)] if environment_was_absent_before_canary else []
        ),
        "observed_publications": [
            str(environment_path), str(budget_path), str(execution_output),
        ],
    }


def _evaluation_stage(
    *, name: str, phase: str, base: list[str], root: Path, sources: object,
    source_publisher: Callable[[Path, str, object], dict[str, object]],
) -> dict[str, object]:
    authority = source_publisher(root, name, sources)
    output = root / f"{name}-result.json"
    temporary = root / f".{name}-result.json.tmp"
    command = [
        *base, "--phase", phase, "--sources", authority["path"],
        "--sources-sha256", authority["sha256"], "--sources-bytes",
        str(authority["bytes"]), "--evidence-root", str(root),
        "--output", str(output), "--temporary", str(temporary),
        "--publication-stage", name, "--campaign-root", str(root),
    ]
    stage = _stage(name, command, root, terminal_path=output)
    stage["sources_authority"] = authority
    stage["evidence_root"] = root
    return stage


def write_status_marker_atomic(
    path: Path,
    value: Mapping[str, object],
    *,
    publication_guard: Callable[[str, Path, bytes], None] = (
        lambda _name, _path, _payload: None
    ),
) -> None:
    """Write operator progress only; no validator or resume decision trusts it."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("campaign status marker path differs")
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(temporary)
    try:
        payload = _canonical_json(dict(value))
        publication_guard("campaign:controller-status-temporary", temporary, payload)
        publication_guard("campaign:controller-status", path, payload)
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


class SubprocessStageExecutor:
    """Own exactly one original Popen until it reaches a terminal state."""

    def __init__(
        self,
        *,
        checkout_root: Path,
        marker_writer: Callable[[dict[str, object]], None],
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        cancelled: Callable[[], bool] = lambda: False,
        killpg: Callable[[int, int], None] = os.killpg,
        poll_seconds: float = 5.0,
        registered_environment: Mapping[str, object] | None = None,
    ) -> None:
        if poll_seconds <= 0 or poll_seconds > 55:
            raise ValueError("campaign polling interval differs")
        self.checkout_root = checkout_root
        self.marker_writer = marker_writer
        self.popen = popen
        self.sleep = sleep
        self.monotonic = monotonic
        self.cancelled = cancelled
        self.killpg = killpg
        self.poll_seconds = poll_seconds
        self.child_environment: dict[str, str] | None = None
        if registered_environment is not None:
            self.set_registered_environment(registered_environment)

    def set_registered_environment(self, environment: Mapping[str, object]) -> None:
        deterministic = environment.get("deterministic_execution")
        if (
            type(deterministic) is not dict
            or deterministic.get("cublas_workspace_config") != ":4096:8"
        ):
            raise ValueError("registered child deterministic environment differs")
        child = dict(os.environ)
        current = child.get("CUBLAS_WORKSPACE_CONFIG")
        if current not in (None, ":4096:8"):
            raise ValueError("registered child deterministic environment differs")
        child["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.child_environment = child

    def __call__(self, stage: dict[str, object]) -> dict[str, object]:
        command = stage["command"]
        if (
            type(command) is not list
            or not command
            or not all(type(item) is str for item in command)
        ):
            raise ValueError("campaign command differs")
        started = self.monotonic()
        kwargs: dict[str, object] = {
            "cwd": self.checkout_root,
            "start_new_session": True,
        }
        if self.child_environment is not None:
            kwargs["env"] = dict(self.child_environment)
        process = self.popen(command, **kwargs)
        status: int | None = None
        try:
            while True:
                status = process.poll()
                elapsed = self.monotonic() - started
                self.marker_writer({
                    "state": "running", "stage": stage["name"], "pid": process.pid,
                    "elapsed_seconds": elapsed,
                    "last_child_progress": stage.get("progress_path"),
                })
                if status is not None:
                    break
                if self.cancelled():
                    self.killpg(process.pid, signal.SIGTERM)
                    status = process.wait()
                    break
                self.sleep(self.poll_seconds)
        except BaseException:
            if status is None:
                self.killpg(process.pid, signal.SIGTERM)
                process.wait()
            raise
        terminal: object = None
        terminal_path = stage.get("terminal_path")
        if status == 0 and isinstance(terminal_path, Path):
            raw = terminal_path.read_bytes()
            terminal = json.loads(raw)
        return {"exit_code": status, "terminal": terminal}


def _stage(
    name: str, command: list[str], root: Path, *, terminal_path: Path | None = None
) -> dict[str, object]:
    destination = root / name
    return {
        "name": name,
        "command": command,
        "destination": destination,
        "terminal_path": terminal_path or destination / "terminal.json",
        "progress_path": str(destination / "progress.json"),
    }


def _train_command(
    base: list[str], *, mode: str, training_seed: int, holdout_seed: int,
    stop: int, output: Path, resume: Path | None = None,
) -> list[str]:
    command = [
        *base, "--classifier-init", mode, "--seed", str(training_seed),
        "--holdout-seed", str(holdout_seed), "--holdout-fraction", "0.2",
        "--epochs", "16", "--stop-after-epoch", str(stop),
        "--output-dir", str(output), "--run-receipt", str(output / "run-receipt.json"),
        "--publication-stage", output.name, "--campaign-root", str(output.parent),
    ]
    if resume is not None:
        command.extend([
            "--resume", str(resume / "epoch-0004.pt"),
            "--parent-run-receipt", str(resume / "run-receipt.json"),
            "--parent-initialization-receipt", str(resume / "initialization-receipt.json"),
        ])
    return command


def _profile_stages(
    *, prefix: str, root: Path, base: list[str], control: Path, candidate: Path,
    config_path: Path, runtime_decision: str,
) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for index, arm in enumerate(QUALITY_PROFILE_ORDER):
        arm_root = control if arm == "control" else candidate
        name = f"{prefix}-profile-{arm}-{0 if index < 2 else 1}"
        command = apply_runtime_selection([
            *base, "--runtime-mode", "current", "--run-checkpoint",
            str(arm_root / "epoch-0016.pt"), "--run-receipt",
            str(arm_root / "run-receipt.json"), "--output", str(root / name / "terminal.json"),
            "--config", str(config_path),
            "--publication-stage", name, "--campaign-root", str(root),
        ], runtime_decision, profile=True)
        stages.append(_stage(name, command, root))
    return stages


def _required_config(config: object) -> dict[str, object]:
    if (
        type(config) is not dict
        or config.get("schema") != "unicom-fepf-run-config-v1"
        or config.get("runtime_order") != list(RUNTIME_ORDER)
        or config.get("confirmation_pairs") != [list(pair) for pair in CONFIRMATION_PAIRS]
        or type(config.get("commands")) is not dict
    ):
        raise ValueError("campaign config differs")
    return config


def _execute(
    stage: dict[str, object], *, executor: Callable[[dict[str, object]], dict[str, object]],
    terminal_validator: Callable[[dict[str, object], object], None],
    prior_terminals: Mapping[str, object], marker_writer: Callable[[dict[str, object]], None],
) -> tuple[int, object]:
    name = str(stage["name"])
    marker_writer({"state": "starting", "stage": name})
    if name in prior_terminals:
        terminal = prior_terminals[name]
        terminal_validator(stage, terminal)
        marker_writer({"state": "resumed", "stage": name})
        return 0, terminal
    result = executor(stage)
    code = result["exit_code"]
    if type(code) is not int:
        raise ValueError("campaign exit code differs")
    if code != 0:
        marker_writer({"state": "failed", "stage": name, "exit_code": code})
        return code, None
    terminal = result["terminal"]
    terminal_validator({**stage, "fresh_execution": True}, terminal)
    marker_writer({"state": "terminal", "stage": name})
    return 0, terminal


def run_campaign(
    config: object,
    *,
    executor: Callable[[dict[str, object]], dict[str, object]],
    terminal_validator: Callable[[dict[str, object], object], None],
    through_stage: str = "confirmation",
    marker_writer: Callable[[dict[str, object]], None] = lambda _value: None,
    prior_terminals: Mapping[str, object] | None = None,
    runtime_selector: Callable[[object], str] | None = None,
    source_publisher: Callable[[Path, str, object], dict[str, object]] | None = None,
    config_path: Path | None = None,
    capacity_guard: Callable[[], None] = lambda: None,
    post_canary_guard: Callable[[], None] = lambda: None,
) -> int:
    value = _required_config(config)
    if through_stage not in {"runtime", "exploratory", "confirmation"}:
        raise ValueError("campaign through-stage differs")
    prior = prior_terminals or {}
    root = Path(value["artifact_root"])
    commands = value["commands"]
    source_publisher = source_publisher or publish_evaluation_sources
    config_path = config_path or Path("docs/unicom_fepf_run_config.json").resolve()

    def run(stage: dict[str, object]) -> tuple[int, object]:
        capacity_guard()
        return _execute(
            stage, executor=executor, terminal_validator=terminal_validator,
            prior_terminals=prior, marker_writer=marker_writer,
        )

    canary_command = [
        *list(value.get("cuda_canary_command", commands.get("cuda_canary", []))),
        "--publication-stage", "cuda-canary", "--campaign-root", str(root),
    ]
    code, _ = run(_stage(
        "cuda-canary", canary_command, root,
        terminal_path=root / value.get("cuda_canary_receipt", "preflight/cuda_canary_v1.json"),
    ))
    if code:
        return code
    environment_authority = value.get("cuda_canary_environment")
    if type(environment_authority) is dict:
        environment_path = Path(environment_authority["path"])
        environment_payload = environment_path.read_bytes()
        environment = json.loads(environment_payload)
        if (
            environment_path.is_symlink()
            or not environment_path.is_file()
            or environment_payload != _canonical_json(environment)
        ):
            raise ValueError("post-canary environment authority differs")
        resolve_canary_environment_commands(value, _sha256(environment_payload))
        commands = value["commands"]
        setter = getattr(executor, "set_registered_environment", None)
        if callable(setter):
            setter(environment)
    post_canary_guard()
    runtime_terminals = []
    for index, command in enumerate(commands["runtime"]):
        name = f"runtime-{index:02d}"
        output = root / name / "terminal.json"
        resolved = [str(output) if item == "{output}" else item for item in command]
        resolved.extend(("--publication-stage", name, "--campaign-root", str(root)))
        code, terminal = run(_stage(name, resolved, root))
        if code:
            return code
        runtime_terminals.append(terminal)
    decision = (
        runtime_selector(tuple(runtime_terminals))
        if runtime_selector is not None
        else select_runtime_from_receipts(tuple(runtime_terminals), checkout_root=Path.cwd())
    )
    if through_stage == "runtime":
        marker_writer({"state": "complete", "through_stage": "runtime"})
        return 0

    train = apply_runtime_selection(list(commands["train"]), decision, profile=False)
    control4 = root / "exploratory-control-stage4"
    candidate4 = root / "exploratory-candidate-stage4"
    for name, mode, destination in (
        ("exploratory-control-stage4", "imprinted", control4),
        ("exploratory-candidate-stage4", "fepf_mean", candidate4),
    ):
        command = _train_command(
            train, mode=mode, training_seed=0, holdout_seed=0, stop=4,
            output=destination,
        )
        code, _ = run(_stage(name, command, root, terminal_path=destination / "run-receipt.json"))
        if code:
            return code
    config_payload = config_path.read_bytes() if config_path.is_file() else _canonical_json(value)
    config_authority = {
        "path": str(config_path.resolve()), "sha256": _sha256(config_payload),
        "bytes": len(config_payload),
    }
    epoch4_sources = [{
        "training_seed": 0, "holdout_seed": 0,
        "control_root": control4.relative_to(root).as_posix(),
        "candidate_root": candidate4.relative_to(root).as_posix(),
        "quality_profiles": [], "config": config_authority,
    }]
    code, epoch4 = run(_evaluation_stage(
        name="exploratory-epoch4-decision", phase="epoch4",
        base=list(commands["evaluate"]), root=root, sources=epoch4_sources,
        source_publisher=source_publisher,
    ))
    if code:
        return code
    if epoch4["decision"] == "CLOSE_EPOCH4":
        marker_writer({"state": "complete", "decision": "CLOSE_EPOCH4"})
        return 0
    if epoch4["decision"] != "PASS_TO_RESUME":
        raise ValueError("epoch-4 decision differs")
    control16 = root / "exploratory-control-stage16"
    candidate16 = root / "exploratory-candidate-stage16"
    for name, mode, destination, parent in (
        ("exploratory-control-stage16", "imprinted", control16, control4),
        ("exploratory-candidate-stage16", "fepf_mean", candidate16, candidate4),
    ):
        code, _ = run(_stage(name, _train_command(
            train, mode=mode, training_seed=0, holdout_seed=0, stop=16,
            output=destination, resume=parent,
        ), root, terminal_path=destination / "run-receipt.json"))
        if code:
            return code
    for stage in _profile_stages(
        prefix="exploratory", root=root, base=list(commands["profile_quality"]),
        control=control16, candidate=candidate16, config_path=config_path,
        runtime_decision=decision,
    ):
        code, _ = run(stage)
        if code:
            return code
    exploratory_profiles = [
        f"exploratory-profile-{arm}-{0 if index < 2 else 1}/terminal.json"
        for index, arm in enumerate(QUALITY_PROFILE_ORDER)
    ]
    exploratory_sources = [{
        "training_seed": 0, "holdout_seed": 0,
        "control_root": control16.relative_to(root).as_posix(),
        "candidate_root": candidate16.relative_to(root).as_posix(),
        "quality_profiles": exploratory_profiles,
    }]
    code, exploratory_result = run(_evaluation_stage(
        name="exploratory-decision", phase="exploratory",
        base=list(commands["evaluate"]), root=root, sources=exploratory_sources,
        source_publisher=source_publisher,
    ))
    if code:
        return code
    if exploratory_result["decision"] != "PROMOTE":
        marker_writer({"state": "complete", "decision": exploratory_result["decision"]})
        return 0
    random_root = root / "exploratory-random-stage16"
    code, _ = run(_stage("exploratory-random-stage16", _train_command(
        train, mode="fepf_random", training_seed=0, holdout_seed=0, stop=16,
        output=random_root,
    ), root, terminal_path=random_root / "run-receipt.json"))
    if code or through_stage == "exploratory":
        marker_writer({"state": "complete", "through_stage": "exploratory"})
        return code

    for pair_index, (training_seed, holdout_seed) in enumerate(CONFIRMATION_PAIRS):
        prefix = f"confirmation-{pair_index}"
        control = root / f"{prefix}-control"
        candidate = root / f"{prefix}-candidate"
        for name, mode, destination in (
            (f"{prefix}-control", "imprinted", control),
            (f"{prefix}-candidate", "fepf_mean", candidate),
        ):
            code, _ = run(_stage(name, _train_command(
                train, mode=mode, training_seed=training_seed,
                holdout_seed=holdout_seed, stop=16, output=destination,
            ), root, terminal_path=destination / "run-receipt.json"))
            if code:
                return code
        for stage in _profile_stages(
            prefix=prefix, root=root, base=list(commands["profile_quality"]),
            control=control, candidate=candidate, config_path=config_path,
            runtime_decision=decision,
        ):
            code, _ = run(stage)
            if code:
                return code
    confirmation_sources = []
    for pair_index, (training_seed, holdout_seed) in enumerate(CONFIRMATION_PAIRS):
        prefix = f"confirmation-{pair_index}"
        confirmation_sources.append({
            "training_seed": training_seed, "holdout_seed": holdout_seed,
            "control_root": f"{prefix}-control", "candidate_root": f"{prefix}-candidate",
            "quality_profiles": [
                f"{prefix}-profile-{arm}-{0 if index < 2 else 1}/terminal.json"
                for index, arm in enumerate(QUALITY_PROFILE_ORDER)
            ],
        })
    code, _ = run(_evaluation_stage(
        name="confirmation-decision", phase="confirmation",
        base=list(commands["evaluate"]), root=root, sources=confirmation_sources,
        source_publisher=source_publisher,
    ))
    marker_writer({"state": "complete", "through_stage": "confirmation"})
    return code


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--through-stage", choices=("runtime", "exploratory", "confirmation"),
                        default="confirmation")
    parser.add_argument("--authority-preflight-only", action="store_true")
    parser.add_argument("--non-authentic-synthesized-authorities", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    cancelled = False

    def request_cancel(_signum: int, _frame: object) -> None:
        nonlocal cancelled
        cancelled = True

    previous_handlers = {
        signum: signal.signal(signum, request_cancel)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        config = json.loads(args.config.read_bytes())
        builder = _load_module(
            Path(__file__).with_name("build_unicom_fepf_run_config.py"),
            "fepf_builder_handoff",
        )
        if args.non_authentic_synthesized_authorities:
            if not args.authority_preflight_only:
                raise ValueError("non-authentic authority seam is preflight-only")
            builder.validate_non_authentic_synthesized_membership(
                args.config, Path.cwd()
            )
        else:
            builder.validate_config_membership(args.config, Path.cwd())
        validate_registered_command_vectors(config, checkout_root=Path.cwd())
        if args.non_authentic_synthesized_authorities:
            builder.validate_exact_publication_budget(
                config, config.get("publication_budget")
            )
        else:
            builder.validate_external_exact_publication_budget(
                config, config.get("publication_budget")
            )
        first_launch = not os.path.lexists(config["artifact_root"])
        if first_launch:
            builder.validate_first_launch_absence(config)
        root = prepare_campaign_storage(
            config, physical_admission=not args.authority_preflight_only
        )
        marker_path = root / "controller-status.json"
        status_publisher = BudgetedPublisher(
            campaign_root=root,
            budget_path=root / config["publication_budget_path"],
            budget_sha256=config["publication_budget_sha256"],
            exact_budget=config["publication_budget"],
        )
        def marker(value: dict[str, object]) -> None:
            write_status_marker_atomic(
                marker_path,
                value,
                publication_guard=lambda name, path, payload: (
                    status_publisher.validate_payload(
                        name=name, destination=path, payload=payload
                    )
                ),
            )
        validate = RegisteredTerminalValidator(checkout_root=Path.cwd(), config=config)
        prior = load_campaign_resume(config)
        prevalidate_campaign_resume(
            config, prior, terminal_validator=validate, checkout_root=Path.cwd()
        )
        if args.authority_preflight_only:
            return 0

        executor = SubprocessStageExecutor(
            checkout_root=Path.cwd(), marker_writer=marker,
            cancelled=lambda: cancelled,
        )

        return run_campaign(
            config, executor=executor, terminal_validator=validate,
            through_stage=args.through_stage, marker_writer=marker,
            prior_terminals=prior,
            runtime_selector=lambda receipts: select_runtime_from_receipts(
                receipts, checkout_root=Path.cwd()
            ),
            config_path=args.config.resolve(),
            capacity_guard=lambda: require_campaign_remaining_capacity(config, root),
            post_canary_guard=lambda: run_fresh_process_contract_preflight(
                checkout_root=Path.cwd(), config_path=args.config.resolve(),
                artifact_root=root,
            ),
            source_publisher=lambda source_root, name, sources: (
                require_campaign_remaining_capacity(config, root),
                publish_evaluation_sources(
                    source_root, name, sources, config=config
                ),
            )[1],
        )
    except Exception as error:
        print(f"FEPF campaign failed: {error}", file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
