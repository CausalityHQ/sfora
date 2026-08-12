"""Faithful, minimal adapter for the pinned upstream DADA In-Shop recipe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import secrets
import stat
import subprocess
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DADA_REVISION = "726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8"
INSHOP_CONFIG_SHA256 = "2685672b2a42faef74d5ee3af0cecc035741728379ea147ef1197af777ff2160"

_EXPECTED_INSHOP_CONFIG: dict[str, object] = {
    "dataset": "inshop",
    "n_epochs": 200,
    "batch_size": 180,
    "fd2_bn": True,
    "fc2_bn": True,
    "pos_class_mix": True,
    "arch": "resnet50_layernorm_double",
    "d_dis_ratio": 0.01,
    "decay": 0.0001,
    "dis_decay": 0.00005,
    "fc_fc2_dim": 4096,
    "fd_fc1_dim": 512,
    "loss_oproxy_neg_alpha": 200,
    "loss_oproxy_pos_alpha": 40,
    "lr": 0.00012,
    "lr_reduce_rate": 0.25,
    "lr_reduce_step": 40,
    "oproxy_ratio": 0.0075,
    "mix_alpha": 3,
    "mix_beta": 3,
    "warmup": 5,
    "store_improvements": True,
}


@dataclass(frozen=True)
class DadaSource:
    checkout: Path
    revision: str
    config_path: Path
    config_sha256: str
    config: dict[str, object]


@dataclass(frozen=True)
class DadaSmokeRequest:
    python: Path
    source: DadaSource
    smoke_config: Path
    dataset_root: Path
    output_root: Path
    save_name: str
    gpu: int
    seed: int
    dependency_root: Path | None = None


@dataclass(frozen=True)
class DadaProgress:
    completed_epochs: int
    optimizer_steps: int
    last_loss: float
    last_recall_at_1: float
    last_epoch_seconds: float


@dataclass(frozen=True)
class DadaChildResult:
    returncode: int
    stdout: str
    elapsed_seconds: float
    peak_gpu_memory_mib: int | None


def _run_git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(checkout), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{name} is missing") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} is not a regular file")


def _require_literal_config(value: Any) -> dict[str, object]:
    if type(value) is not dict or list(value) != list(_EXPECTED_INSHOP_CONFIG):
        raise ValueError("DADA In-Shop config keys/order differ")
    for key, expected in _EXPECTED_INSHOP_CONFIG.items():
        actual = value[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"DADA In-Shop config field differs: {key}")
    return dict(value)


def validate_dada_source(checkout: Path) -> DadaSource:
    checkout = checkout.resolve(strict=True)
    if _run_git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("DADA checkout has dirty tracked bytes")
    revision = _run_git(checkout, "rev-parse", "HEAD")
    if revision != DADA_REVISION:
        raise ValueError("DADA revision differs")

    main_path = checkout / "main.py"
    config_path = checkout / "configs" / "inshop.yaml"
    _require_regular_file(main_path, "DADA main.py")
    _require_regular_file(config_path, "DADA In-Shop config")
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != INSHOP_CONFIG_SHA256:
        raise ValueError("DADA In-Shop config SHA-256 differs")
    config = _require_literal_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    return DadaSource(
        checkout=checkout,
        revision=revision,
        config_path=config_path,
        config_sha256=config_sha256,
        config=config,
    )


def validate_inshop_dataset_root(dataset_root: Path) -> Path:
    root = dataset_root.resolve(strict=True)
    try:
        inshop = (root / "inshop").resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("DADA In-Shop dataset is missing") from exc
    if not inshop.is_dir():
        raise ValueError("DADA In-Shop dataset is not a directory")
    partition = inshop / "list_eval_partition.txt"
    try:
        rows = partition.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError("DADA In-Shop partition is missing") from exc
    if len(rows) < 3 or not rows[2].split():
        raise ValueError("DADA In-Shop partition has no registered rows")
    relative_image = rows[2].split()[0]
    try:
        registered_image = (inshop / relative_image).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("DADA In-Shop registered image is missing") from exc
    if not registered_image.is_file():
        raise ValueError("DADA In-Shop registered image is not a file")
    return root


def build_smoke_config(
    source: DadaSource,
    destination: Path,
    *,
    epochs: int = 6,
) -> str:
    if type(epochs) is not int or epochs != 6:
        raise ValueError("DADA smoke epoch count must be exactly 6")
    payload = dict(source.config)
    payload["n_epochs"] = epochs
    encoded = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            destination.unlink()
        raise
    return _sha256_file(destination)


def build_dada_command(request: DadaSmokeRequest) -> tuple[str, ...]:
    if type(request.seed) is not int or request.seed != 0:
        raise ValueError("seed must be exactly 0")
    if type(request.gpu) is not int or request.gpu != 0:
        raise ValueError("gpu must be exactly 0")
    checkout = str(request.source.checkout)
    main_path = str(request.source.checkout / "main.py")
    dependency = ""
    if request.dependency_root is not None:
        dependency_root = request.dependency_root.resolve(strict=True)
        if not dependency_root.is_dir() or dependency_root.is_symlink():
            raise ValueError("DADA dependency root must be a real directory")
        dependency = f"sys.path.insert(0,{str(dependency_root)!r});"
    bootstrap = (
        "import runpy,sys;"
        f"{dependency}"
        f"sys.path.insert(0,{checkout!r});"
        f"runpy.run_path({main_path!r},run_name='__main__')"
    )
    return (
        str(request.python),
        "-I",
        "-B",
        "-c",
        bootstrap,
        "--source_path",
        str(request.dataset_root),
        "--save_path",
        str(request.output_root),
        "--save_name",
        request.save_name,
        "--config",
        str(request.smoke_config),
        "--gpu",
        str(request.gpu),
        "--seed",
        str(request.seed),
    )


_TRAIN_RE = re.compile(
    r"\[Train Epoch (?P<epoch>\d+)\].*\[(?P<done>\d+)/(?P<total>\d+).*"
    r"DML:(?P<loss>[-+A-Za-z0-9.eE]+)"
)
_RECALL_RE = re.compile(r"e_recall@1:\s*(?P<recall>[-+A-Za-z0-9.eE]+)")
_EPOCH_RUNTIME_RE = re.compile(
    r"Total Epoch Runtime:\s*(?P<seconds>[-+A-Za-z0-9.eE]+)s"
)


def parse_dada_log(lines: Iterable[str]) -> DadaProgress:
    epochs: dict[int, tuple[int, int, float]] = {}
    recalls: list[float] = []
    epoch_runtimes: list[float] = []
    for raw_line in lines:
        line = str(raw_line)
        if "Traceback (most recent call last):" in line:
            raise ValueError("DADA child traceback")
        if "CUDA out of memory" in line:
            raise ValueError("DADA child CUDA out of memory")
        for fragment in line.split("\r"):
            train_match = _TRAIN_RE.search(fragment)
            if train_match is not None:
                done = int(train_match.group("done"))
                total = int(train_match.group("total"))
                if done < 0 or total <= 0 or done > total:
                    raise ValueError("DADA optimizer progress is incomplete")
                loss = float(train_match.group("loss"))
                if not math.isfinite(loss):
                    raise ValueError("DADA loss is non-finite")
                epoch = int(train_match.group("epoch"))
                epochs[epoch] = (done, total, loss)
            recall_match = _RECALL_RE.search(fragment)
            if recall_match is not None:
                recall = float(recall_match.group("recall"))
                if not math.isfinite(recall):
                    raise ValueError("DADA recall is non-finite")
                recalls.append(recall)
            runtime_match = _EPOCH_RUNTIME_RE.search(fragment)
            if runtime_match is not None:
                runtime = float(runtime_match.group("seconds"))
                if not math.isfinite(runtime) or runtime <= 0.0:
                    raise ValueError("DADA epoch runtime is invalid")
                epoch_runtimes.append(runtime)
    if not epochs:
        raise ValueError("DADA optimizer progress is absent")
    if not recalls:
        raise ValueError("DADA evaluation R@1 is absent")
    if sorted(epochs) != list(range(len(epochs))):
        raise ValueError("DADA epoch sequence differs")
    if len(recalls) != len(epochs) or len(epoch_runtimes) != len(epochs):
        raise ValueError("DADA per-epoch evidence differs")
    ordered = [epochs[epoch] for epoch in sorted(epochs)]
    if any(done != total for done, total, _loss in ordered):
        raise ValueError("DADA optimizer progress is incomplete")
    return DadaProgress(
        completed_epochs=len(epochs),
        optimizer_steps=sum(done for done, _total, _loss in ordered[1:]),
        last_loss=ordered[-1][2],
        last_recall_at_1=recalls[-1],
        last_epoch_seconds=epoch_runtimes[-1],
    )


def _execute_dada_child(request: DadaSmokeRequest) -> DadaChildResult:
    log_path = request.output_root / "dada-child.log"
    started = time.perf_counter()
    peak: int | None = None
    with log_path.open("xb") as log_handle:
        process = subprocess.Popen(
            build_dada_command(request),
            cwd=request.source.checkout,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                memory = _query_gpu_memory(pid=process.pid)
                if memory is not None:
                    peak = memory if peak is None else max(peak, memory)
                time.sleep(0.5)
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    return DadaChildResult(
        returncode=returncode,
        stdout=log_path.read_text(encoding="utf-8", errors="replace"),
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_mib=peak,
    )


def _parse_gpu_memory(output: str, *, pid: int) -> int | None:
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            observed_pid = int(fields[0])
            memory = int(fields[1])
        except ValueError:
            continue
        if observed_pid == pid and memory > 0:
            return memory
    return None


def _query_gpu_memory(*, pid: int) -> int | None:
    try:
        observed = subprocess.run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if observed.returncode != 0:
        return None
    return _parse_gpu_memory(observed.stdout, pid=pid)


def _probe_environment(request: DadaSmokeRequest) -> dict[str, object]:
    code = (
        "import json,platform,torch;"
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':str(torch.__version__),'cuda':str(torch.version.cuda),"
        "'device':torch.cuda.get_device_name(0)}))"
    )
    completed = subprocess.run(
        (str(request.python), "-I", "-B", "-c", code),
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if type(value) is not dict or tuple(value) != ("python", "torch", "cuda", "device"):
        raise ValueError("DADA runtime probe schema differs")
    if any(type(value[key]) is not str or not value[key] for key in value):
        raise ValueError("DADA runtime probe value differs")
    return value


def _checkpoint_is_reloadable(request: DadaSmokeRequest, checkpoint: Path) -> bool:
    completed = subprocess.run(
        _build_checkpoint_reload_command(request, checkpoint),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _build_checkpoint_reload_command(
    request: DadaSmokeRequest, checkpoint: Path
) -> tuple[str, ...]:
    checkout = str(request.source.checkout)
    dependency = ""
    if request.dependency_root is not None:
        dependency_root = request.dependency_root.resolve(strict=True)
        dependency = f"sys.path.insert(0,{str(dependency_root)!r});"
    code = (
        "import sys;"
        f"{dependency}"
        f"sys.path.insert(0,{checkout!r});"
        "import torch;"
        "x=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
        "assert type(x) is dict and 'state_dict' in x"
    )
    return (
        str(request.python),
        "-I",
        "-B",
        "-c",
        code,
        str(checkpoint),
    )


_REPORT_KEYS = (
    "schema_version",
    "status",
    "source",
    "config",
    "command",
    "environment",
    "process",
    "progress",
    "evaluation",
    "resources",
    "projection",
    "failure",
)


def _base_report(status: str, failure: str | None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "source": {
            "revision": DADA_REVISION,
            "config_sha256": INSHOP_CONFIG_SHA256,
        },
        "config": {"dataset": "inshop", "epochs": 6, "batch_size": 180, "seed": 0},
        "command": [],
        "environment": {
            "python": platform.python_version(),
            "torch": "NOT_OBSERVED",
            "cuda": "NOT_OBSERVED",
            "device": "NOT_OBSERVED",
        },
        "process": {"returncode": 2, "elapsed_seconds": 0.0},
        "progress": {
            "completed_epochs": 0,
            "optimizer_steps": 0,
            "last_loss": None,
            "last_epoch_seconds": None,
        },
        "evaluation": {"last_recall_at_1": None, "checkpoint": None},
        "resources": {"peak_gpu_memory_mib": None},
        "projection": {"projected_full_run_seconds": None},
        "failure": failure,
    }


def invalid_dada_smoke_report(failure: str) -> dict[str, object]:
    if type(failure) is not str or not failure:
        raise ValueError("DADA failure must be a nonempty string")
    return _base_report("INVALID", failure)


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_dada_smoke_report(value: object) -> None:
    if type(value) is not dict or tuple(value) != _REPORT_KEYS:
        raise ValueError("DADA smoke report top-level schema differs")
    report = value
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("DADA smoke schema version differs")
    if report["status"] not in {"PASS", "INCOMPATIBLE", "INVALID"}:
        raise ValueError("DADA smoke status differs")
    expected_nested = {
        "source": ("revision", "config_sha256"),
        "config": ("dataset", "epochs", "batch_size", "seed"),
        "environment": ("python", "torch", "cuda", "device"),
        "process": ("returncode", "elapsed_seconds"),
        "progress": (
            "completed_epochs",
            "optimizer_steps",
            "last_loss",
            "last_epoch_seconds",
        ),
        "evaluation": ("last_recall_at_1", "checkpoint"),
        "resources": ("peak_gpu_memory_mib",),
        "projection": ("projected_full_run_seconds",),
    }
    for name, keys in expected_nested.items():
        nested = report[name]
        if type(nested) is not dict or tuple(nested) != keys:
            raise ValueError(f"DADA smoke {name} schema differs")
    source = report["source"]
    config = report["config"]
    process = report["process"]
    progress = report["progress"]
    evaluation = report["evaluation"]
    projection = report["projection"]
    assert isinstance(source, dict)
    assert isinstance(config, dict)
    assert isinstance(process, dict)
    assert isinstance(progress, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(projection, dict)
    if source != {"revision": DADA_REVISION, "config_sha256": INSHOP_CONFIG_SHA256}:
        raise ValueError("DADA smoke source differs")
    if type(report["command"]) is not list or any(
        type(item) is not str for item in report["command"]
    ):
        raise TypeError("DADA smoke command differs")
    if (
        type(config["dataset"]) is not str
        or config["dataset"] != "inshop"
        or type(config["epochs"]) is not int
        or config["epochs"] != 6
        or type(config["batch_size"]) is not int
        or config["batch_size"] != 180
        or type(config["seed"]) is not int
        or config["seed"] < 0
    ):
        raise ValueError("DADA smoke config differs")
    environment = report["environment"]
    resources = report["resources"]
    assert isinstance(environment, dict)
    assert isinstance(resources, dict)
    if any(type(environment[key]) is not str or not environment[key] for key in environment):
        raise TypeError("DADA smoke environment differs")
    if type(process["returncode"]) is not int:
        raise TypeError("DADA smoke return code differs")
    for name in ("completed_epochs", "optimizer_steps"):
        if type(progress[name]) is not int or progress[name] < 0:
            raise TypeError(f"DADA smoke {name} differs")
    peak = resources["peak_gpu_memory_mib"]
    if peak is not None and (type(peak) is not int or peak <= 0):
        raise ValueError("DADA peak GPU memory differs")
    _finite_float(process["elapsed_seconds"], "elapsed_seconds")
    status = report["status"]
    if status == "PASS":
        if process["returncode"] != 0:
            raise ValueError("DADA PASS return code differs")
        if progress["completed_epochs"] != 6 or progress["optimizer_steps"] <= 0:
            raise ValueError("DADA PASS progress differs")
        _finite_float(progress["last_loss"], "last_loss")
        last_epoch_seconds = _finite_float(
            progress["last_epoch_seconds"], "last_epoch_seconds", positive=True
        )
        _finite_float(evaluation["last_recall_at_1"], "last_recall_at_1")
        projected = _finite_float(
            projection["projected_full_run_seconds"],
            "projected_full_run_seconds",
            positive=True,
        )
        expected = last_epoch_seconds * 200.0
        if projected != expected:
            raise ValueError("DADA projected full-run duration differs")
        if type(evaluation["checkpoint"]) is not str or not evaluation["checkpoint"]:
            raise ValueError("DADA checkpoint differs")
        if report["failure"] is not None:
            raise ValueError("DADA PASS failure must be null")
    elif type(report["failure"]) is not str or not report["failure"]:
        raise ValueError("DADA failed report needs a failure string")


def run_dada_smoke(request: DadaSmokeRequest) -> dict[str, object]:
    environment = _probe_environment(request)
    child = _execute_dada_child(request)
    if child.returncode != 0:
        report = _base_report("INCOMPATIBLE", f"DADA child exited {child.returncode}")
        report["command"] = list(build_dada_command(request))
        report["environment"] = environment
        report["process"] = {
            "returncode": child.returncode,
            "elapsed_seconds": float(child.elapsed_seconds),
        }
        validate_dada_smoke_report(report)
        return report
    try:
        progress = parse_dada_log(child.stdout.splitlines())
    except ValueError as exc:
        return _invalid_after_child(request, environment, child, str(exc))
    if progress.completed_epochs != 6:
        return _invalid_after_child(
            request,
            environment,
            child,
            "DADA smoke did not complete exactly six epochs",
        )
    checkpoints = sorted(request.output_root.rglob("*.pth.tar"))
    if not checkpoints:
        return _invalid_after_child(
            request,
            environment,
            child,
            "DADA smoke produced no checkpoint",
        )
    checkpoint = checkpoints[0]
    if not _checkpoint_is_reloadable(request, checkpoint):
        return _invalid_after_child(
            request,
            environment,
            child,
            "DADA smoke checkpoint is not reloadable",
        )
    elapsed = float(child.elapsed_seconds)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "source": {
            "revision": request.source.revision,
            "config_sha256": request.source.config_sha256,
        },
        "config": {"dataset": "inshop", "epochs": 6, "batch_size": 180, "seed": request.seed},
        "command": list(build_dada_command(request)),
        "environment": environment,
        "process": {"returncode": 0, "elapsed_seconds": elapsed},
        "progress": {
            "completed_epochs": progress.completed_epochs,
            "optimizer_steps": progress.optimizer_steps,
            "last_loss": float(progress.last_loss),
            "last_epoch_seconds": float(progress.last_epoch_seconds),
        },
        "evaluation": {
            "last_recall_at_1": float(progress.last_recall_at_1),
            "checkpoint": str(checkpoint),
        },
        "resources": {"peak_gpu_memory_mib": child.peak_gpu_memory_mib},
        "projection": {
            "projected_full_run_seconds": progress.last_epoch_seconds * 200.0
        },
        "failure": None,
    }
    validate_dada_smoke_report(report)
    return report


def _invalid_after_child(
    request: DadaSmokeRequest,
    environment: dict[str, object],
    child: DadaChildResult,
    failure: str,
) -> dict[str, object]:
    report = _base_report("INVALID", failure)
    report["command"] = list(build_dada_command(request))
    report["environment"] = environment
    report["process"] = {
        "returncode": child.returncode,
        "elapsed_seconds": float(child.elapsed_seconds),
    }
    report["resources"] = {"peak_gpu_memory_mib": child.peak_gpu_memory_mib}
    validate_dada_smoke_report(report)
    return report


def _strict_json_object(data: bytes) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise ValueError("DADA report contains a duplicate key")
            result[key] = item
        return result

    def reject(token: str) -> None:
        raise ValueError(f"DADA report contains non-finite number {token}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    if type(value) is not dict:
        raise TypeError("DADA report must be an object")
    return value


def publish_dada_smoke_report(path: Path, payload: dict[str, object]) -> None:
    validate_dada_smoke_report(payload)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("DADA report parent must be a real directory")
    encoded = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    published = False
    try:
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        persisted = _strict_json_object(path.read_bytes())
        validate_dada_smoke_report(persisted)
        if persisted != payload:
            raise ValueError("persisted DADA report differs")
    except Exception:
        if published:
            with suppress(FileNotFoundError):
                if path.stat().st_ino == temporary.stat().st_ino:
                    path.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
