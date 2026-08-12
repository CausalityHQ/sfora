from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_inshop_ahncr.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_ahncr", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load AHNCR evaluator")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(values: np.ndarray) -> np.ndarray:
    return np.asarray(values / np.linalg.norm(values, axis=1, keepdims=True), dtype=np.float32)


def _archive(path: Path) -> None:
    rng = np.random.Generator(np.random.PCG64(17))
    rows: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for label in range(60):
        center = rng.normal(size=16)
        center /= np.linalg.norm(center)
        count = 3 if label < 55 else 1
        for occurrence in range(count):
            row = center + 0.08 * rng.normal(size=16)
            rows.append(row)
            labels.append(label)
            ids.append(f"train-{label:03d}-{occurrence}")
    np.savez(
        path,
        embeddings=_unit(np.asarray(rows, dtype=np.float32)),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(ids),
        split=np.asarray("train"),
        checkpoint_sha256=np.asarray("a" * 64),
    )


@pytest.fixture
def pinned_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_cli_surface_has_no_query_gallery_or_result_inputs() -> None:
    parsed = CLI._parse_args(["--train", "train.npz", "--output", "out.json"])
    assert vars(parsed) == {
        "train": Path("train.npz"),
        "output": Path("out.json"),
        "block_size": 256,
    }
    with pytest.raises(SystemExit):
        CLI._parse_args(
            [
                "--train",
                "train.npz",
                "--output",
                "out.json",
                "--query",
                "forbidden.npz",
            ]
        )


def test_build_report_uses_only_registered_train_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_threads: None
) -> None:
    train = tmp_path / "train.npz"
    _archive(train)
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", _sha256(train))
    forbidden = {"query.npz", "gallery.npz", "inshop_alsp_seed0.json"}
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name in forbidden:
            raise AssertionError(f"forbidden input reached: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = CLI.build_ahncr_report(train, block_size=7)
    validated = CLI.validate_ahncr_report(json.loads(json.dumps(report)))
    assert validated["schema_version"] == "inshop-ahncr-v1"
    assert validated["input"]["sha256"] == _sha256(train)
    assert validated["split"]["evaluation_label_count"] == 11
    assert validated["split"]["query_count"] == 11
    assert validated["split"]["gallery_count"] == 22
    assert list(validated["arms"]) == [
        "raw",
        "ahncr",
        "global_mean",
        "positive_expansion",
        "unary_cohort_density",
        "symmetric_ad_norm",
    ]
    assert len(validated["null"]["shuffled_gains"]) == 20


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("decision", "passes_falsifier"), True),
        (("arms", "ahncr", "gain"), 0.123),
        (("arms", "ahncr", "p_value"), 0.123),
        (("null", "shuffled_p95"), 0.123),
        (("split", "query_count"), 999),
    ],
)
def test_validator_rejects_inconsistent_relations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pinned_threads: None,
    path: tuple[str, ...],
    value: object,
) -> None:
    train = tmp_path / "train.npz"
    _archive(train)
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", _sha256(train))
    payload = CLI.build_ahncr_report(train, block_size=7)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        CLI.validate_ahncr_report(payload)


def test_atomic_writer_reloads_and_does_not_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_threads: None
) -> None:
    train = tmp_path / "train.npz"
    _archive(train)
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", _sha256(train))
    payload = CLI.build_ahncr_report(train, block_size=7)
    output = tmp_path / "result.json"
    CLI.write_json_atomic(output, payload, validator=CLI.validate_ahncr_report)
    first = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o600
    assert first.endswith(b"\n")
    with pytest.raises(FileExistsError):
        CLI.write_json_atomic(output, payload, validator=CLI.validate_ahncr_report)
    assert output.read_bytes() == first
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
