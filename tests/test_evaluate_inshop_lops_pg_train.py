from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sfora.idgp import assign_label_folds

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_inshop_lops_pg_train.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_lops_pg_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _labels_in_fold(fold: int, count: int) -> list[int]:
    result: list[int] = []
    label = 0
    while len(result) < count:
        values = np.asarray([label], dtype=np.int64)
        if assign_label_folds(values)[label] == fold:
            result.append(label)
        label += 1
    return result


def _archive(path: Path) -> None:
    labels = np.asarray(
        [label for fold in (1, 2, 3) for label in _labels_in_fold(fold, 45) for _ in range(4)],
        dtype=np.int64,
    )
    rng = np.random.default_rng(91)
    embeddings = rng.normal(size=(labels.size, 8)).astype(np.float64)
    example_ids = np.asarray([f"train-{index}" for index in range(labels.size)])
    np.savez(path, embeddings=embeddings, labels=labels, example_ids=example_ids)


def test_build_report_is_confirmation_only_and_relational(tmp_path: Path) -> None:
    archive = tmp_path / "train.npz"
    _archive(archive)
    report = cli.build_report(archive, require_registered_hash=False, bootstrap_replicates=100)
    validated = cli.validate_report(report, bootstrap_replicates=100)
    assert validated["protocol"]["folds"] == [1, 2, 3]
    assert {row["fold"] for row in validated["rows"]} == {1, 2, 3}
    assert validated["status"] in {"PASS", "KILL"}

    mutated = cli.json.loads(cli.json.dumps(report))
    cohort = next(row for row in mutated["rows"] if any(row["primary_mask"]))
    primary = cohort["primary_mask"].index(True)
    cohort["margin_changes"]["lops_pg"][primary] += 0.1
    with pytest.raises(ValueError):
        cli.validate_report(mutated, bootstrap_replicates=100)


def test_parser_rejects_query_gallery_and_writer_is_no_clobber(tmp_path: Path) -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--query", "q.npz", "--output", "x.json"])
    output = tmp_path / "result.json"
    cli.write_report(output, {"value": 1}, validator=lambda value: value)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.write_report(output, {"value": 2}, validator=lambda value: value)
    assert output.read_bytes() == before
