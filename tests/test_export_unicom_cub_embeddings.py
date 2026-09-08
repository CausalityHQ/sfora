from __future__ import annotations

import hashlib
import importlib.util
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_unicom_cub_embeddings.py"
_SPEC = importlib.util.spec_from_file_location("export_unicom_cub_embeddings", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_cub_fixture(root: Path) -> None:
    images = root / "images"
    for class_id, name in ((1, "one"), (2, "two"), (3, "bird"), (4, "other")):
        directory = images / f"{class_id:03d}.{name}"
        directory.mkdir(parents=True)
        for suffix in ("a", "b"):
            (directory / f"{suffix}.jpg").write_bytes(f"{class_id}-{suffix}".encode())
    paths = [
        "001.one/a.jpg",
        "001.one/b.jpg",
        "002.two/a.jpg",
        "002.two/b.jpg",
        "003.bird/a.jpg",
        "003.bird/b.jpg",
        "004.other/a.jpg",
        "004.other/b.jpg",
    ]
    (root / "images.txt").write_text(
        "".join(f"{index} {path}\n" for index, path in enumerate(paths, start=1)),
        encoding="utf-8",
    )
    labels = (1, 1, 2, 2, 3, 3, 4, 4)
    (root / "image_class_labels.txt").write_text(
        "".join(f"{index} {label}\n" for index, label in enumerate(labels, start=1)),
        encoding="utf-8",
    )
    (root / "classes.txt").write_text(
        "1 001.one\n2 002.two\n3 003.bird\n4 004.other\n",
        encoding="utf-8",
    )


def _write_cub_archive(path: Path, root: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        archive.add(root, arcname="CUB_200_2011")


def test_parse_cub_records_uses_disjoint_first100_last100_protocol(tmp_path: Path) -> None:
    _write_cub_fixture(tmp_path)

    records = _MODULE.parse_cub_records(
        tmp_path,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_total_classes=4,
        train_class_max=2,
    )

    assert tuple(record.split for record in records) == ("train",) * 4 + ("test",) * 4
    assert tuple(record.image_id for record in records) == tuple(range(1, 9))
    assert tuple(record.label for record in records) == (1, 1, 2, 2, 3, 3, 4, 4)
    assert tuple(record.class_name for record in records) == (
        "001.one",
        "001.one",
        "002.two",
        "002.two",
        "003.bird",
        "003.bird",
        "004.other",
        "004.other",
    )
    authority = b"".join(
        f"{record.split}\0{record.image_id}\0{record.label}\0{record.class_name}\0"
        f"{record.relative_path}\n".encode()
        for record in records
    )
    assert _MODULE.ordered_record_sha256(records) == hashlib.sha256(authority).hexdigest()


def test_extracted_content_is_bound_byte_for_byte_to_verified_archive(tmp_path: Path) -> None:
    root = tmp_path / "CUB_200_2011"
    root.mkdir()
    _write_cub_fixture(root)
    archive = tmp_path / "CUB_200_2011.tgz"
    _write_cub_archive(archive, root)
    records = _MODULE.parse_cub_records(
        root,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_total_classes=4,
        train_class_max=2,
    )

    first = _MODULE.verify_extracted_cub_matches_archive(archive, root, records)
    second = _MODULE.verify_extracted_cub_matches_archive(archive, root, records)
    assert first == second
    assert len(first) == 64
    (root / "images/003.bird/a.jpg").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="content"):
        _MODULE.verify_extracted_cub_matches_archive(archive, root, records)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing-image", "regular"),
        ("duplicate-path", "duplicate"),
        ("label-id", "identity"),
        ("class-name", "class"),
        ("singleton", "class cardinality"),
    ),
)
def test_parse_cub_records_rejects_protocol_drift(
    tmp_path: Path, mutation: str, match: str
) -> None:
    _write_cub_fixture(tmp_path)
    if mutation == "missing-image":
        (tmp_path / "images/003.bird/a.jpg").unlink()
    elif mutation == "duplicate-path":
        value = (tmp_path / "images.txt").read_text()
        (tmp_path / "images.txt").write_text(
            value.replace("8 004.other/b.jpg", "8 004.other/a.jpg"), encoding="utf-8"
        )
    elif mutation == "label-id":
        value = (tmp_path / "image_class_labels.txt").read_text()
        (tmp_path / "image_class_labels.txt").write_text(
            value.replace("8 4", "9 4"), encoding="utf-8"
        )
    elif mutation == "class-name":
        value = (tmp_path / "classes.txt").read_text()
        (tmp_path / "classes.txt").write_text(
            value.replace("3 003.bird", "3 wrong"), encoding="utf-8"
        )
    else:
        for name in ("images.txt", "image_class_labels.txt"):
            lines = (tmp_path / name).read_text().splitlines()
            (tmp_path / name).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _MODULE.parse_cub_records(
            tmp_path,
            expected_counts=(4, 4),
            expected_classes=(2, 2),
            expected_total_classes=4,
            train_class_max=2,
        )


def test_export_and_load_cub_embeddings_preserve_exact_authority(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_cub_fixture(root)
    records = _MODULE.parse_cub_records(
        root,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_total_classes=4,
        train_class_max=2,
    )
    output = tmp_path / "bundle.npz"

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        return np.asarray(
            [[float(path.read_bytes()[0]), float(len(path.name)), 1.0] for path in paths],
            dtype=np.float32,
        )

    _MODULE.export_cub_embeddings(
        records,
        encode,
        {
            "model_identifier": "fixture-model",
            "model_revision": "ab" * 20,
            "checkpoint_sha256": "cd" * 32,
            "transform": "fixture transform",
            "dataset_archive_sha256": _MODULE.CUB_ARCHIVE_SHA256,
            "dataset_archive_md5": _MODULE.CUB_ARCHIVE_MD5,
            "dataset_content_sha256": "ef" * 32,
            "cub_exporter_source_sha256": "12" * 32,
            "sop_exporter_source_sha256": "34" * 32,
        },
        output,
        batch_size=3,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
    )

    loaded = _MODULE.load_cub_embedding_archive(
        output,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_dimension=3,
    )
    metadata = loaded["metadata"]
    assert metadata["schema"] == "sfora-unicom-cub-embeddings-v1"
    assert metadata["protocol"] == "classes-001-100-train-101-200-test"
    assert metadata["export_batch_size"] == 3
    assert metadata["ordered_record_sha256"] == _MODULE.ordered_record_sha256(records)
    assert loaded["train_labels"].tolist() == [1, 1, 2, 2]
    assert loaded["test_labels"].tolist() == [3, 3, 4, 4]
    assert loaded["test_class_names"].tolist() == [
        "003.bird",
        "003.bird",
        "004.other",
        "004.other",
    ]


def test_cub_export_rejects_nonfinite_encoder_and_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_cub_fixture(root)
    records = _MODULE.parse_cub_records(
        root,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_total_classes=4,
        train_class_max=2,
    )
    output = tmp_path / "bundle.npz"
    metadata = {
        "model_identifier": "fixture-model",
        "model_revision": "ab" * 20,
        "checkpoint_sha256": "cd" * 32,
        "transform": "fixture transform",
        "dataset_archive_sha256": _MODULE.CUB_ARCHIVE_SHA256,
        "dataset_archive_md5": _MODULE.CUB_ARCHIVE_MD5,
        "dataset_content_sha256": "ef" * 32,
        "cub_exporter_source_sha256": "12" * 32,
        "sop_exporter_source_sha256": "34" * 32,
    }

    with pytest.raises(ValueError, match="encoded batch"):
        _MODULE.export_cub_embeddings(
            records,
            lambda paths: np.full((len(paths), 3), np.nan, dtype=np.float32),
            metadata,
            output,
            expected_counts=(4, 4),
            expected_classes=(2, 2),
        )


def test_cub_export_never_deletes_or_overwrites_concurrent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_cub_fixture(root)
    records = _MODULE.parse_cub_records(
        root,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_total_classes=4,
        train_class_max=2,
    )
    output = tmp_path / "bundle.npz"
    partial = output.with_name(output.name + ".partial")
    metadata = {
        "model_identifier": "fixture-model",
        "model_revision": "ab" * 20,
        "checkpoint_sha256": "cd" * 32,
        "transform": "fixture transform",
        "dataset_archive_sha256": _MODULE.CUB_ARCHIVE_SHA256,
        "dataset_archive_md5": _MODULE.CUB_ARCHIVE_MD5,
        "dataset_content_sha256": "ef" * 32,
        "cub_exporter_source_sha256": "12" * 32,
        "sop_exporter_source_sha256": "34" * 32,
    }
    real_open = Path.open

    def racing_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == partial and mode == "xb" and not partial.exists():
            partial.write_bytes(b"other-run")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        _MODULE.export_cub_embeddings(
            records,
            lambda paths: np.ones((len(paths), 3), dtype=np.float32),
            metadata,
            output,
            expected_counts=(4, 4),
            expected_classes=(2, 2),
        )
    assert partial.read_bytes() == b"other-run"
    assert not output.exists()
    assert not output.exists()
    output.write_bytes(b"caller-owned")
    with pytest.raises(FileExistsError):
        _MODULE.export_cub_embeddings(
            records,
            lambda paths: np.ones((len(paths), 3), dtype=np.float32),
            metadata,
            output,
            expected_counts=(4, 4),
            expected_classes=(2, 2),
        )


def test_cub_cli_is_explicit_local_only() -> None:
    args = _MODULE.parse_args(
        [
            "--unicom-checkout",
            "/checkout",
            "--checkpoint",
            "/checkpoint.pt",
            "--dataset-root",
            "/dataset",
            "--dataset-archive",
            "/dataset.tgz",
            "--dataset-archive-sha256",
            "ab" * 32,
            "--model",
            "b16",
            "--output",
            "/output.npz",
            "--execute-export",
        ]
    )
    assert args.dataset_archive == Path("/dataset.tgz")
    assert args.execute_export is True
    with pytest.raises(SystemExit):
        _MODULE.parse_args(["--dataset-root", "/dataset"])


def test_unicom_checkout_must_be_exact_revision_and_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(("ab" * 20 + "\n", ""))
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=next(responses)),
    )
    _MODULE._require_clean_git_checkout(tmp_path, "ab" * 20)

    responses = iter(("ab" * 20 + "\n", "?? untracked.py\n"))
    with pytest.raises(ValueError, match="checkout authority"):
        _MODULE._require_clean_git_checkout(tmp_path, "ab" * 20)


def test_main_removes_published_export_when_post_audit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive.tgz"
    archive.write_bytes(b"archive")
    output = tmp_path / "output.npz"
    args = SimpleNamespace(
        batch_size=1,
        checkpoint=tmp_path / "checkpoint.pt",
        dataset_archive=archive,
        dataset_archive_sha256=_MODULE.CUB_ARCHIVE_SHA256,
        dataset_root=tmp_path / "root",
        model="b16",
        output=output,
        unicom_checkout=tmp_path / "checkout",
    )
    authority = SimpleNamespace(
        checkpoint_sha256="cd" * 32,
        identifier="fixture",
        load_name="fixture",
        revision="ab" * 20,
    )
    monkeypatch.setattr(_MODULE, "parse_args", lambda _arguments: args)
    monkeypatch.setattr(_MODULE, "CUB_ARCHIVE_BYTES", len(b"archive"))
    monkeypatch.setattr(
        _MODULE,
        "_hash_file",
        lambda _path, algorithm: (
            _MODULE.CUB_ARCHIVE_SHA256 if algorithm == "sha256" else _MODULE.CUB_ARCHIVE_MD5
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "export_unicom_sop_embeddings",
        SimpleNamespace(
            __file__=str(_SCRIPT),
            _official_encoder=lambda *_args: object(),
            model_authority=lambda _model: authority,
        ),
    )
    monkeypatch.setattr(_MODULE, "_require_clean_git_checkout", lambda *_args: None)
    monkeypatch.setattr(_MODULE, "parse_cub_records", lambda _root: (object(),))
    audits = iter(("ef" * 32, ValueError("post audit failed")))

    def audit(*_args: object) -> str:
        value = next(audits)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(_MODULE, "verify_extracted_cub_matches_archive", audit)
    monkeypatch.setattr(
        _MODULE,
        "export_cub_embeddings",
        lambda *_args, **_kwargs: output.write_bytes(b"published"),
    )

    with pytest.raises(ValueError, match="post audit failed"):
        _MODULE.main([])
    assert not output.exists()
