import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sfora.publication import (
    HfPublishConfig,
    build_hf_publish_bundle,
    publish_hf_bundle,
)


def _write_fixture_project(tmp_path: Path) -> None:
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "README.md").write_text("# sfora\n", encoding="utf-8")
    (tmp_path / "reports" / "archive").mkdir(parents=True)
    (tmp_path / "reports" / "archive" / "result.json").write_text(
        '{"name": "sentence-transformer-training"}\n',
        encoding="utf-8",
    )
    (tmp_path / "reports" / "REPORT.md").write_text("# Report\n", encoding="utf-8")
    (tmp_path / "reports" / "site").mkdir()
    (tmp_path / "reports" / "site" / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "src" / "sfora").mkdir(parents=True)
    (tmp_path / "src" / "sfora" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "sfora" / "__pycache__").mkdir()
    (tmp_path / "src" / "sfora" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "__pycache__").mkdir()
    (tmp_path / "tests" / "__pycache__" / "test_smoke.pyc").write_bytes(b"cache")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_remote.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sfora'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def test_build_hf_publish_bundle_copies_publishable_files(tmp_path: Path) -> None:
    _write_fixture_project(tmp_path)
    output_dir = tmp_path / "dist" / "hf_publish"

    bundle = build_hf_publish_bundle(
        HfPublishConfig(
            repo_id="romanbartusiak/sfora",
            project_root=tmp_path,
            output_dir=output_dir,
        )
    )

    assert bundle.bundle_dir == output_dir
    assert (output_dir / "README.md").read_text(encoding="utf-8") == "# sfora\n"
    assert (output_dir / "reports" / "archive" / "result.json").exists()
    assert (output_dir / "src" / "sfora" / "__init__.py").exists()
    assert (output_dir / "tests" / "test_smoke.py").exists()
    assert (output_dir / "scripts" / "run_remote.sh").exists()
    manifest = json.loads((output_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["repo_id"] == "romanbartusiak/sfora"
    assert "reports/archive/result.json" in manifest["files"]
    assert "reports/site/index.html" in manifest["files"]
    assert "tests/test_smoke.py" in manifest["files"]
    assert "scripts/run_remote.sh" in manifest["files"]
    assert manifest["sha256"]["README.md"] == (
        "59b8ac7b251f45560d66161e9638ec124d2ef347fabef6a61396f8207f920d2a"
    )
    assert set(manifest["sha256"]) == set(manifest["files"]) - {"MANIFEST.json"}
    assert not any("__pycache__" in file for file in manifest["files"])
    assert not any(file.endswith(".pyc") for file in manifest["files"])
    assert sorted(bundle.files) == manifest["files"]


@dataclass
class FakeHfApi:
    created: list[dict[str, Any]] = field(default_factory=list)
    uploaded: list[dict[str, Any]] = field(default_factory=list)

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> str:
        self.created.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
            }
        )
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        commit_message: str,
    ) -> str:
        self.uploaded.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "folder_path": folder_path,
                "commit_message": commit_message,
            }
        )
        return "commit-123"


def test_publish_hf_bundle_can_upload_with_injected_api(tmp_path: Path) -> None:
    _write_fixture_project(tmp_path)
    fake_api = FakeHfApi()

    result = publish_hf_bundle(
        HfPublishConfig(
            repo_id="romanbartusiak/sfora",
            project_root=tmp_path,
            output_dir=tmp_path / "dist" / "hf_publish",
            dry_run=False,
            private=True,
            commit_message="Publish sfora report",
        ),
        api_factory=lambda _token: fake_api,
    )

    assert result.uploaded is True
    assert result.repo_url == "https://huggingface.co/romanbartusiak/sfora"
    assert fake_api.created == [
        {
            "repo_id": "romanbartusiak/sfora",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert fake_api.uploaded[0]["commit_message"] == "Publish sfora report"


class FailingHfApi:
    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> str:
        raise ValueError("401 Unauthorized")

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        commit_message: str,
    ) -> str:
        raise AssertionError("upload_folder should not run after create_repo fails")


def test_publish_hf_bundle_reports_upload_failures_without_traceback(tmp_path: Path) -> None:
    _write_fixture_project(tmp_path)

    try:
        publish_hf_bundle(
            HfPublishConfig(
                repo_id="romanbartusiak/sfora",
                project_root=tmp_path,
                output_dir=tmp_path / "dist" / "hf_publish",
                dry_run=False,
            ),
            api_factory=lambda _token: FailingHfApi(),
        )
    except RuntimeError as error:
        assert "Failed to publish Hugging Face bundle" in str(error)
        assert "401 Unauthorized" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
