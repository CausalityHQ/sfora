import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

RepoType = Literal["model", "dataset", "space"]


class HfPublishConfig(BaseModel):
    """Configuration for building and optionally uploading a Hugging Face bundle."""

    repo_id: str
    repo_type: RepoType = "model"
    project_root: Path = Field(default_factory=Path.cwd)
    output_dir: Path = Path("dist/hf_publish")
    private: bool = False
    dry_run: bool = True
    token: str | None = None
    commit_message: str = "Publish sfora report"


@dataclass(frozen=True)
class HfPublishBundle:
    """Local publish bundle metadata."""

    repo_id: str
    repo_type: RepoType
    bundle_dir: Path
    files: list[str]


@dataclass(frozen=True)
class HfPublishResult:
    """Result from preparing and optionally uploading a Hugging Face bundle."""

    bundle: HfPublishBundle
    uploaded: bool
    repo_url: str | None
    commit_url: str | None


class HfApiLike(Protocol):
    """Subset of the Hugging Face Hub API used by publication."""

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> str:
        """Create or reuse a Hub repository."""

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        commit_message: str,
    ) -> str:
        """Upload a local folder to the Hub."""


HfApiFactory = Callable[[str | None], HfApiLike]

_DEFAULT_FILE_PATHS = (
    Path("hf/README.md"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("docs/results.md"),
    Path("reports/REPORT.md"),
)
_DEFAULT_DIR_PATHS = (
    Path("src/sfora"),
    Path("tests"),
    Path("scripts"),
    Path("reports/archive"),
    Path("reports/site"),
)


def build_hf_publish_bundle(config: HfPublishConfig) -> HfPublishBundle:
    """Build a deterministic local folder ready for Hugging Face Hub upload."""
    project_root = config.project_root.resolve()
    output_dir = _resolve_output_dir(config.output_dir, project_root)
    if output_dir.exists():
        # Only ever delete a directory we created (carries our marker) or an empty
        # one — never clobber a pre-existing directory full of unrelated files.
        marker = output_dir / _BUNDLE_MARKER
        if not marker.exists() and any(output_dir.iterdir()):
            raise ValueError(
                f"{output_dir} exists and is not a sfora HF bundle (no {_BUNDLE_MARKER} "
                "marker) and is not empty; refusing to delete it. Remove it manually or "
                "choose another --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _BUNDLE_MARKER).write_text("sfora hugging-face bundle output\n")

    copied_files: list[str] = []
    for source in _DEFAULT_FILE_PATHS:
        source_path = project_root / source
        if source_path.exists():
            destination = _destination_path(source, output_dir)
            _copy_file(source_path, destination)
            copied_files.append(_relative_posix(destination, output_dir))

    for source in _DEFAULT_DIR_PATHS:
        source_path = project_root / source
        if not source_path.exists():
            continue
        for file_path in sorted(
            path for path in source_path.rglob("*") if path.is_file() and _is_publishable(path)
        ):
            relative_path = file_path.relative_to(project_root)
            destination = output_dir / relative_path
            _copy_file(file_path, destination)
            copied_files.append(_relative_posix(destination, output_dir))

    copied_files = sorted(set(copied_files + ["MANIFEST.json"]))
    manifest_path = output_dir / "MANIFEST.json"
    manifest_payload = {
        "repo_id": config.repo_id,
        "repo_type": config.repo_type,
        "files": copied_files,
        "sha256": _sha256_manifest(output_dir, copied_files),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return HfPublishBundle(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        bundle_dir=output_dir,
        files=sorted(copied_files),
    )


def publish_hf_bundle(
    config: HfPublishConfig,
    *,
    api_factory: HfApiFactory | None = None,
) -> HfPublishResult:
    """Build and optionally upload a Hugging Face publish bundle."""
    bundle = build_hf_publish_bundle(config)
    if config.dry_run:
        return HfPublishResult(bundle=bundle, uploaded=False, repo_url=None, commit_url=None)

    factory = api_factory or _load_hf_api
    api = factory(config.token)
    try:
        repo_url = api.create_repo(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            private=config.private,
            exist_ok=True,
        )
        commit_url = api.upload_folder(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            folder_path=str(bundle.bundle_dir),
            commit_message=config.commit_message,
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to publish Hugging Face bundle. "
            "Check that your token has write access to the target repo: "
            f"{error}"
        ) from error
    return HfPublishResult(bundle=bundle, uploaded=True, repo_url=repo_url, commit_url=commit_url)


_BUNDLE_MARKER = ".sfora-hf-bundle"


def _resolve_output_dir(output_dir: Path, project_root: Path) -> Path:
    resolved = (output_dir if output_dir.is_absolute() else project_root / output_dir).resolve()
    # Never operate on the project root, one of its ancestors, or a filesystem root:
    # the bundle build recursively deletes this directory, so an unsafe path (e.g.
    # `--output-dir .`) would wipe the repository.
    if resolved == project_root or resolved in project_root.parents or resolved == resolved.parent:
        raise ValueError(
            f"refusing to use {resolved} as the Hugging Face bundle output directory: "
            "it is the project root, an ancestor, or a filesystem root. Choose a "
            "dedicated subdirectory (e.g. reports/hf-bundle)."
        )
    return resolved


def _destination_path(source: Path, output_dir: Path) -> Path:
    if source == Path("hf/README.md"):
        return output_dir / "README.md"
    if source == Path("README.md"):
        return output_dir / "PROJECT_README.md"
    return output_dir / source


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _is_publishable(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return False
    return path.suffix not in {".pyc", ".pyo"}


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256_manifest(output_dir: Path, files: list[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for file_name in files:
        if file_name == "MANIFEST.json":
            continue
        checksums[file_name] = sha256((output_dir / file_name).read_bytes()).hexdigest()
    return checksums


def _load_hf_api(token: str | None) -> HfApiLike:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to publish to Hugging Face: "
            "uv sync --group dev --extra research"
        ) from error
    return HfApi(token=token)
