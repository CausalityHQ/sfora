"""Authenticated connected bundle → unchanged packed wire → exact native top-10.

Native parity, quality and latency require independent qualification. In
particular, a loader failure before an endpoint exists has no release API:
only proven registry ownership and finished owned-frame references are cleaned.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import threading
import uuid
from collections.abc import Iterable
from contextlib import suppress
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sfora.cutile_int8 import CutilePackedInt8Gallery

# Supported registry writers (including integrations replacing an owned entry)
# must acquire this lock. Arbitrary unsynchronized sys.modules writes are outside
# that ownership contract.
_REGISTRY_LOCK = threading.RLock()

_TRAINER = "train_siglip2_connected_mlp.py"
_CODE = {
    _TRAINER,
    "test_siglip2_connected_mlp.py",
    "qualify_siglip2_substrate_cpu.py",
    "extract_siglip2_vision_source.py",
    "train_siglip2_cached_readout.py",
    "train_siglip2_substrate_adaptation.py",
    "prototype_residual_readout.py",
    "quadratic_readout.py",
    "joint_relational_compaction.py",
}
_FILES = {"vision.pt", "endpoint.pt", "processor.json"}
_MANIFEST = {
    "schema",
    "code",
    "files",
    "endpoint_state_sha256",
    "environment",
    "encoder_identity",
    "base_vision_sha256",
    "vision_sha256",
    "scope",
}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checked_file(path: Path, digest: str, owned: bool = False) -> Path:
    _require(
        isinstance(path, Path)
        and path.is_absolute()
        and path.resolve() == path
        and path.is_file()
        and not path.is_symlink(),
        "canonical regular file required",
    )
    _require(
        isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "explicit SHA256 required"
    )
    _require(not owned or path.stat().st_nlink == 1, "single-link owned bundle required")
    with path.open("rb") as stream:
        _require(
            hashlib.file_digest(stream, "sha256").hexdigest() == digest,
            "current file bytes differ: " + str(path),
        )
    return path


def _read_checked(path: Path, digest: str, owned: bool = False) -> bytes:
    raw = _checked_file(path, digest, owned).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == digest, "file changed before use")
    return raw


def _json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    _require(len(result) == len(pairs), "duplicate manifest key")
    return result


class ConnectedCompactIndex:
    """Own one genuine CUDA endpoint and a copied resident native gallery.

    Manifest/gallery/native disk inputs are pinned at startup. Bundle helper
    and processor-backend sources remain required by genuine per-call checks.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._module: ModuleType | None = None
        self._endpoint: dict[str, Any] | None = None
        self._gallery: CutilePackedInt8Gallery | None = None
        self._owned: dict[str, ModuleType] = {}
        self._apis: dict[str, tuple[FunctionType, CodeType]] = {}
        self._release: FunctionType | None = None
        self._release_modules: tuple[ModuleType, ...] = ()
        self._guards: tuple[tuple[Path, str, bool], ...] = ()

    @classmethod
    def from_bundle(
        cls,
        *,
        bundle_dir: Path,
        expected_bundle_sha256: str,
        gallery_path: Path,
        expected_gallery_sha256: str,
        gallery_count: int,
        native_library_path: Path,
        expected_native_library_sha256: str,
    ) -> ConnectedCompactIndex:
        """Load explicit caller-pinned bytes; device, dimensions and k are fixed."""
        self = cls()
        try:
            _require(
                isinstance(bundle_dir, Path)
                and bundle_dir.is_absolute()
                and bundle_dir.resolve() == bundle_dir
                and bundle_dir.is_dir(),
                "canonical bundle directory required",
            )
            _require(
                type(gallery_count) is int and gallery_count >= 10, "gallery count must be >=10"
            )
            manifest_path = bundle_dir / "bundle.json"
            raw = _read_checked(manifest_path, expected_bundle_sha256, True)
            manifest = json.loads(
                raw,
                object_pairs_hook=_json_pairs,
                parse_constant=lambda value: _require(False, "nonfinite manifest"),
            )
            _require(
                isinstance(manifest, dict)
                and manifest.keys() == _MANIFEST
                and manifest["schema"] == "siglip2-connected-mlp-bundle-v1"
                and isinstance(manifest["code"], dict)
                and manifest["code"].keys() == _CODE
                and isinstance(manifest["files"], dict)
                and manifest["files"].keys() == _FILES,
                "unsupported connected bundle schema/closure",
            )
            for name, digest in (manifest["code"] | manifest["files"]).items():
                _checked_file(bundle_dir / name, digest, True)
            trainer_path, trainer_sha = bundle_dir / _TRAINER, manifest["code"][_TRAINER]
            source = _read_checked(trainer_path, trainer_sha, True)
            gallery_wire = _read_checked(gallery_path, expected_gallery_sha256)
            _checked_file(native_library_path, expected_native_library_sha256)
            self._guards = ((trainer_path, trainer_sha, True),)
            name = "_sfora_connected_compact_" + uuid.uuid4().hex
            spec = importlib.util.spec_from_file_location(name, trainer_path)
            _require(
                spec is not None and spec.loader is not None, "connected loader origin required"
            )
            self._module = importlib.util.module_from_spec(cast(ModuleSpec, spec))
            with _REGISTRY_LOCK:
                _require(name not in sys.modules, "fresh connected loader namespace required")
                self._owned[name] = self._module
                sys.modules[name] = self._module
                exec(compile(source, str(trainer_path), "exec"), vars(self._module))
            for api in (
                "load_inference",
                "load_authenticated",
                "inference_outputs",
                "release_inference",
            ):
                fn = getattr(self._module, api)
                _require(
                    type(fn) is FunctionType and fn.__globals__ is vars(self._module),
                    "genuine connected public function required",
                )
                self._apis[api] = (fn, fn.__code__)
            fn, code = self._apis["release_inference"]
            self._release = FunctionType(
                code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
            )
            # Imports are deliberately lazy; source-only checks substitute these seams.
            # Preserve the original order of these lazy imports.
            from sfora.joint_relational_compaction import PackedInt8Embeddings  # noqa: I001
            from sfora.cutile_int8 import CutilePackedInt8Gallery

            packed = PackedInt8Embeddings.from_bytes(
                gallery_wire, count=gallery_count, dimensions=128
            )
            # ponytail: serialize helper registration at startup; an original
            # synchronized registrar is needed only for parallel bundle loading.
            with _REGISTRY_LOCK:
                self._endpoint = cast(
                    dict[str, Any],
                    self._apis["load_inference"][0](bundle_dir, expected_bundle_sha256, "cuda"),
                )
                self._release_modules = tuple(self._endpoint["modules"].values())
                self._remember(self._release_modules)
            self._check_current()
            _checked_file(native_library_path, expected_native_library_sha256)
            self._gallery = CutilePackedInt8Gallery.open_packed(native_library_path, packed)
            return self
        except BaseException as error:
            self._capture_failure(error)
            self._close_after_error(error)
            raise

    def _remember(self, modules: Iterable[object]) -> None:
        for module in modules:
            if type(module) is ModuleType:
                self._owned[module.__name__] = module

    def _capture_failure(self, error: BaseException | None) -> None:
        """Use exact authenticated loader frames, never a registry diff or sweep."""
        trace = cast(BaseException, error).__traceback__
        frames = []
        while trace is not None:
            frame = trace.tb_frame
            if self._module is not None and frame.f_globals is vars(self._module):
                frames.append(frame)
                if frame.f_code is self._apis.get("load_inference", (None, None))[1]:
                    self._remember(frame.f_locals.get("modules", {}).values())
                    endpoint = frame.f_locals.get("endpoint")
                    if isinstance(endpoint, dict):
                        self._endpoint = endpoint
                        self._release_modules = tuple(endpoint["modules"].values())
                if frame.f_code is self._apis.get("load_authenticated", (None, None))[1]:
                    self._remember((frame.f_locals.get("module"),))
            trace = trace.tb_next
        # Only finished frames belonging to proven owned source modules are cleared.
        # Traceback locations and exception/cause objects remain available to callers.
        namespaces = tuple(vars(module) for module in self._owned.values())
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            trace = error.__traceback__
            while trace is not None:
                if any(trace.tb_frame.f_globals is namespace for namespace in namespaces):
                    frames.append(trace.tb_frame)
                trace = trace.tb_next
            error = error.__cause__ or error.__context__
        for frame in frames:
            # An active frame retains its own lifetime; never clear it.
            with suppress(RuntimeError):
                frame.clear()

    def _check_current(self) -> None:
        for guard in self._guards:
            _checked_file(*guard)
        _require(
            all(sys.modules.get(name) is module for name, module in self._owned.items()),
            "owned connected registry changed",
        )
        for name, (fn, code) in self._apis.items():
            _require(
                getattr(cast(ModuleType, self._module), name) is fn
                and fn.__code__ is code
                and fn.__globals__ is vars(cast(ModuleType, self._module)),
                "connected public callable changed",
            )
        _require(
            cast(ModuleType, self._module).__file__ == str(self._guards[0][0])
            and cast(ModuleSpec, cast(ModuleType, self._module).__spec__).origin
            == cast(ModuleType, self._module).__file__
            and cast(ModuleSpec, cast(ModuleType, self._module).__spec__).name
            == cast(ModuleType, self._module).__name__,
            "connected loader origin changed",
        )

    def _close_after_error(self, error: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note("connected cleanup also failed: " + repr(cleanup))

    def search_images(self, images: list[object] | tuple[object, ...]) -> tuple[object, object]:
        """Return unchanged native IDs/scores for 1..32 PIL image objects."""
        with self._lock:
            if self._closed:
                raise RuntimeError("connected compact index is closed")
            _require(isinstance(images, (list, tuple)), "PIL image batch length must be 1..32")
            images = list(images)
            _require(1 <= len(images) <= 32, "PIL image batch length must be 1..32")
            from PIL.Image import Image

            _require(
                all(isinstance(image, Image) for image in images), "PIL image objects required"
            )
            try:
                self._check_current()
                from sfora.joint_relational_compaction import PackedInt8Embeddings

                output = self._apis["inference_outputs"][0](self._endpoint, images)
                queries = PackedInt8Embeddings.from_bytes(
                    output["wire"], count=len(images), dimensions=128
                )
                return cast(
                    tuple[object, object],
                    cast("CutilePackedInt8Gallery", self._gallery).search_packed(queries, k=10),
                )
            except BaseException as error:
                self._capture_failure(error)
                self._close_after_error(error)
                raise

    def _check_release_registry(self, modules: tuple[ModuleType, ...]) -> None:
        # Corresponds to EACH original release pop(...) is module predicate.
        # Evaluate the complete saved inventory, even if an earlier entry differs.
        identities = [sys.modules.get(module.__name__) is module for module in modules]
        owned = [sys.modules.get(name) is module for name, module in self._owned.items()]
        _require(all(identities) and all(owned), "owned serving registry changed")

    def close(self) -> None:
        """Close once; report tampering only after genuine cleanup is attempted."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors = []
            modules = self._release_modules
            with _REGISTRY_LOCK:
                for check in (self._check_current, lambda: self._check_release_registry(modules)):
                    try:
                        if self._module is not None:
                            check()
                    except BaseException as error:
                        errors.append(error)
            try:
                if self._gallery is not None:
                    self._gallery.close()
            except BaseException as error:
                errors.append(error)
            try:
                if self._endpoint is not None:
                    # Delegate ONLY registry removal. The original complete module
                    # tuple is checked explicitly above AND below, never filtered.
                    # The authenticated release retains processor-cache authority,
                    # cache clearing/emptiness, endpoint clearing, GC, every resource
                    # lifetime predicate and CUDA cleanup using its original globals.
                    self._endpoint["modules"] = {}
                    cast(FunctionType, self._release)(self._endpoint)
            except BaseException as error:
                errors.append(error)
            finally:
                if self._endpoint is not None:
                    self._endpoint.clear()
                with _REGISTRY_LOCK:
                    try:
                        self._check_release_registry(modules)
                    except BaseException as error:
                        errors.append(error)
                    # Identity check + delete is atomic for supported writers.
                    for name, module in self._owned.items():
                        if sys.modules.get(name) is module:
                            del sys.modules[name]
                for failure in errors:
                    self._capture_failure(failure)
                self._endpoint = self._gallery = self._module = self._release = None
                self._owned.clear()
                self._apis.clear()
                self._release_modules = ()
            if errors:
                for failure in errors[1:]:
                    errors[0].add_note("connected cleanup also failed: " + repr(failure))
                raise errors[0]

    def __enter__(self) -> ConnectedCompactIndex:
        _require(not self._closed, "connected compact index is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
