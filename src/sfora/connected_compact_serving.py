"""Authenticated connected bundle → unchanged packed wire → exact native top-10.

Native parity, quality and latency require independent qualification. In
particular, a loader failure before an endpoint exists has no release API:
only proven registry ownership and finished owned-frame references are cleaned.
"""

from __future__ import annotations

import hashlib
import ast
import copy
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


def _installed_authority() -> tuple[Path, str]:
    return (
        Path(__file__).absolute().parent / "_connected_inference_authority.py",
        "6e1027127d827f031db6673cac668a8da943f0395f652319d28dce5d8afc536b",
    )


def _literal_state(value: object) -> object:
    """Snapshot only mutable literal globals; modules/callables retain identity."""
    if type(value) is dict:
        return dict, tuple((_literal_state(key), _literal_state(item)) for key, item in cast(dict[object, object], value).items())
    if type(value) in (tuple, list):
        return type(value), tuple(_literal_state(item) for item in cast(tuple[object, ...], value))
    if type(value) in (set, frozenset):
        return type(value), frozenset(_literal_state(item) for item in cast(set[object] | frozenset[object], value))
    return type(value), value


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
        self._namespaces: list[tuple[ModuleType | type[Any], dict[str, object], dict[str, object]]] = []
        self._callables: list[tuple[FunctionType, CodeType, object, object, object, object, object]] = []
        self._shared: tuple[ModuleType, ...] = ()

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
            authority_path, authority_sha = _installed_authority()
            authority_source = _read_checked(authority_path, authority_sha)
            # The authenticated record is literal data and never binds the bridge.
            tree = ast.parse(authority_source)
            _require(
                all(
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    for node in tree.body
                ),
                "literal installed inference authority required",
            )
            record = {
                node.targets[0].id: ast.literal_eval(node.value)
                for node in tree.body if isinstance(node, ast.Assign)
            }
            _require(
                record.keys() == {"SCHEMA", "HISTORICAL_CODE", "SOURCE_SYMBOLS", "PACKED_SOURCE_SYMBOLS",
                                  "SUBSTITUTIONS", "RUNTIME_SHA256", "PACKED_SHA256"}
                and record["SCHEMA"] == "sfora-connected-inference-extraction-v1"
                and tuple(sorted(manifest["code"].items())) == record["HISTORICAL_CODE"],
                "unsupported historical inference closure",
            )
            runtime_path = Path(__file__).resolve().parent / "connected_inference.py"
            packed_path = runtime_path.parent / "packed_int8.py"
            source = _read_checked(runtime_path, record["RUNTIME_SHA256"])
            _checked_file(packed_path, record["PACKED_SHA256"])
            gallery_wire = _read_checked(gallery_path, expected_gallery_sha256)
            _checked_file(native_library_path, expected_native_library_sha256)
            self._guards = (
                (runtime_path, record["RUNTIME_SHA256"], False),
                (authority_path, authority_sha, False),
                (packed_path, record["PACKED_SHA256"], False),
                *((bundle_dir / name, digest, True) for name, digest in manifest["code"].items()),
            )
            name = "_sfora_connected_compact_" + uuid.uuid4().hex
            spec = importlib.util.spec_from_file_location(name, runtime_path)
            _require(
                spec is not None and spec.loader is not None, "connected loader origin required"
            )
            self._module = importlib.util.module_from_spec(cast(ModuleSpec, spec))
            with _REGISTRY_LOCK:
                _require(name not in sys.modules, "fresh connected loader namespace required")
                self._owned[name] = self._module
                sys.modules[name] = self._module
                exec(compile(source, str(runtime_path), "exec", dont_inherit=True), vars(self._module))
            self._snapshot(self._module, source)
            self._module._bind_runtime(
                record["HISTORICAL_CODE"], tuple((str(path), sha) for path, sha, owned in self._guards[:3])
            )
            # The one permitted namespace change occurs during authenticated binding.
            self._namespaces.clear()
            self._callables.clear()
            self._snapshot(self._module, source)
            for api in ("load_inference", "inference_outputs", "release_inference"):
                fn = getattr(self._module, api)
                _require(
                    type(fn) is FunctionType and fn.__globals__ is vars(self._module),
                    "genuine connected public function required",
                )
                self._apis[api] = (fn, fn.__code__)
            # Validate scope, environment and every v1 data/source predicate
            # before the shared packing import can load Torch or NumPy.
            self._apis["admit_bundle"][0](bundle_dir, expected_bundle_sha256)
            fn, code = self._apis["release_inference"]
            # Cleanup uses an independent saved namespace, including genuine helpers.
            # Mutable function objects and runtime globals cannot replace that graph.
            release_globals = dict(fn.__globals__)
            for key, value in tuple(release_globals.items()):
                if type(value) is FunctionType and value.__globals__ is fn.__globals__:
                    release_globals[key] = FunctionType(
                        value.__code__, release_globals, value.__name__, copy.deepcopy(value.__defaults__), value.__closure__
                    )
                    release_globals[key].__kwdefaults__ = copy.deepcopy(value.__kwdefaults__)
            self._release = FunctionType(
                code, release_globals, fn.__name__, fn.__defaults__, fn.__closure__
            )
            # Imports are deliberately lazy; source-only checks substitute these seams.
            # Preserve the original order of these lazy imports.
            from sfora.packed_int8 import PackedInt8Embeddings  # noqa: I001
            from sfora.cutile_int8 import CutilePackedInt8Gallery

            shared = sys.modules["sfora.packed_int8"]
            _require(
                shared.__file__ == str(packed_path) and shared.__spec__.origin == str(packed_path),
                "canonical shared packing origin differs",
            )
            self._shared = (shared,)
            self._snapshot(shared, _read_checked(packed_path, record["PACKED_SHA256"]))
            self._check_current()

            packed = PackedInt8Embeddings.from_bytes(
                gallery_wire, count=gallery_count, dimensions=128
            )
            # ponytail: serialize helper registration at startup; an original
            # synchronized registrar is needed only for parallel bundle loading.
            with _REGISTRY_LOCK:
                self._check_current()
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
                if module not in self._shared:
                    self._owned[module.__name__] = module

    def _snapshot(self, module: ModuleType, source: bytes) -> None:
        namespace = vars(module)
        expected = compile(source, module.__file__, "exec", dont_inherit=True)
        declarations = {node.name for node in ast.parse(source).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        codes = {code.co_name: code for code in expected.co_consts
                 if isinstance(code, CodeType) and code.co_name in declarations}
        for key, code in codes.items():
            value = namespace.get(key)
            if type(value) is FunctionType:
                _require(value.__code__ == code and value.__globals__ is namespace,
                         "installed inference callable source differs")
                self._save_callable(value)
                if module is self._module:
                    self._apis[key] = (value, value.__code__)
            elif type(value) is type:
                methods = {c.co_name: c for c in code.co_consts if isinstance(c, CodeType)}
                class_values = dict(vars(value))
                self._namespaces.append((value, class_values, {
                    name: _literal_state(item) for name, item in class_values.items()
                    if type(item) in (dict, list, set, tuple)
                }))
                for name, item in class_values.items():
                    item = item.__func__ if isinstance(item, (classmethod, staticmethod)) else item
                    item = item.fget if isinstance(item, property) else item
                    if type(item) is FunctionType:
                        _require(name not in methods or item.__code__ == methods[name]
                                 and item.__globals__ is namespace, "canonical packing method source differs")
                        self._save_callable(item)
            else:
                _require(False, "installed inference definition missing")
        values = dict(namespace)
        literals = {
            key: _literal_state(value) for key, value in values.items()
            if key != "__builtins__" and type(value) in (dict, list, set, tuple)
        }
        self._namespaces.append((module, values, literals))

    def _save_callable(self, value: FunctionType) -> None:
        self._callables.append((value, value.__code__, value.__defaults__, value.__kwdefaults__, value.__closure__,
                                _literal_state(value.__defaults__), _literal_state(value.__kwdefaults__)))

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
            all(sys.modules.get(name) is module for name, module in self._owned.items())
            and all(sys.modules.get(module.__name__) is module for module in self._shared),
            "owned connected registry changed",
        )
        for name, (fn, code) in self._apis.items():
            _require(
                getattr(cast(ModuleType, self._module), name) is fn
                and fn.__code__ is code
                and fn.__globals__ is vars(cast(ModuleType, self._module)),
                "connected public callable changed",
            )
        for fn, code, defaults, kwdefaults, closure, default_state, kwdefault_state in self._callables:
            _require(
                fn.__code__ is code and fn.__defaults__ is defaults
                and fn.__kwdefaults__ is kwdefaults and fn.__closure__ is closure
                and _literal_state(fn.__defaults__) == default_state
                and _literal_state(fn.__kwdefaults__) == kwdefault_state,
                "installed inference callable state changed",
            )
        for module, values, literals in self._namespaces:
            namespace = vars(module)
            _require(
                namespace.keys() == values.keys()
                and all(namespace[key] is value for key, value in values.items())
                and all(_literal_state(namespace[key]) == state for key, state in literals.items()),
                "installed inference globals changed",
            )
        for module in self._shared:
            _require(
                module.__file__ == str(self._guards[2][0]) and module.__spec__.origin == module.__file__
                and module.__spec__.name == module.__name__, "canonical shared packing origin changed",
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
                from sfora.packed_int8 import PackedInt8Embeddings

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
                self._namespaces.clear()
                self._callables.clear()
                self._shared = ()
            if errors:
                for failure in errors[1:]:
                    errors[0].add_note("connected cleanup also failed: " + repr(failure))
                raise errors[0]

    def __enter__(self) -> ConnectedCompactIndex:
        _require(not self._closed, "connected compact index is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
