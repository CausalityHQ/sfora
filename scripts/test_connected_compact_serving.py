#!/usr/bin/env python3
"""Source-only wrapper checks; no native/model/quality/latency qualification."""

import __future__
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import tempfile
import threading
from pathlib import Path
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
from unittest.mock import patch

ACTUAL_SOURCE = (
    Path(__file__).resolve().parents[1] / "src/sfora/connected_inference.py"
).read_text()
RELEASE_SOURCE = ast.get_source_segment(
    ACTUAL_SOURCE,
    next(
        node
        for node in ast.parse(ACTUAL_SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name == "release_inference"
    ),
)

ROOT = Path(__file__).resolve().parents[1]
CODE_NAMES = {
    "train_siglip2_connected_mlp.py",
    "test_siglip2_connected_mlp.py",
    "qualify_siglip2_substrate_cpu.py",
    "extract_siglip2_vision_source.py",
    "train_siglip2_cached_readout.py",
    "train_siglip2_substrate_adaptation.py",
    "prototype_residual_readout.py",
    "quadratic_readout.py",
    "joint_relational_compaction.py",
}
SOURCE = (
    r"""import importlib.util
from pathlib import Path
import sys
import time
import gc
import weakref
from functools import lru_cache
import _connected_test_events as events
FIXTURE_FLAGS = {"empty": False, "numeric": 1, "nested": {1: [1]}, "frozen": frozenset({1})}

def _bind_runtime(historical_code, runtime_guards):
    assert historical_code == events.historical_code
    assert tuple(Path(path).name for path, digest in runtime_guards) == (
        "connected_inference.py", "_connected_inference_authority.py", "packed_int8.py")

def admit_bundle(directory, digest):
    assert directory == events.bundle
    assert digest == events.digest or digest == events.admission_digest

def load_authenticated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if name in sys.modules:
        raise ValueError("helper collision")
    sys.modules[name] = module
    events.helpers.append(module)
    if events.fail == "partial_helper":
        raise ValueError("partial helper")
    return module

def load_inference(directory, digest, device):
    assert device == "cuda" and directory == events.bundle and digest == events.digest
    events.calls.append("load")
    if events.fail == "load_cancel":
        raise events.cancellation
    modules = {}
    for number in range(2):
        name = "_connected_serving_" + str(time.time_ns()) + "_" + str(number)
        modules[str(number)] = load_authenticated(name, Path(__file__).parent / "helper.py")
    if events.fail == "before_endpoint":
        raise ValueError("before endpoint")
    class Atom:
        pass
    class Resource:
        def __init__(self):
            self.parameter, self.buffer = Atom(), Atom()
        def parameters(self):
            return (self.parameter,)
        def buffers(self):
            return (self.buffer,)
    cache = lru_cache()(lambda value: value)
    endpoint = {"modules": modules, "alive": True, "guards": {}, "processor_cache": cache}
    for key in ("model", "processor_object", "head_object", "A", "C", "mu_train"):
        endpoint[key] = Resource()
    endpoint["processor_object"].cache = cache
    cache(endpoint["model"])
    events.cache = cache
    events.refs = [weakref.ref(endpoint[key])
                   for key in ("model", "processor_object", "head_object", "A", "C", "mu_train")]
    events.refs += [weakref.ref(value) for key in ("model", "head_object")
                   for value in (*endpoint[key].parameters(), *endpoint[key].buffers())]
    if events.fail == "after_endpoint":
        raise ValueError("after endpoint")
    return endpoint

def inference_outputs(endpoint, images):
    assert endpoint["alive"] and 1 <= len(images) <= 32
    assert type(images) is list
    events.calls.append("infer")
    if events.started is not None:
        events.started.set()
        assert events.resume.wait(5), "test did not release search"
    if events.fail == "infer_cancel":
        raise events.cancellation
    if events.fail == "infer":
        raise ValueError("inference failure")
    wire = events.wire if events.wire is not None else b"\x81" * (130 * len(images))
    events.returned_wire = wire
    return {"wire": wire, "raw": object(), "unit": object(), "codes": object(),
            "inverse_norms": object()}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def _processor_cache(processor, guards, *, empty=False):
    events.calls.append("release")
    if events.swap is not None:
        events.swap()
    if events.fail == "release":
        raise ValueError("release failure")
    if events.fail == "cache_authority":
        return object()
    require(not empty or processor.cache.cache_info().currsize == 0, "cache must initially be empty")
    return processor.cache
"""
    + "\n"
    + RELEASE_SOURCE
    + "\n"
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fails(call, label):
    try:
        call()
    except (ValueError, RuntimeError, OSError):
        return
    raise AssertionError("accepted " + label)


class NoNativeImports(importlib.abc.MetaPathFinder):
    hook = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" and self.hook is not None:
            self.hook()
        if fullname.split(".")[0] in {"torch", "numpy", "PIL", "sfora"}:
            raise AssertionError("real native/package import: " + fullname)


def teardown_regressions(bridge, load, events, Image, result, opened, no_owned_registry):
    # Calling the mutable saved fn after rejection must fail this check.
    index = load()
    release = index._module.release_inference
    globals_before = dict(release.__globals__)

    def altered(endpoint):
        raise AssertionError("rejected release code executed")

    release.__code__ = altered.__code__
    fails(lambda index=index: index.search_images([Image()]), "release code mutation")
    assert events.cache.cache_info().currsize == 0, "genuine cache cleanup skipped"
    assert all(ref() is None for ref in events.refs), "genuine lifetime cleanup skipped"
    assert all(release.__globals__.get(key) is value for key, value in globals_before.items())
    assert opened[-1].closes == 1
    index.close()
    no_owned_registry()

    # Namespace/default mutation must reject while saved genuine teardown still runs.
    for mutation in ('defaults', 'kwdefaults', 'numeric_bool_literal', 'bool_numeric_default', 'numeric_bool_key', 'list_tuple_literal', 'frozenset_numeric_bool', 'helper_code', 'helper_global', 'literal_global', 'extra_global'):
        index = load()
        module = index._module
        if mutation == 'defaults':
            module.inference_outputs.__defaults__ = (None,)
        elif mutation == 'kwdefaults':
            module._processor_cache.__kwdefaults__['empty'] = True
        elif mutation == 'numeric_bool_literal':
            module.FIXTURE_FLAGS['numeric'] = True
        elif mutation == 'bool_numeric_default':
            module._processor_cache.__kwdefaults__['empty'] = 0
        elif mutation == 'numeric_bool_key':
            nested = module.FIXTURE_FLAGS['nested']
            value = nested.pop(1)
            nested[True] = value
        elif mutation == 'list_tuple_literal':
            module.FIXTURE_FLAGS['nested'][1] = (1,)
        elif mutation == 'frozenset_numeric_bool':
            module.FIXTURE_FLAGS['frozen'] = frozenset({True})
        elif mutation == 'helper_code':
            module._processor_cache.__code__ = (lambda processor, guards: object()).__code__
        elif mutation == 'helper_global':
            module._processor_cache = lambda processor, guards: object()
        elif mutation == 'literal_global':
            module.FIXTURE_FLAGS['empty'] = True
        else:
            module.added = True
        if mutation in ("numeric_bool_literal", "bool_numeric_default", "numeric_bool_key", "list_tuple_literal", "frozenset_numeric_bool"):
            fails(index._check_current, mutation)
        fails(lambda index=index: index.search_images([Image()]), mutation)
        assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
        assert opened[-1].closes == 1
        index.close()
        no_owned_registry()

    registry_regressions(bridge, load, events, Image, result, opened, no_owned_registry)


def registry_regressions(bridge, load, events, Image, result, opened, no_owned_registry):
    # Empty/filter-only helper checks must fail for plain-close tampering too.
    for during, ordinal in ((True, 0), (True, 1), (False, 0), (False, 1)):
        index = load()
        module = tuple(index._endpoint["modules"].values())[ordinal]
        foreign = ModuleType(module.__name__)

        def swap(module=module, foreign=foreign):
            with getattr(bridge, "_REGISTRY_LOCK", threading.RLock()):
                sys.modules[module.__name__] = foreign

        if during:
            events.swap = swap
        else:
            swap()
        fails(index.close, "plain close registry tampering")
        assert sys.modules.get(module.__name__) is foreign, (
            "foreign helper removed during cache work"
        )
        assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
        assert opened[-1].closes == 1
        index.close()
        del sys.modules[module.__name__]
        events.swap = None
        no_owned_registry()

    # Supported writers and close share ownership synchronization.
    index = load()
    module = next(iter(index._endpoint["modules"].values()))
    foreign = ModuleType(module.__name__)
    entered, finished, errors = threading.Event(), threading.Event(), []

    def close_with_writer():
        entered.set()
        try:
            index.close()
        except ValueError as error:
            errors.append(error)
        finally:
            finished.set()

    with bridge._REGISTRY_LOCK:
        closer = threading.Thread(target=close_with_writer)
        closer.start()
        assert entered.wait(5) and not finished.wait(0.05)
        assert opened[-1].closes == 0
        sys.modules[module.__name__] = foreign
    closer.join(5)
    assert not closer.is_alive() and len(errors) == 1
    assert sys.modules.get(module.__name__) is foreign and opened[-1].closes == 1
    del sys.modules[module.__name__]
    no_owned_registry()

    # All original non-registry gates remain live, including failure reporting.
    for failure in ("cache_authority", "lifetime", "parameter_lifetime"):
        index = load()
        survivor = (
            index._endpoint["model"]
            if failure == "lifetime"
            else index._endpoint["head_object"].parameter
            if failure == "parameter_lifetime"
            else None
        )
        events.fail = failure
        fails(index.close, "original release " + failure)
        assert opened[-1].closes == 1
        if survivor is not None:
            assert events.cache.cache_info().currsize == 0
        index.close()
        survivor = None
        events.cache.cache_clear()
        events.fail = None
        no_owned_registry()

    index = load()
    assert index.search_images((Image(),)) is result
    index.close()
    assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
    no_owned_registry()


def compile_regressions(bridge, args, bundle, manifest, Packed, no_owned_registry):
    # The actual package serializer must use the package's own compiler flags.
    actual = (ROOT / 'src/sfora/connected_inference.py').read_bytes()
    spec = importlib.util.spec_from_file_location('_package_compile_subject', ROOT / 'src/sfora/connected_inference.py')
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    try:
        spec.loader.exec_module(runtime)
        expected = next(code for code in compile(actual, runtime.__file__, 'exec', dont_inherit=True).co_consts
                        if isinstance(code, CodeType) and code.co_name == 'fingerprint')
        assert runtime.fingerprint.__code__ == expected
        assert runtime.fingerprint.__defaults__ == (None, None)
        assert runtime.fingerprint.__globals__ is vars(runtime)
        inherited = compile(actual, runtime.__file__, 'exec', flags=__future__.annotations.compiler_flag)
        changed = next(code for code in inherited.co_consts if isinstance(code, CodeType) and code.co_name == 'fingerprint')
        assert changed != expected, 'serializer compiler-flag negative did not differ'
    finally:
        del sys.modules[spec.name]


def main():
    assert not any(n in sys.modules for n in ("torch", "numpy", "PIL", "sfora"))
    spec = importlib.util.spec_from_file_location(
        "connected_bridge_test_subject", ROOT / "src/sfora/connected_compact_serving.py"
    )
    assert spec is not None and spec.loader is not None, "bridge implementation missing"
    assert Path(spec.origin).is_file(), "bridge implementation missing"
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    events = ModuleType("_connected_test_events")
    events.calls, events.helpers = [], []
    events.fail = events.wire = events.started = events.resume = events.swap = None
    events.cancellation = events.cleanup_error = None
    events.admission_digest = None
    parsed, opened = [], []
    result = (object(), object())

    class Image:
        pass

    class Packed:
        @classmethod
        def from_bytes(cls, wire, *, count, dimensions):
            assert dimensions == 128 and type(count) is int
            if type(wire) is not bytes or len(wire) != count * 130:
                raise ValueError("wire authority")
            value = SimpleNamespace(wire=wire, count=count, dimensions=dimensions)
            parsed.append(value)
            return value

    class Gallery:
        @classmethod
        def open_packed(cls, path, embeddings):
            assert path == events.library and embeddings.count == 10
            events.calls.append("open")
            if events.fail == "open":
                raise ValueError("open failure")
            value = cls()
            value.closes = 0
            opened.append(value)
            return value

        def search_packed(self, embeddings, *, k):
            assert k == 10 and embeddings.wire is events.returned_wire
            events.calls.append("search")
            if events.fail == "search":
                raise ValueError("search failure")
            return result

        def close(self):
            self.closes += 1
            events.calls.append("close")
            if events.cleanup_error is not None:
                raise events.cleanup_error
            if events.fail == "close":
                raise ValueError("close failure")

    stubs = {"_connected_test_events": events}
    for name in (
        "sfora",
        "PIL",
        "PIL.Image",
        "sfora.cutile_int8",
        "sfora.packed_int8",
    ):
        stubs[name] = ModuleType(name)
    stubs["PIL.Image"].Image = Image
    stubs["sfora.cutile_int8"].CutilePackedInt8Gallery = Gallery
    stubs["sfora.packed_int8"].PackedInt8Embeddings = Packed
    guard = NoNativeImports()
    sys.meta_path.insert(0, guard)
    try:
        with (
            patch.dict(sys.modules, stubs),
            tempfile.TemporaryDirectory(
                prefix="connected-bridge-", dir="/home/rb/agents/handoffs"
            ) as scratch,
        ):
            root = Path(scratch)
            bundle = root / "bundle"
            bundle.mkdir()
            for name in CODE_NAMES:
                (bundle / name).write_bytes(
                    b"raise AssertionError(\"historical evidence executed\")\n"
                )
            for name in ("vision.pt", "endpoint.pt", "processor.json"):
                (bundle / name).write_bytes(b"synthetic owned bytes")
            manifest = {
                "schema": "siglip2-connected-mlp-bundle-v1",
                "code": {name: sha((bundle / name).read_bytes()) for name in CODE_NAMES},
                "files": {
                    name: sha((bundle / name).read_bytes())
                    for name in ("vision.pt", "endpoint.pt", "processor.json")
                },
                "endpoint_state_sha256": "a" * 64,
                "environment": {},
                "encoder_identity": {},
                "base_vision_sha256": "b" * 64,
                "vision_sha256": "c" * 64,
                "scope": {},
            }
            (bundle / "bundle.json").write_text(json.dumps(manifest))
            gallery, library = root / "gallery.bin", root / "library.so"
            gallery.write_bytes(b"G" * 1300)
            library.write_bytes(b"synthetic native bytes")
            installed = root / "installed"
            installed.mkdir()
            runtime_path = installed / "connected_inference.py"
            runtime_path.write_text(SOURCE)
            (installed / "helper.py").write_bytes(b"# installed ownership fixture\n")
            packed_path = installed / "packed_int8.py"
            packed_path.write_bytes(b"# shared packing ownership fixture\n")
            authority_path = installed / "_connected_inference_authority.py"
            def install_authority(code=None):
                historical = tuple(sorted((manifest["code"] if code is None else code).items()))
                record = {"SCHEMA": "sfora-connected-inference-extraction-v1", "HISTORICAL_CODE": historical,
                          "SOURCE_SYMBOLS": (), "PACKED_SOURCE_SYMBOLS": (), "SUBSTITUTIONS": (),
                          "RUNTIME_SHA256": sha(runtime_path.read_bytes()), "PACKED_SHA256": sha(packed_path.read_bytes())}
                authority_path.write_text('\n'.join(key + ' = ' + repr(value) for key, value in record.items()) + '\n')
                events.historical_code = historical
                return authority_path, sha(authority_path.read_bytes())
            authority = install_authority()
            bridge._installed_authority = lambda: authority
            bridge.__file__ = str(installed / 'connected_compact_serving.py')
            packed_module = stubs['sfora.packed_int8']
            packed_module.__file__ = str(packed_path)
            packed_module.__spec__ = importlib.util.spec_from_file_location(packed_module.__name__, packed_path)
            events.bundle, events.digest, events.library = (
                bundle,
                sha((bundle / "bundle.json").read_bytes()),
                library,
            )
            args = dict(
                bundle_dir=bundle,
                expected_bundle_sha256=events.digest,
                gallery_path=gallery,
                expected_gallery_sha256=sha(gallery.read_bytes()),
                gallery_count=10,
                native_library_path=library,
                expected_native_library_sha256=sha(library.read_bytes()),
            )
            trainer = bundle / "train_siglip2_connected_mlp.py"
            registry_before = dict(sys.modules)

            def load(**changes):
                return bridge.ConnectedCompactIndex.from_bundle(**(args | changes))

            def no_owned_registry():
                assert all(sys.modules.get(m.__name__) is not m for m in events.helpers)
                assert not any(
                    n.startswith("_sfora_connected_compact_") and n not in registry_before
                    for n in sys.modules
                )

            if "--teardown-only" in sys.argv or "--registry-only" in sys.argv:
                check = (
                    registry_regressions if "--registry-only" in sys.argv else teardown_regressions
                )
                check(bridge, load, events, Image, result, opened, no_owned_registry)
                return

            compile_regressions(bridge, args, bundle, manifest, Packed, no_owned_registry)
            if "--compile-only" in sys.argv:
                return

            # Missing/wrong admission must prevent any copied source execution.
            for field in (
                "expected_bundle_sha256",
                "expected_gallery_sha256",
                "expected_native_library_sha256",
            ):
                for bad in ("0" * 64, "G" * 64, None):
                    before = len(events.calls)
                    fails(lambda field=field, bad=bad: load(**{field: bad}), field)
                    assert len(events.calls) == before
            for bad in (True, 9, 10.0):
                fails(lambda bad=bad: load(gallery_count=bad), "gallery count")
            for field, path in (
                ("bundle_dir", bundle),
                ("gallery_path", gallery),
                ("native_library_path", library),
            ):
                alias = root / (field + "-alias")
                alias.symlink_to(path)
                fails(lambda field=field, alias=alias: load(**{field: alias}), "symlink " + field)
                fails(
                    lambda field=field, path=path: load(**{field: Path(path.name)}),
                    "relative " + field,
                )
                fails(
                    lambda field=field, path=path: load(**{field: str(path)}), "non-Path " + field
                )
            original_manifest = (bundle / "bundle.json").read_bytes()
            for changes in ({"schema": "unknown"}, {"code": {}}, {"files": {}}, {"extra": True}):
                raw = json.dumps(manifest | changes).encode()
                (bundle / "bundle.json").write_bytes(raw)
                before = len(events.calls)
                fails(lambda raw=raw: load(expected_bundle_sha256=sha(raw)), "unsupported manifest")
                assert len(events.calls) == before
            (bundle / "bundle.json").write_bytes(original_manifest)
            raw = original_manifest[:-1] + b', "schema":"siglip2-connected-mlp-bundle-v1"}'
            (bundle / "bundle.json").write_bytes(raw)
            fails(lambda raw=raw: load(expected_bundle_sha256=sha(raw)), "duplicate manifest key")
            (bundle / "bundle.json").write_bytes(original_manifest)
            original_source = trainer.read_bytes()
            trainer.write_bytes(
                original_source + b"\nraise AssertionError('executed tampered source')\n"
            )
            fails(load, "copied source hash")
            trainer.write_bytes(original_source)
            changed_source = original_source + b'# unknown historical closure\n'
            trainer.write_bytes(changed_source)
            changed_manifest = manifest | {'code': manifest['code'] | {trainer.name: sha(changed_source)}}
            raw = json.dumps(changed_manifest).encode()
            (bundle / 'bundle.json').write_bytes(raw)
            before = len(events.calls)
            fails(lambda: load(expected_bundle_sha256=sha(raw)), 'unknown authenticated historical closure')
            assert len(events.calls) == before
            trainer.write_bytes(original_source)
            (bundle / 'bundle.json').write_bytes(original_manifest)
            linked = root / "hardlink"
            linked.hardlink_to(trainer)
            fails(load, "multiply linked owned source")
            linked.unlink()
            gallery_wire = gallery.read_bytes()
            gallery.write_bytes(b"malformed gallery")
            before = len(events.calls)
            fails(
                lambda: load(expected_gallery_sha256=sha(gallery.read_bytes())),
                "gallery wire framing",
            )
            assert len(events.calls) == before
            gallery.write_bytes(gallery_wire)

            teardown_regressions(bridge, load, events, Image, result, opened, no_owned_registry)

            # Exact wire parsing and native result identity; instances never collide.
            first, second = load(), load()
            assert first._module.__name__ != second._module.__name__
            assert first.search_images([Image()]) is result
            assert parsed[-1].count == 1 and parsed[-1].wire is events.returned_wire
            assert second.search_images(tuple(Image() for _ in range(32))) is result
            assert parsed[-1].count == 32 and parsed[-1].wire is events.returned_wire
            first.close()
            assert second.search_images([Image()]) is result
            first.close()
            second.close()
            assert all(value.closes == 1 for value in opened)
            fails(lambda: first.search_images([Image()]), "search after close")
            no_owned_registry()
            foreign = ModuleType("_sfora_connected_compact_" + "d" * 32)
            with (
                patch.dict(sys.modules, {foreign.__name__: foreign}),
                patch.object(bridge.uuid, "uuid4", return_value=SimpleNamespace(hex="d" * 32)),
            ):
                fails(load, "loader registry collision")
                assert sys.modules[foreign.__name__] is foreign

            # External disk inputs are authenticated at load; only source is hot-path guarded.
            for path in (bundle / "bundle.json", gallery, library):
                original = path.read_bytes()
                path.write_bytes(original + b"x")
                fails(load, "startup bytes " + path.name)
                path.write_bytes(original)
            for path in (trainer, runtime_path, authority_path, packed_path):
                for kind in ("bytes", "symlink"):
                    index = load()
                    original = path.read_bytes()
                    target = root / "tamper-target"
                    if kind == "bytes":
                        path.write_bytes(original + b"x")
                    else:
                        target.write_bytes(original)
                        path.unlink()
                        path.symlink_to(target)
                    fails(
                        lambda index=index: index.search_images([Image()]),
                        "live " + kind + " " + path.name,
                    )
                    assert opened[-1].closes == 1 and events.calls[-1] == "release"
                    path.unlink()
                    path.write_bytes(original)
                    if target.exists():
                        target.unlink()
                    no_owned_registry()

            # A foreign registry replacement survives cleanup, including helper replacement.
            for helper in (False, True):
                index = load()
                owned = next(iter(index._endpoint["modules"].values())) if helper else index._module
                replacement = ModuleType(owned.__name__)
                sys.modules[owned.__name__] = replacement
                fails(
                    lambda index=index: index.search_images([Image()]),
                    "foreign registry replacement",
                )
                assert sys.modules[owned.__name__] is replacement
                del sys.modules[owned.__name__]
                no_owned_registry()

            index = load()
            shared = sys.modules['sfora.packed_int8']
            foreign_packing = ModuleType('sfora.packed_int8')
            sys.modules['sfora.packed_int8'] = foreign_packing
            try:
                fails(lambda: index.search_images([Image()]), 'shared canonical packing registry replacement')
                assert sys.modules['sfora.packed_int8'] is foreign_packing
            finally:
                sys.modules['sfora.packed_int8'] = shared
            assert sys.modules['sfora.packed_int8'] is shared
            no_owned_registry()

            # Unchanged public callable objects/code are part of source admission.
            index = load()
            index._module.inference_outputs = lambda *a: {"wire": b""}
            fails(lambda index=index: index.search_images([Image()]), "public callable replacement")
            no_owned_registry()
            index = load()
            index._module.__file__ = str(root / "foreign-loader.py")
            fails(
                lambda index=index: index.search_images([Image()]),
                "copied loader origin replacement",
            )
            no_owned_registry()

            for failure in ("partial_helper", "before_endpoint", "after_endpoint", "open"):
                events.fail = failure
                before = events.calls.count("release")
                fails(load, failure)
                assert events.calls.count("release") == before + int(
                    failure in ("after_endpoint", "open")
                )
                no_owned_registry()
            events.fail = None

            # Run the actual installed loader through admission to its first denied
            # native import; genuine package frames must release all owned references.
            saved_runtime = runtime_path.read_bytes()
            runtime_path.write_bytes((ROOT / 'src/sfora/connected_inference.py').read_bytes())
            constructor = installed / 'constructor.py'
            constructor.write_bytes(b'# authenticated constructor evidence\n')
            actual_source = runtime_path.read_text()
            constants = {node.targets[0].id: ast.literal_eval(node.value)
                         for node in ast.parse(actual_source).body if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id in {'SCOPE_SHA256', 'CONTROL_SHA256'}}
            actual_manifest = manifest | {
                'scope': {'arm': 'control', 'manifest_sha256': constants['SCOPE_SHA256'],
                          'arm_sha256': constants['CONTROL_SHA256']},
                'environment': {'packages': {name: {'root': str(installed)} for name in (
                    'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision')},
                    'files': {str(constructor): sha(constructor.read_bytes())}, 'native_files': {},
                    'vision_constructor': str(constructor)}}
            (bundle / 'bundle.json').write_text(json.dumps(actual_manifest))
            authority = install_authority()
            retained = []
            class GenuineProbe(bridge.ConnectedCompactIndex):
                def _snapshot(self, module, source):
                    if module is self._module:
                        import weakref
                        retained.append(weakref.ref(module))
                    return super()._snapshot(module, source)
            try:
                GenuineProbe.from_bundle(**(args | {'expected_bundle_sha256': sha((bundle / 'bundle.json').read_bytes())}))
            except AssertionError as error:
                assert str(error) == 'real native/package import: torch'
                trace = error.__traceback__
                owned_frames = []
                while trace is not None:
                    if trace.tb_frame.f_code.co_name == 'load_inference':
                        owned_frames.append(trace.tb_frame)
                    trace = trace.tb_next
                assert owned_frames and all(not frame.f_locals for frame in owned_frames)
                del owned_frames, trace
            else:
                raise AssertionError('actual installed pre-return failure accepted')
            def foreign_failure():
                foreign_marker = result
                raise LookupError('foreign failure diagnostic')
            def denied_native():
                try:
                    foreign_failure()
                except LookupError as cause:
                    raise ValueError('actual package pre-return rejection') from cause
            guard.hook = denied_native
            try:
                GenuineProbe.from_bundle(**(args | {'expected_bundle_sha256': sha((bundle / 'bundle.json').read_bytes())}))
            except ValueError as error:
                assert str(error) == 'actual package pre-return rejection'
                assert isinstance(error.__cause__, LookupError)
                trace = error.__cause__.__traceback__
                while trace.tb_frame.f_code.co_name != 'foreign_failure':
                    trace = trace.tb_next
                assert trace.tb_frame.f_locals['foreign_marker'] is result
                trace = error.__traceback__
                owned_frames = []
                while trace is not None:
                    if trace.tb_frame.f_code.co_name == 'load_inference':
                        owned_frames.append(trace.tb_frame)
                    trace = trace.tb_next
                assert owned_frames and all(not frame.f_locals for frame in owned_frames)
                del owned_frames, trace
            else:
                raise AssertionError('genuine error chain lost')
            finally:
                guard.hook = None
            import gc
            gc.collect()
            assert retained and all(ref() is None for ref in retained), 'package runtime retained after failure'
            no_owned_registry()
            runtime_path.write_bytes(saved_runtime)
            authority = install_authority()
            (bundle / 'bundle.json').write_bytes(original_manifest)

            for failure in ("infer", "search", "close", "release"):
                index = load()
                events.fail = failure
                fails(
                    index.close
                    if failure in ("close", "release")
                    else lambda index=index: index.search_images([Image()]),
                    failure,
                )
                assert opened[-1].closes == 1 and events.calls[-1] == "release"
                index.close()
                no_owned_registry()
                events.fail = None
            for phase in ('load_cancel', 'infer_cancel'):
                index = None if phase == 'load_cancel' else load()
                events.fail = phase
                events.cancellation = KeyboardInterrupt('original cancellation')
                events.cleanup_error = ValueError('gallery cleanup failure') if index is not None else None
                original = events.cancellation
                try:
                    load() if index is None else index.search_images([Image()])
                except KeyboardInterrupt as error:
                    assert error is original, 'cleanup replaced original cancellation'
                    if index is not None:
                        assert any('gallery cleanup failure' in note for note in error.__notes__)
                        assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
                        assert opened[-1].closes == 1
                else:
                    raise AssertionError('cancellation swallowed')
                if index is not None:
                    index.close()
                no_owned_registry()
                events.fail = events.cancellation = events.cleanup_error = None

            for wire in (b"", b"X" * 129, b"X" * 131, bytearray(b"X" * 130)):
                index = load()
                events.wire = wire
                fails(lambda index=index: index.search_images([Image()]), "malformed output wire")
                assert opened[-1].closes == 1 and events.calls[-1] == "release"
                no_owned_registry()
            events.wire = None
            for images in ([], [Image()] * 33, [object()], "image"):
                index = load()
                fails(lambda index=index, images=images: index.search_images(images), "image input")
                assert opened[-1].closes == 0 and index.search_images([Image()]) is result
                index.close()
            with load() as index:
                assert index.search_images([Image()]) is result
            assert opened[-1].closes == 1

            # Close cannot destroy an active request; the lifecycle lock is reentrant.
            index = load()
            events.started, events.resume = threading.Event(), threading.Event()
            request = [Image()]
            errors, answers = [], []
            close_started, closed = threading.Event(), threading.Event()

            def search():
                try:
                    answers.append(index.search_images(request))
                except BaseException as error:
                    errors.append(error)

            def close():
                close_started.set()
                try:
                    index.close()
                    closed.set()
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=search)
            thread.start()
            assert events.started.wait(5)
            request.extend(Image() for _ in range(32))
            closer = threading.Thread(target=close)
            closer.start()
            assert close_started.wait(5)
            assert not closed.wait(0.05) and opened[-1].closes == 0
            events.resume.set()
            thread.join(5)
            closer.join(5)
            assert not thread.is_alive() and not closer.is_alive()
            assert not errors and answers == [result] and closed.is_set() and opened[-1].closes == 1
            assert parsed[-1].count == 1, "caller mutation changed accepted image snapshot"
            events.started = events.resume = None
            index = load()
            with index._lock:
                assert index.search_images([Image()]) is result
                index.close()
            no_owned_registry()
    finally:
        sys.meta_path.remove(guard)
    assert not any(n in sys.modules for n in ("torch", "numpy", "PIL", "sfora"))
    print(
        "PASS: connected bridge source-only admission, exact wire, failure cleanup, "
        "registry ownership, lifecycle lock"
    )


if __name__ == "__main__":
    main()
