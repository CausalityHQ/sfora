#!/usr/bin/env python3
"""Source-only wrapper checks; no native/model/quality/latency qualification."""

import ast
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ACTUAL_SOURCE = (
    Path(__file__).resolve().parents[1] / "scripts/train_siglip2_connected_mlp.py"
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
    modules = {}
    for number in range(2):
        name = "_connected_serving_" + str(time.time_ns()) + "_" + str(number)
        modules[str(number)] = load_authenticated(name, directory / "quadratic_readout.py")
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
    if events.fail == "infer":
        raise ValueError("inference failure")
    wire = events.wire if events.wire is not None else b"\x81" * (130 * len(images))
    events.returned_wire = wire
    return {"wire": wire, "raw": object(), "unit": object(), "codes": object(),
            "inverse_norms": object()}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def _processor_cache(processor, guards):
    events.calls.append("release")
    if events.swap is not None:
        events.swap()
    if events.fail == "release":
        raise ValueError("release failure")
    if events.fail == "cache_authority":
        return object()
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
    def find_spec(self, fullname, path=None, target=None):
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
            if events.fail == "close":
                raise ValueError("close failure")

    stubs = {"_connected_test_events": events}
    for name in (
        "sfora",
        "PIL",
        "PIL.Image",
        "sfora.cutile_int8",
        "sfora.joint_relational_compaction",
    ):
        stubs[name] = ModuleType(name)
    stubs["PIL.Image"].Image = Image
    stubs["sfora.cutile_int8"].CutilePackedInt8Gallery = Gallery
    stubs["sfora.joint_relational_compaction"].PackedInt8Embeddings = Packed
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
                    SOURCE.encode() if name == "train_siglip2_connected_mlp.py" else b"# helper\n"
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
            for path in (trainer,):
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

            # The actual trainer runs only through finite helper registration, before
            # endpoint/Torch construction. This proves exact pre-return ownership capture.
            actual_source = (ROOT / "scripts/train_siglip2_connected_mlp.py").read_bytes()
            constants = {
                node.targets[0].id: ast.literal_eval(node.value)
                for node in ast.parse(actual_source).body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"SCOPE_SHA256", "CONTROL_SHA256"}
            }
            saved = {name: (bundle / name).read_bytes() for name in CODE_NAMES}
            installed = root / "installed"
            installed.mkdir()
            constructor = installed / "constructor.py"
            constructor.write_bytes(b"# synthetic installed source\n")

            def foreign_failure():
                foreign_marker = result  # noqa: F841 - asserted through the preserved foreign traceback.
                raise LookupError("foreign failure diagnostic")

            events.foreign_failure = foreign_failure
            failure_source = (
                b"import sys, _connected_test_events as events\n"
                b"events.helpers.append(sys.modules[__name__])\n"
                b"try:\n    events.foreign_failure()\n"
                b"except LookupError as error:\n"
                b"    raise ValueError('actual authenticated helper registration failure')"
                b" from error\n"
            )
            for name in CODE_NAMES:
                (bundle / name).write_bytes(
                    actual_source if name == trainer.name else failure_source
                )
            actual_manifest = manifest | {
                "code": {name: sha((bundle / name).read_bytes()) for name in CODE_NAMES},
                "scope": {
                    "arm": "control",
                    "manifest_sha256": constants["SCOPE_SHA256"],
                    "arm_sha256": constants["CONTROL_SHA256"],
                },
                "environment": {
                    "packages": {
                        name: {"root": str(installed)}
                        for name in (
                            "torch",
                            "numpy",
                            "PIL",
                            "transformers",
                            "safetensors",
                            "torchvision",
                        )
                    },
                    "files": {str(constructor): sha(constructor.read_bytes())},
                    "native_files": {},
                    "vision_constructor": str(constructor),
                },
            }
            (bundle / "bundle.json").write_text(json.dumps(actual_manifest))
            before = len(events.helpers)
            try:
                load(expected_bundle_sha256=sha((bundle / "bundle.json").read_bytes()))
            except ValueError as error:
                assert str(error) == "actual authenticated helper registration failure"
                names = []
                trace = error.__traceback__
                while trace is not None:
                    names.append(trace.tb_frame.f_code.co_name)
                    trace = trace.tb_next
                assert "load_inference" in names and "load_authenticated" in names
                assert isinstance(error.__cause__, LookupError)
                assert str(error.__cause__) == "foreign failure diagnostic"
                trace = error.__cause__.__traceback__
                while trace.tb_frame.f_code.co_name != "foreign_failure":
                    trace = trace.tb_next
                assert trace.tb_frame.f_locals["foreign_marker"] is result
            else:
                raise AssertionError("actual helper registration failure accepted")
            assert len(events.helpers) == before + 1
            no_owned_registry()
            for name, raw in saved.items():
                (bundle / name).write_bytes(raw)
            (bundle / "bundle.json").write_bytes(original_manifest)

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
