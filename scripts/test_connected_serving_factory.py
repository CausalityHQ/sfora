#!/usr/bin/env python3
"""Genuine artifact factory reachability; no native or performance qualification."""

import ast
import hashlib
import importlib.abc
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "src/sfora"
EVIDENCE = (
    ROOT
    / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
    / "connected-installed-reader-preparation-v1"
)
HELPERS = (
    "connected_gallery_provenance",
    "connected_serving_artifact",
    "connected_serving_admission",
    "connected_artifact_identity",
    "connected_installed_environment",
)


def package(name):
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = []
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    module.__spec__ = spec
    assert name not in sys.modules
    sys.modules[name] = module
    return module


def load(name, filename, raw=None):
    path = SOURCES / filename
    raw = path.read_bytes() if raw is None else raw
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert name not in sys.modules
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    return module, raw, hashlib.sha256(raw).hexdigest()


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch",
            "numpy",
            "PIL",
            "transformers",
            "safetensors",
            "torchvision",
        } or fullname in {"sfora.packed_int8", "sfora.cutile_int8"}:
            raise AssertionError("native/packing import before qualified boundary: " + fullname)
        return None


def staged_lifecycle(bridge, namespace, folder, args, selected=None):
    """Closed source substitutions before snapshotting; never native parity."""
    stage = folder / "installed-standins"
    stage.mkdir()
    for name in HELPERS:
        (stage / (name + ".py")).write_bytes((SOURCES / (name + ".py")).read_bytes())
    runtime = (SOURCES / "connected_inference.py").read_text()
    tree = ast.parse(runtime)
    payload_source = """def _serving_load_payload(prepared):
    import _serving_factory_fake_events as events
    class Resource:
        def parameters(self): return ()
        def buffers(self): return ()
    class Processor(Resource): pass
    processor = Processor()
    cache = lru_cache(maxsize=10)(lambda value:value)
    processor.cache = cache
    cache(processor)
    endpoint = {'model':Resource(),'processor_object':processor,'processor_cache':cache,
        'head_object':Resource(),'A':Resource(),'C':Resource(),'mu_train':Resource(),
        'modules':{'runtime':sys.modules[__name__]},'guards':prepared['guards'],'_serving':prepared}
    names = ('model','processor_object','head_object','A','C','mu_train')
    events.refs = [weakref.ref(endpoint[name]) for name in names]
    events.loaded = True
    events.calls.append('load')
    return endpoint"""
    cache_source = """def _processor_cache(processor, guards, *, empty=False):
    import _serving_factory_fake_events as events
    events.calls.append('cache')
    if events.mode=='release': raise events.primary
    return processor.cache"""
    for name, replacement in (
        ("_serving_load_payload", payload_source),
        ("_processor_cache", cache_source),
    ):
        node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        old = ast.get_source_segment(runtime, node)
        assert runtime.count(old) == 1
        runtime = runtime.replace(old, replacement)
    (stage / "connected_inference.py").write_text(runtime)
    packed = """class PackedInt8Embeddings:
    @classmethod
    def from_bytes(cls, raw, *, count, dimensions):
        import _serving_factory_fake_events as events
        assert events.loaded and dimensions==128 and count==33 and len(raw)==count*130
        events.calls.append('pack')
        if events.mode=='pack-bytes':
            for path in events.paths:
                events.changed[path] = path.read_bytes()
                path.write_bytes(events.changed[path]+b'changed during packing')
            raise events.primary
        if events.mode=='pack': raise events.primary
        return cls()
"""
    native = """class CutilePackedInt8Gallery:
    @classmethod
    def open_packed(cls, path, packed):
        import _serving_factory_fake_events as events
        events.calls.append('open')
        if events.mode=='open': raise events.primary
        return cls()
    def close(self):
        import _serving_factory_fake_events as events
        events.calls.append('gallery-close')
        if events.mode=='gallery-close': raise events.primary
"""
    (stage / "packed_int8.py").write_text(packed)
    (stage / "cutile_int8.py").write_text(native)

    def sha(raw):
        return hashlib.sha256(raw).hexdigest()

    authority = (SOURCES / "_connected_inference_authority.py").read_text()
    records = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(authority).body
        if isinstance(node, ast.Assign)
    }
    for key, replacement in (
        ("RUNTIME_SHA256", sha(runtime.encode())),
        ("PACKED_SHA256", sha(packed.encode())),
    ):
        assert authority.count(records[key]) == 1
        authority = authority.replace(records[key], replacement)
    (stage / "_connected_inference_authority.py").write_text(authority)
    bridge_source = (SOURCES / "connected_compact_serving.py").read_text()
    old_sha = bridge._installed_authority()[1]
    assert bridge_source.count(old_sha) == 1
    bridge_source = bridge_source.replace(old_sha, sha(authority.encode()))
    (stage / "connected_compact_serving.py").write_text(bridge_source)
    events = ModuleType("_serving_factory_fake_events")
    assert events.__name__ not in sys.modules
    sys.modules[events.__name__] = events

    class FakeLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            assert events.loaded, "packing/native wrapper executed before loader returned"
            events.calls.append(module.__name__ + "-import")
            raw = Path(module.__file__).read_bytes()
            exec(compile(raw, module.__file__, "exec", dont_inherit=True), vars(module))

    class FakeImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"sfora.packed_int8", "sfora.cutile_int8"}:
                return importlib.util.spec_from_file_location(
                    fullname, stage / (fullname.rsplit(".", 1)[1] + ".py"), loader=FakeLoader()
                )
            return None

    finder = FakeImports()
    exit_reads = {"active": False, "names": []}

    def audit(event, args):
        if exit_reads["active"] and event == "open" and isinstance(args[0], str):
            exit_reads["names"].append(args[0])

    sys.addaudithook(audit)
    original = sys.modules.pop("sfora.connected_compact_serving")
    assert original is bridge
    spec = importlib.util.spec_from_file_location(
        "sfora.connected_compact_serving", stage / "connected_compact_serving.py"
    )
    installed = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = installed
    exec(
        compile(
            bridge_source, str(stage / "connected_compact_serving.py"), "exec", dont_inherit=True
        ),
        vars(installed),
    )
    sys.modules["sfora"].connected_compact_serving = installed
    sys.meta_path.insert(0, finder)
    try:
        cases = (
            "normal",
            "pack",
            "open",
            "release",
            "gallery-close",
            "runtime-mutation",
            "runtime-mutation-bytes",
            "helper-mutation-bytes",
            "context-mutation",
            "foreign-helper",
            "setup",
            "setup-copy",
            "setup-checker",
            "setup-publish",
            "pack-bytes",
        )
        for mode in (selected,) if selected else cases:
            events.mode, events.calls, events.refs, events.loaded = mode, [], [], False
            events.primary = RuntimeError("original " + mode + " sentinel")
            events.paths = (
                namespace["directory"] / "endpoint.pt",
                namespace["environment"].target / "torch/__init__.py",
            )
            events.changed = {}
            events.forged = []
            foreign = None
            exit_reads["names"].clear()
            setup_lines = {
                "setup": 'context = self._endpoint["_serving"]',
                "setup-copy": "release_context = context.copy()",
                "setup-checker": 'release_context["exit_check"] = FunctionType(',
                "setup-publish": 'self._artifact["release_context"] = release_context',
            }

            target_line = setup_lines.get(mode)

            def trace_setup(frame, event, arg, target_line=target_line):
                if (
                    event == "line"
                    and frame.f_globals is vars(installed)
                    and frame.f_code.co_name == "from_serving_artifact"
                    and bridge_source.splitlines()[frame.f_lineno - 1].strip() == target_line
                ):
                    exit_reads["active"] = True
                    raise events.primary
                return trace_setup

            index = None
            caught = None
            try:
                if mode in setup_lines:
                    sys.settrace(trace_setup)
                index = installed.ConnectedCompactIndex.from_serving_artifact(**args)
                assert events.calls.index("load") < events.calls.index("sfora.packed_int8-import")
                assert all(
                    Path(path).name not in {"endpoint.pt", "vision.pt", "torch/__init__.py"}
                    for path, _, _ in index._guards
                )
                if mode == "runtime-mutation":
                    index._module.release_inference = lambda endpoint: None
                if mode in ("runtime-mutation-bytes", "helper-mutation-bytes"):
                    for path in events.paths:
                        events.changed[path] = path.read_bytes()
                        path.write_bytes(events.changed[path] + b"changed before close")
                    if mode == "runtime-mutation-bytes":
                        index._module._serving_full_exit = lambda *a, **k: events.forged.append(
                            True
                        )
                        index._module.require = lambda *a, **k: events.forged.append(True)
                        index._module._binding = None
                    else:
                        index._artifact["helpers"][2].admit_serving_artifact = lambda *a, **k: (
                            events.forged.append(True)
                        )
                if mode == "context-mutation":
                    context = index._endpoint["_serving"]
                    context["directory"] = "/forged-context"
                    context["fragment_sha256"]["origin.json"] = "0" * 64
                if mode == "foreign-helper":
                    module = index._artifact["helpers"][2]
                    foreign = ModuleType(module.__name__)
                    foreign.__file__, foreign.__spec__ = module.__file__, module.__spec__
                    sys.modules[module.__name__] = foreign
                index.close()
            except BaseException as error:
                caught = error
            finally:
                sys.settrace(None)
                exit_reads["active"] = False
                for path, raw in events.changed.items():
                    path.write_bytes(raw)
            if mode in ("normal", "context-mutation"):
                assert caught is None, repr(caught)
                assert events.calls.count("cache") == events.calls.count("gallery-close") == 1
                index.close()
            elif mode == "runtime-mutation":
                assert isinstance(caught, ValueError) and "callable changed" in str(caught), repr(
                    caught
                )
                assert events.calls.count("cache") == 1, events.calls
            elif mode in ("runtime-mutation-bytes", "helper-mutation-bytes", "foreign-helper"):
                assert isinstance(caught, ValueError), repr(caught)
                assert events.calls.count("cache") == 1 and not events.forged, events.calls
            else:
                assert caught is events.primary, repr(caught) + repr(
                    getattr(caught, "__notes__", ())
                )
            if mode in setup_lines:
                assert "endpoint.pt" in exit_reads["names"], (
                    "setup skipped fresh artifact exit",
                    getattr(caught, "__notes__", ()),
                )
                assert any(Path(name).name == "__init__.py" for name in exit_reads["names"]), (
                    "setup skipped fresh installed-file exit",
                    getattr(caught, "__notes__", ()),
                )
            if mode == "pack-bytes":
                assert any(
                    "serving exit installed environment failed" in note
                    for note in getattr(caught, "__notes__", ())
                ), ("nested byte-failure note lost", getattr(caught, "__notes__", ()))
            if mode in ("runtime-mutation-bytes", "helper-mutation-bytes"):
                notes = getattr(caught, "__notes__", ())
                assert any("serving exit artifact failed" in note for note in notes), notes
                assert any("serving exit installed environment failed" in note for note in notes), (
                    notes
                )
            assert all(ref() is None for ref in events.refs), (
                mode,
                events.calls,
                getattr(caught, "__notes__", ()),
            )
            remaining = [
                name for name in sys.modules if name.startswith("_sfora_connected_artifact_")
            ]
            if foreign is not None:
                assert remaining == [foreign.__name__] and sys.modules[foreign.__name__] is foreign
                # Only the test created this stand-in; the factory must retain
                # it. Dispose it after checking that ownership result.
                del sys.modules[foreign.__name__]
            else:
                assert not remaining, mode
            for name in ("sfora.packed_int8", "sfora.cutile_int8"):
                sys.modules.pop(name, None)
                child = name.rsplit(".", 1)[1]
                if hasattr(sys.modules["sfora"], child):
                    delattr(sys.modules["sfora"], child)
        print(
            "PASS actual factory/independent release lifecycle seams "
            + str((selected,) if selected else cases)
            + " with authenticated stand-in source; native UNRUN",
            flush=True,
        )
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop(events.__name__, None)
        sys.modules["sfora.connected_compact_serving"] = original
        sys.modules["sfora"].connected_compact_serving = original


def main():
    assert __debug__
    assert len(sys.argv) == 1 or (len(sys.argv) == 3 and sys.argv[1] == "--lifecycle-case")
    selected = sys.argv[2] if len(sys.argv) == 3 else None
    before = dict(sys.modules)
    blocker = NoNative()
    sys.meta_path.insert(0, blocker)
    environment = None
    try:
        package("sfora")
        bridge, _, _ = load("sfora.connected_compact_serving", "connected_compact_serving.py")
        package("_draft_helpers")
        helpers = tuple(load("_draft_helpers." + name, name + ".py")[0] for name in HELPERS)
        authority = ast.parse((SOURCES / "_connected_inference_authority.py").read_text())
        record = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in authority.body
            if isinstance(node, ast.Assign)
        }
        with tempfile.TemporaryDirectory(prefix="serving-factory-") as tmp:
            folder = Path(tmp)
            fixture = (
                EVIDENCE / "assembled-candidate/sfora-serving-prepare-fixture.py"
            ).read_text()
            fixture = fixture[: fixture.index("    prepared = runtime._serving_prepare")]
            namespace = dict(
                root=ROOT,
                folder=folder,
                historical=record["HISTORICAL_CODE"],
                helpers=helpers,
                load=load,
                binding=None,
                sys=sys,
            )
            exec(
                compile(fixture + "finally:\n    pass\n", "<original-fixture-builder>", "exec"),
                namespace,
            )
            environment = namespace["environment"]
            args = {
                key: value for key, value in namespace["kwargs"].items() if key != "serving_helpers"
            }
            library = folder / "opaque-native-library.so"
            library.write_bytes(b"never loaded")
            args.update(
                serving_dir=namespace["directory"],
                native_library_path=library,
                expected_native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            )
            sentinel = None
            reached, owners, inventories = [], [], []
            auditing = {"active": False}
            reads = []

            def audit(event, args):
                if auditing["active"] and event == "open" and isinstance(args[0], str):
                    reads.append(args[0])

            sys.addaudithook(audit)
            changed = {}
            mode = "normal"

            def early_trace(frame, event, arg):
                if (
                    event == "line"
                    and frame.f_globals is vars(bridge)
                    and frame.f_code.co_name == "from_serving_artifact"
                    and (SOURCES / "connected_compact_serving.py")
                    .read_text()
                    .splitlines()[frame.f_lineno - 1]
                    .strip()
                    == "self._module._bind_runtime("
                ):
                    raise sentinel
                return early_trace

            def trace(frame, event, arg):
                if event == "call" and frame.f_code.co_name == "_serving_load_payload":
                    prepared = frame.f_locals["prepared"]
                    assert prepared["installed"] == namespace["expected"]
                    assert len(prepared["admitted"]["files"]) == 10
                    owner = prepared["checker"].__self__
                    owner._check_current()
                    owners.append(owner)
                    inventories.append(tuple(owner._owned.items()))
                    reached.append(True)
                    auditing["active"] = True
                    if mode == "bytes":
                        for path in (
                            namespace["directory"] / "endpoint.pt",
                            environment.target / "torch/__init__.py",
                        ):
                            changed[path] = path.read_bytes()
                            path.write_bytes(changed[path] + b"mutated after admission")
                    raise sentinel
                return trace

            # First RED is a missing public factory. GREEN must reach the real
            # loader with actual helper snapshots, without substituting its API.
            factory = bridge.ConnectedCompactIndex.from_serving_artifact
            if selected in (None, "pre-native"):
                sentinel = RuntimeError("before binding, no native resources")
                sys.settrace(early_trace)
                try:
                    factory(**args)
                except BaseException as error:
                    assert error is sentinel and not getattr(error, "__notes__", ()), repr(
                        error
                    ) + repr(getattr(error, "__notes__", ()))
                else:
                    raise AssertionError("early binding sentinel did not reject")
                finally:
                    sys.settrace(None)
                assert not any(
                    name.startswith("_sfora_connected_artifact_") for name in sys.modules
                )
                foreign_bridge = ModuleType(bridge.__name__)
                vars(foreign_bridge).update(vars(bridge))
                sys.modules[bridge.__name__] = foreign_bridge
                try:
                    try:
                        factory(**args)
                    except ValueError as error:
                        assert "canonical installed connected index" in str(error)
                    else:
                        raise AssertionError("foreign canonical bridge registry accepted")
                finally:
                    sys.modules[bridge.__name__] = bridge
            pre_native = selected in (None, "pre-native")
            for mode in ("normal", "normal", "bytes", "normal") if pre_native else ():
                sentinel = RuntimeError("stop before the first payload/native instruction")
                reads.clear()
                sys.settrace(trace)
                try:
                    factory(**args)
                except BaseException as error:
                    assert error is sentinel, repr(error) + repr(getattr(error, "__notes__", ()))
                    notes = getattr(error, "__notes__", ())
                    if mode == "bytes":
                        assert any("serving exit artifact failed" in note for note in notes), notes
                        assert any(
                            "serving exit installed environment failed" in note for note in notes
                        ), notes
                    else:
                        assert not notes, notes
                else:
                    raise AssertionError("pre-native sentinel did not reject")
                finally:
                    sys.settrace(None)
                    auditing["active"] = False
                    for path, raw in changed.items():
                        path.write_bytes(raw)
                    changed.clear()
                assert owners[-1]._closed and owners[-1]._module is None
                assert all(sys.modules.get(name) is not module for name, module in inventories[-1])
                # A trace-function exception disables tracing in CPython. The
                # audit observer proves fresh exit reads instead of inventing
                # an exit-call count from that disabled trace.
                assert any(Path(path).name == "endpoint.pt" for path in reads), reads
                if mode != "bytes":
                    assert any(Path(path).name == "__init__.py" for path in reads), reads
            assert len(reached) == (4 if pre_native else 0)
            assert not any(name.startswith("_sfora_connected_artifact_") for name in sys.modules)
            assert not any(
                name.split(".")[0]
                in {"torch", "numpy", "PIL", "transformers", "safetensors", "torchvision"}
                for name in sys.modules
            )
            if pre_native:
                print(
                    "PASS factory admission and pre-native stop; "
                    "fresh artifact/environment exit, registry cleanup, byte-mutation rejection "
                    "and restored composition; native UNRUN",
                    flush=True,
                )
            if selected != "pre-native":
                staged_lifecycle(bridge, namespace, folder, args, selected)
    finally:
        sys.settrace(None)
        if environment is not None:
            environment.doCleanups()
        sys.meta_path.remove(blocker)
        for name, _module in tuple(sys.modules.items()):
            if name not in before:
                del sys.modules[name]
        assert all(sys.modules.get(name) is module for name, module in before.items())


if __name__ == "__main__":
    main()
