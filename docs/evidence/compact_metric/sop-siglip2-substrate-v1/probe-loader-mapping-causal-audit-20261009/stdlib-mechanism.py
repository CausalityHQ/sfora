import ast, copy, gc, hashlib, json, mmap, os, resource, signal, sys, tempfile, time, traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path("/home/rb/worktrees/sfora-probe-loader-mapping-causal-audit-20261009")
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
signal.alarm(15)
started = time.monotonic()
SOURCES = {}
EXTRACTED = []

def extract(file, names, namespace, owner=None):
    path = ROOT/file
    raw = path.read_bytes()
    SOURCES[file] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    body = tree.body if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    nodes = [next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name) for name in names]
    before = [ast.dump(n, include_attributes=False) for n in nodes]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    assert before == [ast.dump(n, include_attributes=False) for n in nodes]
    EXTRACTED.extend(file + ":" + (owner + "." if owner else "") + name for name in names)

class Tensor:
    def __init__(self, storage):
        self.storage = storage
        self.device = SimpleNamespace(type="cpu")
        self.requires_grad = False
        self._version = 0; self.dtype = "fake.float32"; self.shape = (len(storage),)
    def to(self, device="cpu", copy=False):
        if WORLD.mode == "copy_error" and isinstance(self.storage, mmap.mmap):
            raise RuntimeError("injected copy failure")
        return Tensor(bytes(self.storage)) if copy else self
    def __deepcopy__(self, memo):
        result = Tensor(bytes(self.storage)); memo[id(self)] = result; return result
    def is_contiguous(self): return True
    def data_ptr(self): return id(self.storage)
    def numel(self): return len(self.storage)
    def element_size(self): return 1
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def reshape(self, *args): return self
    def view(self, *args): return self
    def numpy(self): return bytes(self.storage)
    def copy_(self, other): self.storage = bytes(other.storage)

class Model:
    def state_dict(self): return {"model_vision": True}
    def requires_grad_(self, value): return self
    def eval(self): return self
    def train(self): return self
    def to(self, device): return self

pages_space = {}
extract("scripts/train_siglip2_substrate_adaptation.py", ["copy", "consume"], pages_space, "CheckpointPages")
class Pages:
    copy = pages_space["copy"]
    consume = pages_space["consume"]
    def __init__(self, stream): self.fd = stream.fileno()
    def release(self, address, count):
        assert count >= 0
        WORLD.consumed += 1

space = {"Path": Path, "os": os, "copy": copy, "gc": gc, "time": time}
extract("scripts/train_siglip2_connected_probe.py", ["require", "mapping_absent", "owned_copy", "inference_readout_tree", "load_inference"], space)
extract("scripts/qualify_connected_probe_serving.py", ["clear_frames"], space)
extract("scripts/qualify_connected_serving_requests.py", ["raise_failures"], space)
fingerprint_space = {"hashlib": hashlib}
extract("scripts/train_siglip2_substrate_adaptation.py", ["fingerprint"], fingerprint_space)
extract("scripts/train_siglip2_substrate_adaptation.py", ["load_vision"], space)
pages_space["require"] = space["require"]
trainer_tree = ast.parse((ROOT/"scripts/train_siglip2_connected_probe.py").read_bytes())
for name in ("INFERENCE_KEYS", "INFERENCE_SCHEMA", "CONTROL_SHA256", "SERVING_FILES"):
    node = next(n for n in trainer_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    space[name] = ast.literal_eval(node.value)
space["ARMS"] = ast.literal_eval(next(n.value.elts[0] for n in trainer_tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and n.targets[0].elts[0].id == "ARMS"))

class World:
    def __init__(self, directory, mode):
        self.directory, self.mode = directory, mode
        self.external = None; self.consumed = 0; self.events = []
    def load(self, path, **kwargs):
        assert kwargs == dict(map_location="cpu", weights_only=True, mmap=True)
        with Path(path).open("rb") as stream:
            storage = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
        t = lambda: Tensor(storage)
        if self.mode == "external_owner": self.external = t()
        disk = {k: None for k in space["INFERENCE_KEYS"]}
        disk.update(schema=space["INFERENCE_SCHEMA"], arm="control", fixed_sha256="fixed",
            config={}, buffers={"position": t()}, processor={"config": {}}, head={"weight": t()},
            A=t(), C=t(), means=[t(), (t(),)], mu_train=t(), common_statistics={"nested": [t(), (t(),)]},
            mu_train_provenance={"role": "FIT"}, numerical_flags={}, base_vision={"sha256": "vision"},
            encoder={"overlay": t()}, encoder_identity={"runtime": "structure"}, vision_sha256="vision",
            scope={"arm": "control", "payload": {"scope_sha256": space["CONTROL_SHA256"], "class_names": ["x"]*1008}})
        # A tensor in metadata is a stronger deepcopy falsifier than the actual scalar metadata.
        disk["processor"]["metadata_tensor"] = t()
        return disk
    def fingerprint(self, value):
        if isinstance(value, dict) and "model_vision" in value: return "vision"
        if isinstance(value, dict) and "schema" in value: return "endpoint" if "fixed_sha256" in value else "fixed"
        return "readout"
    def construct(self, source, original, context, config, buffers, processor_config, base, overlay):
        if self.mode == "construct_error": raise RuntimeError("injected constructor failure with endpoint overlay")
        return Model(), SimpleNamespace(), SimpleNamespace(currsize=0), {}, "structure"
    def head(self, arm, tensors):
        assert arm == "control"
        if self.mode == "head_error": raise RuntimeError("injected head failure after independent copies")
        return Model()
    def trace(self, frame, event, arg):
        if frame.f_globals is space and frame.f_code is space["load_inference"].__code__:
            if event == "exception":
                self.events.append({"event": "loader_exception", "type": arg[0].__name__, "message": str(arg[1])})
            if event == "line" and frame.f_lineno == MAPPING_LINE:
                active = sys.exception()
                self.events.append({"event": "before_original_mapping_predicate", "disk_deleted": "disk" not in frame.f_locals,
                    "active_exception": None if active is None else type(active).__name__ + ":" + str(active),
                    "copied_present": "copied" in frame.f_locals, "endpoint_present": "endpoint" in frame.f_locals})
        return self.trace

node = next(n for n in trainer_tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_inference")
MAPPING_LINE = next(n.lineno for n in ast.walk(node) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "mapping_absent")
results = []
rendering_checks = []
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    path = directory/"endpoint.pt"; path.write_bytes(b"x"*4096)
    for mode in ("success", "external_owner", "copy_error", "construct_error", "head_error"):
        WORLD = World(directory, mode)
        torch = SimpleNamespace(Tensor=Tensor, load=WORLD.load, nn=SimpleNamespace(Parameter=lambda value, requires_grad: value))
        original = SimpleNamespace(fingerprint=WORLD.fingerprint, CheckpointPages=Pages)
        source = SimpleNamespace(numerical_flags=lambda: {})
        modules = {"train_siglip2_substrate_adaptation.py": original, "qualify_siglip2_substrate_cpu.py": source,
            "extract_siglip2_vision_source.py": SimpleNamespace(), "train_siglip2_cached_readout.py": SimpleNamespace(head_from=WORLD.head)}
        manifest = {"endpoint_state_sha256": "endpoint", "vision_sha256": "vision", "encoder_identity": {"runtime": "structure"},
            "base_vision_sha256": "vision", "environment": {"packages": {}, "vision_constructor": "fake"},
            "files": {"vision.pt": "fake"}, "code": {name: "fake" for name in space["SERVING_FILES"] | {"joint_relational_compaction.py"}}}
        space.update(admit_bundle=lambda *args: (manifest, {}), load_authenticated=lambda name, path, *args: modules.get(path.name, SimpleNamespace()),
            construct_encoder=WORLD.construct, encoder_facts=lambda *args, **kwargs: {"vision_sha256": "vision"})
        error = state = None
        previous = sys.gettrace()
        with patch.dict(sys.modules, {"torch": torch}):
            sys.settrace(WORLD.trace)
            try:
                state = space["load_inference"](directory, "fake", "cuda")
            except BaseException as caught:
                error = caught
            finally: sys.settrace(previous)
        assert sys.gettrace() is previous
        if mode == "success":
            assert error is None, (repr(error), WORLD.events)
            space["mapping_absent"](path)
            assert WORLD.consumed == 8, WORLD.consumed
            assert not isinstance(state["A"].storage, mmap.mmap)
            assert not isinstance(state["processor"]["metadata_tensor"].storage, mmap.mmap)
        elif mode == "external_owner":
            assert type(error) is ValueError and error.__context__ is None
            assert WORLD.external is not None
        elif mode in ("copy_error", "construct_error"):
            assert type(error) is ValueError and type(error.__context__) is RuntimeError
            assert "injected" in str(error.__context__)
            assert WORLD.events[-2]["event"] == "before_original_mapping_predicate"
            assert WORLD.events[-2]["active_exception"].startswith("RuntimeError:")
        else:
            assert type(error) is RuntimeError and error.__context__ is None
            space["mapping_absent"](path)
        results.append({"mode": mode, "returned": state is not None,
            "error": None if error is None else type(error).__name__ + ":" + str(error),
            "context": None if error is None or error.__context__ is None else type(error.__context__).__name__ + ":" + str(error.__context__),
            "events": WORLD.events, "actual_pages_copy_calls": WORLD.consumed})
        if error is not None:
            space["clear_frames"](error)
            if mode in ("external_owner", "copy_error", "construct_error"):
                try: space["raise_failures"]([error, ValueError("observed native difference must be exact four")])
                except BaseException as grouped:
                    rendered = "".join(traceback.format_exception(grouped))
                    assert "observed native difference must be exact four" in rendered
                    assert "checkpoint mapping survived independent copy/release" in rendered
                    assert "RuntimeError: injected" not in rendered
                    rendering_checks.append({"mode": mode, "earlier_context_hidden_by_explicit_exit_group": mode != "external_owner"})
                    space["clear_frames"](grouped)
        WORLD.external = state = error = None
        gc.collect(); space["mapping_absent"](path)
    # Actual recursion and actual copy method independently preserve tuple/list and never adopt mmap aliases.
    WORLD = World(directory, "success")
    with path.open("rb") as stream: mm = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    with patch.dict(sys.modules, {"torch": SimpleNamespace(Tensor=Tensor)}):
        value = {"a": [Tensor(mm), (Tensor(mm),)], "scalar": 3}
        out = space["owned_copy"](value, SimpleNamespace(copy=lambda v,d: pages_space["copy"](SimpleNamespace(consume=lambda v: None),v,d)))
        assert type(out["a"]) is list and type(out["a"][1]) is tuple and out["scalar"] == 3
        assert all(type(v.storage) is bytes for v in (out["a"][0], out["a"][1][0]))
    # Actual typed hash walks aliases, returns only a scalar digest, and retains no memo/cache.
    with patch.dict(sys.modules, {"torch": SimpleNamespace(Tensor=Tensor, uint8="fake.uint8")}):
        hashed = fingerprint_space["fingerprint"](value)
        assert type(hashed) is str and len(hashed) == 64
    del value, mm; gc.collect(); space["mapping_absent"](path)
    # Actual strict loader hooks close over vision only while load_vision is alive.
    class Handle:
        def __init__(self, model, callback): self.model, self.callback = model, callback
        def remove(self): self.model.hooks.remove(self.callback)
    class VisionModel:
        def __init__(self): self.hooks = []; self.weight = Tensor(b"owned")
        def named_modules(self): return [("", self)]
        def named_parameters(self, recurse=False): return [("weight", self.weight)]
        def named_buffers(self, recurse=False): return []
        def register_load_state_dict_post_hook(self, callback):
            self.hooks.append(callback); return Handle(self, callback)
        def load_state_dict(self, vision, strict):
            assert strict is True
            self.weight.copy_(vision["weight"])
            for callback in self.hooks: callback(self, None)
    with path.open("rb") as stream: mm = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    vision = {"weight": Tensor(mm)}; model = VisionModel()
    space["load_vision"](model, vision, SimpleNamespace(consume=lambda value: None))
    assert model.hooks == [] and type(model.weight.storage) is bytes
    del vision, mm; gc.collect(); space["mapping_absent"](path)
report = {"status": "STDLIB_MECHANISM_CHECKS_PASS_ORIGINAL_CAUSE_UNPROVEN", "seconds": time.monotonic()-started,
    "as_bytes": 1024**3, "source_sha256": SOURCES, "unmodified_ast_seams": EXTRACTED, "results": results, "exception_rendering_checks": rendering_checks,
    "limitations": "Fake native dependency globals and Torch tensors; actual mmap and unmodified extracted seams. No original checkpoint or native run; no attribution to original failure."}
Path("/tmp/sfora-probe-loader-mapping-causal-check.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
