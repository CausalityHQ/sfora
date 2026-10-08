"""Source-only check: python3 -B -S scripts/test_packed_int8_core_extraction.py.

No third-party code executes. Tensor stand-ins test admission and import wiring,
not numerical behavior; genuine Torch/wire/native parity belongs to the root.
"""

import ast
import builtins
import hashlib
import importlib
import pickle
import struct
import symtable
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
HISTORICAL = "sfora.joint_relational_compaction"
# Complete definition bytes and ASTs pinned to f7bb3f9ee8e9e2471f1044934ee8b5e37858d5ad.
PINS = {
    "_PACKED_INT8_ARTIFACT_MAGIC": (
        "6f267e324b2afbdf779c0b31cea93f76fd1e05bf0114664c99e7132ba180e9d0",
        "ef679b13e31bf3a19ac60e2287cb680d95f52d3aabe73757e9edefa5e22231e0"),
    "_SHA256_BYTES": (
        "42124fc1e52f203f151a610c9d6f4ae5283d46723c2912663bc57256eb15ec55",
        "ef5a634e07ec333c10673717e0af64983bf7d6f61bcef2a1da4d7cf3cf8ab1b5"),
    "_unit_rows": (
        "95cec4310531180999dd26e0c1c5716bb76445b42958b1682758caa1e8eed160",
        "637aa103f56005520d46a50115cad62ce79c4665cf2f8a6ec58442e25721670a"),
    "PackedInt8Embeddings": (
        "db23275bbc8f0a82bdc4fe0429c79d9a92b38cc41b548f616b31fba0bdd56afa",
        "4330092effdf3f288015d614031b21ff435f8ab38006247d0e422bfd1c66aff3"),
    "fixed_int8_unit_codes": (
        "5c398642753281dc94acf91f88576c14e74fd70969cb5a5565b394daca08b346",
        "2a464b16c3c01cf7ce63c74d9a5eb128b92fd0195dcabd9ed019b3eba20440a3"),
    "pack_int8_unit_embeddings": (
        "546da6e737458ff7dcde92389b1376b8e7dc409d3795547d8fca299664e68e85",
        "bfa6c8919e9b4c82b50563a1ca63ce3cf4f71313ec93dfffa64a30da2808545a"),
}
BASE_SHA = "4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67"


def definitions(source):
    result = {}
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in PINS:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            result[name] = (node, "".join(lines[first - 1:node.end_lineno]))
    return result


class Flag:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


class Tensor:
    """Admission-only tensor: numerical operations deliberately absent."""

    def __init__(self, shape=(1, 2), dtype="int8", device="cpu", contiguous=True,
                 finite=True):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.contiguous_value = contiguous
        self.finite = finite

    def is_contiguous(self):
        return self.contiguous_value

    def numpy(self):
        return self

    def clone(self):
        return self

    def contiguous(self):
        return self


class DependencyGuard:
    def __init__(self):
        self.loaded = []
        self.allow_historical = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("sfora."):
            allowed = {"sfora.packed_int8", "sfora.packed_int8_search", "sfora.cutile_int8"}
            if self.allow_historical:
                allowed.add(HISTORICAL)
            if fullname not in allowed:
                raise AssertionError(f"research dependency imported: {fullname}")
            self.loaded.append(fullname)
        elif fullname != "sfora" and fullname.partition(".")[0] not in sys.stdlib_module_names:
            raise AssertionError(f"third-party execution attempted: {fullname}")
        return None


@contextmanager
def fake_dependencies():
    names = ("torch", "torch.nn", "torch.nn.functional", "numpy", "numpy.typing",
             "numpy.ctypeslib")
    saved = {k: v for k, v in sys.modules.items()
             if k == "sfora" or k.startswith("sfora.") or k in names}
    for name in saved:
        del sys.modules[name]
    modules = {name: ModuleType(name) for name in names}
    torch = modules["torch"]
    torch.Tensor = Tensor
    torch.int8, torch.float16, torch.float32 = "int8", "float16", "float32"
    torch.isfinite = lambda value: Flag(value.finite)
    torch.nn = modules["torch.nn"]
    torch.nn.Module = type("Module", (), {})
    torch.nn.functional = modules["torch.nn.functional"]
    modules["numpy.typing"].NDArray = object
    modules["numpy.ctypeslib"].ndpointer = lambda **kwargs: None
    guard = DependencyGuard()
    sys.modules.update(modules)
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.meta_path.insert(0, guard)
    try:
        yield guard
    finally:
        sys.meta_path.remove(guard)
        sys.path.remove(str(SOURCE_ROOT))
        for name in list(sys.modules):
            if name == "sfora" or name.startswith("sfora.") or name in names:
                del sys.modules[name]
        sys.modules.update(saved)


class ExtractionTests(unittest.TestCase):
    def test_consumer_byte_inverses(self):
        pins = {
            "cutile_int8.py": "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409",
            "packed_int8_search.py": "24a665c1eb0b043c258ec181d7c451b0584ba4665da2be978eafb823ae24da61",
        }
        for filename, expected in pins.items():
            source = (SOURCE_ROOT / "sfora" / filename).read_text()
            self.assertNotIn("sfora.joint_relational_compaction", source)
            restored = source.replace("from sfora.packed_int8 import PackedInt8Embeddings",
                                      "from sfora.joint_relational_compaction import PackedInt8Embeddings")
            self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(), expected)
        source = (SOURCE_ROOT / "sfora/__init__.py").read_text()
        source = source.replace(
            '_PACKED_INT8_EXPORTS = frozenset({"PackedInt8Embeddings", "pack_int8_unit_embeddings"})\n\n',
            "")
        source = source.replace(
            '    if name in _PACKED_INT8_EXPORTS:\n'
            '        module = import_module("sfora.packed_int8")\n'
            '        value = cast(object, getattr(module, name))\n'
            '        globals()[name] = value\n'
            '        return value\n', "")
        source = source.replace('_RELATIONAL_COMPACTION_EXPORTS = frozenset(\n    {\n',
                                '_RELATIONAL_COMPACTION_EXPORTS = frozenset(\n    {\n'
                                '        "PackedInt8Embeddings",\n')
        source = source.replace('        "fit_relational_linear_encoder",\n',
                                '        "fit_relational_linear_encoder",\n'
                                '        "pack_int8_unit_embeddings",\n')
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         "224ea495fa7d5e000a5ced9d744434cb4724a9ecdfb96de0e2963b9cc2b2c32e")

    def test_exact_definitions_and_byte_inverse(self):
        source = (SOURCE_ROOT / "sfora/packed_int8.py").read_text()
        moved = definitions(source)
        self.assertEqual(set(moved), set(PINS))
        for name, (node, text) in moved.items():
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), PINS[name][0], name)
            self.assertEqual(hashlib.sha256(ast.dump(node).encode()).hexdigest(), PINS[name][1], name)
        # Undo only declared extraction plumbing; recover every original byte.
        old = (SOURCE_ROOT / "sfora/joint_relational_compaction.py").read_text()
        tree = ast.parse(old)
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                   and n.module == "sfora.packed_int8"]
        self.assertEqual(len(imports), 1)
        node = imports[0]
        self.assertEqual({n.name for n in node.names}, set(PINS))
        lines = old.splitlines(keepends=True)
        lines[node.lineno - 1:node.end_lineno] = [
            moved["_PACKED_INT8_ARTIFACT_MAGIC"][1] + moved["_SHA256_BYTES"][1]
            + "\n\n" + moved["_unit_rows"][1]]
        restored = "".join(lines)
        restored = restored.replace("import math\n", "import hashlib\nimport math\n")
        restored = restored.replace("from typing import cast\n",
                                    "from pathlib import Path\nfrom typing import cast\n")
        anchor = "class JointRelationalEncoder"
        self.assertEqual(restored.count(anchor), 1)
        restored = restored.replace(anchor, moved["PackedInt8Embeddings"][1] + "\n\n" + anchor)
        restored += "\n\n" + moved["fixed_int8_unit_codes"][1] + "\n\n" + moved["pack_int8_unit_embeddings"][1]
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(), BASE_SHA)

    def test_complete_dependency_closure(self):
        source = (SOURCE_ROOT / "sfora/packed_int8.py").read_text()
        tree = ast.parse(source)
        imports = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update({alias.asname or alias.name: alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                imports.update({alias.asname or alias.name: f"{node.module}.{alias.name}"
                                for alias in node.names})
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertIn(node.name, PINS)
            elif isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):
                    self.assertIn(node.targets[0].id, PINS)
                else:
                    self.assertIsInstance(node.targets[0], ast.Attribute)
            else:
                self.assertIs(node, tree.body[0])
                self.assertIsInstance(node, ast.Expr)
                self.assertIsInstance(node.value, ast.Constant)
        self.assertEqual(imports, {
            "annotations": "__future__.annotations", "hashlib": "hashlib", "struct": "struct",
            "dataclass": "dataclasses.dataclass", "Path": "pathlib.Path", "np": "numpy",
            "torch": "torch", "F": "torch.nn.functional"})
        # Resolve every referenced global in genuine nested scopes without third-party imports.
        allowed = set(imports) | set(PINS) | set(vars(builtins)) | {"__name__"}
        pending = [symtable.symtable(source, "packed_int8.py", "exec")]
        while pending:
            table = pending.pop()
            pending.extend(table.get_children())
            for symbol in table.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    self.assertIn(symbol.get_name(), allowed)
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Attribute)]
        self.assertEqual({n.targets[0].value.id for n in assignments}, set(PINS) - {
            "_PACKED_INT8_ARTIFACT_MAGIC", "_SHA256_BYTES"})
        for node in assignments:
            self.assertEqual(node.targets[0].attr, "__module__")
            self.assertEqual(ast.literal_eval(node.value), HISTORICAL)

    def test_real_module_identity_and_historical_pickle(self):
        with fake_dependencies() as guard:
            package = importlib.import_module("sfora")
            self.assertEqual(guard.loaded, [])
            cls = package.PackedInt8Embeddings
            canonical = importlib.import_module("sfora.packed_int8")
            self.assertIs(cls, canonical.PackedInt8Embeddings)
            self.assertIs(package.pack_int8_unit_embeddings, canonical.pack_int8_unit_embeddings)
            self.assertNotIn(HISTORICAL, sys.modules)
            search = importlib.import_module("sfora.packed_int8_search")
            native = importlib.import_module("sfora.cutile_int8")
            self.assertIs(search.PackedInt8Embeddings, cls)
            self.assertNotIn(HISTORICAL, sys.modules)
            # Execute genuine native wrapper imports with only its FFI entry points replaced.
            value = object.__new__(cls)
            object.__setattr__(value, "codes", Tensor())
            object.__setattr__(value, "inverse_norms", Tensor((1,), "float16"))
            gallery = object.__new__(native.CutilePackedInt8Gallery)
            gallery.search = lambda codes, norms, k: (codes, norms, k)
            self.assertEqual(gallery.search_packed(value), (value.codes, value.inverse_norms, 10))
            original = native.CutilePackedInt8Gallery.__dict__["open"]
            native.CutilePackedInt8Gallery.open = classmethod(lambda cls, path, codes, norms:
                                                             (path, codes, norms))
            try:
                self.assertEqual(native.CutilePackedInt8Gallery.open_packed(Path("unused"), value),
                                 (Path("unused"), value.codes, value.inverse_norms))
            finally:
                native.CutilePackedInt8Gallery.open = original
            self.assertNotIn(HISTORICAL, sys.modules)
            # A valid exact-type value is also admitted by the real CPU wrapper.
            cpu = search.CpuPackedInt8Gallery.open_packed(value)
            self.assertIs(cpu._codes, value.codes)
            guard.allow_historical = True
            historical = importlib.import_module(HISTORICAL)
            for name in PINS:
                self.assertIs(getattr(historical, name), getattr(canonical, name))
            for name in ("PackedInt8Embeddings", "_unit_rows", "fixed_int8_unit_codes",
                         "pack_int8_unit_embeddings"):
                symbol = getattr(canonical, name)
                self.assertEqual(symbol.__module__, HISTORICAL)
                self.assertEqual(symbol.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(symbol)), symbol)
            wire = pickle.dumps(value, protocol=4)
            self.assertIn(HISTORICAL.encode(), wire)
            self.assertIs(type(pickle.loads(wire)), cls)
            # A historical GLOBAL reference from before extraction resolves unchanged.
            self.assertIs(pickle.loads(b"csfora.joint_relational_compaction\nPackedInt8Embeddings\n."), cls)

    def test_admission_and_wire_negatives(self):
        with fake_dependencies():
            packed = importlib.import_module("sfora.packed_int8")
            cls = packed.PackedInt8Embeddings
            norms = Tensor((1,), "float16")
            for codes in (object(), Tensor(dtype="float32"), Tensor(device="cuda"),
                          Tensor(shape=(0, 2)), Tensor(shape=(1, 1)), Tensor(shape=(2,)),
                          Tensor(contiguous=False)):
                with self.assertRaisesRegex(ValueError, "embedding authority"):
                    cls(codes, norms)
            for inverse in (object(), Tensor((1,), "float32"), Tensor((2,), "float16"),
                            Tensor((1,), "float16", device="cuda"),
                            Tensor((1,), "float16", contiguous=False),
                            Tensor((1,), "float16", finite=False)):
                with self.assertRaisesRegex(ValueError, "embedding authority"):
                    cls(Tensor(), inverse)
            for value in (object(), Tensor(), Tensor((1, 2), "float16"),
                          Tensor((0, 2), "float32"), Tensor((1, 1), "float32"),
                          Tensor((2,), "float32"), Tensor((1, 2), "float32", finite=False)):
                with self.assertRaisesRegex(ValueError, "quantization authority"):
                    packed.fixed_int8_unit_codes(value)
            for wire, count, dimensions in ((bytearray(4), 1, 2), (b"", 1, 2),
                                            (b"12345", 1, 2), (b"123", 1, 2),
                                            (b"1234", True, 2), (b"", 0, 2),
                                            (b"123", 1, 1), (b"1234", 1, 2.0)):
                with self.assertRaisesRegex(ValueError, "byte authority"):
                    cls.from_bytes(wire, count=count, dimensions=dimensions)
            value = object.__new__(cls)
            object.__setattr__(value, "codes", Tensor())
            class Foreign(cls):
                pass
            for foreign in (object(), object.__new__(Foreign)):
                with self.assertRaisesRegex(ValueError, "similarity authority"):
                    value.cosine_similarity(foreign)
                search = importlib.import_module("sfora.packed_int8_search")
                with self.assertRaisesRegex(ValueError, "gallery authority"):
                    search.CpuPackedInt8Gallery.open_packed(foreign)
                native = importlib.import_module("sfora.cutile_int8")
                with self.assertRaisesRegex(ValueError, "gallery authority"):
                    native.CutilePackedInt8Gallery.open_packed(Path("unused"), foreign)
                gallery = object.__new__(native.CutilePackedInt8Gallery)
                with self.assertRaisesRegex(ValueError, "query authority"):
                    gallery.search_packed(foreign)
            with self.assertRaisesRegex(ValueError, "artifact path"):
                cls.load("unused")
            with self.assertRaisesRegex(ValueError, "artifact path"):
                value.save("unused")
            magic = packed._PACKED_INT8_ARTIFACT_MAGIC
            good_payload = magic + struct.pack("<QQ", 1, 2) + b"1234"
            seal = lambda payload: payload + hashlib.sha256(payload).digest()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "packed.bin"
                for artifact in (b"", seal(b"x" * len(magic) + good_payload[len(magic):]),
                                 good_payload + b"x" * 32, seal(good_payload + b"x"),
                                 seal(magic + struct.pack("<QQ", 0, 2)),
                                 seal(magic + struct.pack("<QQ", 1, 1) + b"123")):
                    path.write_bytes(artifact)
                    with self.assertRaisesRegex(ValueError, "artifact differs"):
                        cls.load(path)


if __name__ == "__main__":
    unittest.main()
