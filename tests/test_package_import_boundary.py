"""Stdlib-only checks: python3 -B -S tests/test_package_import_boundary.py."""

import ast
import hashlib
import importlib
import importlib.util
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/sfora/__init__.py").read_text()

# Export ledger and inverse hashes from f5aa175168c6cf1cd6c2ba865414aec0e4717e45.
EAGER_EXPORTS = {
    "sfora.ablation": "SyntheticAblationConfig SyntheticAblationResult SyntheticAblationTrial "
    "run_synthetic_ablation write_ablation_report",
    "sfora.api": "SforaProjector fit_sfora_projection",
    "sfora.compose": "Head Identity Join L2Normalize Pca Pipeline Projection RetrievalReport "
    "compare evaluate grid",
    "sfora.data": "ImageExample TextExample TextGroupTriplet TextTriplet "
    "load_image_retrieval_examples load_imdb_examples mine_group_triplets mine_triplets "
    "select_balanced_examples select_labeled_image_examples",
    "sfora.encoder_ablation": "EncoderAblationConfig EncoderAblationResult EncoderAblationTrial "
    "run_encoder_ablation write_encoder_ablation_report",
    "sfora.encoder_training": "EncoderTrainingConfig EncoderTrainingMethodMetrics "
    "EncoderTrainingResult run_encoder_training run_encoder_training_on_split "
    "write_encoder_training_report",
    "sfora.evaluation": "EmbeddingSpaceDiagnostics ProbeScore RetrievalScore "
    "embedding_space_diagnostics_on_split linear_probe_score linear_probe_score_on_split "
    "retrieval_score_on_split",
    "sfora.experiments": "ExperimentResult MethodMetrics SyntheticExperimentConfig "
    "TrainableSyntheticExperimentConfig run_synthetic_experiment "
    "run_trainable_synthetic_experiment write_experiment_report",
    "sfora.image_benchmark": "ImageBenchmarkConfig ImageBenchmarkMethodMetrics "
    "ImageBenchmarkResult ImageObjective ImageRetrievalMetrics "
    "image_self_retrieval_score objective_display_name "
    "run_image_benchmark write_image_benchmark_report",
    "sfora.losses": "group_triplet_margin_loss triplet_margin_loss",
    "sfora.publication": "HfPublishBundle HfPublishConfig HfPublishResult build_hf_publish_bundle "
    "publish_hf_bundle",
    "sfora.remote": "RemoteRunConfig RemoteRunPlan RemoteStep build_remote_run_plan "
    "write_remote_run_plan",
    "sfora.report": "ReportConfig build_html_report build_markdown_report build_site_data "
    "write_hf_model_card write_html_report write_markdown_report write_site_data",
    "sfora.text_baselines": "SentenceTransformerBaselineConfig SentenceTransformerModelSuiteConfig "
    "TextBaselineConfig TextBaselineResult TextMethodMetrics run_sentence_transformer_baseline "
    "run_sentence_transformer_model_suite run_text_baseline write_text_baseline_report",
    "sfora.training": "ProjectionHeadTrainingConfig ProjectionHeadTrainingResult "
    "ProjectionTrainingConfig ProjectionTrainingResult train_embedding_table train_projection_head",
}
EAGER_EXPORTS = {module: names.split() for module, names in EAGER_EXPORTS.items()}


class DependencyGuard:
    def __init__(self):
        self.denied = []
        self.package_attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("sfora."):
            self.package_attempts.append(fullname)
        root = fullname.partition(".")[0]
        if root != "sfora" and root not in sys.stdlib_module_names:
            self.denied.append(fullname)
            raise AssertionError(f"nonstdlib import attempted: {fullname}")
        return None


class ModuleStandins:
    def __init__(self, exports):
        self.modules = {}
        self.loaded = []
        for module_name, names in exports.items():
            module = ModuleType(module_name)
            for name in names:
                setattr(module, name, object())
            self.modules[module_name] = module

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return self.modules[spec.name]

    def exec_module(self, module):
        self.loaded.append(module.__name__)


@contextmanager
def fresh_package(standins=None):
    saved = {name: value for name, value in sys.modules.items()
             if name == "sfora" or name.startswith("sfora.")}
    for name in saved:
        del sys.modules[name]
    old_path = sys.path[:]
    old_meta_path = sys.meta_path[:]
    guard = DependencyGuard()
    sys.path.insert(0, str(ROOT / "src"))
    sys.meta_path.insert(0, guard)
    if standins is not None:
        sys.meta_path.insert(1, standins)
    try:
        yield importlib.import_module("sfora"), guard
    finally:
        sys.meta_path[:] = old_meta_path
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "sfora" or name.startswith("sfora."):
                del sys.modules[name]
        sys.modules.update(saved)


def old_lazy_exports():
    tree = ast.parse(SOURCE)
    groups = {node.targets[0].id: ast.literal_eval(node.value.args[0])
              for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == "frozenset"}
    resolver = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
    return {node.body[0].value.args[0].value: groups[node.test.comparators[0].id]
            for node in resolver.body if isinstance(node, ast.If)}


def ast_snapshot(value):
    if isinstance(value, ast.AST):
        return (type(value).__name__, [(field, ast_snapshot(child))
                for field, child in ast.iter_fields(value) if field != "type_params"])
    if isinstance(value, list):
        return [ast_snapshot(child) for child in value]
    return value


class PackageImportBoundaryTests(unittest.TestCase):
    def test_fresh_interpreter_import_attempts_no_dependencies_or_submodules(self):
        result = subprocess.run(
            [sys.executable, "-B", "-S", str(Path(__file__).resolve()), "--import-probe"],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_eager_export_resolves_exact_module_object_and_is_cached(self):
        for module_name, names in EAGER_EXPORTS.items():
            for name in names:
                with self.subTest(module=module_name, name=name):
                    standins = ModuleStandins(EAGER_EXPORTS)
                    with fresh_package(standins) as (package, guard):
                        self.assertEqual(standins.loaded, [])
                        self.assertNotIn(name, vars(package))
                        expected = getattr(standins.modules[module_name], name)
                        self.assertIs(getattr(package, name), expected)
                        self.assertEqual(standins.loaded, [module_name])
                        self.assertIs(getattr(package, name), expected)
                        self.assertIs(vars(package)[name], expected)
                        self.assertIs(importlib.import_module(module_name),
                                      standins.modules[module_name])
                        self.assertEqual(guard.denied, [])

    def test_from_import_and_module_import_preserve_identity(self):
        standins = ModuleStandins(EAGER_EXPORTS)
        with fresh_package(standins) as (package, guard):
            namespace = {}
            exec("from sfora import SforaProjector, fit_sfora_projection, api", namespace)
            module = standins.modules["sfora.api"]
            self.assertIs(namespace["SforaProjector"], module.SforaProjector)
            self.assertIs(namespace["fit_sfora_projection"], module.fit_sfora_projection)
            self.assertIs(namespace["api"], module)
            self.assertIs(package.api, module)
            self.assertEqual(guard.denied, [])

    def test_unknown_names_raise_without_loading_modules(self):
        with fresh_package() as (package, guard):
            for name in ("not_a_sfora_export", "__missing__"):
                with self.assertRaises(AttributeError) as error:
                    getattr(package, name)
                self.assertEqual(str(error.exception),
                                 f"module 'sfora' has no attribute {name!r}")
            self.assertEqual(guard.package_attempts, [])
            self.assertEqual(guard.denied, [])

    def test_dir_exposes_public_names_without_importing_them(self):
        with fresh_package() as (package, guard):
            self.assertTrue(set(package.__all__).issubset(dir(package)))
            self.assertIn("__name__", dir(package))
            self.assertEqual(guard.package_attempts, [])
            self.assertEqual(guard.denied, [])

    def test_existing_lazy_exports_and_star_import_keep_object_identity(self):
        exports = dict(EAGER_EXPORTS, **old_lazy_exports())
        standins = ModuleStandins(exports)
        with fresh_package(standins) as (package, guard):
            namespace = {}
            exec("from sfora import *", namespace)
            self.assertEqual(set(namespace) - {"__builtins__"}, set(package.__all__))
            for module_name, names in exports.items():
                for name in names:
                    with self.subTest(name=name):
                        expected = getattr(standins.modules[module_name], name)
                        if name in package.__all__:
                            self.assertIs(namespace[name], expected)
                        self.assertIs(getattr(package, name), expected)
            self.assertEqual(set(standins.loaded), set(exports))
            self.assertEqual(len(standins.loaded), len(exports))
            self.assertEqual(guard.denied, [])

    def test_all_bytes_and_original_ast_survive_inverse(self):
        source = SOURCE
        packed_group = '_PACKED_INT8_EXPORTS = frozenset({"PackedInt8Embeddings", "pack_int8_unit_embeddings"})\n\n'
        packed_branch = '\n'.join([
            '    if name in _PACKED_INT8_EXPORTS:',
            '        module = import_module("sfora.packed_int8")',
            '        value = cast(object, getattr(module, name))',
            '        globals()[name] = value',
            '        return value', ''])
        relational = '\n'.join([
            '_RELATIONAL_COMPACTION_EXPORTS = frozenset(', '    {',
            '        "RelationalLinearEncoder",', '        "RelationalLinearTrainingConfig",',
            '        "fit_relational_linear_compaction",', '        "fit_relational_linear_encoder",',
            '    }', ')'])
        original_relational = relational.replace('    {\n', '    {\n        "PackedInt8Embeddings",\n').replace(
            '        "fit_relational_linear_encoder",\n',
            '        "fit_relational_linear_encoder",\n        "pack_int8_unit_embeddings",\n')
        for current, original in ((packed_group, ''), (packed_branch, ''),
                                  (relational, original_relational)):
            self.assertEqual(source.count(current), 1)
            source = source.replace(current, original)
        tree = ast.parse(source)
        all_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                        and node.targets[0].id == "__all__")
        all_bytes = ast.get_source_segment(source, all_node).encode()
        self.assertEqual(hashlib.sha256(all_bytes).hexdigest(),
                         "6c87e434ff75524550af48ccf1883fd5a9ea0b3bfee9d12680a6f1b11523aee4")
        original_nodes = []
        type_imports = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                node.names = [alias for alias in node.names if alias.name != "TYPE_CHECKING"]
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                self.assertEqual(node.test.id, "TYPE_CHECKING")
                type_imports = node.body
                original_nodes.extend(node.body)
                continue
            if isinstance(node, ast.Assign) and node.targets[0].id == "_CORE_EXPORTS":
                continue
            if isinstance(node, ast.FunctionDef) and node.name == "__dir__":
                continue
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                node.body = [part for part in node.body if not isinstance(part, ast.For)]
                old_bytes = "\n".join(ast.get_source_segment(source, part) for part in node.body)
                self.assertEqual(hashlib.sha256(old_bytes.encode()).hexdigest(),
                                 "df0ca8121b36735c1764512b2014bc89f8990227b54ae4cb596dfc60fc2b6b69")
            original_nodes.append(node)
        self.assertEqual({node.module: [alias.asname or alias.name for alias in node.names]
                          for node in type_imports}, EAGER_EXPORTS)
        tree.body = original_nodes
        self.assertEqual(hashlib.sha256(repr(ast_snapshot(tree)).encode()).hexdigest(),
                         "b1817713b8db1475023b6db9d18f4ba681d44c4004476c67c335708f2139584f")


if __name__ == "__main__":
    if sys.argv[1:] == ["--import-probe"]:
        with fresh_package() as (package, guard):
            assert Path(package.__file__).resolve() == ROOT / "src/sfora/__init__.py"
            assert not guard.package_attempts, guard.package_attempts
            assert not guard.denied, guard.denied
    else:
        unittest.main()
