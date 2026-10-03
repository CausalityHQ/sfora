#!/usr/bin/env python3
"""Bounded stdlib admissions; native numerical witnesses run only in CPU phase."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
from tempfile import TemporaryDirectory
from contextlib import contextmanager, redirect_stdout
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("train_siglip2_compact_ranking.py")
if PATH.exists():
    spec = importlib.util.spec_from_file_location("compact_test_driver", PATH)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
else:
    driver = SimpleNamespace()


class FreshOriginAuditTests(unittest.TestCase):
    """Genuine private audit, reader and collector over stdlib origin fixtures."""
    @classmethod
    def setUpClass(cls):
        path = PATH.with_name('test_siglip2_nearest_ranking.py')
        spec = importlib.util.spec_from_file_location('compact_origin_fixtures', path)
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    @contextmanager
    def composition(self):
        with TemporaryDirectory() as directory, patch.dict(sys.modules):
            f = self.fixtures.NativeAdmissionFixture(Path(directory))
            # Bind the genuine collector before the real API freezes dependencies.
            f.source = f.module('qualify_siglip2_substrate_cpu.py')
            f.legacy['source_driver'] = f.source
            f.context['guards'][f.source.__file__] = hashlib.sha256(Path(f.source.__file__).read_bytes()).hexdigest()
            f.legacy['prior']['source_driver'] = f.source
            modules, cpu, warm = {}, {'files': {}, 'modules': {}}, {'files': {}, 'modules': {}}
            for name, proof in (('cpu', cpu), ('warm', warm)):
                path = Path(directory) / (name + '.py')
                path.write_bytes(b'# original origin fixture\n')
                alias = 'compact_origin_fixture.' + name
                spec = importlib.util.spec_from_file_location(alias, path)
                modules[alias] = importlib.util.module_from_spec(spec)
                proof['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                proof['modules'][alias] = str(path)
            f.legacy['selected']['packages'] = {'compact_origin_fixture': {'root': directory}}
            f.legacy['selected']['source_cpu']['origins'] = cpu
            f.legacy['warm_record']['origins'] = warm
            f.cpu, f.warm, f.loaded = cpu, warm, modules
            f.observed = {**cpu['files'], **warm['files'], **f.files}
            f.mapped = list(f.files)
            f.readers, f.reads = [], []
            f.original_state = [(m, dict(vars(m))) for m in (f.old, f.fitter, f.original, f.source, f.extract)]
            read_text, open_file = Path.read_text, Path.open

            def maps(path, *args, **kwargs):
                if str(path) == '/proc/self/maps':
                    return '\n'.join('0-1 r--p 0 00:00 0 ' + p for p in f.mapped)
                return read_text(path, *args, **kwargs)

            def observed_open(path, *args, **kwargs):
                if str(path) in f.observed:
                    code = sys._getframe(1).f_code
                    if code is f.extract.sha.__code__:
                        f.reads.append(('origin_sha', str(path)))
                    elif code is f.old.bound_file.__code__:
                        f.reads.append(('duplicate_sha', str(path)))
                return open_file(path, *args, **kwargs)

            def acquire(context):
                api = self.fixtures.driver.native_source_api(context)

                def audit(legacy, *args, **kwargs):
                    f.readers.append(kwargs.get('admission'))
                    return api.audit_origins(legacy, *args, **kwargs)

                return SimpleNamespace(audit_origins=audit)

            f.context.update(nearest=SimpleNamespace(native_source_api=acquire), args=SimpleNamespace(phase='mechanics'))
            with patch.dict(sys.modules, modules), patch.object(Path, 'read_text', maps), patch.object(Path, 'open', observed_open):
                f.api = f.admit()
                yield f
                f.unchanged_originals(self)
                self.assertFalse(any(n.split('.')[0] in driver.NATIVE for n in sys.modules))

    def boundaries(self, f):
        # Execute each actual compact audit expression with the real private API.
        # Whole-module correspondence below protects all surrounding operations.
        result = []
        for fn in ast.parse(PATH.read_text()).body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call) and
                            isinstance(n.func, ast.Attribute) and n.func.attr == 'audit_origins'),
                           key=lambda n: n.lineno)
            for call in calls:
                code = compile(ast.Expression(body=call), str(PATH), 'eval')

                def invoke(code=code):
                    api = f.context['nearest'].native_source_api(f.context)
                    return eval(code, vars(driver), {'context': f.context, 'legacy': f.legacy,
                                'args': f.context['args'], 'api': api})

                result.append((fn.name, invoke))
        self.assertEqual([name for name, _ in result], ['prepare_native', 'gpu_run', 'exit_rehash', 'exit_rehash', 'run'])
        return result

    def test_fresh_original_reader_and_one_genuine_origin_hash_per_boundary(self):
        with self.composition() as f:
            legacy_guards, prior_guards = dict(f.legacy['guards']), dict(f.legacy['prior']['guards'])
            f.api.audit_origins(f.legacy)
            self.assertCountEqual(f.reads, [(kind, p) for p in f.observed
                                           for kind in ('origin_sha', 'duplicate_sha')])
            f.admission.verified.update(f.observed)  # Startup cache cannot qualify a later audit.
            for phase in ('cpu', 'mechanics', 'train'):
                f.context['args'].phase = phase
                for name, invoke in self.boundaries(f):
                    with self.subTest(phase=phase, boundary=name):
                        f.reads.clear()
                        invoke()
                        reader = f.readers[-1]
                        self.assertIs(type(reader), f.original.FlatAdmission)
                        self.assertIsNot(reader, f.admission)
                        self.assertTrue(all(reader is not old for old in f.readers[:-1]))
                        self.assertEqual(reader.verified, set(f.observed))
                        self.assertEqual(reader.entries, {p: (h, Path(p).stat().st_size) for p, h in f.observed.items()})
                        self.assertEqual(reader.json_bytes, {})
                        self.assertCountEqual(f.reads, [('origin_sha', p) for p in f.observed])
                        self.assertEqual(f.legacy['guards'], {**legacy_guards, **f.observed})
                        self.assertEqual(f.legacy['prior']['guards'], {**prior_guards, **f.observed})
                        self.assertEqual(f.legacy['origins']['files'], f.observed)
            self.assertEqual(len(f.readers), 15)

    def test_private_audit_predicates_and_guard_correspondence(self):
        with self.composition() as f:
            unknown = Path(f.root) / 'unknown.so'
            unknown.write_bytes(b'unknown')
            supplemental = next(iter(f.files))
            cases = [
                ('accepted', {'require_exact': True}, None),
                ('initial', {'initial': True}, 'original CPU'),
                ('initial_accepted', {'initial': True}, None),
                ('file_conflict', {}, 'conflicting original'),
                ('module_conflict', {}, 'conflicting original'),
                ('unknown_file', {}, 'unknown or changed'),
                ('unknown_module', {}, 'unknown or changed'),
                ('outside_package', {}, 'loaded native module origin differs'),
                ('legacy_guard', {}, 'conflicting'),
                ('supplement_guard', {}, 'conflicting'),
                ('prior_guard', {}, 'original native origin changed'),
                ('missing_one', {'require_exact': True}, 'exact four'),
                ('cpu_subset', {'require_exact': False}, None),
                ('missing_native', {}, 'missing native'),
            ]
            original_guards = dict(f.legacy['guards'])
            for name, kwargs, error in cases:
                outcomes = []
                for admitted in (False, True):
                    f.legacy['guards'] = dict(original_guards)
                    f.legacy['prior']['guards'] = {}
                    f.legacy['selected']['source_cpu']['origins'] = copy.deepcopy(f.cpu)
                    f.legacy['warm_record']['origins'] = copy.deepcopy(f.warm)
                    f.legacy.pop('origins', None)
                    f.mapped = list(f.files)
                    loaded = dict(f.loaded)
                    if name == 'initial_accepted':
                        f.legacy['selected']['source_cpu']['origins'] = {
                            'files': dict(f.observed), 'modules': {**f.cpu['modules'], **f.warm['modules']}}
                    elif name == 'file_conflict':
                        f.legacy['warm_record']['origins']['files'].update({p: '0' * 64 for p in f.cpu['files']})
                    elif name == 'module_conflict':
                        f.legacy['warm_record']['origins']['modules'].update({n: '/conflict.py' for n in f.cpu['modules']})
                    elif name == 'unknown_file':
                        f.mapped.append(str(unknown))
                    elif name == 'unknown_module':
                        loaded['compact_origin_fixture.unknown'] = next(iter(f.loaded.values()))
                    elif name == 'outside_package':
                        loaded['compact_origin_fixture.unknown'] = SimpleNamespace(__file__=f.source.__file__)
                    elif name == 'legacy_guard':
                        f.legacy['guards'][next(iter(f.cpu['files']))] = '0' * 64
                    elif name == 'supplement_guard':
                        f.legacy['guards'][supplemental] = '0' * 64
                    elif name == 'prior_guard':
                        f.legacy['prior']['guards'][next(iter(f.cpu['files']))] = '0' * 64
                    elif name in ('missing_one', 'cpu_subset', 'missing_native'):
                        f.mapped.remove(supplemental)
                        if name == 'missing_native':
                            alias = 'compact_origin_fixture.supplemental'
                            loaded[alias] = SimpleNamespace(__file__=supplemental)
                            f.legacy['warm_record']['origins']['modules'][alias] = supplemental
                    reader = f.original.FlatAdmission() if admitted else None
                    with self.subTest(case=name, admitted=admitted), patch.dict(sys.modules, loaded):
                        if error is None:
                            f.api.audit_origins(f.legacy, admission=reader, **kwargs)
                        else:
                            with self.assertRaisesRegex(ValueError, error):
                                f.api.audit_origins(f.legacy, admission=reader, **kwargs)
                        outcomes.append((dict(f.legacy['guards']), dict(f.legacy['prior']['guards']),
                                         copy.deepcopy(f.legacy.get('origins'))))
                self.assertEqual(outcomes[0], outcomes[1], name)

    def test_current_bytes_tamper_and_collector_failures_propagate(self):
        with self.composition() as f:
            path = Path(next(iter(f.cpu['files'])))
            saved, stamp = path.read_bytes(), path.stat()
            sentinel = ValueError('origin read failed')
            real_open = Path.open
            for name, invoke in self.boundaries(f):
                with self.subTest(boundary=name):
                    invoke()
                    try:
                        path.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
                        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        self.assertEqual(path.stat().st_size, len(saved))
                        self.assertEqual(path.stat().st_mtime_ns, stamp.st_mtime_ns)
                        with self.assertRaisesRegex(ValueError, 'unknown or changed'):
                            invoke()
                    finally:
                        path.write_bytes(saved)

                    def failed_open(value, *args, **kwargs):
                        if value == path and sys._getframe(1).f_code is f.extract.sha.__code__:
                            raise sentinel
                        return real_open(value, *args, **kwargs)

                    with patch.object(Path, 'open', failed_open), self.assertRaises(ValueError) as caught:
                        invoke()
                    self.assertIs(caught.exception, sentinel)

                    def failed_dependency(value, *args, **kwargs):
                        if str(value) == f.source.__file__:
                            raise sentinel
                        return real_open(value, *args, **kwargs)

                    with patch.object(Path, 'open', failed_dependency), self.assertRaises(ValueError) as caught:
                        invoke()
                    self.assertIs(caught.exception, sentinel)
                    invoke()


class ContractTests(unittest.TestCase):
    def api(self, name):
        self.assertTrue(hasattr(driver, name), "missing bounded trainer API: " + name)
        return getattr(driver, name)

    def launch(self, phase="cpu", arm="control", seed=179061):
        self.api("check_launch")
        unit = {"receipt": {"path": "/unit/receipt.json", "sha256": "a" * 64},
                "log": {"path": "/unit/log", "sha256": "b" * 64}, "unit": "test-unit",
                "invocation_id": "c" * 32, "service_seconds": 1.,
                "native_peak_rss_kib": 10, "both_locks_held": True}
        value = {"schema": driver.AUTHORITY_SCHEMA, "execution_sha256": "d" * 64,
                 "phase": phase, "arm": arm, "seed": seed,
                 "nearest": copy.deepcopy(driver.NEAREST), "fitter": copy.deepcopy(driver.FITTER),
                 "accepted": copy.deepcopy(driver.ACCEPTED), "readout": copy.deepcopy(driver.READOUT),
                 "recipe": copy.deepcopy(driver.RECIPE), "resource_policy": driver.policy(phase),
                 "both_locks_held": True, "native_authority": {"path": "/native.json", "sha256": "e" * 64},
                 "selected_cpu": None if phase == "cpu" else copy.deepcopy(unit),
                 "selected_mechanics": {a: copy.deepcopy(unit) for a in driver.ARMS} if phase == "train" else None}
        args = SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256="d" * 64)
        return value, args

    def test_fixed_launch_and_seed_prerequisites(self):
        check = self.api("check_launch")
        for phase, seed in [("cpu", 179061), ("mechanics", 179061), ("train", 179069)]:
            value, args = self.launch(phase, seed=seed)
            check(value, args)
            for key, bad in [("seed", 179070), ("recipe", {}), ("nearest", {}),
                             ("readout", {}), ("both_locks_held", False)]:
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    check({**value, key: bad}, args)
            with self.assertRaises(ValueError):
                check({**value, "extra": True}, args)
        for phase, arm, seed in [("cpu", "candidate", 179061), ("cpu", "control", 179069),
                                 ("mechanics", "control", 179069)]:
            value, args = self.launch(phase, arm, seed)
            with self.assertRaises(ValueError):
                check(value, args)
        value, args = self.launch("train")
        for missing in [None, {}, {"candidate": value["selected_mechanics"]["candidate"]}]:
            with self.assertRaises((ValueError, TypeError, AttributeError)):
                check({**value, "selected_mechanics": missing}, args)

    def test_current_file_bytes_symlink_fifo_and_restored_mtime(self):
        bound = self.api("bound_file")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "member"
            path.write_bytes(b"accepted bytes")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            guards = {}
            self.assertEqual(bound(guards, path, sha), path)
            stamp = path.stat()
            path.write_bytes(b"modified bytes")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.assertRaises(ValueError):
                bound(guards, path, sha)
            link = root / "link"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                bound({}, link, hashlib.sha256(path.read_bytes()).hexdigest())
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                bound({}, fifo, sha)
            with self.assertRaises(ValueError):
                bound({}, Path("relative"), sha)

    def test_strict_json_and_exact_two_file_closure(self):
        parse = self.api("strict_json")
        closure = self.api("closure")
        for raw in ["{\"a\":1,\"a\":2}", "{\"a\":NaN}"]:
            with self.assertRaises(ValueError):
                parse(raw)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            code = {}
            for name in driver.FILES:
                (root / name).write_text("# tiny fixture\n")
                code[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            manifest = root / "execution.json"
            manifest.write_text(json.dumps(code))
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(closure(root, sha, driver.FILES, {}), code)
            for names in [set(), {"../outside"}, driver.FILES | {"third.py"}]:
                with self.assertRaises(ValueError):
                    closure(root, sha, names, {})
            (root / next(iter(driver.FILES))).write_text("# replaced\n")
            with self.assertRaises(ValueError):
                closure(root, sha, driver.FILES, {})

    def test_global_both_view_denominators(self):
        denominators = self.api("loss_denominators")
        self.assertEqual(denominators(25), (128, 2.5))
        self.assertEqual(denominators(0), (128, None))
        for invalid in [-1, 65, True, 2.5]:
            with self.assertRaises(ValueError):
                denominators(invalid)
        # Independent full-batch scalar oracle, including singleton-only micros.
        valid_groups = [16, 8, 0, 1] * 2
        hinge_groups = [[.1] * n for n in valid_groups]
        batch, rank = denominators(25)
        partial = math.fsum(math.fsum(g) / rank for g in hinge_groups)
        self.assertAlmostEqual(partial, math.fsum(sum(hinge_groups, [])) / (.05 * 50))
        self.assertAlmostEqual(math.fsum([16 / (batch * 3)] * 8), 1 / 3)

    def test_original_miner_ties_singletons_and_nonfinite(self):
        self.api("NEAREST")
        source = PATH.with_name("train_siglip2_nearest_ranking.py")
        spec = importlib.util.spec_from_file_location("compact_mining_oracle", source)
        nearest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nearest)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), driver.NEAREST["code"][source.name])
        choose = nearest.select_nearest
        self.assertEqual(choose([1., .5, .5, .5], [0, 0, 0, 1], [9, 8, 2, 4], 0), (2, 3))
        self.assertEqual(choose([.7, .7, .7], [0, 1, 2], [9, 2, 5], 0), (-1, 1))
        for scores, rows in [([1., float("nan")], [0, 1]), ([1., 2.], [1, 1])]:
            with self.assertRaises(ValueError):
                choose(scores, [0, 1], rows, 0)

    def test_complete_unit_and_resource_policy(self):
        check = self.api("check_unit")
        value, _ = self.launch("train")
        unit = value["selected_cpu"]
        check(unit)
        for key, bad in [("both_locks_held", False), ("service_seconds", float("inf")),
                         ("invocation_id", "unknown"), ("receipt", {"path": "/a", "sha256": "guess"})]:
            with self.assertRaises(ValueError):
                check({**unit, key: bad})
        self.assertEqual(driver.policy("cpu")["seconds"], 500)
        self.assertEqual(driver.policy("train")["seconds"], 300)
        self.assertEqual(driver.policy("train")["host_bytes"], 8 * 1024**3)
        with self.assertRaises(ValueError):
            driver.policy("quality")

    def test_updated_A_current_bytes_binding(self):
        own = self.api("own_A")
        # Metadata stand-in: no Torch import. A noninitial update can still be
        # changed through a .data-like alias without a version/counter change.
        context = {"nearest": SimpleNamespace(fingerprint=lambda c, v, **kw:
                   hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest())}
        state = {"A": {"bytes": [1., 2.]}, "counter": 0}
        own(context, state, admit=True)
        state["A"]["bytes"][0] = 3.
        state["counter"] = 1
        own(context, state, advanced=True)
        own(context, state)
        state["A"]["bytes"][0] = 4.
        with self.assertRaises(ValueError):
            own(context, state)
        with self.assertRaises(ValueError):
            own(context, state, advanced=True)
        state["A"]["bytes"][0] = 3.
        own(context, state)
        state["counter"] = 2
        with self.assertRaises(ValueError):
            own(context, state)

    def test_terminal_rejects_partial_false_and_nonfinite_cpu_proof(self):
        check = self.api("check_terminal_record")
        launch, _ = self.launch()
        source, flags = {"source": "fixture"}, {"flags": "fixture"}
        ident = {"method": driver.method(launch), "source": source, "arm": "control", "seed": 179061,
                 "device": "cpu", "parameter_names": ["A"], "parameter_shapes": [[128, 160]],
                 "numerical_flags": flags}
        record = {"schema": driver.SCHEMA, "phase": "cpu", "arm": "control", "seed": 179061,
                  "launch": launch, "source": source, "identity": ident, "code": {n: "a" * 64 for n in driver.FILES},
                  "authority": {"path": "/authority.json", "sha256": "b" * 64}, "authority_sha256": "b" * 64,
                  "resource_policy": driver.policy("cpu"), "optimizer_members": 1, "trainable_scalars": 20480,
                  "frozen_vision_members": 448, "quality_read": False, "total_training_core_seconds": 1.,
                  "wall_seconds": 2., "process_peak_rss_kib": 100, "numerical_flags": flags,
                  "completed_step": 0, "cuda_initialized": False, "peak_cuda_allocated_bytes": 0,
                  "invocation": {"optimize": 0, "cuda_visible_devices": ""},
                  "gradients": [{"seed": seed, "mse": 1., "rank": .1, "active": 1, "K": 64,
                                "control_gradient_norm": 1., "ranking_gradient_norm": .2,
                                "candidate_minus_control_equals_rank": True,
                                "micro16_global_reduction_exact": True} for seed in driver.SEEDS],
                  "checkpoint": {"path": "/unit/initializer.pt", "sha256": "c" * 64},
                  "bundle": {"path": "/unit/bundle/bundle.json", "sha256": "d" * 64},
                  "inference_state_sha256": "e" * 64,
                  "input_guards": {"/unit/initializer.pt": "c" * 64, "/unit/bundle/bundle.json": "d" * 64}}
        required = ("pass", "strict_reload_exact", "exit_rehash_pass", "sequential_model_ownership",
                    "forward_oracle_exact", "native_training_inference_exact", "inference_artifact_independent",
                    "bundle_original_dependencies_denied", "both_locks_held_in_parent_authority",
                    "initial_arm_parity", "cpu_serialization_exact", "bypass_version_tamper_rejected",
                    "malformed_state_rejected", "native_role_mutation_rejected", "native_loss_reduction_exact")
        record.update({k: True for k in required})
        check(record, launch, "cpu", "control", 179061)
        for key in required:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check({**record, key: False}, launch, "cpu", "control", 179061)
        for bad in [[], record["gradients"][:1]]:
            with self.assertRaises(ValueError):
                check({**record, "gradients": bad}, launch, "cpu", "control", 179061)
        for key in ("mse", "control_gradient_norm", "ranking_gradient_norm"):
            bad = copy.deepcopy(record)
            bad["gradients"][0][key] = float("inf")
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(bad, launch, "cpu", "control", 179061)

    def test_bundle_owned_files_and_forbidden_dependencies(self):
        admit = self.api("admit_bundle")
        self.api("deny_training_dependencies")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            code = {}
            for name in driver.FILES | driver.SERVING_FILES | {"joint_relational_compaction.py"}:
                source = PATH.parent.parent / "src/sfora" / name if name == "joint_relational_compaction.py" else PATH.with_name(name)
                shutil.copyfile(source, bundle / name)
                code[name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            files = {}
            for name in ("endpoint.pt", "vision.pt", "processor.json"):
                (bundle / name).write_bytes(b"tiny admission fixture")
                files[name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            packages = {n: {"root": str(root / "installed" / n)} for n in driver.NATIVE - {"sfora"}}
            constructor = Path(packages["transformers"]["root"]) / "modeling.py"
            constructor.parent.mkdir(parents=True)
            constructor.write_text("# installed fixture")
            environment = {"packages": packages, "files": {str(constructor): hashlib.sha256(constructor.read_bytes()).hexdigest()},
                           "native_files": {}, "vision_constructor": str(constructor)}
            value = {"schema": driver.BUNDLE_SCHEMA, "code": code, "files": files,
                     "environment": environment, "vision_inventory": [], "endpoint_state_sha256": "c" * 64}
            def publish(v):
                (bundle / "bundle.json").write_text(json.dumps(v))
                return hashlib.sha256((bundle / "bundle.json").read_bytes()).hexdigest()
            sha = publish(value)
            self.assertEqual(admit(bundle, sha)[0], value)
            for bad in [{**value, "teacher": {}}, {**value, "code": {**code, "third.py": "a" * 64}},
                        {**value, "files": {"endpoint.pt": files["endpoint.pt"]}}]:
                with self.assertRaises(ValueError):
                    admit(bundle, publish(bad))
            sha = publish(value)
            (bundle / "vision.pt").unlink()
            (bundle / "vision.pt").symlink_to(constructor)
            with self.assertRaises(ValueError):
                admit(bundle, sha)
            warm = root / "warm.pt"
            warm.write_bytes(b"forbidden original state")
            # RECORD is outside every package root, but original native
            # ownership admission authenticated this exact metadata inventory.
            site = root / "installed"
            record = site / "aiohappyeyeballs-2.6.2.dist-info" / "RECORD"
            record.parent.mkdir()
            record.write_text("aiohappyeyeballs/__init__.py,,\n")
            owner_record = site / "nvidia_cudnn_cu13-9.20.0.48.dist-info" / "RECORD"
            owner_record.parent.mkdir()
            owner_record.write_text("nvidia/cudnn/lib/libcudnn.so,,\n")
            metadata = [owner_record.parent / name for name in ("METADATA", "WHEEL")]
            for path in metadata:
                path.write_text("qualified vendor metadata\n")
            # Neither a lookalike nor a training payload elsewhere in the
            # installed tree becomes runtime metadata by its path alone.
            lookalike = site / "training.dist-info" / "RECORD"
            lookalike.parent.mkdir()
            lookalike.write_text("original training input\n")
            training = site / "optimizer.pt"
            training.write_bytes(b"original optimizer")
            runtime = [record, owner_record, *metadata]
            guarded = [warm, lookalike, training, *runtime]
            guards = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded}
            proof = {"authority": {"installed_site_root": str(site)},
                     "installed_record_ownership": {"records": [str(record), str(owner_record)],
                         "owners": {str(site / "nvidia/cudnn/lib/libcudnn.so"): [str(owner_record)]}},
                     "input_guards": {str(p): {"sha256": guards[str(p)], "size_bytes": p.stat().st_size}
                                      for p in runtime}}
            proof_path = root / "native-proof.json"
            proof_path.write_text(json.dumps(proof))
            proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
            authority_path = root / "native-authority.json"
            authority_path.write_text(json.dumps({"proof": {"path": str(proof_path), "sha256": proof_sha}}))
            context = {"guards": guards,
                       "nearest": SimpleNamespace(NATIVE_PROOF_PINS={"proof": proof_sha}),
                       "launch": {"native_authority": {"path": str(authority_path),
                           "sha256": hashlib.sha256(authority_path.read_bytes()).hexdigest()}}}
            with driver.deny_training_dependencies(context, bundle, environment):
                self.assertEqual(record.read_text(), "aiohappyeyeballs/__init__.py,,\n")
                for path in runtime:
                    self.assertTrue(path.read_bytes())
                for path in (warm, lookalike, training, authority_path, proof_path):
                    with self.subTest(denied=path), self.assertRaises(ValueError):
                        path.read_bytes()
                self.assertEqual(constructor.read_text(), "# installed fixture")
            self.assertEqual(warm.read_bytes(), b"forbidden original state")
            for path in (record, proof_path, authority_path):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.subTest(tamper=path), self.assertRaises(ValueError):
                    with driver.deny_training_dependencies(context, bundle, environment):
                        pass
                path.write_bytes(original)
            for bad_guard in (None, "0" * 64):
                broken = {**context, "guards": dict(context["guards"])}
                if bad_guard is None:
                    broken["guards"].pop(str(record))
                else:
                    broken["guards"][str(record)] = bad_guard
                with self.subTest(guard=bad_guard), self.assertRaises(ValueError):
                    with driver.deny_training_dependencies(broken, bundle, environment):
                        pass
            broken = {**context, "nearest": SimpleNamespace(NATIVE_PROOF_PINS={"proof": "0" * 64})}
            with self.assertRaises(ValueError):
                with driver.deny_training_dependencies(broken, bundle, environment):
                    pass
            # A symlink/parent alias cannot inherit an authenticated allowance.
            original = record.read_bytes()
            record.unlink()
            record.symlink_to(lookalike)
            with self.assertRaises(ValueError):
                with driver.deny_training_dependencies(context, bundle, environment):
                    pass
            record.unlink()
            record.write_bytes(original)
            wrong_environment = copy.deepcopy(environment)
            wrong_environment["packages"]["torch"]["root"] = str(root / "other" / "torch")
            with self.assertRaises(ValueError):
                with driver.deny_training_dependencies(context, bundle, wrong_environment):
                    pass

    def test_completion_timing_preserves_checks(self):
        # Whole-module AST at fa2a8bb9; no general call/statement normalization.
        baseline = "d33ab2444ad0b66064c9f038524f4738544b6f03be932ff9df0bb897cbfd759f"
        reader_calls = [
            ('prepare_native', "context['nearest'].native_source_api(context).audit_origins(legacy)",
             "legacy['original'].FlatAdmission()", 1),
            ('gpu_run', "api.audit_origins(context['legacy'], require_exact=True)",
             "context['legacy']['original'].FlatAdmission()", 1),
            ('exit_rehash', "api.audit_origins(context['legacy'], require_exact=context['args'].phase != 'cpu')",
             "context['legacy']['original'].FlatAdmission()", 2),
            ('run', "api.audit_origins(legacy, require_exact=args.phase != 'cpu')",
             "legacy['original'].FlatAdmission()", 1),
        ]
        allowed_readers = {}
        for fn, text, reader, count in reader_calls:
            original = ast.parse(text, mode='eval').body
            changed = copy.deepcopy(original)
            changed.keywords.insert(0, ast.keyword(arg='admission', value=ast.parse(reader, mode='eval').body))
            allowed_readers[(fn, ast.dump(changed))] = (original, count)
        restored_readers = {}

        class RestoreAuditReaders(ast.NodeTransformer):
            function = None

            def visit_FunctionDef(self, node):
                prior, self.function = self.function, node.name
                result = self.generic_visit(node)
                self.function = prior
                return result

            def visit_Call(self, node):
                key = (self.function, ast.dump(node))
                if key in allowed_readers:
                    restored_readers[key] = restored_readers.get(key, 0) + 1
                    return copy.deepcopy(allowed_readers[key][0])
                return self.generic_visit(node)

        original_calls = RestoreAuditReaders().visit(ast.parse(PATH.read_text()))
        self.assertEqual(restored_readers, {key: count for key, (_, count) in allowed_readers.items()})
        phases = {
            "cpu_witnesses": {"cpu_bundle_qualification"},
            "gpu_run": {"gpu_bundle_qualification", "post_calibration_api_authentication",
                        "post_calibration_origin_audit"},
            "qualify_bundle": {"bundle_image_loading", "bundle_loader_authentication",
                               "bundle_dependency_denial", "bundle_loader", "bundle_native_forward",
                               "bundle_oracle", "bundle_release"},
            "run": {"post_run_api_authentication", "post_run_origin_audit", "origin_guard_promotion"},
            "exit_rehash": {"post_exit_api_authentication", "post_exit_origin_audit"},
        }
        splits = {"post_calibration_api_authentication": "post_calibration_origin_audit",
                  "post_run_api_authentication": "post_run_origin_audit",
                  "post_exit_api_authentication": "post_exit_origin_audit"}
        case = self
        seen, acquisitions = set(), set()

        class StripTimers(ast.NodeTransformer):
            function = None

            def visit_FunctionDef(self, node):
                prior, self.function = self.function, node.name
                result = self.generic_visit(node)
                self.function = prior
                return result

            def visit_With(self, node):
                call = node.items[0].context_expr
                name = (call.args[1].value if isinstance(call, ast.Call) and
                        isinstance(call.func, ast.Name) and call.func.id == "timed" and
                        len(call.args) == 2 and isinstance(call.args[1], ast.Constant) else None)
                if name not in set().union(*phases.values()):
                    return self.generic_visit(node)
                case.assertIn(name, phases.get(self.function, set()))
                case.assertNotIn((self.function, name), seen)
                seen.add((self.function, name))
                case.assertEqual(len(node.items), 1)
                case.assertIsNone(node.items[0].optional_vars)
                case.assertEqual(ast.dump(call.args[0]), ast.dump(ast.Name(id="context", ctx=ast.Load())))
                case.assertEqual(call.keywords, [])
                if name in splits:
                    case.assertEqual(len(node.body), 1)
                    assignment = node.body[0]
                    expected = ast.parse("api = context['nearest'].native_source_api(context)").body[0]
                    case.assertEqual(ast.dump(assignment), ast.dump(expected))
                    acquisitions.add(id(assignment))
                self.generic_visit(node)
                return node.body

        restored = StripTimers().visit(original_calls)
        self.assertEqual(seen, {(fn, name) for fn, names in phases.items() for name in names})
        reversed_splits = []

        def reverse_splits(node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    reverse_splits(value)
                elif isinstance(value, list):
                    result = []
                    index = 0
                    while index < len(value):
                        child = value[index]
                        if isinstance(child, ast.AST):
                            reverse_splits(child)
                        if id(child) in acquisitions:
                            audit = value[index + 1]
                            case.assertIsInstance(audit, ast.Expr)
                            case.assertEqual(ast.dump(audit.value.func),
                                ast.dump(ast.parse("api.audit_origins").body[0].value))
                            audit.value.func.value = child.value
                            reversed_splits.append(child)
                            result.append(audit)
                            index += 2
                        else:
                            result.append(child)
                            index += 1
                    setattr(node, field, result)

        reverse_splits(restored)
        self.assertEqual(len(reversed_splits), 3)
        self.assertEqual(hashlib.sha256(ast.dump(restored).encode()).hexdigest(), baseline)
        original_exit = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == "exit_rehash")
        timed_exit = next(n for n in ast.parse(PATH.read_text()).body
                          if isinstance(n, ast.FunctionDef) and n.name == "exit_rehash")

        def compile_exit(node):
            namespace = dict(vars(driver))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])),
                         str(PATH), "exec"), namespace)
            return namespace["exit_rehash"]

        with TemporaryDirectory() as directory:
            root = Path(directory)

            def fixture(name):
                path = root / name
                path.write_bytes(name.encode())
                return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

            guards = dict(fixture(name) for name in ("first-input", "second-input", "last-input"))
            supplemental = fixture("supplemental-library")
            origin = fixture("observed-origin")
            fitter = fixture("fitter-union")
            helper = fixture("helper")

            def make_closure(name, names):
                path = root / name
                path.mkdir()
                code = {}
                for member in sorted(names):
                    (path / member).write_bytes(member.encode())
                    code[member] = hashlib.sha256((path / member).read_bytes()).hexdigest()
                execution = path / "execution.json"
                execution.write_text(json.dumps(code))
                return {"root": str(path), "code": code,
                        "execution_sha256": hashlib.sha256(execution.read_bytes()).hexdigest()}

            own = make_closure("own", driver.FILES)
            nearest = make_closure("nearest", driver.NEAREST["code"])
            sentinel = ValueError("final native failure")
            real_bound = driver.bound_file

            def exercise(function, phase="mechanics", fail=None):
                trace, captures = [], io.StringIO()
                acquisitions_count = 0

                def bound(values, path, sha):
                    trace.append(("hash", str(path), sha))
                    return real_bound(values, path, sha)

                def authenticate():
                    trace.append(("authenticate",))
                    bound({}, *supplemental)

                def audit(legacy, *, admission=None, require_exact):
                    self.assertIs(legacy, context["legacy"])
                    trace.append(("audit", require_exact))
                    authenticate()
                    bound({}, *origin)
                    if acquisitions_count == 2 and fail == "audit":
                        raise sentinel

                def exit_fitter(value):
                    self.assertIs(value, context["fit_context"])
                    trace.append(("fitter_exit",))
                    authenticate()
                    bound({}, *fitter)

                def acquire(value):
                    nonlocal acquisitions_count
                    self.assertIs(value, context)
                    acquisitions_count += 1
                    trace.append(("api",))
                    authenticate()
                    if acquisitions_count == 2 and fail == "api":
                        raise sentinel
                    return SimpleNamespace(audit_origins=audit, exit_rehash=exit_fitter)

                def helper_guard(value):
                    self.assertIs(value, context)
                    trace.append(("helper",))
                    bound({}, *helper)

                bindings = {**vars(driver), "bound_file": bound}
                bindings["read_json"] = FunctionType(driver.read_json.__code__, bindings)
                real_closure = FunctionType(driver.closure.__code__, bindings)

                def closure(path, sha, names, values):
                    trace.append(("closure", str(path)))
                    return real_closure(path, sha, names, values)

                context = {"root": Path(own["root"]), "code": own["code"], "guards": dict(guards),
                           "args": SimpleNamespace(phase=phase, execution_sha256=own["execution_sha256"]),
                           "legacy": {"original": SimpleNamespace(FlatAdmission=object)},
                           "fit_context": {}, "started": driver.time.perf_counter(),
                           "phase_seconds": {}, "nearest": SimpleNamespace(native_source_api=acquire,
                               require_no_model=lambda value: trace.append(("no_model",)))}
                error = None
                with patch.dict(function.__globals__, {"bound_file": bound, "closure": closure,
                        "helper_guard": helper_guard, "NEAREST": nearest}), redirect_stdout(captures):
                    try:
                        function(context)
                    except ValueError as caught:
                        error = caught
                events = [json.loads(line) for line in captures.getvalue().splitlines()]
                return trace, error, events, context["phase_seconds"]

            reference = compile_exit(original_exit)
            for phase in ("cpu", "mechanics", "train"):
                expected, error, _, _ = exercise(reference, phase)
                self.assertIsNone(error)
                actual, error, events, seconds = exercise(driver.exit_rehash, phase)
                self.assertIsNone(error)
                self.assertEqual(actual, expected)
                self.assertEqual([row for row in actual if row[0] == "audit"],
                                 [("audit", phase != "cpu")] * 2)
                self.assertEqual([row[1] for row in actual if row[0] == "hash" and row[1] in guards], list(guards))
                self.assertEqual([row for row in actual if row[0] == "api"], [("api",)] * 2)
                self.assertEqual([row for row in actual if row[0] == "authenticate"], [("authenticate",)] * 5)
                self.assertEqual([(e["phase"], e["boundary"]) for e in events], [
                    ("source_exit_rehash", "begin"), ("source_exit_rehash", "end"),
                    ("own_exit_rehash", "begin"), ("own_exit_rehash", "end"),
                    ("post_exit_api_authentication", "begin"), ("post_exit_api_authentication", "end"),
                    ("post_exit_origin_audit", "begin"), ("post_exit_origin_audit", "end")])
                self.assertEqual(set(seconds), {"source_exit_rehash", "own_exit_rehash",
                                               "post_exit_api_authentication", "post_exit_origin_audit"})
            for failure in ("api", "audit"):
                expected, error, _, _ = exercise(reference, fail=failure)
                self.assertIs(error, sentinel)
                actual, error, _, _ = exercise(driver.exit_rehash, fail=failure)
                self.assertIs(error, sentinel)
                self.assertEqual(actual, expected)
            for path in guards:
                member = Path(path)
                saved, stamp = member.read_bytes(), member.stat()
                try:
                    member.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
                    os.utime(member, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    self.assertEqual(member.stat().st_mtime_ns, stamp.st_mtime_ns)
                    for function in (reference, driver.exit_rehash):
                        with self.subTest(tamper=path):
                            self.assertIsInstance(exercise(function)[1], ValueError)
                finally:
                    member.write_bytes(saved)

            # These mutants must fail the same trace/failure contract.
            omitted = copy.deepcopy(timed_exit)
            own_timer = next(n for n in omitted.body if isinstance(n, ast.With) and
                             n.items[0].context_expr.args[1].value == "own_exit_rehash")
            own_timer.body[0].body = [ast.Pass()]
            earlier = copy.deepcopy(timed_exit)
            earlier.body[1:1] = earlier.body[-2:]
            del earlier.body[-2:]
            swallowed = copy.deepcopy(timed_exit)
            swallowed.body[-1] = ast.Try(body=[swallowed.body[-1]], handlers=[ast.ExceptHandler(
                type=ast.Name(id="ValueError", ctx=ast.Load()), body=[ast.Pass()])], orelse=[], finalbody=[])
            expected = exercise(reference)[0]
            for name, mutant in (("omit_hash", omitted), ("earlier_final_audit", earlier)):
                with self.subTest(mutant=name), self.assertRaises(AssertionError):
                    self.assertEqual(exercise(compile_exit(mutant))[0], expected)
            with self.subTest(mutant="swallowed_failure"), self.assertRaises(AssertionError):
                self.assertIs(exercise(compile_exit(swallowed), fail="audit")[1], sentinel)

    def test_no_native_import_help_and_optimized_rejection(self):
        self.api("parser")
        self.assertFalse(any(n.split(".")[0] in driver.NATIVE for n in sys.modules))
        self.assertEqual(driver.parser().parse_args(["--execution-sha256", "a" * 64,
                         "--authority", "/a", "--authority-sha256", "b" * 64,
                         "--phase", "train", "--arm", "candidate", "--seed", "179069",
                         "--output", "/out"]).seed, 179069)
        for flags in [[], ["-O"], ["-OO"]]:
            result = subprocess.run([sys.executable, *flags, str(PATH), "--help"], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0 if not flags else 1)
            if flags:
                self.assertIn(b"optimized mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
