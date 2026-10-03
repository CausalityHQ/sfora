#!/usr/bin/env python3
"""Bounded stdlib source/stand-in falsifiers; no Torch/native qualification."""
import ast
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import weakref
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('train_siglip2_nearest_ranking.py')
spec = importlib.util.spec_from_file_location('nearest_test_driver', PATH)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Tensor:
    """Small scalar/list arithmetic stand-in; rejects every negative gather."""
    dtype = 'float32'
    device = SimpleNamespace(type='cpu')

    def __init__(self, value):
        self.value = list(value) if isinstance(value, tuple) else value

    def item(self):
        return self.value

    def tolist(self):
        return self.value

    def cpu(self):
        return self

    def detach(self):
        return self

    def binary(self, other, op):
        other = other.value if isinstance(other, Tensor) else other
        def apply(a, b):
            if isinstance(a, list):
                return [apply(x, y) for x, y in zip(a, b, strict=True)] if isinstance(b, list) else [apply(x, b) for x in a]
            if isinstance(b, list):
                return [apply(a, y) for y in b]
            return op(a, b)
        return Tensor(apply(self.value, other))

    def __add__(self, v):
        return self.binary(v, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, v):
        return self.binary(v, lambda a, b: a - b)

    def __mul__(self, v):
        return self.binary(v, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, v):
        return self.binary(v, lambda a, b: a / b)

    def __ge__(self, v):
        return self.binary(v, lambda a, b: a >= b)

    def __gt__(self, v):
        return self.binary(v, lambda a, b: a > b)

    def square(self):
        return self * self

    def sum(self, axis=None):
        def total(v):
            return sum(total(x) for x in v) if isinstance(v, list) else v
        return Tensor([total(row) for row in self.value] if axis == 1 else total(self.value))

    def all(self):
        def all_items(v):
            return all(all_items(x) for x in v) if isinstance(v, list) else bool(v)
        return Tensor(all_items(self.value))

    def any(self):
        return Tensor(any(self.value))

    def __int__(self):
        return int(self.value)

    def __getitem__(self, key):
        key = key.value if isinstance(key, Tensor) else key
        if isinstance(key, list):
            if all(type(i) is bool for i in key):
                return Tensor([v for v, keep in zip(self.value, key, strict=True) if keep])
            if any(i < 0 for i in key):
                raise AssertionError('invalid positive was indexed')
            return Tensor([self.value[i] for i in key])
        if type(key) is int and key < 0:
            raise AssertionError('invalid positive was indexed')
        return Tensor(self.value[key])

    @property
    def T(self):
        return Tensor([list(row) for row in zip(*self.value, strict=True)])

    def __matmul__(self, other):
        return Tensor([[sum(a * b for a, b in zip(row, col, strict=True)) for col in zip(*other.value, strict=True)]
                       for row in self.value])


def fake_torch():
    torch = ModuleType('torch')
    nn, functional = ModuleType('torch.nn'), ModuleType('torch.nn.functional')
    torch.float32 = 'float32'
    torch.tensor = lambda v, **kwargs: Tensor(v)
    torch.no_grad = lambda: nullcontext()
    torch.autocast = lambda *args, **kwargs: nullcontext()
    torch.isfinite = lambda v: v.binary(0, lambda a, _: math.isfinite(a))
    functional.normalize = lambda v, dim: Tensor([[x / math.sqrt(sum(y * y for y in row)) for x in row] for row in v.value])
    functional.relu = lambda v: v.binary(0, lambda a, _: max(0., a))
    nn.functional = functional
    torch.nn = nn
    return {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional}


class RestoreFixture:
    """Tiny metadata-only native boundary doubles; run real fresh/restore/identity.

    No storage is allocated for declared native parameter shapes. Transfer replaces
    parameter objects; Adam preserves CPU step references as the native loader does.
    """
    class Value:
        dtype = 'torch.float32'
        requires_grad, grad_fn = False, None

        def __init__(self, value, shape=(), device='cpu'):
            self.value, self.shape = copy.deepcopy(value), shape
            self.device = SimpleNamespace(type=device)

        def to(self, device, copy=False):
            device = device if isinstance(device, str) else device.type
            if copy or device != self.device.type:
                result = self.clone()
                result.device = SimpleNamespace(type=device)
                return result
            return self

        def clone(self):
            return copy.deepcopy(self)

        def detach(self):
            return self

        def tolist(self):
            return self.value

        def requires_grad_(self, enabled):
            self.requires_grad = enabled
            return self

        def numel(self):
            return math.prod(self.shape)

        def copy_(self, other):
            self.value = copy.deepcopy(other.value)

    def __init__(self, case, device, strict_failure=False):
        self.case, self.device, self.strict_failure = case, device, strict_failure
        self.events, self.old_bindings = [], []
        fixture = self

        class Buffer(self.Value):
            def copy_(self, other):
                fixture.events.append(('buffer_copy', self.device.type))
                super().copy_(other)

        class Model:
            def __init__(self):
                self.parameters = {**{f'frozen{i}': fixture.Value(0., (1,)) for i in range(444)},
                    **{n: fixture.Value(0., shape) for n, shape in zip(driver.NAMES, driver.SHAPES, strict=True)}}
                self.buffers = {'positions': Buffer([0, 1], (2,))}
                self.config = SimpleNamespace(to_dict=lambda: {'source': 'pinned'})

            def requires_grad_(self, enabled):
                for value in self.parameters.values():
                    value.requires_grad_(enabled)

            def named_parameters(self):
                return self.parameters.items()

            def named_buffers(self):
                return self.buffers.items()

            def state_dict(self):
                return self.parameters

            def eval(self):
                self.training = False

            def to(self, device):
                fixture.events.append(('transfer', device))
                if device == 'cuda':
                    fixture.old_bindings.append([weakref.ref(p) for p in self.parameters.values()])
                    self.parameters = {n: p.to(device, copy=True) for n, p in self.parameters.items()}
                    self.buffers = {n: p.to(device, copy=True) for n, p in self.buffers.items()}
                return self

        class Head:
            def __init__(self, tensors):
                self.tensors = copy.deepcopy(tensors)

            def requires_grad_(self, enabled):
                return self

            def to(self, device):
                self.tensors = {n: p.to(device) for n, p in self.tensors.items()}
                return self

            def train(self):
                return self

            def state_dict(self):
                return self.tensors

        class Adam:
            def __init__(self, params, **defaults):
                self.defaults, self.state = defaults, {}
                self.param_groups = [{'params': params, **defaults}]
                fixture.events.append(('adam', params[0].device.type))

            def state_dict(self):
                return {'state': {i: self.state[p] for i, p in enumerate(self.param_groups[0]['params']) if p in self.state},
                        'param_groups': [{**self.defaults, 'params': list(range(4))}]}

            def load_state_dict(self, saved):
                fixture.events.append(('moments', fixture.device))
                self.state = {self.param_groups[0]['params'][i]: member for i, member in saved['state'].items()}

        class Scaler:
            def __init__(self, device, init_scale):
                self.saved = {'scale': init_scale, '_growth_tracker': 0} if device == 'cuda' else {}

            def state_dict(self):
                return copy.deepcopy(self.saved)

            def load_state_dict(self, saved):
                fixture.events.append(('scaler', fixture.device))
                self.saved = copy.deepcopy(saved)

        class Pages:
            def __init__(self, stream):
                pass

            def consume(self, value):
                case.assertEqual(value.device.type, 'cpu')

            def copy(self, value, device='cpu'):
                result = value.to(device, copy=True)
                self.consume(value)
                case.assertIsNot(result, value)
                return result

        def encode(value):
            if isinstance(value, self.Value):
                return ['Tensor', value.dtype, value.shape, value.value]
            if isinstance(value, dict):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            return value

        def fingerprint(value, consumed=None):
            if consumed:
                def visit(v):
                    if isinstance(v, self.Value):
                        consumed(v)
                    elif isinstance(v, dict):
                        for child in v.values():
                            visit(child)
                    elif isinstance(v, (tuple, list)):
                        for child in v:
                            visit(child)
                visit(value)
            return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()

        def load_vision(model, vision, pages):
            current = next(iter(model.parameters.values())).device.type
            self.events.append(('strict_load', current))
            if strict_failure:
                raise ValueError('strict fixture load failed')
            case.assertEqual(model.parameters.keys(), vision.keys())
            for name, value in model.parameters.items():
                value.copy_(vision[name])
                pages.consume(vision[name])

        def fresh_source(prior):
            self.events.append(('source', 'cpu'))
            return Model(), SimpleNamespace(), ['pinned roles']

        launch = {k: {} for k in ('execution_sha256', 'fitter', 'accepted', 'recipe', 'seed')}
        initial = {k: {} for k in driver.STATIC_KEYS}
        initial.update(provenance={'encoder': {'source_proof': {'runtime': {'source': 'pinned'}}}},
                       head={'weight': self.Value(3.)}, config={'source': 'pinned'},
                       buffers={'positions': self.Value([0, 1], (2,))}, target=self.Value([0]),
                       classifier=self.Value(4.), A=self.Value(5.), means={'fixed': self.Value(6.)},
                       original_rows=self.Value([4]), teachers={'counts': self.Value([1])})
        source = SimpleNamespace(fresh_source=fresh_source,
            model_facts=lambda *args: {'source': 'pinned'}, numerical_flags=lambda: {'pinned': True})
        self.context = {'initial': initial, 'source': {'pinned': True}, 'launch': launch,
            'flags': {'pinned': True}, 'guards': {}, 'started': time.perf_counter(), 'phase_seconds': {},
            'old': SimpleNamespace(clone_tree=lambda v, device='cpu': self.clone_tree(v, device)),
            'legacy': {'source_driver': source, 'prior': {},
                'selected': {'packages': {}, 'cached': SimpleNamespace(head_from=lambda arm, tensors: Head(tensors))},
                'original': SimpleNamespace(CheckpointPages=Pages, load_vision=load_vision,
                    fingerprint=fingerprint, runtime=lambda *args: {'source': 'pinned'})}}
        self.modules = fake_torch()
        torch = self.modules['torch']
        torch.optim = SimpleNamespace(AdamW=Adam)
        torch.amp = SimpleNamespace(GradScaler=Scaler)
        torch.random = SimpleNamespace(get_rng_state=lambda: self.Value([1]),
            set_rng_state=lambda v: self.events.append(('cpu_rng', 'cpu')))
        torch.cuda = SimpleNamespace(get_rng_state_all=lambda: [self.Value([2])],
            set_rng_state_all=lambda v: self.events.append(('cuda_rng', 'cuda')), is_initialized=lambda: False)
        def load(*args, **kwargs):
            saved = copy.deepcopy(self.disk)
            self.loaded_steps = [saved['optimizer']['state'][i]['step'] for i in range(4)]
            return saved
        torch.load = load

    def clone_tree(self, value, device):
        if isinstance(value, self.Value):
            return value.to(device, copy=True)
        if isinstance(value, dict):
            return {k: self.clone_tree(v, device) for k, v in value.items()}
        return copy.deepcopy(value)

    def run(self, directory):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, self.modules))
            output = io.StringIO()
            stack.enter_context(redirect_stdout(output))
            # Payload admission/native integrity are separate tested boundaries;
            # retain the actual complete identity and payload hashes in this fixture.
            stack.enter_context(patch.object(driver, 'check_payload', side_effect=lambda c, d, i, s:
                self.case.assertEqual((d['identity'], d['counter']), (i, s))))
            stack.enter_context(patch.object(driver, 'integrity', side_effect=self.check_bindings))
            real_identity = driver.identity
            def identity(context, state):
                self.events.append(('identity', next(iter(state['model'].parameters.values())).device.type))
                return real_identity(context, state)
            stack.enter_context(patch.object(driver, 'identity', side_effect=identity))
            state = driver.fresh(self.context, 'control', self.device)
            ident = driver.identity(self.context, state)
            for name in driver.NAMES:
                state['model'].parameters[name].value = 7.
            state['counter'] = 1
            for _, p in state['params']:
                state['optimizer_object'].state[p] = {'step': self.Value(1.),
                    'exp_avg': self.Value(2., p.shape, self.device), 'exp_avg_sq': self.Value(3., p.shape, self.device)}
            if self.device == 'cuda':
                state['scaler_object'].saved['_growth_tracker'] = 1
            self.disk = self.clone_tree(driver.payload(self.context, state, ident), 'cpu')
            digest = driver.fingerprint(self.context, self.disk)
            driver.release(self.context, state)
            self.events.clear()
            self.old_bindings.clear()
            output.seek(0)
            output.truncate()
            path = Path(directory) / 'fixture.pt'
            path.write_bytes(b'tiny stand-in archive')
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.case.assertLess(len(repr(self.disk).encode()), 16 * 1024**2)
            result = driver.restore(self.context, path, sha, digest, ident, 1)
            self.snapshots = [json.loads(line) for line in output.getvalue().splitlines()
                              if json.loads(line)['event'] == 'NEAREST_RESTORE_MEMORY']
            self.case.assertEqual(driver.fingerprint(self.context, driver.payload(self.context, result, ident)), digest)
            return result

    def check_bindings(self, context, state, ident):
        params = dict(state['model'].named_parameters())
        self.case.assertEqual(state['device'], self.device)
        for i, (name, p) in enumerate(state['params']):
            self.case.assertIs(p, params[name])
            self.case.assertIs(state['optimizer_object'].param_groups[0]['params'][i], p)
            step = state['optimizer_object'].state[p]['step']
            self.case.assertEqual(step.device.type, 'cpu')
            self.case.assertIsNot(step, self.loaded_steps[i])
            for key in ('exp_avg', 'exp_avg_sq'):
                self.case.assertEqual(state['optimizer_object'].state[p][key].device.type, self.device)
        self.case.assertTrue(all(ref() is None for group in self.old_bindings for ref in group))
        for value in (state['target'], state['classifier'], state['A'], state['means']['fixed'],
                      state['teachers']['counts'], state['head_object'].tensors['weight']):
            self.case.assertEqual(value.device.type, self.device)


class StartupAdmissionFixture:
    """Real pinned original reader/terminal adapter; only tiny on-disk facts."""
    def __init__(self, root):
        self.root, self.read_bytes = root, {}
        self.fitter = self.module('fit_siglip2_prototype_residual.py', driver.FITTER['code']['fit_siglip2_prototype_residual.py'])
        self.original = self.module('train_siglip2_substrate_adaptation.py', self.fitter.TERMINAL_SOURCE_SHA)
        self.init = self.module('initialize_siglip2_substrate_fit.py')
        self.old = self.module('train_siglip2_quadratic_readout.py', self.fitter.ORIGINAL_CODE['train_siglip2_quadratic_readout.py'])
        self.admission = self.original.FlatAdmission()
        self.admission.init = self.init
        self.bulk = [self.write('bulk-' + str(i), b'bulk' * 1024) for i in range(2)]
        self.historical = {}
        self.context = {'root': root, 'args': SimpleNamespace(execution_sha256='d' * 64),
            'guards': {}, 'code': {n: 'e' * 64 for n in driver.FILES}, 'source': {'fixture': 'source'},
            'fitter': self.fitter, 'old': self.old, 'terminals': {}, 'terminal_cgroups': {},
            'phase_seconds': {}, 'started': time.perf_counter()}
        original_guards = {self.original.__file__: self.fitter.TERMINAL_SOURCE_SHA}
        self.legacy = {'original': self.original, 'admission': self.admission, 'guards': original_guards,
            'selected': {'source_cpu': {'invocation': {'python': '/python', 'python_sha256': 'f' * 64,
                'python_version': 'fixture'}, 'numerical_flags': {'fixture': True}}}, 'invocations': set()}
        self.context.update(legacy=self.legacy, fit_context={'legacy': self.legacy, 'guards': original_guards})
        self.context['required_guards'] = {self.bulk[0]['path']: self.bulk[0]['sha256']}
        self.cpu = self.terminal('cpu', 'control', 1)
        self.context['launch'] = self.cpu[1]['launch']
        self.mechanics = self.terminal('mechanics', 'control', 2)
        self.context['launch'] = self.mechanics[1]['launch']

    @staticmethod
    def module(name, sha=None):
        path = PATH.with_name(name)
        raw = path.read_bytes()
        if sha is not None:
            driver.require(hashlib.sha256(raw).hexdigest() == sha, 'fixture source pin differs')
        spec = importlib.util.spec_from_file_location('_startup_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(path), 'exec'), vars(module))
        return module

    def write(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}

    def write_json(self, name, value):
        return self.write(name, json.dumps(value, allow_nan=False).encode())

    @staticmethod
    def cgroup(unit, peak):
        return {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
            'memory.max': str(8 * 1024**3), 'memory.current': '1', 'memory.peak': str(peak),
            'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
            'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0'}}

    def terminal(self, phase, arm, ordinal):
        unit = {'unit': 'fixture-' + str(ordinal), 'invocation_id': format(ordinal, '032x'),
            'service_seconds': 4., 'native_peak_rss_kib': 2, 'both_locks_held': True}
        selected = None if phase == 'cpu' else self.cpu[0]
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64, 'phase': phase,
            'arm': arm, 'seed': driver.SEED, 'fitter': copy.deepcopy(driver.FITTER),
            'accepted': copy.deepcopy(driver.ACCEPTED), 'recipe': copy.deepcopy(driver.RECIPE),
            'resource_policy': driver.policy(phase), 'both_locks_held': True,
            'native_authority': {'path': '/native-authority.json', 'sha256': 'a' * 64},
            'selected_cpu': copy.deepcopy(selected), 'selected_mechanics': None}
        authority = self.write_json('authority-' + str(ordinal), launch)
        output = self.root / ('stage-' + str(ordinal))
        output.mkdir()
        final = {**self.cgroup(unit['unit'], 3), 'invocation_id': unit['invocation_id']}
        log = [f"Running as unit: {unit['unit']}.service; invocation ID: {unit['invocation_id']}",
            '\tExit status: 0', 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
            '\tSwaps: 0', 'Memory swap peak: 0B', 'Service runtime: 4.0s',
            '\tMaximum resident set size (kbytes): 2', 'FINAL_CGROUP ' + json.dumps(final)]
        unit['log'] = self.write('log-' + str(ordinal), ('\n'.join(log) + '\n').encode())
        flags = self.legacy['selected']['source_cpu']['numerical_flags']
        record = {'schema': driver.SCHEMA, 'phase': phase, 'arm': arm, 'seed': driver.SEED,
            'execution_sha256': 'd' * 64, 'launch': launch, 'resource_policy': driver.policy(phase),
            'optimizer_members': 4, 'trainable_scalars': 9921872, 'quality_read': False,
            'total_training_core_seconds': 1., 'wall_seconds': 3., 'process_peak_rss_kib': 1,
            'code': self.context['code'], 'source': self.context['source'], 'authority': authority,
            'authority_sha256': authority['sha256'], 'numerical_flags': flags,
            'identity': {'method': driver.method(launch), 'parameter_names': driver.NAMES,
                'source': self.context['source'], 'numerical_flags': flags},
            'invocation': {**self.legacy['selected']['source_cpu']['invocation'], 'optimize': 0,
                'invocation_id': unit['invocation_id'], 'cuda_visible_devices': '' if phase == 'cpu' else '0',
                'cublas_workspace_config': ':4096:8', 'argv': driver.cli(self.root, authority['path'],
                    authority['sha256'], 'd' * 64, phase, arm, output)},
            'input_guards': {f['path']: f['sha256'] for f in self.bulk[:ordinal]},
            'cgroup_before': self.cgroup(unit['unit'], 1), 'cgroup_after': self.cgroup(unit['unit'], 2),
            'checkpoint': None, 'inference_checkpoint': None}
        for name in ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'sequential_model_ownership',
            'forward_oracle_exact', 'native_training_inference_exact', 'both_locks_held_in_parent_authority'):
            record[name] = True
        if phase == 'cpu':
            record.update(completed_step=0, cuda_initialized=False, peak_cuda_allocated_bytes=0)
            for name in ('initial_arm_parity', 'cpu_serialization_exact', 'bypass_version_tamper_rejected',
                'malformed_state_rejected', 'native_loss_reduction_exact', 'native_role_mutation_rejected'):
                record[name] = True
        else:
            record.update(completed_step=17, cuda_initialized=True, peak_cuda_allocated_bytes=1,
                source_substitution_rejected=True, inference_artifact_independent=True,
                training_state_discarded=True, replay_exact=True)
            record['steps'] = [{'step': i, 'batch': list(range(64)), 'images': [None] * 4,
                'mined': [None] * 4, 'all_four_updated': True, 'gradient_norms': {n: 1. for n in driver.NAMES},
                'scale': 128., 'core_seconds': .01, 'seconds': .02, 'mse': 0., 'rank': 0.,
                'active_hinges': 1, 'ranking_gradient_norms': {n: 1. for n in driver.NAMES}} for i in range(1, 18)]
            record['resumed_steps'] = copy.deepcopy(record['steps'][8:])
        unit['receipt'] = self.write_json(str(output.relative_to(self.root) / 'receipt.json'), record)
        return unit, record

    def rewrite(self, unit, record):
        unit['receipt'] = self.write_json(str(Path(unit['receipt']['path']).relative_to(self.root)), record)

    def count_reads(self):
        fixture, original_open = self, Path.open

        class Stream:
            def __init__(self, stream, path):
                self.stream, self.path = stream, str(path)

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def read(self, *args):
                raw = self.stream.read(*args)
                fixture.read_bytes[self.path] = fixture.read_bytes.get(self.path, 0) + len(raw)
                return raw

            def readinto(self, buffer):
                count = self.stream.readinto(buffer)
                fixture.read_bytes[self.path] = fixture.read_bytes.get(self.path, 0) + count
                return count

        def opened(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return Stream(stream, path) if args and args[0] == 'rb' else stream
        return patch.object(Path, 'open', opened)


class CurrentByteFixture:
    """Byte/layout stand-ins execute the genuine pinned serializer without Torch."""
    class Buffer(bytearray):
        pass

    class View:
        def __init__(self, fixture, raw, device='cuda:0', group=False):
            self.fixture, self.raw, self.device, self.group = fixture, CurrentByteFixture.Buffer(raw), device, group

        def reshape(self, *shape):
            return self

        def view(self, dtype):
            assert dtype == 'uint8'
            return self

        def numel(self):
            return len(self.raw)

        def cpu(self):
            self.fixture.events.append(('copy', self.device, len(self.raw)))
            if self.fixture.copy_failure:
                raise ValueError('copy failed')
            host = type(self)(self.fixture, self.raw, 'cpu', self.group)
            self.fixture.hosts.append(weakref.ref(host.raw))
            return host

        def contiguous(self):
            return self

        def numpy(self):
            return self.raw

    class Value:
        _version = 0

        def __init__(self, fixture, storage, shape, dtype='torch.uint8', width=1,
                     device='cuda:0', indices=None):
            self.fixture, self.storage = fixture, bytearray(storage)
            self.shape, self.dtype, self.width, self.device = shape, dtype, width, device
            self.indices = list(range(math.prod(shape))) if indices is None else indices
            self.is_cuda = device.startswith('cuda')
            self.reads = 0

        def data_ptr(self):
            return id(self.storage)

        def numel(self):
            return math.prod(self.shape)

        def element_size(self):
            return self.width

        def detach(self):
            return self

        def contiguous(self):
            self.reads += 1
            raw = b''.join(self.storage[i * self.width:(i + 1) * self.width] for i in self.indices)
            view = CurrentByteFixture.View(self.fixture, raw, self.device)
            self.fixture.views.append(weakref.ref(view))
            return view

        def cpu(self):
            self.fixture.events.append(('serial', self.device, self.numel() * self.width))
            return self.contiguous()

    def __init__(self, directory):
        self.events, self.hosts, self.views = [], [], []
        self.copy_failure = False
        path = Path(directory) / 'original.py'
        path.write_bytes(PATH.with_name('train_siglip2_substrate_adaptation.py').read_bytes())
        spec = importlib.util.spec_from_file_location('_current_original', path)
        self.original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.original)
        self.bindings = dict(vars(self.original))
        self.codes = {k: v.__code__ for k, v in self.bindings.items() if isinstance(v, driver.FunctionType)}
        self.context = {'legacy': {'original': self.original}}
        self.torch = ModuleType('torch')
        self.torch.Tensor, self.torch.uint8 = self.Value, 'uint8'
        self.torch.cat = self.cat

    def cat(self, views):
        if any(ref() is not None for ref in self.hosts):
            raise AssertionError('previous group host storage survived')
        self.events.append(('cat', tuple(v.numel() for v in views)))
        return self.View(self, b''.join(v.raw for v in views), views[0].device, True)

    def value(self, raw, shape=None, **kwargs):
        width = kwargs.get('width', 1)
        return self.Value(self, raw, (len(raw) // width,) if shape is None else shape, **kwargs)

    def small_groups(self):
        # Only the private test clone uses eight bytes. Production stays literal64MiB.
        node = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'current_cuda_bytes')
        dump = lambda n: ast.dump(n, include_attributes=False)
        limit = ast.parse('64 * 1024**2', mode='eval').body
        matches = [n for n in ast.walk(node) if dump(n) == dump(limit)]
        assert len(matches) == 1
        class Limit(ast.NodeTransformer):
            def visit_BinOp(self, n):
                return ast.copy_location(ast.Constant(8), n) if dump(n) == dump(limit) else self.generic_visit(n)
        namespace = dict(vars(driver))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[Limit().visit(node)], type_ignores=[])),
                     str(PATH), 'exec'), namespace)
        return patch.object(driver, 'current_cuda_bytes', namespace['current_cuda_bytes'])

    def unchanged(self, case):
        case.assertEqual(vars(self.original), self.bindings)
        case.assertEqual({k: v.__code__ for k, v in vars(self.original).items()
                          if isinstance(v, driver.FunctionType)}, self.codes)
        case.assertEqual(hashlib.sha256(Path(self.original.__file__).read_bytes()).hexdigest(),
                         'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543')


class CurrentByteTests(unittest.TestCase):
    def test_current_byte_order_dtype_shape_alias_stride_empty_mixed_groups(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            a = f.value(b'0123456789', (3,), indices=[1, 4, 7])
            b = f.value(b'abcdefgh', (2, 2), dtype='torch.bfloat16', width=2)
            c = f.value(b'CPU', device='cpu')
            d = f.value(b'xyz', device='cuda:1')
            empty = f.value(b'', (0, 4), dtype='torch.float32', width=4)
            # Repr-sorted dicts, nested sequences and repeated alias occurrences.
            value = {'z': (b, c, d, empty), 'a': [a, a]}
            with patch.dict(sys.modules, {'torch': f.torch}):
                expected = f.original.fingerprint(value)
                f.events.clear()
                before = a.reads
                with f.small_groups():
                    self.assertEqual(driver.fingerprint(f.context, value), expected)
                copies = [e for e in f.events if e[0] == 'copy']
                self.assertEqual(copies, [('copy', 'cuda:0', 6), ('copy', 'cuda:0', 8),
                                         ('copy', 'cuda:1', 3)])
                self.assertEqual(a.reads - before, 2)
                self.assertEqual([e for e in f.events if e[0] == 'serial'], [('serial', 'cpu', 3)])
                self.assertTrue(all(sum(e[1]) <= 8 for e in f.events if e[0] == 'cat'))
                self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                self.assertNotEqual(expected, f.original.fingerprint({**value, 'z': (b, c, d)}))
                old_shape = b.shape
                b.shape = (4,)
                self.assertNotEqual(expected, driver.fingerprint(f.context, value))
                b.shape, b.dtype = old_shape, 'torch.float16'
                self.assertNotEqual(expected, driver.fingerprint(f.context, value))
            f.unchanged(self)

    def test_fresh_current_reads_ignore_unchanged_version_and_release_groups(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            value = [f.value(b'abcd'), f.value(b'efgh'), f.value(b'ijkl')]
            with patch.dict(sys.modules, {'torch': f.torch}), f.small_groups():
                first = driver.fingerprint(f.context, value)
                reads = [v.reads for v in value]
                value[0].storage[0] = ord('z')
                self.assertEqual(value[0]._version, 0)
                second = driver.fingerprint(f.context, value)
                self.assertNotEqual(first, second)
                self.assertEqual([v.reads - n for v, n in zip(value, reads)], [1, 1, 1])
                self.assertEqual(second, f.original.fingerprint(value))
                self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
            f.unchanged(self)

    def test_empty_cpu_callbacks_kwargs_and_oversized_are_original_serial(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            with patch.dict(sys.modules, {'torch': f.torch}):
                empty = f.value(b'', (0, 3), dtype='torch.float32', width=4)
                expected = f.original.fingerprint(empty)
                f.events.clear()
                self.assertEqual(driver.fingerprint(f.context, empty), expected)
                self.assertEqual(f.events, [])
                cpu = f.value(b'cpu', device='cpu')
                with patch.object(driver, 'current_byte_adapter', side_effect=AssertionError('CPU adapter')):
                    self.assertEqual(driver.fingerprint(f.context, cpu), f.original.fingerprint(cpu))
                seen = []
                callback_value = [f.value(b'cuda'), cpu]
                self.assertEqual(driver.fingerprint(f.context, callback_value, consumed=seen.append),
                                 f.original.fingerprint(callback_value))
                self.assertEqual(seen, callback_value)
                frozen = {(empty.data_ptr(), 0, str(empty.dtype), tuple(empty.shape)): ('frozen', (), 'cached')}
                self.assertEqual(driver.fingerprint(f.context, empty, frozen=frozen),
                                 f.original.fingerprint(empty, frozen=frozen))
                for kwargs in ({'consumed': None}, {'frozen': None}, {'unsupported': True}):
                    with patch.object(driver, 'current_byte_adapter', side_effect=AssertionError('kwargs adapter')):
                        if 'unsupported' in kwargs:
                            with self.assertRaises(TypeError):
                                driver.fingerprint(f.context, empty, **kwargs)
                        else:
                            self.assertEqual(driver.fingerprint(f.context, empty, **kwargs), expected)
                big = f.value(b'ab')
                big.numel = lambda: 64 * 1024**2 + 1
                f.events.clear()
                driver.fingerprint(f.context, [big, f.value(b'cd')])
                self.assertFalse(any(e[0] in ('cat', 'copy') for e in f.events))
                self.assertEqual(len([e for e in f.events if e[0] == 'serial']), 2)
            f.unchanged(self)

    def test_copy_and_serializer_errors_close_groups_even_with_traceback(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            value = [f.value(b'abcd'), f.value(b'efgh'), f.value(b'ijkl')]
            with patch.dict(sys.modules, {'torch': f.torch}), f.small_groups():
                f.copy_failure = True
                try:
                    driver.fingerprint(f.context, value)
                except ValueError as error:
                    self.assertIn('copy failed', str(error))
                    self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                else:
                    self.fail('copy failure was swallowed')
                f.copy_failure = False
                class BadRepr:
                    def __repr__(self):
                        raise ValueError('repr failed')
                try:
                    driver.fingerprint(f.context, [*value, BadRepr()])
                except ValueError as error:
                    self.assertIn('repr failed', str(error))
                    self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                else:
                    self.fail('serializer failure was swallowed')
                self.assertEqual(driver.fingerprint(f.context, value), f.original.fingerprint(value))
            f.unchanged(self)

    def test_original_source_live_function_globals_defaults_and_origin_negatives(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            value = f.value(b'ab')
            with patch.dict(sys.modules, {'torch': f.torch}):
                fn = f.original.fingerprint
                for attr, replacement in (('__code__', (lambda value, frozen=None, consumed=None: 'bad').__code__),
                                          ('__defaults__', ({}, None))):
                    saved = getattr(fn, attr)
                    try:
                        setattr(fn, attr, replacement)
                        with self.assertRaisesRegex(ValueError, 'original'):
                            driver.fingerprint(f.context, value)
                    finally:
                        setattr(fn, attr, saved)
                replacement = driver.FunctionType(fn.__code__, dict(fn.__globals__), fn.__name__, fn.__defaults__)
                with patch.object(f.original, 'fingerprint', replacement), self.assertRaisesRegex(ValueError, 'original'):
                    driver.fingerprint(f.context, value)
                for name, replacement in (('hashlib', SimpleNamespace(sha256=lambda *a: None)),
                                          ('memoryview', lambda x: memoryview(b'bad')), ('sorted', lambda x, **k: x),
                                          ('__builtins__', dict(vars(__import__('builtins'))))):
                    with patch.dict(vars(f.original), {name: replacement}), self.assertRaisesRegex(ValueError, 'original'):
                        driver.fingerprint(f.context, value)
                with patch.object(f.original.__spec__, 'origin', '/bad'), self.assertRaisesRegex(ValueError, 'original'):
                    driver.fingerprint(f.context, value)
                with patch.object(f.original.hashlib, 'sha256', side_effect=lambda *a, **k: None), \
                        self.assertRaisesRegex(ValueError, 'original'):
                    driver.current_byte_adapter(f.original, lambda item: memoryview(b''))
                raw = Path(f.original.__file__).read_bytes()
                Path(f.original.__file__).write_bytes(raw + b'\n')
                with self.assertRaisesRegex(ValueError, 'original'):
                    driver.fingerprint(f.context, value)
                Path(f.original.__file__).write_bytes(raw)
            f.unchanged(self)

    def test_cpu_and_unsupported_keys_do_not_repeat_repr_or_custom_traversal(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            class ChangingKey:
                calls = 0
                def __repr__(self):
                    self.calls += 1
                    return str(self.calls)
            with patch.dict(sys.modules, {'torch': f.torch}):
                for device in ('cpu', 'cuda:0'):
                    key = ChangingKey()
                    value = {key: f.value(b'ab', device=device)}
                    expected = f.original.fingerprint(value)
                    expected_calls = key.calls
                    key.calls = 0
                    self.assertEqual(driver.fingerprint(f.context, value), expected)
                    self.assertEqual(key.calls, expected_calls)
                class TensorKey(f.Value):
                    calls = 0
                    def __repr__(self):
                        self.calls += 1
                        self.storage[0] += 1
                        return 'tensor-key'
                key = TensorKey(f, b'ab', (2,))
                value = {key: f.value(b'cd')}
                expected = f.original.fingerprint(value)
                expected_calls = key.calls
                key.calls, key.storage[0] = 0, ord('a')
                self.assertEqual(driver.fingerprint(f.context, value), expected)
                self.assertEqual(key.calls, expected_calls)
                class CustomDict(dict):
                    calls = 0
                    def __iter__(self):
                        self.calls += 1
                        return super().__iter__()
                value = CustomDict(a=f.value(b'ab', device='cpu'))
                expected = f.original.fingerprint(value)
                expected_calls = value.calls
                value.calls = 0
                self.assertEqual(driver.fingerprint(f.context, value), expected)
                self.assertEqual(value.calls, expected_calls)
            f.unchanged(self)

    def test_group_limit_explicit_close_byte_size_and_occurrence_order_fail_closed(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            a, b = f.value(b'abcd'), f.value(b'efgh')
            with patch.dict(sys.modules, {'torch': f.torch}), f.small_groups():
                chunks = driver.current_cuda_bytes([(a, 4), (b, 4)])
                tensor, raw = next(chunks)
                self.assertIs(tensor, a)
                self.assertEqual(raw.tobytes(), b'abcd')
                self.assertEqual([e for e in f.events if e[0] == 'copy'], [('copy', 'cuda:0', 8)])
                chunks.close()
                with self.assertRaises(ValueError):
                    raw.tobytes()
                self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                for count, message in ((9, 'exceeds byte group limit'), (3, 'size changed')):
                    chunks = driver.current_cuda_bytes([(a, count)])
                    with self.assertRaisesRegex(ValueError, message):
                        next(chunks)
                    chunks.close()
                    self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                with patch.object(f.torch, 'cat', side_effect=ValueError('cat failed')):
                    with self.assertRaisesRegex(ValueError, 'cat failed'):
                        driver.fingerprint(f.context, [a, b])
                    self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                with patch.object(driver, 'current_cuda_occurrences', return_value=[(b, 4), (a, 4)]):
                    with self.assertRaisesRegex(ValueError, 'occurrence order differs'):
                        driver.fingerprint(f.context, [a, b])
                    self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
                with patch.object(driver, 'current_cuda_occurrences', return_value=[(a, 4), (a, 4)]):
                    with self.assertRaisesRegex(ValueError, 'occurrences not exhausted'):
                        driver.fingerprint(f.context, a)
                    self.assertTrue(all(ref() is None for ref in f.hosts + f.views))
            f.unchanged(self)

    def test_source_boundary_rejects_fingerprint_and_helper_inventory_mutants(self):
        tree = ast.parse(PATH.read_text())
        self.assertEqual(startup_source_boundary(copy.deepcopy(tree)),
                         '2b749548e57aa010826665a38e4e145adaeefed9658a3d8a86f470134f312a72')
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
        fn.body.append(ast.parse('unrelated_predicate = False').body[0])
        with self.assertRaisesRegex(ValueError, 'exact current-byte fingerprint dispatch'):
            startup_source_boundary(tree)
        tree = ast.parse(PATH.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'current_cuda_bytes')
        tree.body.append(copy.deepcopy(fn))
        with self.assertRaisesRegex(ValueError, 'exact added current-byte definition'):
            startup_source_boundary(tree)

    def test_exact_raw_ast_replacement_and_inverse(self):
        with TemporaryDirectory() as directory:
            f = CurrentByteFixture(directory)
            fn = driver.current_byte_adapter(f.original, lambda item: memoryview(b''))
            self.assertIsNot(fn.__globals__, vars(f.original))
            self.assertEqual(hashlib.sha256(ast.dump(fn.__current_byte_ast__, include_attributes=False).encode()).hexdigest(),
                             'bcb97676f8800cd5d7e4048dde059d2d3a47d80673b8556f85616383701f3f12')
            before = ast.parse('item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()', mode='eval').body
            after = ast.parse('_nearest_cuda_raw(item) if item.is_cuda else ' + ast.unparse(before), mode='eval').body
            tree = copy.deepcopy(fn.__current_byte_ast__)
            dump = lambda n: ast.dump(n, include_attributes=False)
            class Inverse(ast.NodeTransformer):
                count = 0
                def visit_IfExp(self, node):
                    if dump(node) == dump(after):
                        self.count += 1
                        return copy.deepcopy(before)
                    return self.generic_visit(node)
            inverse = Inverse()
            restored = inverse.visit(tree)
            self.assertEqual(inverse.count, 1)
            self.assertEqual(hashlib.sha256(dump(restored).encode()).hexdigest(),
                             '3de225c57984a5ee3f292ecddce7a1516154938d5b4d415e692837abda5b2d0f')
            f.unchanged(self)


def current_byte_source_boundary(tree):
    """Remove exactly the scheduling helpers and reverse only the pinned dispatch."""
    added = {'current_cuda_occurrences', 'current_cuda_bytes', 'current_byte_adapter'}
    for name in added:
        driver.require(sum(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body) == 1,
                       'exact added current-byte definition required')
    dispatch = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint']
    driver.require(len(dispatch) == 1 and
        hashlib.sha256(ast.dump(dispatch[0], include_attributes=False).encode()).hexdigest() ==
        '9f2d65d5391b9bf9a3ceb0f3a55dfe65acc9c89be0d509368b2e6dd335bf2e81',
        'exact current-byte fingerprint dispatch required')
    original = ast.parse("""def fingerprint(context, value, **kwargs):
    return context['legacy']['original'].fingerprint(value, **kwargs)
""").body[0]
    tree.body = [original if node is dispatch[0] else node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name in added)]
    return tree



def native_source_boundary(tree):
    """Invert only exact-four authority/dispatch edits; retain prior AST hashes."""
    dump = lambda n: ast.dump(n, include_attributes=False)
    added = {'bind_native_authority', 'native_source_api'}
    for name in added:
        driver.require(sum(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body) == 1,
                       'exact native definition required')
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in added) and
                 not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                      t.id in {'NATIVE_PROOF_PINS', 'NATIVE_MEMBERS'} for t in n.targets))]
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == 'types':
            driver.require(sum(a.name == 'MappingProxyType' for a in n.names) == 1, 'exact native import required')
            n.names = [a for a in n.names if a.name != 'MappingProxyType']
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LAUNCH_KEYS' for t in n.targets):
            driver.require(sum(isinstance(v, ast.Constant) and v.value == 'native_authority' for v in n.value.elts) == 1,
                           'exact native launch key required')
            n.value.elts = [v for v in n.value.elts if not (isinstance(v, ast.Constant) and v.value == 'native_authority')]
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, statement in [('check_launch', "file_fact(launch['native_authority'])"),
                            ('authority', 'native_source_api(context)')]:
        node = functions[name]
        matches = [n for n in node.body if dump(n) == dump(ast.parse(statement).body[0])]
        driver.require(len(matches) == 1, 'exact native authority binding required')
        node.body.remove(matches[0])
    replacements = {
        "dict(original['source'])": ("original['source']", 1),
        'native_source_api(context).audit_origins(legacy)': ("context['old'].audit_origins(legacy)", 2),
        "native_source_api(context).exit_rehash(context['fit_context'])":
            ("context['fitter'].exit_rehash(context['fit_context'])", 1)}
    class Inverse(ast.NodeTransformer):
        def __init__(self): self.counts = dict.fromkeys(replacements, 0)
        def visit_Call(self, node):
            for before, (after, _) in replacements.items():
                if dump(node) == dump(ast.parse(before, mode='eval').body):
                    self.counts[before] += 1
                    return ast.copy_location(ast.parse(after, mode='eval').body, node)
            return self.generic_visit(node)
    inverse = Inverse()
    tree = inverse.visit(tree)
    driver.require(all(inverse.counts[n] == count for n, (_, count) in replacements.items()),
                   'exact native dispatch/source substitutions required')
    return tree


def startup_source_boundary(tree):
    """Invert only the authorized startup dispatch to retain the prior AST proof."""
    tree = current_byte_source_boundary(native_source_boundary(tree))
    added = {'authenticate_startup_reader', 'startup_admission_adapter'}
    for name in added:
        driver.require(sum(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body) == 1,
                       'exact added startup definition required')
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and
        n.name in {'read_json', 'admit_terminal', *added})]
    authority = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authority')
    dump = lambda node: ast.dump(node, include_attributes=False)
    assignment = ast.parse('startup = startup_admission_adapter(fitter, guards)').body[0]
    matches = [n for n in authority.body if dump(n) == dump(assignment)]
    driver.require(len(matches) == 1, 'exact startup construction required')
    authority.body.remove(matches[0])
    counts = {'authority': 0, 'admit_terminal': 0}
    for node in ast.walk(authority):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                isinstance(node.func.value, ast.Name) and node.func.value.id == 'startup':
            driver.require(node.func.attr in counts, 'unexpected startup dispatch')
            counts[node.func.attr] += 1
            node.func.value.id = 'fitter'
    driver.require(counts == {'authority': 1, 'admit_terminal': 1}, 'exact two startup dispatches required')
    return hashlib.sha256(dump(tree).encode()).hexdigest()


class FitterStartupFixture(StartupAdmissionFixture):
    """Four real admission sweeps, with archived terminal metadata and tiny FILEs.

    Only fixture locations/encoder byte size are retargeted in private globals;
    every validator body, log parser and original FlatAdmission stays genuine.
    """
    def __init__(self, root):
        super().__init__(root)
        self.evidence = PATH.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1'
        self.adapter = driver.startup_admission_adapter(self.fitter, {})
        self.namespace = self.adapter.authority.__globals__
        self.original_members = dict(vars(self.fitter))
        self.original_codes = {n: v.__code__ for n, v in self.original_members.items()
                               if isinstance(v, driver.FunctionType)}
        self.original_classes = {n: dict(vars(v)) for n, v in self.original_members.items()
                                 if isinstance(v, type) and v.__module__ == self.fitter.__name__}
        self.reader_members = dict(vars(self.original))
        self.reader_methods = dict(vars(self.original.FlatAdmission))
        training = {'root': str(root / 'historical'), 'code': {}}
        Path(training['root']).mkdir()
        for name, sha in self.fitter.HISTORICAL_LINEAR['training']['code'].items():
            raw = (self.evidence / 'train-source-v3' / name).read_bytes()
            driver.require(hashlib.sha256(raw).hexdigest() == sha, 'historical fixture source pin differs')
            training['code'][name] = self.write('historical/' + name, raw)['sha256']
        training['execution_sha256'] = self.write_json('historical/execution.json', training['code'])['sha256']
        solver = self.write('solver.py', (self.evidence / 'solver-source-v1/foundation_adapter.py').read_bytes())
        driver.require(solver['sha256'] == self.fitter.SOLVER_SHA, 'solver fixture pin differs')
        self.context = {'args': SimpleNamespace(execution_sha256='d' * 64), 'root': root,
            'code': copy.deepcopy(driver.FITTER['code']), 'legacy': self.legacy, 'old': self.old,
            'source': {'fixture': 'source'}, 'guards': dict(self.legacy['guards']),
            'original_required_guards': {self.bulk[0]['path']: self.bulk[0]['sha256']},
            'terminals': {}, 'terminal_cgroups': {}, 'phase_seconds': {}, 'unit_started': time.perf_counter()}
        self.context['guards'].update(self.context['original_required_guards'])
        self.namespace['HISTORICAL_LINEAR'] = {'training': training, 'endpoint': {}}
        self.records = {}
        self.hist_cpu = self.fitter_terminal('cpu-v3-receipt.json', 'cpu', 'linear', 11, training, solver)
        self.hist_fit = self.fitter_terminal('fit-linear-v1-receipt.json', 'fit', 'linear', 12, training, solver,
                                              self.hist_cpu[0])
        self.namespace['HISTORICAL_LINEAR']['endpoint'] = {'arm': 'linear', 'launch': self.hist_fit[1]['authority'],
            'terminal': self.hist_fit[0], 'checkpoint': self.hist_fit[1]['checkpoint'],
            'terminal_state_sha256': self.hist_fit[1]['terminal_state_sha256']}
        # Clones retain every authenticated validation instruction. The fixture
        # cannot materialize the real 1.7GB retained encoder within its 16MiB cap.
        validators = dict(vars(self.fitter), HISTORICAL_LINEAR=self.namespace['HISTORICAL_LINEAR'],
            ENCODER_CHECKPOINT=self.bulk[0], ENCODER_CHECKPOINT_BYTES=4096)
        for name in ('check_launch', 'check_run_metadata', 'check_terminal_record'):
            fn = getattr(self.fitter, name)
            validators[name] = driver.FunctionType(fn.__code__, validators, name)
        self.namespace['check_terminal_record'] = validators['check_terminal_record']
        self.signed_cpu = self.fitter_terminal('signed-concat-cpu-v2/receipt.json', 'cpu', 'linear', 13,
            {'root': str(root), 'execution_sha256': 'd' * 64, 'code': self.context['code']}, solver)
        self.concat = self.fitter_terminal('signed-concat-fit-concat-v1/receipt.json', 'fit', 'concat', 14,
            {'root': str(root), 'execution_sha256': 'd' * 64, 'code': self.context['code']}, solver, self.signed_cpu[0])
        self.context['launch'] = self.concat[1]['launch']

    def fitter_terminal(self, name, phase, arm, ordinal, training, solver, selected=None):
        record = json.loads((self.evidence / name).read_text())
        launch = record['launch']
        launch.update(execution_sha256=training['execution_sha256'], selected_cpu=copy.deepcopy(selected), ridge_solver=solver)
        signed = record['schema'] == self.fitter.SCHEMA
        if signed:
            launch['historical_linear'] = copy.deepcopy(self.namespace['HISTORICAL_LINEAR'])
            if phase == 'cpu':
                record['native_linear_parity']['historical_linear'] = copy.deepcopy(self.namespace['HISTORICAL_LINEAR'])
        record.update(code=copy.deepcopy(training['code']), execution_sha256=training['execution_sha256'],
                      source=self.context['source'], numerical_flags=self.legacy['selected']['source_cpu']['numerical_flags'])
        output = self.root / ('fitter-stage-' + str(ordinal))
        output.mkdir()
        record['output'] = str(output)
        record['authority'] = self.write_json('fitter-authority-' + str(ordinal), launch)
        record['authority_sha256'] = record['authority']['sha256']
        record['invocation'].update(self.legacy['selected']['source_cpu']['invocation'], optimize=0,
            invocation_id=format(ordinal, '032x'), cuda_visible_devices='',
            argv=self.fitter.cli(training['root'], record['authority']['path'], record['authority']['sha256'],
                training['execution_sha256'], phase, arm, output))
        facts = record['arms'] if phase == 'cpu' else {arm: record}
        identity_method = {k: launch[k] for k in ('execution_sha256', 'original_reference', 'original_cpu',
            'ridge_solver', 'warm_start', 'partition', 'recipe', *(['historical_linear'] if signed else []))}
        for fact in facts.values():
            fact['identity'].update(method=identity_method, source=self.context['source'],
                numerical_flags=record['numerical_flags'])
        inventory = {**self.context['original_required_guards'],
            str(Path(training['root']) / 'execution.json'): training['execution_sha256'],
            **{str(Path(training['root']) / n): h for n, h in training['code'].items()},
            record['authority']['path']: record['authority']['sha256'], solver['path']: solver['sha256']}
        if signed:
            # Signed exact3 source FILEs are actual pinned bytes, also tiny.
            for n, h in training['code'].items():
                self.write(n, PATH.with_name(n).read_bytes())
            self.write_json('execution.json', training['code'])
            inventory.pop(str(self.root / 'execution.json'))  # No closure read in these two functions.
            record['encoder_retention'].update(self.bulk[0], bytes=4096, pages_populated=1, page_bytes=4096)
        record['input_guards'] = inventory
        if ordinal == 14:
            record['input_guards'][self.bulk[1]['path']] = self.bulk[1]['sha256']
        if phase == 'fit':
            record['checkpoint'] = self.write(str(output.relative_to(self.root) / 'resume.pt'), b'checkpoint')
            record['input_guards'][record['checkpoint']['path']] = record['checkpoint']['sha256']
        unit = {'unit': 'fitter-fixture-' + str(ordinal), 'invocation_id': format(ordinal, '032x'),
            'service_seconds': math.ceil(record['wall_seconds']) + 1,
            'native_peak_rss_kib': record['process_peak_rss_kib'] + 1, 'both_locks_held': True}
        record['cgroup_before'], record['cgroup_after'] = self.cgroup(unit['unit'], 1), self.cgroup(unit['unit'], 2)
        final = {**self.cgroup(unit['unit'], 3), 'invocation_id': unit['invocation_id']}
        log = [f"Running as unit: {unit['unit']}.service; invocation ID: {unit['invocation_id']}",
            '\tExit status: 0', 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
            '\tSwaps: 0', 'Memory swap peak: 0B', f"Service runtime: {unit['service_seconds']}s",
            f"\tMaximum resident set size (kbytes): {unit['native_peak_rss_kib']}", 'FINAL_CGROUP ' + json.dumps(final)]
        unit['log'] = self.write('fitter-log-' + str(ordinal), ('\n'.join(log) + '\n').encode())
        unit['receipt'] = self.write_json(str(output.relative_to(self.root) / 'receipt.json'), record)
        self.records[ordinal] = (unit, record)
        return unit, record

    def run_historical(self):
        self.adapter.admit_historical_linear(self.context)
        self.context['required_guards'] = dict(self.context['guards'])
        for unit, record in (self.signed_cpu, self.concat):
            record['input_guards'].update(self.context['required_guards'])
            self.rewrite(unit, record)
        self.concat[1]['launch']['selected_cpu'] = copy.deepcopy(self.signed_cpu[0])
        self.concat[1]['authority'] = self.write_json('fitter-authority-14', self.concat[1]['launch'])
        self.concat[1]['authority_sha256'] = self.concat[1]['authority']['sha256']
        self.concat[1]['input_guards'][self.concat[1]['authority']['path']] = self.concat[1]['authority']['sha256']
        self.concat[1]['invocation']['argv'] = self.fitter.cli(self.root, self.concat[1]['authority']['path'],
            self.concat[1]['authority']['sha256'], 'd' * 64, 'fit', 'concat', Path(self.concat[0]['receipt']['path']).parent)
        self.rewrite(*self.concat)

    def all_sweeps(self):
        self.admission.bound_file(self.historical_guards, self.bulk[0]['path'], self.bulk[0]['sha256'])
        self.run_historical()
        self.adapter.admit_terminal(self.context, self.signed_cpu[0], 'cpu', 'linear')
        self.adapter.admit_terminal(self.context, self.concat[0], 'fit', 'concat')

    def quadratic_exit(self):
        # Exercise the real fresh-reader/three-inventory exit implementation.
        # Image resolution and unrelated source closures are bounded doubles.
        source = SimpleNamespace(fit_rows=lambda *a: [], bootstrap=lambda *a: (None, {}))
        prior = {'source_driver': source, 'extract': None, 'fit': {}, 'all_images': [], 'images': [],
            'guards': {self.bulk[0]['path']: self.bulk[0]['sha256']}, 'root': self.root,
            'args': SimpleNamespace(execution_sha256='d' * 64), 'code': {}, 'own_root': self.root,
            'export_args': SimpleNamespace(execution_sha256='d' * 64), 'own_code': {}}
        exporter = SimpleNamespace(FILES=set(), closure=lambda *a: {}, file_json=lambda *a: {},
            selected_manifest=lambda *a: {'original_rows': []}, image_rows_node=lambda *a: None)
        genuine = {'prior': prior, 'reference': SimpleNamespace(bootstrap=lambda *a: {}),
            'guards': {self.bulk[1]['path']: self.bulk[1]['sha256']}, 'root': self.root,
            'args': SimpleNamespace(execution_sha256='d' * 64), 'code': {},
            'launch': {'partition': {}, 'image_rows': {'path': str(self.root / 'rows.py')}},
            'selected': {'original_rows': [], 'resolved_paths': []}}
        context = {'original': self.original, 'selected': {'exporter': exporter, 'genuine': genuine},
            'guards': self.context['guards'], 'root': self.root,
            'args': SimpleNamespace(phase='fit', execution_sha256='d' * 64),
            'code': self.context['code'], 'phase_seconds': {}}
        with patch.object(self.old, 'audit_origins'), patch.object(self.original.FlatAdmission, 'all_fit_images', return_value=[]), \
                patch.object(self.old, 'closure', return_value=context['code']):
            self.old.exit_rehash(context)

    def unchanged(self, case):
        case.assertEqual(vars(self.fitter).keys(), self.original_members.keys())
        for name, value in self.original_members.items():
            case.assertIs(vars(self.fitter)[name], value)
        for name, code in self.original_codes.items():
            case.assertIs(getattr(self.fitter, name).__code__, code)
        for name, members in self.original_classes.items():
            actual = vars(getattr(self.fitter, name))
            case.assertEqual(actual.keys(), members.keys())
            for key, value in members.items():
                case.assertIs(actual[key], value)
        case.assertEqual(vars(self.original).keys(), self.reader_members.keys())
        for name, value in self.reader_members.items():
            case.assertIs(vars(self.original)[name], value)
        case.assertEqual(vars(self.original.FlatAdmission).keys(), self.reader_methods.keys())
        for name, value in self.reader_methods.items():
            case.assertIs(vars(self.original.FlatAdmission)[name], value)



class NativeAdmissionFixture(StartupAdmissionFixture):
    """Actual pinned audit, quadratic exit, fitter exit and fresh reader; no Torch."""
    def __init__(self, root):
        super().__init__(root)
        import base64
        import csv
        self.site = root / 'site'
        owner = self.site / 'nvidia_cudnn_cu13-9.20.0.48.dist-info'
        owner.mkdir(parents=True)
        self.members = {}
        rows = []
        for i, name in enumerate(sorted(driver.NATIVE_MEMBERS)):
            member = 'nvidia/cudnn/lib/' + name
            path = self.site / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('native' + str(i)).encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.members[member] = {'sha256': digest, 'size_bytes': path.stat().st_size}
            rows.append([member, 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('='), str(path.stat().st_size)])
        (owner / 'METADATA').write_text('Name: nvidia-cudnn-cu13\nVersion: 9.20.0.48\n')
        (owner / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: py3-none-manylinux_2_27_aarch64\n')
        with (owner / 'RECORD').open('w', newline='') as stream: csv.writer(stream).writerows(rows)
        self.records = [str(owner / 'RECORD')]
        inputs = {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'size_bytes': p.stat().st_size}
                  for p in (*[self.site / n for n in self.members], owner / 'METADATA', owner / 'WHEEL', owner / 'RECORD')}
        self.files = {str(self.site / n): v['sha256'] for n, v in self.members.items()}
        wheel = {'sha256': 'a' * 64, 'size_bytes': 10}
        self.proof = {'pass': True, 'exit_rehash_pass': True, 'native_imported': False,
            'original_native_authority_modified': False, 'invocation': {'invocation_id': '1' * 32},
            'authority': {'unit': 'vendor-test', 'installed_site_root': str(self.site), 'official_wheel': wheel,
                          'installed_evidence': self.write_json('installed.json', {})},
            'wheel': wheel, 'input_guards': inputs,
            'comparison': {'selected_members': self.members, 'selected_record_entries_equal': True,
                'metadata': {n: {'bytes_equal': True, 'installed_sha256': inputs[str(owner / n)]['sha256'],
                                'wheel_sha256': inputs[str(owner / n)]['sha256']} for n in ('METADATA', 'WHEEL')}},
            'installed_record_ownership': {'records': self.records, 'owners': {p: self.records.copy() for p in self.files}}}
        log = self.write('vendor.log', b'accepted vendor terminal\n')
        self.authority = {'schema': 'siglip2-nearest-native-source-v1', 'proof': self.write_json('vendor-proof.json', self.proof),
            'log': log, 'decision': self.write_json('vendor-decision.json', {
                'decision': 'PASS_VENDOR_BYTE_PROVENANCE_ONLY', 'exit_code': 0,
                'proof_sha256': self.write_json('vendor-proof.json', self.proof)['sha256'], 'log_sha256': log['sha256'],
                'unit': 'vendor-test', 'invocation_id': '1' * 32, 'memory_events_zero': True, 'swap_bytes': 0, 'service_seconds': 1.})}
        self.pins = {n: self.authority[n]['sha256'] for n in driver.NATIVE_PROOF_PINS}
        self.context['launch'] = {'native_authority': self.write_json('native-authority.json', self.authority)}
        self.inventory = root / 'observed.json'
        self.set_origins(self.files)
        source_path = root / 'origin-source.py'
        source_path.write_text('from pathlib import Path\nimport json\n'
            f'INVENTORY = Path({str(self.inventory)!r})\n'
            'def imported_origins(extract, packages):\n'
            '    origins = json.loads(INVENTORY.read_text())\n'
            '    origins["files"] = {p: extract.sha(p) for p in origins["files"]}\n'
            '    return origins\n'
            'def fit_rows(extract, fit): return []\n'
            'def bootstrap(root, sha): return None, {}\n')
        spec = importlib.util.spec_from_file_location('_native_fixture_source', source_path)
        self.source = importlib.util.module_from_spec(spec)
        exec(compile(source_path.read_bytes(), str(source_path), 'exec'), vars(self.source))
        self.extract = self.module('extract_siglip2_vision_source.py')
        guards = self.context['guards']
        for module in (self.old, self.fitter, self.original, self.source, self.extract):
            guards[module.__file__] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        oldroot, fitroot = root / 'old', root / 'fit'
        oldroot.mkdir(); fitroot.mkdir()
        def code_closure(destination, names):
            for name in names: (destination / name).write_bytes(PATH.with_name(name).read_bytes())
            code = {n: hashlib.sha256((destination / n).read_bytes()).hexdigest() for n in names}
            fact = self.write_json(str(destination.relative_to(root) / 'execution.json'), code)
            return code, fact['sha256']
        oldcode, oldsha = code_closure(oldroot, self.old.FILES)
        fitcode, fitsha = code_closure(fitroot, self.fitter.FILES)
        # 13283 distinct zero-byte image FILEs exercise the genuine FIT cardinality/reader predicate.
        images = root / 'images'; images.mkdir()
        paths = [images / str(i) for i in range(13283)]
        for path in paths: path.touch()
        prior = {'source_driver': self.source, 'extract': self.extract,
            'fit': {'dataset_root': str(images), 'rows': [{'relative_path': p.name, 'image_sha256': hashlib.sha256(b'').hexdigest()} for p in paths]},
            'all_images': paths, 'images': [], 'guards': {}, 'root': root, 'code': {}, 'args': SimpleNamespace(execution_sha256='a' * 64),
            'own_root': root, 'own_code': {}, 'export_args': SimpleNamespace(execution_sha256='a' * 64)}
        exporter = SimpleNamespace(FILES=set(), closure=lambda *a: {}, file_json=lambda *a: {},
            selected_manifest=lambda *a: {'original_rows': []}, image_rows_node=lambda *a: None)
        genuine = {'prior': prior, 'reference': SimpleNamespace(bootstrap=lambda *a: {}), 'guards': {},
            'root': root, 'args': SimpleNamespace(execution_sha256='a' * 64), 'code': {},
            'launch': {'partition': {}, 'image_rows': {'path': str(root / 'rows.json')}},
            'selected': {'original_rows': [], 'resolved_paths': []}}
        self.legacy.update(source_driver=self.source, extract=self.extract, prior=prior,
            warm_record={'origins': {'files': {}, 'modules': {}}},
            selected={'packages': {}, 'source_cpu': {'origins': {'files': {}, 'modules': {}}}, 'genuine': genuine, 'exporter': exporter},
            root=oldroot, args=SimpleNamespace(phase='mechanics', execution_sha256=oldsha), code=oldcode, phase_seconds={})
        self.context['fit_context'].update(legacy=self.legacy, old=self.old, root=fitroot, code=fitcode,
            args=SimpleNamespace(phase='fit', execution_sha256=fitsha), phase_seconds={}, unit_started=time.perf_counter())
        self.fitter.prepare_readout(self.context['fit_context'])
        self.api = None
        self.original_state = [(m, dict(vars(m))) for m in (self.old, self.fitter, self.original, self.source, self.extract)]
    def set_origins(self, files, native=None, modules=None):
        self.inventory.write_text(json.dumps({'files': files, 'native_files': list(files) if native is None else native,
                                              'modules': modules or {}}))
    def admit(self):
        with patch.object(driver, 'NATIVE_PROOF_PINS', self.pins):
            self.api = driver.native_source_api(self.context)
        return self.api
    def unchanged_originals(self, case):
        for module, members in self.original_state:
            case.assertEqual(vars(module).keys(), members.keys())
            case.assertTrue(all(vars(module)[n] is v for n, v in members.items()))


class NearestRankingTests(unittest.TestCase):

    def test_native_actual_adapters_exact_four_and_exit_mutation_matrix(self):
        started = time.perf_counter()
        with TemporaryDirectory() as directory, patch.dict(sys.modules):
            f = NativeAdmissionFixture(Path(directory))
            archived_source = f.context['source']
            with self.assertRaisesRegex(ValueError, 'unknown or changed'):
                f.old.audit_origins(f.legacy)
            api = f.admit()
            self.assertIsNot(f.context['source'], archived_source)
            self.assertNotIn('native_authority', archived_source)
            api.audit_origins(f.legacy, require_exact=True)
            with self.assertRaisesRegex(ValueError, 'original CPU'):
                api.audit_origins(f.legacy, initial=True)
            with redirect_stdout(io.StringIO()):
                api.exit_rehash(f.context['fit_context'])
            for files, native, modules, error in [
                (f.files, [], {}, 'missing native'),
                ({**f.files, f.bulk[0]['path']: f.bulk[0]['sha256']}, list(f.files), {}, 'unknown or changed'),
                (dict(list(f.files.items())[:-1]), list(f.files), {}, 'exact four'),
                (f.files, list(f.files), {'torch.unknown': '/unknown.py'}, 'unknown or changed')]:
                f.set_origins(files, native, modules)
                with self.assertRaisesRegex(ValueError, error): api.audit_origins(f.legacy, require_exact=True)
                if error != 'exact four':
                    f.context['fit_context']['phase_seconds'].clear()
                    with self.assertRaisesRegex(ValueError, error): api.exit_rehash(f.context['fit_context'])
            f.set_origins(f.files)
            for owner, name in [(f.source, 'imported_origins'), (f.extract, 'sha'), (f.original.FlatAdmission, 'bound_file')]:
                with patch.object(owner, name, lambda *a: {}):
                    with self.assertRaisesRegex(ValueError, 'binding changed|dependency changed'):
                        api.exit_rehash(f.context['fit_context'])
            for name, replacement in [('source_driver', SimpleNamespace(imported_origins=lambda *a: {'files': {}, 'modules': {}, 'native_files': []})),
                                      ('extract', SimpleNamespace(sha=f.extract.sha))]:
                with patch.dict(f.legacy, {name: replacement}):
                    with self.assertRaisesRegex(ValueError, 'context dependency binding changed'):
                        api.audit_origins(f.legacy)
            with patch.object(f.extract.hashlib, 'sha256', lambda *a: None):
                with self.assertRaisesRegex(ValueError, 'hash dependency changed'): api.exit_rehash(f.context['fit_context'])
            with patch.dict(f.fitter.ORIGINAL_CODE, {'extra.py': 'a' * 64}):
                with self.assertRaisesRegex(ValueError, 'global contents changed'): api.exit_rehash(f.context['fit_context'])
            fn = f.source.imported_origins
            code = fn.__code__
            try:
                fn.__code__ = (lambda *a: {}).__code__
                with self.assertRaisesRegex(ValueError, 'dependency changed'): api.audit_origins(f.legacy)
            finally: fn.__code__ = code
            # The dispatch dictionary is private; its exported closure still lets the falsifier tamper it.
            fitter_exit = next(c.cell_contents for c in api.exit_rehash.__closure__ if
                              isinstance(c.cell_contents, driver.FunctionType) and c.cell_contents.__name__ == 'exit_rehash')
            private = fitter_exit.__globals__
            with patch.dict(private, {'_nearest_quadratic_exit': lambda *a: None}):
                with self.assertRaisesRegex(ValueError, 'global binding changed'): api.exit_rehash(f.context['fit_context'])
            original_audit = private['_nearest_quadratic_exit'].__globals__['audit_origins']
            with patch.dict(private['_nearest_quadratic_exit'].__globals__, {'audit_origins': lambda *a, **kw: None}):
                with self.assertRaisesRegex(ValueError, 'global binding changed'): api.exit_rehash(f.context['fit_context'])
            self.assertIs(private['_nearest_quadratic_exit'].__globals__['audit_origins'], original_audit)
            for path in (Path(next(iter(f.files))), Path(f.authority['proof']['path']), Path(f.source.__file__)):
                raw, saved = path.read_bytes(), path.stat()
                try:
                    path.write_bytes(b'x' * len(raw)); os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
                    for action in (lambda: api.audit_origins(f.legacy), lambda: api.exit_rehash(f.context['fit_context'])):
                        with self.assertRaisesRegex(ValueError, 'SHA256'): action()
                finally: path.write_bytes(raw)
            with patch.dict(f.context['native_source_owned'], {'authenticate': lambda: None}):
                with self.assertRaisesRegex(ValueError, 'owned API/supplement changed'):
                    driver.native_source_api(f.context)
            with patch.object(api, 'audit_origins', lambda *a, **kw: None):
                with self.assertRaisesRegex(ValueError, 'owned API/supplement changed'): api.exit_rehash(f.context['fit_context'])
            original_kwdefaults = api.audit_origins.__kwdefaults__
            try:
                api.audit_origins.__kwdefaults__ = {'require_exact': True}
                with self.assertRaisesRegex(ValueError, 'private function changed'): api.exit_rehash(f.context['fit_context'])
            finally: api.audit_origins.__kwdefaults__ = original_kwdefaults
            defaults = fitter_exit.__defaults__
            try:
                fitter_exit.__defaults__ = (object(),)
                with self.assertRaisesRegex(ValueError, 'private function changed'): api.exit_rehash(f.context['fit_context'])
            finally: fitter_exit.__defaults__ = defaults
            private_audit = next(c.cell_contents for c in api.audit_origins.__closure__ if
                                 isinstance(c.cell_contents, driver.FunctionType) and c.cell_contents.__name__ == 'audit_origins')
            with patch.dict(private_audit.__globals__, {'_nearest_supplement': {'files': {}, 'modules': {}}}):
                with self.assertRaisesRegex(ValueError, 'global binding changed'): api.exit_rehash(f.context['fit_context'])
            prior = f.legacy['selected']['genuine']['prior']
            f.context['fit_context']['phase_seconds'].clear()
            prior['images'] = ['FIT failure']
            with self.assertRaisesRegex(ValueError, 'FIT image resolution'):
                api.exit_rehash(f.context['fit_context'])
            prior['images'] = []
            genuine = f.legacy['selected']['genuine']
            f.context['fit_context']['phase_seconds'].clear()
            genuine['selected']['original_rows'] = [0]
            with self.assertRaisesRegex(ValueError, 'TRAIN mapping'):
                api.exit_rehash(f.context['fit_context'])
            genuine['selected']['original_rows'] = []
            f.legacy['warm_record']['origins']['files'][next(iter(f.files))] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'conflicting original'):
                api.audit_origins(f.legacy)
            f.legacy['warm_record']['origins']['files'].clear()
            f.unchanged_originals(self)
            self.assertLess(sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()), 16 * 1024**2)
        self.assertLess(time.perf_counter() - started, 15)


    def test_native_vendor_record_metadata_semantics(self):
        with TemporaryDirectory() as directory, patch.dict(sys.modules):
            f = NativeAdmissionFixture(Path(directory))
            baseline_proof = copy.deepcopy(f.proof)
            record = Path(f.records[0]); metadata = record.with_name('METADATA')
            record_raw, metadata_raw = record.read_bytes(), metadata.read_bytes()
            base_source = dict(f.context['source'])
            for kind, error in [('duplicate', 'unique installed'), ('hash', 'RECORD hash/size'),
                                ('size', 'RECORD hash/size'), ('owner', 'unique installed'),
                                ('version', 'distribution differs')]:
                record.write_bytes(record_raw); metadata.write_bytes(metadata_raw)
                proof = copy.deepcopy(baseline_proof)
                if kind == 'duplicate': record.write_bytes(record_raw + record_raw.splitlines(keepends=True)[0])
                elif kind == 'hash': record.write_bytes(record_raw.replace(b'sha256=', b'sha512=', 1))
                elif kind == 'size': record.write_bytes(record_raw.replace(b',7', b',8', 1))
                elif kind == 'owner': proof['installed_record_ownership']['owners'][next(iter(f.files))] = []
                else:
                    metadata.write_bytes(metadata_raw.replace(b'9.20.0.48', b'9.20.0.49'))
                for path in (record, metadata):
                    value = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size}
                    proof['input_guards'][str(path)] = value
                    if path == metadata:
                        proof['comparison']['metadata']['METADATA'].update(installed_sha256=value['sha256'], wheel_sha256=value['sha256'])
                authority = copy.deepcopy(f.authority)
                authority['proof'] = f.write_json('vendor-proof.json', proof)
                decision = json.loads(Path(authority['decision']['path']).read_text())
                decision['proof_sha256'] = authority['proof']['sha256']
                authority['decision'] = f.write_json('vendor-decision.json', decision)
                launch = {'native_authority': f.write_json('native-authority.json', authority)}
                context = {**f.context, 'guards': {}, 'source': base_source, 'launch': launch}
                pins = {n: authority[n]['sha256'] for n in driver.NATIVE_PROOF_PINS}
                with patch.object(driver, 'NATIVE_PROOF_PINS', pins), self.assertRaisesRegex(ValueError, error):
                    driver.bind_native_authority(context)

    def test_native_source_owned_api_exists(self):
        self.assertTrue(callable(getattr(driver, "native_source_api", None)), "owned native API missing")

    def test_fitter_startup_exact_ast_inverse_and_private_namespace(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            before = dict(vars(fixture.fitter))
            adapter = driver.startup_admission_adapter(fixture.fitter, {})
            original = {n.name: n for n in ast.parse(Path(fixture.fitter.__file__).read_bytes()).body
                        if isinstance(n, ast.FunctionDef)}
            pins = {'authority': 'ad95ed5ddb58e40230e4ef2a942569bddfdf1da727bd00ed6311372ab5ac98d6',
                'admit_historical_linear': '22293a825a9f8ad1c24e75abe41464b72c19aaa8e44c65b34ec35c4d247d05ec',
                'admit_terminal': 'c8722d6eed5469a76dce1a26a19036ff69a54da9f55d8b999f08d3e81b5302b0'}
            for name, pin in pins.items():
                node = copy.deepcopy(getattr(adapter, name).__startup_ast__)
                self.assertEqual(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(), pin)
                changed = 0
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and \
                            ast.unparse(call.func) == "context['legacy']['admission'].bound_file":
                        self.assertEqual(ast.unparse(call), "context['legacy']['admission'].bound_file(guards, path, digest)")
                        call.func = ast.Name(id='bound_file', ctx=ast.Load())
                        changed += 1
                self.assertEqual(changed, 0 if name == 'authority' else 1)
                self.assertEqual(ast.dump(node, include_attributes=False), ast.dump(original[name], include_attributes=False))
            namespace = adapter.authority.__globals__
            self.assertIsNot(namespace, vars(fixture.fitter))
            for name, value in before.items():
                self.assertIs(vars(fixture.fitter)[name], value)
                if name not in pins:
                    self.assertIs(namespace[name], value)

    def test_fitter_startup_entry_rejects_reader_source_method_and_global_mutants(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            adapter = driver.startup_admission_adapter(fixture.fitter, {})
            calls = (lambda: adapter.admit_historical_linear(fixture.context['fit_context']),
                     lambda: adapter.admit_terminal(fixture.context['fit_context'], {}, 'fit', 'concat'))
            for call in calls:
                for name in ('__init__', 'canonical', 'digest_string', 'register', 'bound_file',
                             'read_json', 'descriptor_json', 'admit_terminal'):
                    with self.subTest(entry=call, method=name), patch.object(fixture.admission, name, lambda *a: None):
                        with self.assertRaisesRegex(ValueError, 'reader method changed'):
                            call()
                    method = getattr(fixture.original.FlatAdmission, name)
                    foreign = driver.FunctionType(method.__code__, dict(vars(fixture.original)), name)
                    if name in ('canonical', 'digest_string'):
                        foreign = staticmethod(foreign)
                    with patch.object(fixture.original.FlatAdmission, name, foreign):
                        with self.assertRaisesRegex(ValueError, 'reader method changed'):
                            call()
                fn = fixture.original.bound_file
                for replacement in (lambda *a: None, driver.FunctionType(fn.__code__, dict(vars(fixture.original)))):
                    with patch.object(fixture.original, 'bound_file', replacement):
                        with self.assertRaisesRegex(ValueError, 'global bound_file changed'):
                            call()
                code = fn.__code__
                try:
                    fn.__code__ = (lambda *a: None).__code__
                    with self.assertRaisesRegex(ValueError, 'global bound_file changed'):
                        call()
                finally:
                    fn.__code__ = code
                with patch.object(fixture.original.__spec__, 'origin', str(fixture.root / 'wrong.py')):
                    with self.assertRaisesRegex(ValueError, 'actual original startup'):
                        call()
                source = fixture.write('changed-reader.py', Path(fixture.original.__file__).read_bytes() + b'\n')
                with patch.object(fixture.original, '__file__', source['path']), \
                        patch.object(fixture.original.__spec__, 'origin', source['path']), \
                        patch.dict(fixture.context['fit_context']['guards'], {source['path']: fixture.fitter.TERMINAL_SOURCE_SHA}):
                    with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                        call()

    def test_fitter_startup_constructor_rejects_full_source_and_live_function_mutants(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            for name in ('authority', 'admit_historical_linear', 'admit_terminal'):
                fn = getattr(fixture.fitter, name)
                for replacement in (lambda *a: None, driver.FunctionType(fn.__code__, dict(vars(fixture.fitter)))):
                    with self.subTest(name=name), patch.object(fixture.fitter, name, replacement):
                        with self.assertRaisesRegex(ValueError, 'actual startup fitter function/body differs'):
                            driver.startup_admission_adapter(fixture.fitter, {})
                code = fn.__code__
                try:
                    fn.__code__ = (lambda *a: None).__code__
                    with self.assertRaisesRegex(ValueError, 'actual startup fitter function/body differs'):
                        driver.startup_admission_adapter(fixture.fitter, {})
                finally:
                    fn.__code__ = code
            source = fixture.write('changed-fitter.py', Path(fixture.fitter.__file__).read_bytes() + b'\n')
            with patch.object(fixture.fitter, '__file__', source['path']), \
                    patch.object(fixture.fitter.__spec__, 'origin', source['path']):
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    driver.startup_admission_adapter(fixture.fitter, {})
            with patch.object(fixture.fitter.__spec__, 'origin', '/wrong'):
                with self.assertRaisesRegex(ValueError, 'source origin differs'):
                    driver.startup_admission_adapter(fixture.fitter, {})

    def test_fitter_startup_inventory_terminal_and_invocation_mutants(self):
        for stage in ('historical', 'signed'):
            for mutation in ('sha', 'size', 'stage', 'unseen-bytes', 'required', 'terminal', 'duplicate', 'source'):
                with self.subTest(stage=stage, mutation=mutation), TemporaryDirectory() as directory, patch.dict(sys.modules):
                    fixture = FitterStartupFixture(Path(directory))
                    if stage == 'signed':
                        fixture.run_historical()
                    unit, record = fixture.hist_fit if stage == 'historical' else fixture.concat
                    fact = fixture.bulk[1]
                    record['input_guards'][fact['path']] = fact['sha256']
                    if mutation in ('sha', 'size'):
                        fixture.admission.bound_file({}, fact['path'], fact['sha256'])
                    if mutation == 'sha':
                        record['input_guards'][fact['path']] = '0' * 64
                    elif mutation == 'size':
                        Path(fact['path']).write_bytes(b'bulk' * 1024 + b'x')
                    elif mutation == 'stage':
                        fixture.context['guards'][fact['path']] = '0' * 64
                    elif mutation == 'unseen-bytes':
                        Path(fact['path']).write_bytes(b'FAIL' * 1024)
                    elif mutation == 'required':
                        record['input_guards'].pop(fixture.bulk[0]['path'])
                    elif mutation == 'terminal':
                        record['pass'] = False
                    elif mutation == 'duplicate':
                        fixture.legacy['invocations'].add(unit['invocation_id'])
                    elif mutation == 'source':
                        record['source'] = {'fixture': 'substituted'}
                    fixture.rewrite(unit, record)
                    with self.assertRaises(ValueError):
                        if stage == 'historical':
                            fixture.adapter.admit_historical_linear(fixture.context)
                        else:
                            fixture.adapter.admit_terminal(fixture.context, unit, 'fit', 'concat')
                    if mutation == 'unseen-bytes':
                        self.assertNotIn(fact['path'], fixture.admission.verified)
                    fixture.unchanged(self)

    def test_startup_boundary_rejects_unrelated_predicate_and_extra_dispatch_mutants(self):
        original_sha = '2b749548e57aa010826665a38e4e145adaeefed9658a3d8a86f470134f312a72'
        for name in ('policy', 'check_launch', 'exit_rehash', 'restore', 'integrity'):
            tree = ast.parse(PATH.read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            fn.body.append(ast.parse('unrelated_predicate = False').body[0])
            self.assertNotEqual(startup_source_boundary(tree), original_sha)
        tree = ast.parse(PATH.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authority')
        extra = next(n for n in fn.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and
                     ast.unparse(n.value.func) == 'startup.authority')
        fn.body.append(copy.deepcopy(extra))
        with self.assertRaisesRegex(ValueError, 'exact two startup dispatches'):
            startup_source_boundary(tree)

    def test_fitter_startup_four_sweeps_one_authentication_and_fresh_exit(self):
        self.assertTrue(callable(getattr(driver, 'startup_admission_adapter', None)), 'startup adapter is missing')
        started = time.perf_counter()
        with TemporaryDirectory() as directory, patch.dict(sys.modules):
            fixture = FitterStartupFixture(Path(directory))
            initial = {**fixture.context, 'guards': dict(fixture.context['guards']),
                       'terminals': {}, 'terminal_cgroups': {}, 'phase_seconds': {}}
            adapted = {name: getattr(fixture.adapter, name) for name in ('admit_historical_linear', 'admit_terminal')}
            # The same FILEs and predicates first run through the original
            # pinned bodies, then a fresh invocation-owned reader runs the adapter.
            for name in adapted:
                setattr(fixture.adapter, name, driver.FunctionType(getattr(fixture.fitter, name).__code__,
                                                                 fixture.namespace, name))
            fixture.historical_guards = {}
            with fixture.count_reads():
                fixture.all_sweeps()
            self.assertEqual(fixture.read_bytes[fixture.bulk[0]['path']], 5 * 4096)
            self.assertEqual(fixture.read_bytes[fixture.bulk[1]['path']], 4096)
            baseline_required, baseline_union = dict(fixture.context['required_guards']), dict(fixture.context['guards'])
            fixture.unchanged(self)
            fixture.context = initial
            fixture.legacy['invocations'].clear()
            fixture.admission = fixture.original.FlatAdmission()
            fixture.admission.init = fixture.init
            fixture.legacy['admission'] = fixture.admission
            del sys.modules['_prototype_historical_linear']
            for name, function in adapted.items():
                setattr(fixture.adapter, name, function)
            fixture.read_bytes.clear()
            fixture.historical_guards = {}
            with fixture.count_reads():
                fixture.all_sweeps()
            for fact in fixture.bulk:
                self.assertEqual(fixture.read_bytes[fact['path']], 4096, 'one genuine SHA read per unique bulk path')
            self.assertEqual(fixture.historical_guards, {fixture.bulk[0]['path']: fixture.bulk[0]['sha256']})
            self.assertEqual(fixture.context['required_guards'], baseline_required)
            self.assertEqual(fixture.context['guards'], baseline_union)
            self.assertEqual(set(fixture.context['terminal_cgroups']),
                             {'historical_linear:cpu', 'historical_linear:fit', 'cpu:linear', 'fit:concat'})
            self.assertNotIn(fixture.bulk[1]['path'], fixture.context['required_guards'])
            self.assertIn(fixture.bulk[1]['path'], fixture.context['guards'])
            for unit, record in fixture.records.values():
                for p, h in record['input_guards'].items():
                    self.assertEqual(fixture.context['guards'][p], h)
            fixture.unchanged(self)
            # Run the genuine fitter union and nearest exit loops; isolate only
            # unrelated native/source closure work, never their FILE readers.
            fixture.read_bytes.clear()
            with fixture.count_reads():
                fixture.quadratic_exit()
            for fact in fixture.bulk:
                self.assertEqual(fixture.read_bytes[fact['path']], 4096, 'quadratic exit needs a fresh genuine reader')
            fixture.read_bytes.clear()
            with patch.object(fixture.fitter, 'prepare_readout'), patch.object(fixture.old, 'exit_rehash'), \
                    patch.object(fixture.fitter, 'closure', return_value=fixture.context['code']), \
                    fixture.count_reads(), redirect_stdout(io.StringIO()):
                fixture.fitter.exit_rehash(fixture.context)
            for fact in fixture.bulk:
                self.assertEqual(fixture.read_bytes[fact['path']], 4096, 'fitter exit must reread bulk bytes')
            nearest = {**fixture.context, 'fit_context': fixture.context, 'fitter': fixture.fitter, 'models': [],
                       'started': time.perf_counter()}
            fixture.read_bytes.clear()
            with patch.object(driver, 'helper_guard'), patch.object(fixture.fitter, 'exit_rehash'), \
                    patch.object(driver, 'native_source_api', lambda c: fixture.fitter), \
                    patch.object(driver, 'closure', return_value=nearest['code']), \
                    fixture.count_reads(), redirect_stdout(io.StringIO()):
                driver.exit_rehash(nearest)
            for fact in fixture.bulk:
                self.assertEqual(fixture.read_bytes[fact['path']], 4096, 'nearest exit must reread bulk bytes')
            fact = fixture.bulk[0]
            stamp = Path(fact['path']).stat()
            Path(fact['path']).write_bytes(b'FAIL' * 1024)
            os.utime(fact['path'], ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            fixture.admission.bound_file({}, fact['path'], fact['sha256'])  # Genuine startup snapshot.
            with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                fixture.quadratic_exit()
            with patch.object(fixture.fitter, 'prepare_readout'), patch.object(fixture.old, 'exit_rehash'), \
                    patch.object(fixture.fitter, 'closure', return_value=fixture.context['code']), redirect_stdout(io.StringIO()):
                # Reset diagnostic markers from the successful earlier audit.
                fixture.context['phase_seconds'].clear()
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    fixture.fitter.exit_rehash(fixture.context)
            with patch.object(driver, 'helper_guard'), patch.object(fixture.fitter, 'exit_rehash'), \
                    patch.object(driver, 'native_source_api', lambda c: fixture.fitter), \
                    patch.object(driver, 'closure', return_value=nearest['code']), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    driver.exit_rehash(nearest)
            fixture.unchanged(self)
            self.assertLess(sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()), 16 * 1024**2)
        self.assertLess(time.perf_counter() - started, 5)

    def launch(self, phase='cpu', arm='control'):
        unit = {'receipt': {'path': '/receipt', 'sha256': 'a' * 64},
                'log': {'path': '/log', 'sha256': 'b' * 64}, 'unit': 'new-unit',
                'invocation_id': 'c' * 32, 'service_seconds': 1., 'native_peak_rss_kib': 1,
                'both_locks_held': True}
        args = SimpleNamespace(execution_sha256='d' * 64, phase=phase, arm=arm, seed=driver.SEED)
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': args.execution_sha256,
            'phase': phase, 'arm': arm, 'seed': driver.SEED, 'fitter': copy.deepcopy(driver.FITTER),
            'accepted': copy.deepcopy(driver.ACCEPTED), 'recipe': copy.deepcopy(driver.RECIPE),
            'resource_policy': driver.policy(phase), 'both_locks_held': True,
            'native_authority': {'path': '/native-authority.json', 'sha256': 'a' * 64},
            'selected_cpu': None if phase == 'cpu' else unit,
            'selected_mechanics': {a: copy.deepcopy(unit) for a in driver.ARMS} if phase == 'train' else None}
        return launch, args

    def test_startup_terminal_reuses_actual_sha_reads_and_separate_inventories(self):
        started = time.perf_counter()
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            with fixture.count_reads():
                fact = fixture.bulk[0]
                fixture.admission.bound_file(fixture.historical, fact['path'], fact['sha256'])
                first = driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
                second = driver.admit_terminal(fixture.context, fixture.mechanics[0], 'mechanics', 'control')
            for fact in fixture.bulk:
                self.assertEqual(fixture.read_bytes[fact['path']], Path(fact['path']).stat().st_size,
                                 'one actual SHA read per unique bulk path')
            self.assertEqual(fixture.historical, {fixture.bulk[0]['path']: fixture.bulk[0]['sha256']})
            self.assertEqual(first['input_guards'], fixture.context['required_guards'])
            self.assertEqual(second['input_guards'], {f['path']: f['sha256'] for f in fixture.bulk})
            for record in (first, second):
                for p, h in record['input_guards'].items():
                    self.assertEqual(fixture.context['guards'][p], h)
            self.assertEqual(set(fixture.context['terminals']), {'cpu:control', 'mechanics:control'})
            self.assertLess(sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()), 16 * 1024**2)
        self.assertLess(time.perf_counter() - started, 5)

    def test_startup_only_source_boundary_preserves_complete_other_ast(self):
        tree = ast.parse(PATH.read_text())
        # Entire 6e5b3f20 module: pins fresh/restore/diagnostics, native/runtime/exit,
        # objectives, science, caps, globals and imports after exact startup inversion.
        self.assertEqual(startup_source_boundary(tree),
                         '2b749548e57aa010826665a38e4e145adaeefed9658a3d8a86f470134f312a72')

    def test_startup_admission_preserves_default_json_and_all_terminal_predicates(self):
        functions = {n.name: n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef)}
        read = functions['read_json']
        read.args.kwonlyargs, read.args.kw_defaults = [], []
        del read.body[1]  # Remove only the explicit reader branch.
        self.assertEqual(hashlib.sha256(ast.dump(read, include_attributes=False).encode()).hexdigest(),
                         '1f56b8649a8138fddb3a4d73b98f7f3b896cdcfb90ef051527c0515f37601bb8')
        terminal = functions['admit_terminal']
        del terminal.body[2:9]  # Remove only the new original-reader authentication block.
        for node in ast.walk(terminal):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'read_json':
                node.keywords = [k for k in node.keywords if k.arg != 'admission']
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    isinstance(node.func.value, ast.Name) and node.func.value.id == 'admission' and node.func.attr == 'bound_file':
                node.func = ast.Name(id='bound_file', ctx=ast.Load())
        self.assertEqual(hashlib.sha256(ast.dump(terminal, include_attributes=False).encode()).hexdigest(),
                         'bb7c838ef29b56665a8d50e34ed2250ca16d2370211e3b49e40d4860741db7ac')

    def test_startup_json_snapshot_strict_parse_caps_and_default_uncached_semantics(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            fact = fixture.write('json', b'{"value":[1]}')
            guards = {}
            first = driver.read_json(fact, guards, admission=fixture.admission)
            first['value'].append(2)
            self.assertEqual(driver.read_json(fact, {}, admission=fixture.admission), {'value': [1]})
            self.assertEqual(guards, {fact['path']: fact['sha256']})
            stamp = Path(fact['path']).stat()
            Path(fact['path']).write_bytes(b'{"value":[2]}')
            os.utime(fact['path'], ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertEqual(driver.read_json(fact, {}, admission=fixture.admission), {'value': [1]})
            with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                driver.read_json(fact, {})
            for i, raw in enumerate((b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}')):
                bad = fixture.write('bad-json-' + str(i), raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    driver.read_json(bad, {}, admission=fixture.admission)
            wrong = fixture.write('wrong-json', b'{"value":1}')
            with self.assertRaisesRegex(ValueError, 'JSON size/SHA256'):
                driver.read_json({**wrong, 'sha256': '0' * 64}, {}, admission=fixture.admission)
            self.assertNotIn(wrong['path'], fixture.admission.verified)
            bounded = fixture.write('bounded-json', b'{"value":1}')
            with self.assertRaisesRegex(ValueError, 'JSON size/SHA256'):
                fixture.admission.read_json(bounded['path'], bounded['sha256'], {}, cap=4)
            self.assertEqual(driver.read_json(bounded, {}, admission=fixture.admission), {'value': 1})
            with self.assertRaisesRegex(ValueError, 'authority JSON too large'):
                fixture.admission.read_json(bounded['path'], bounded['sha256'], {}, cap=4)
            for bad in ({**fact, 'extra': 1}, {**fact, 'path': 'relative'}, {**fact, 'sha256': 'BAD'}, []):
                with self.subTest(fact=bad), self.assertRaisesRegex(ValueError, 'exact FILE'):
                    driver.read_json(bad, {}, admission=fixture.admission)

    def test_startup_terminal_inventory_conflicts_and_first_wrong_bytes(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            def reset():
                fixture.admission = fixture.original.FlatAdmission()
                fixture.admission.init = fixture.init
                fixture.legacy['admission'] = fixture.admission
                fixture.context['guards'] = {}
                fixture.legacy['invocations'].clear()
            fact = fixture.bulk[1]
            for case in ('sha', 'size', 'stage', 'wrong-bytes', 'missing', 'noncanonical'):
                reset()
                unit, record = copy.deepcopy(fixture.mechanics)
                guards = fixture.context['guards']
                path = Path(fact['path'])
                raw = b'bulk' * 1024
                path.write_bytes(raw)
                if case in ('sha', 'size'):
                    fixture.admission.bound_file({}, path, fact['sha256'])
                if case == 'sha':
                    record['input_guards'][str(path)] = '0' * 64
                elif case == 'size':
                    path.write_bytes(raw + b'x')
                elif case == 'stage':
                    guards[str(path)] = '0' * 64
                elif case == 'wrong-bytes':
                    path.write_bytes(b'FAIL' * 1024)
                elif case == 'missing':
                    path.unlink()
                elif case == 'noncanonical':
                    link = fixture.root / 'link'
                    link.symlink_to(path)
                    record['input_guards'].pop(str(path))
                    record['input_guards'][str(link)] = fact['sha256']
                fixture.rewrite(unit, record)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    driver.admit_terminal(fixture.context, unit, 'mechanics', 'control')
                if case == 'wrong-bytes':
                    self.assertNotIn(str(path), fixture.admission.verified)
            # Supplied size and stage authorities remain independent of cached verification.
            reset()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'size differs'):
                fixture.admission.bound_file({}, path, fact['sha256'], size=len(raw) + 1)
            with self.assertRaisesRegex(ValueError, 'stage file authority'):
                fixture.admission.bound_file({str(path): '0' * 64}, path, fact['sha256'])

    def test_startup_reader_substitution_live_code_and_original_adapter_pins(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            adapter = fixture.fitter.original_terminal_reader(fixture.context['fit_context'])
            self.assertEqual(hashlib.sha256(ast.dump(adapter.__terminal_ast__, include_attributes=False).encode()).hexdigest(),
                             fixture.fitter.ADAPTED_TERMINAL_AST_SHA)
            for name in ('__init__', 'canonical', 'digest_string', 'register', 'bound_file',
                         'read_json', 'descriptor_json', 'admit_terminal'):
                original = getattr(fixture.original.FlatAdmission, name)
                with self.subTest(instance=name), patch.object(fixture.admission, name, lambda *a: None):
                    with self.assertRaisesRegex(ValueError, 'reader method changed'):
                        driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
                code = original.__code__
                try:
                    original.__code__ = (lambda *a: None).__code__
                    with self.subTest(class_method=name), self.assertRaisesRegex(ValueError, 'reader method changed'):
                        driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
                finally:
                    original.__code__ = code
            # Same function bound to another reader, foreign globals, and a
            # copied code object with the wrong filename must also be rejected.
            other = fixture.original.FlatAdmission()
            with patch.object(fixture.admission, 'bound_file', other.bound_file):
                with self.assertRaisesRegex(ValueError, 'reader method changed'):
                    driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
            fn = fixture.original.FlatAdmission.register
            replacement = type(fn)(fn.__code__, {'__name__': fixture.original.__name__})
            with patch.object(fixture.original.FlatAdmission, 'register', replacement):
                with self.assertRaisesRegex(ValueError, 'reader method changed'):
                    driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
            code = fn.__code__
            try:
                fn.__code__ = code.replace(co_filename='/substituted')
                with self.assertRaisesRegex(ValueError, 'reader method changed'):
                    driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
            finally:
                fn.__code__ = code
            terminal = fixture.original.FlatAdmission.admit_terminal
            code = terminal.__code__
            try:
                terminal.__code__ = (lambda *a: None).__code__
                with self.assertRaises(ValueError):
                    fixture.fitter.original_terminal_reader(fixture.context['fit_context'])
            finally:
                terminal.__code__ = code
            class Substituted(fixture.original.FlatAdmission):
                pass
            for replacement in (Substituted(), SimpleNamespace(descriptor_json=lambda *a: {})):
                with patch.dict(fixture.legacy, admission=replacement), self.assertRaisesRegex(ValueError, 'actual original'):
                    driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
            with patch.object(fixture.original.__spec__, 'origin', '/substituted'), self.assertRaisesRegex(ValueError, 'actual original'):
                driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')
            with patch.dict(fixture.context['fit_context']['guards'], {fixture.original.__file__: '0' * 64}):
                with self.assertRaisesRegex(ValueError, 'actual original'):
                    driver.admit_terminal(fixture.context, fixture.cpu[0], 'cpu', 'control')

    def test_startup_terminal_predicate_mutants_still_reject(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            common = [('schema', 'wrong'), ('phase', 'train'), ('arm', 'candidate'), ('seed', 1),
                ('optimizer_members', 3), ('trainable_scalars', 1), ('quality_read', True),
                ('pass', False), ('strict_reload_exact', False), ('exit_rehash_pass', False),
                ('sequential_model_ownership', False), ('forward_oracle_exact', False),
                ('native_training_inference_exact', False), ('both_locks_held_in_parent_authority', False),
                ('total_training_core_seconds', 0), ('wall_seconds', 500), ('process_peak_rss_kib', 8 * 1024**2 + 1),
                ('resource_policy', {}), ('code', {}), ('source', {}), ('execution_sha256', '0' * 64),
                ('numerical_flags', {}), ('authority_sha256', '0' * 64), ('input_guards', {})]
            cpu = [('completed_step', 1), ('cuda_initialized', True), ('peak_cuda_allocated_bytes', 1),
                ('initial_arm_parity', False), ('cpu_serialization_exact', False), ('bypass_version_tamper_rejected', False),
                ('malformed_state_rejected', False), ('native_loss_reduction_exact', False), ('native_role_mutation_rejected', False)]
            mechanics = [('completed_step', 16), ('cuda_initialized', False), ('peak_cuda_allocated_bytes', 10_000_000_000),
                ('source_substitution_rejected', False), ('inference_artifact_independent', False),
                ('training_state_discarded', False), ('replay_exact', False), ('steps', []), ('resumed_steps', [])]
            cases = [('cpu', k, v) for k, v in common + cpu] + [('mechanics', k, v) for k, v in mechanics]
            for phase, key, value in cases:
                fixture.admission = fixture.original.FlatAdmission()
                fixture.admission.init = fixture.init
                fixture.legacy['admission'] = fixture.admission
                fixture.legacy['invocations'].clear()
                fixture.context['guards'] = {}
                unit, record = copy.deepcopy(fixture.cpu if phase == 'cpu' else fixture.mechanics)
                record[key] = value
                fixture.rewrite(unit, record)
                with self.subTest(phase=phase, key=key), self.assertRaises((ValueError, KeyError)):
                    driver.admit_terminal(fixture.context, unit, phase, 'control')
            nested = [('invocation', 'optimize', 1), ('invocation', 'python', '/other'),
                ('invocation', 'python_sha256', '0' * 64), ('invocation', 'python_version', 'other'),
                ('invocation', 'argv', []), ('invocation', 'invocation_id', '0' * 32),
                ('invocation', 'cuda_visible_devices', '0'), ('identity', 'method', {}),
                ('identity', 'parameter_names', []), ('identity', 'source', {}), ('identity', 'numerical_flags', {}),
                ('launch', 'both_locks_held', False), ('launch', 'selected_cpu', fixture.cpu[0])]
            for parent, key, value in nested:
                fixture.legacy['admission'] = fixture.original.FlatAdmission()
                fixture.legacy['admission'].init = fixture.init
                fixture.legacy['invocations'].clear()
                fixture.context['guards'] = {}
                unit, record = copy.deepcopy(fixture.cpu)
                record[parent][key] = value
                fixture.rewrite(unit, record)
                with self.subTest(parent=parent, key=key), self.assertRaises(ValueError):
                    driver.admit_terminal(fixture.context, unit, 'cpu', 'control')
            # Authenticated authority bytes must equal the receipt launch.
            fixture.legacy['admission'] = fixture.original.FlatAdmission()
            fixture.legacy['admission'].init = fixture.init
            fixture.context['guards'] = {}
            unit, record = copy.deepcopy(fixture.cpu)
            record['authority'] = fixture.write_json('wrong-authority', {})
            record['authority_sha256'] = record['authority']['sha256']
            record['invocation']['argv'] = driver.cli(fixture.root, record['authority']['path'], record['authority']['sha256'],
                'd' * 64, 'cpu', 'control', Path(unit['receipt']['path']).parent)
            fixture.rewrite(unit, record)
            with self.assertRaisesRegex(ValueError, 'source/CLI/complete guards'):
                driver.admit_terminal(fixture.context, unit, 'cpu', 'control')

    def test_startup_original_unit_log_cgroup_and_invocation_mutants(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            baseline_log = Path(fixture.cpu[0]['log']['path']).read_text()
            cases = ('locks', 'duration', 'rss', 'reused', 'missing-exit', 'duplicate-exit', 'runtime',
                'footer-invocation', 'footer-peak', 'footer-count', 'cgroup-path', 'memory-cap',
                'swap', 'event-low', 'event-high', 'event-max', 'peak-order')
            for case in cases:
                fixture.legacy['admission'] = fixture.original.FlatAdmission()
                fixture.legacy['admission'].init = fixture.init
                fixture.context['guards'] = {}
                fixture.legacy['invocations'].clear()
                unit, record = copy.deepcopy(fixture.cpu)
                log = baseline_log
                if case == 'locks':
                    unit['both_locks_held'] = False
                elif case == 'duration':
                    unit['service_seconds'] = 501
                elif case == 'rss':
                    unit['native_peak_rss_kib'] = 8 * 1024**2 + 1
                elif case == 'reused':
                    fixture.legacy['invocations'].add(unit['invocation_id'])
                elif case == 'missing-exit':
                    log = log.replace('\tExit status: 0\n', '')
                elif case == 'duplicate-exit':
                    log += '\tExit status: 0\n'
                elif case == 'runtime':
                    log = log.replace('Service runtime: 4.0s', 'Service runtime: 3.0s')
                elif case.startswith('footer-'):
                    footer = json.loads(log.split('FINAL_CGROUP ')[1])
                    if case == 'footer-invocation':
                        footer['invocation_id'] = '0' * 32
                    elif case == 'footer-peak':
                        footer['values']['memory.peak'] = '1'
                    log = log.split('FINAL_CGROUP ')[0] + 'FINAL_CGROUP ' + json.dumps(footer) + '\n'
                    if case == 'footer-count':
                        log += 'FINAL_CGROUP ' + json.dumps(footer) + '\n'
                elif case == 'cgroup-path':
                    record['cgroup_before']['path'] = '/other/fixture-1.service'
                elif case == 'memory-cap':
                    record['cgroup_before']['values']['memory.max'] = '1'
                elif case == 'swap':
                    record['cgroup_before']['values']['memory.swap.peak'] = '1'
                elif case.startswith('event-'):
                    event = case.removeprefix('event-')
                    record['cgroup_before']['values']['memory.events'] = record['cgroup_before']['values']['memory.events'].replace(event + ' 0', event + ' 1')
                elif case == 'peak-order':
                    record['cgroup_before']['values']['memory.peak'] = '4'
                unit['log'] = fixture.write('mutant-log', log.encode())
                fixture.rewrite(unit, record)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    driver.admit_terminal(fixture.context, unit, 'cpu', 'control')

    def test_startup_reuse_does_not_reach_real_fresh_exit_byte_reads(self):
        with TemporaryDirectory() as directory:
            fixture = StartupAdmissionFixture(Path(directory))
            for fact in fixture.bulk:
                fixture.admission.bound_file(fixture.context['guards'], fact['path'], fact['sha256'])
            # Execute the original quadratic exit and nearest exit. Only large
            # historical metadata/topology helpers are stand-ins; all byte readers,
            # fresh reader construction and exit inventory loops run unchanged.
            def closure(root, names):
                root.mkdir(exist_ok=True)
                code = {}
                for name in names:
                    raw = b'exit fixture\n'
                    (root / name).write_bytes(raw)
                    code[name] = hashlib.sha256(raw).hexdigest()
                execution = fixture.write_json(str(root.relative_to(fixture.root) / 'execution.json'), code)
                return code, execution['sha256']
            old_root = fixture.root / 'old-exit'
            old_code, old_sha = closure(old_root, fixture.old.FILES)
            code, sha = closure(fixture.root, driver.FILES)
            guards = {f['path']: f['sha256'] for f in fixture.bulk}
            source = SimpleNamespace(fit_rows=lambda *a: [], bootstrap=lambda *a: (None, {}))
            prior = {'guards': guards, 'all_images': [], 'images': [], 'source_driver': source,
                'extract': None, 'fit': None, 'root': fixture.root, 'args': SimpleNamespace(execution_sha256='0' * 64),
                'code': {}, 'own_root': fixture.root, 'export_args': SimpleNamespace(execution_sha256='0' * 64), 'own_code': {}}
            reference = SimpleNamespace(bootstrap=lambda *a: {})
            genuine = {'prior': prior, 'reference': reference, 'guards': dict(guards),
                'root': fixture.root, 'args': SimpleNamespace(execution_sha256='0' * 64),
                'code': {}, 'launch': {'partition': {}, 'image_rows': {'path': '/metadata'}},
                'selected': {'original_rows': [], 'resolved_paths': []}}
            exporter = SimpleNamespace(FILES=set(), closure=lambda *a: {}, file_json=lambda *a: {},
                selected_manifest=lambda *a: {'original_rows': []}, image_rows_node=lambda *a: None)
            old_context = {'original': fixture.original, 'selected': {'exporter': exporter, 'genuine': genuine},
                'guards': dict(guards), 'root': old_root, 'args': SimpleNamespace(execution_sha256=old_sha, phase='mechanics'),
                'code': old_code, 'phase_seconds': {}}
            fixture.context.update(root=fixture.root, code=code, args=SimpleNamespace(execution_sha256=sha))
            fixture.context['fitter'] = SimpleNamespace(exit_rehash=lambda c: fixture.old.exit_rehash(old_context))
            with patch.object(driver, 'helper_guard', lambda c: None), \
                 patch.object(driver, 'native_source_api', lambda c: c['fitter']), \
                 patch.object(fixture.old, 'audit_origins', lambda *a, **k: None), \
                 patch.object(fixture.original.FlatAdmission, 'all_fit_images', lambda *a: []), redirect_stdout(io.StringIO()):
                fixture.read_bytes.clear()
                with fixture.count_reads():
                    fixture.old.exit_rehash(old_context)
                for fact in fixture.bulk:
                    self.assertEqual(fixture.read_bytes[fact['path']], Path(fact['path']).stat().st_size)
                fixture.read_bytes.clear()
                with fixture.count_reads():
                    driver.exit_rehash(fixture.context)
                for fact in fixture.bulk:
                    self.assertEqual(fixture.read_bytes[fact['path']], 2 * Path(fact['path']).stat().st_size)
                fact = fixture.bulk[0]
                path = Path(fact['path'])
                stamp = path.stat()
                path.write_bytes(b'FAIL' * 1024)
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                # The startup cache is a reuse window, not continuous byte checking.
                fixture.admission.bound_file({}, path, fact['sha256'])
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    fixture.old.exit_rehash(old_context)
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    driver.exit_rehash(fixture.context)

    def test_exact_authority_and_phase_admission(self):
        for phase in ('cpu', 'mechanics', 'train'):
            launch, args = self.launch(phase)
            driver.check_launch(launch, args)
            for key, value in [('seed', 1), ('recipe', {}), ('fitter', {}), ('accepted', {}),
                               ('both_locks_held', False), ('resource_policy', {})]:
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    driver.check_launch({**launch, key: value}, args)
        launch, args = self.launch('cpu', 'candidate')
        with self.assertRaises(ValueError):
            driver.check_launch(launch, args)
        launch, args = self.launch('train')
        with self.assertRaises(ValueError):
            driver.check_launch({**launch, 'selected_mechanics': {'control': launch['selected_cpu']}}, args)
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                driver.strict_json(raw)

    def test_source_pins_match_committed_evidence(self):
        root = PATH.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1'
        receipt = json.loads((root / 'signed-concat-selection-score-v1/receipt.json').read_text())
        self.assertEqual(driver.FITTER, receipt['spec']['training'])
        self.assertEqual(driver.ACCEPTED, receipt['spec']['endpoints'][1])
        for name, sha in driver.FITTER['code'].items():
            self.assertEqual(hashlib.sha256(PATH.with_name(name).read_bytes()).hexdigest(), sha)
        self.assertEqual(driver.policy('cpu')['seconds'], 500)
        self.assertEqual(driver.policy('train')['seconds'], 300)
        self.assertEqual(driver.FILES, {PATH.name, Path(__file__).name, 'nearest_ranking_readout.py'})

    def test_current_bytes_reject_same_size_restored_mtime(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_bytes(b'abcd')
            sha = hashlib.sha256(b'abcd').hexdigest()
            driver.bound_file({}, path, sha)
            stamp = path.stat()
            path.write_bytes(b'abce')
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.assertRaises(ValueError):
                driver.bound_file({}, path, sha)
            with self.assertRaises(ValueError):
                driver.bound_file({str(path): 'b' * 64}, path, hashlib.sha256(b'abce').hexdigest())
            link = Path(directory) / 'link'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                driver.bound_file({}, link, hashlib.sha256(b'abce').hexdigest())

    def test_closure_is_exact_and_full_byte_authenticated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'x.py').write_text('x=1\n')
            code = {'x.py': hashlib.sha256((root / 'x.py').read_bytes()).hexdigest()}
            (root / 'execution.json').write_text(json.dumps(code))
            sha = hashlib.sha256((root / 'execution.json').read_bytes()).hexdigest()
            self.assertEqual(driver.closure(root, sha, {'x.py'}, {}), code)
            with self.assertRaises(ValueError):
                driver.closure(root, sha, {'x.py', 'extra.py'}, {})
            (root / 'x.py').write_text('x=2\n')
            with self.assertRaises(ValueError):
                driver.closure(root, sha, {'x.py'}, {})

    def test_self_exclusion_wrong_identity_original_row_ties_singleton(self):
        scores = [1., .8, .8, .8, .8]
        targets, rows = [0, 0, 0, 1, 2], [40, 30, 10, 50, 20]
        self.assertEqual(driver.select_nearest(scores, targets, rows, 0), (2, 4))
        self.assertEqual(driver.select_nearest(scores, targets, rows, 3), (-1, 0))
        with self.assertRaises(ValueError):
            driver.select_nearest([1., float('nan')], [0, 1], [0, 1], 0)
        with self.assertRaises(ValueError):
            driver.select_nearest([1., 1.], [0, 0], [0, 1], 0)
        with self.assertRaises(ValueError):
            driver.select_nearest([1., 1.], [0, 1], [0, 0], 0)

    def test_actual_loss_unequal_micro_valid_counts_matches_B64(self):
        state = {'teachers': {'V': Tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]]),
                             'P': Tensor([[0., 0.]] * 3), 'e0': Tensor(2.)},
                 'target': Tensor([0, 0, 1, 2, 2]), 'target_list': [0, 0, 1, 2, 2],
                 'row_list': [20, 10, 30, 40, 50], 'count_list': [2, 1, 2]}
        batches = [[0] * 16, [1] * 8 + [2] * 8, [2] * 16, [4] + [2] * 15]
        with patch.dict(sys.modules, fake_torch()):
            loss = [driver.loss_terms(state, Tensor([[.6, .8]] * 16), batch, 25) for batch in batches]
            self.assertEqual([info['valid'] for _, _, info in loss], [16, 8, 0, 1])
            self.assertAlmostEqual(sum(m.item() for m, _, _ in loss), .5)
            expected = (24 * (.05 + .8 - .6) + .05) / (25 * .05)
            self.assertAlmostEqual(sum(r.item() for _, r, _ in loss), expected)
            all_invalid = driver.loss_terms(state, Tensor([[.6, .8]] * 16), [2] * 16, 0)
            self.assertEqual(all_invalid[1].item(), 0.)
            self.assertEqual(all_invalid[2]['positive'], [-1] * 16)
            # A microbatch average would overweight the final single valid anchor.
            wrong = sum(r.item() * 25 / valid / 4 for (_, r, _), valid in zip(loss, [16, 8, 1, 1], strict=True))
            self.assertNotAlmostEqual(expected, wrong)

    def test_freeze_all_before_exact_four_roles_with_standin(self):
        class Parameter:
            dtype, grad = 'torch.float32', None
            def __init__(self, shape):
                self.shape, self.requires_grad = shape, True
            def requires_grad_(self, value):
                self.requires_grad = value
                return self
            def numel(self):
                return math.prod(self.shape)
        class Model:
            def __init__(self):
                self.parameters = {**{f'frozen{i}': Parameter((1,)) for i in range(444)},
                                   **{n: Parameter(s) for n, s in zip(driver.NAMES, driver.SHAPES, strict=True)}}
            def requires_grad_(self, value):
                for p in self.parameters.values():
                    p.requires_grad_(value)
            def named_parameters(self):
                return self.parameters.items()
            def state_dict(self):
                return self.parameters
            def eval(self):
                self.training = False
        model = Model()
        pairs = driver.configure_roles(model)
        self.assertEqual([n for n, _ in pairs], driver.NAMES)
        self.assertEqual(sum(p.requires_grad for p in model.parameters.values()), 4)
        self.assertFalse(model.training)
        model.parameters[driver.NAMES[0]].shape = (1,)
        with self.assertRaises(ValueError):
            driver.configure_roles(model)

    def test_state_export_source_separation_and_no_legacy_mutation(self):
        tree = ast.parse(PATH.read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        native = ast.unparse(functions['native_raw'])
        self.assertNotIn('no_grad', native)
        self.assertIn('isolated_A', native)
        self.assertIn('features.detach()', native)
        self.assertNotIn('pooled.detach()', native)
        forward = ast.unparse(functions['update'])
        self.assertIn('torch.autograd.grad(rank', forward)
        self.assertIn("mse + rank if state['arm'] == 'candidate' else mse", forward)
        self.assertNotIn('reset_peak_memory', PATH.read_text())
        for name in ('load_inference', 'inference_payload', 'check_inference'):
            text = ast.unparse(functions[name])
            for forbidden in ('canonical.npy', "['teachers']", "['schedule']", "['target']"):
                self.assertNotIn(forbidden, text)
        self.assertIn('vision', driver.PAYLOAD_KEYS)
        self.assertIn('provenance', driver.PAYLOAD_KEYS)
        self.assertNotIn('teachers', driver.INFERENCE_KEYS)
        self.assertNotIn('optimizer', driver.INFERENCE_KEYS)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        self.assertNotIn(target.value.id, ('fitter', 'old', 'source', 'legacy'))

    def test_payload_rejects_bad_identity_before_native_reads(self):
        with patch.dict(sys.modules, fake_torch()):
            with self.assertRaises(ValueError):
                driver.check_payload({}, {'schema': 'old-immutable-encoder'}, {}, 0)
            with self.assertRaises(ValueError):
                driver.check_optimizer({'optimizer': {'state': {}, 'param_groups': [{'params': [0, 1, 2, 3, 4]}]}},
                                       {'optimizer_groups': [{}]}, 0)
        self.assertTrue(driver.rejected(lambda: driver.require(False, 'tamper'), 'not rejected'))
        with self.assertRaises(ValueError):
            driver.rejected(lambda: None, 'not rejected')

    def test_fresh_optimizer_keeps_cpu_scaler_and_rejects_old_moments(self):
        scaler = {'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
                  'growth_interval': 2000, '_growth_tracker': 0}
        saved = {'optimizer': {'state': {}, 'param_groups': [{'params': [0, 1, 2, 3]}]},
                 'scaler': scaler, 'numerical_flags': {'threads': 8}}
        ident = {'optimizer_groups': [{}], 'device': 'cpu', 'initial_scaler': scaler,
                 'numerical_flags': saved['numerical_flags']}
        with patch.dict(sys.modules, fake_torch()):
            driver.check_optimizer(saved, ident, 0)
            with self.assertRaises(ValueError):
                driver.check_optimizer({**saved, 'scaler': {}}, ident, 0)
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'][0] = {'warm_optimizer': True}
            with self.assertRaises(ValueError):
                driver.check_optimizer(wrong, ident, 0)
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['param_groups'][0]['params'] = [3, 2, 1, 0]
            with self.assertRaises(ValueError):
                driver.check_optimizer(wrong, ident, 0)

    def test_updated_inference_substitution_and_no_concurrent_models(self):
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        saved = {k: {} for k in driver.INFERENCE_KEYS}
        saved.update(schema=driver.INFERENCE_SCHEMA, source={'original': 'pinned'},
                     vision={'mlp': [2., 3.]}, A={'fixed': [1.]})
        saved['vision_sha256'] = digest(saved['vision'])
        saved['fixed_sha256'] = digest({k: saved[k] for k in
            ('source', 'config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means', 'numerical_flags')})
        context = {'source': saved['source'], 'old': SimpleNamespace(finite_tree=lambda value: None)}
        with patch.object(driver, 'fingerprint', side_effect=lambda context, value: digest(value)):
            driver.check_inference(context, saved, digest(saved))
            replaced = {**saved, 'vision': {'mlp': [0., 1.]}}
            with self.assertRaises(ValueError):
                driver.check_inference(context, replaced, digest(saved))
            with self.assertRaises(ValueError):
                driver.check_inference(context, replaced, digest(replaced))
            corrupted = {**saved, 'A': {'fixed': [2.]}}
            with self.assertRaises(ValueError):
                driver.check_inference(context, corrupted, digest(corrupted))
        import weakref
        class Model:
            pass
        model = Model()
        context = {'live_model': weakref.ref(model)}
        with self.assertRaises(ValueError):
            driver.require_no_model(context)
        del model
        driver.require_no_model(context)

    def test_original_image_mapping_rejects_source_and_identity_swap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'relative_path': 'image.jpg', 'train_row': 8, 'image_sha256': 'a' * 64, 'product': 'p'}
            manifest = {'original_rows': [0], 'rows': [row], 'resolved_paths': [str(root / 'image.jpg')], 'targets': [0]}
            fit = {'dataset_root': str(root), 'rows': [row], 'targets': [7], 'class_names': {7: 'p'}}
            state = {'original_rows': Tensor([0]), 'target': Tensor([0]),
                     'partition': {'panels': {'train': {'original_class_ids': [7]}}}}
            context = {'legacy': {'selected': {'genuine': {'selected': manifest}},
                                  'prior': {'fit': fit, 'all_images': [root / 'image.jpg']}}}
            self.assertEqual(driver.canonical_row(context, state, 0)[2]['image_sha256'], 'a' * 64)
            state['target'] = Tensor([1])
            with self.assertRaises(ValueError):
                driver.canonical_row(context, state, 0)
            state['target'] = Tensor([0])
            manifest['resolved_paths'] = ['/outside/image.jpg']
            with self.assertRaises(ValueError):
                driver.canonical_row(context, state, 0)


    def test_restore_cpu_identity_and_strict_load_precede_cuda_transfer(self):
        started = time.perf_counter()
        with TemporaryDirectory() as directory:
            fixture = RestoreFixture(self, 'cuda')
            state = fixture.run(directory)
            events = fixture.events
            self.assertEqual([(v['phase'], v['boundary']) for v in fixture.snapshots],
                [(phase, boundary) for phase in ('payload_validation', 'payload_hash', 'source_construction',
                 'identity', 'strict_load', 'vision_transfer', 'moments_scaler_rng', 'archive_deletion')
                 for boundary in ('begin', 'end')])
            self.assertIn(('identity', 'cpu'), events)
            self.assertIn(('strict_load', 'cpu'), events)
            transfer = events.index(('transfer', 'cuda'))
            self.assertLess(events.index(('identity', 'cpu')), events.index(('strict_load', 'cpu')))
            self.assertLess(events.index(('strict_load', 'cpu')), events.index(('buffer_copy', 'cpu')))
            self.assertLess(events.index(('buffer_copy', 'cpu')), transfer)
            self.assertLess(transfer, events.index(('adam', 'cuda')))
            self.assertLess(events.index(('adam', 'cuda')), events.index(('moments', 'cuda')))
            self.assertLess(events.index(('moments', 'cuda')), events.index(('scaler', 'cuda')))
            self.assertLess(events.index(('scaler', 'cuda')), events.index(('cpu_rng', 'cpu')))
            self.assertEqual(events[-1], ('cuda_rng', 'cuda'))
            state['optimizer_object'].state[state['params'][0][1]]['step'].value = 99.
            self.assertEqual(fixture.disk['optimizer']['state'][0]['step'].value, 1.)
        self.assertLess(time.perf_counter() - started, 5)

    def test_restore_cpu_lifecycle_unchanged_and_strict_failure_propagates(self):
        started = time.perf_counter()
        with TemporaryDirectory() as directory:
            fixture = RestoreFixture(self, 'cpu')
            fixture.run(directory)
            self.assertEqual(fixture.events, [('source', 'cpu'), ('transfer', 'cpu'), ('adam', 'cpu'),
                ('identity', 'cpu'), ('strict_load', 'cpu'), ('buffer_copy', 'cpu'), ('moments', 'cpu'), ('scaler', 'cpu'), ('cpu_rng', 'cpu')])
            fixture = RestoreFixture(self, 'cuda', strict_failure=True)
            with self.assertRaisesRegex(ValueError, 'strict fixture load failed'):
                fixture.run(directory)
            self.assertIn(('strict_load', 'cpu'), fixture.events)
            self.assertNotIn(('transfer', 'cuda'), fixture.events)
            self.assertNotIn(('moments', 'cuda'), fixture.events)
        self.assertLess(time.perf_counter() - started, 5)


    def test_restore_only_source_boundary_preserves_complete_other_ast(self):
        tree = ast.parse(PATH.read_text())
        tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and
            node.name in {'fresh', 'restore', 'restore_memory_snapshot', 'read_json', 'admit_terminal'})]
        # Committed 40426ca3 AST, excluding lifecycle and startup admission edits.
        self.assertEqual(startup_source_boundary(tree),
                         'b3f24d66d4e381490efa83ed10e742b4b74616a2753e01c83003a2b25c13e362')

    def test_raw_memory_diagnostic_preserves_failure_events_and_propagates_load_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / 'proc-cgroup'
            group.write_text('0::/unit\n')
            unit = root / 'unit'
            unit.mkdir()
            values = {'memory.current': '123', 'memory.peak': '456', 'memory.max': '8589934592',
                'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                'memory.events': 'low 0\nhigh 0\nmax 194\noom 0\noom_kill 0',
                'memory.stat': 'anon 12\nfile 34\nfile_mapped 5'}
            for name, value in values.items():
                (unit / name).write_text(value)
            def diagnostic_path(path):
                return group if path == '/proc/self/cgroup' else root
            with patch.object(driver, 'Path', side_effect=diagnostic_path), patch('builtins.print') as logged:
                with self.assertRaisesRegex(ValueError, 'strict load failed'):
                    with driver.restore_memory_snapshot({'started': time.perf_counter()}, 'strict_load'):
                        raise ValueError('strict load failed')
                self.assertEqual(logged.call_count, 2)
                rows = [json.loads(call.args[0]) for call in logged.call_args_list]
                self.assertEqual([row['boundary'] for row in rows], ['begin', 'end'])
                self.assertTrue(all(call.kwargs['flush'] for call in logged.call_args_list))
                self.assertTrue(all(row['cgroup']['values'] == values for row in rows))
                self.assertTrue(all((unit / name).read_text() == value for name, value in values.items()))
                # Missing diagnostic files cannot turn an original failure into success.
                (unit / 'memory.stat').unlink()
                with self.assertRaisesRegex(ValueError, 'strict load failed'):
                    with driver.restore_memory_snapshot({'started': time.perf_counter()}, 'strict_load'):
                        raise ValueError('strict load failed')
                self.assertIn('error', json.loads(logged.call_args.args[0])['cgroup'])

    def test_help_optimization_rejection_and_syntax(self):
        ast.parse(PATH.read_text())
        ast.parse(Path(__file__).read_text())
        help_result = subprocess.run([sys.executable, str(PATH), '--help'], capture_output=True, text=True, timeout=5)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('--phase {cpu,mechanics,train}', help_result.stdout)
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, flag, str(PATH), '--help'], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('optimized mode is forbidden', result.stderr)


if __name__ == '__main__':
    unittest.main()
