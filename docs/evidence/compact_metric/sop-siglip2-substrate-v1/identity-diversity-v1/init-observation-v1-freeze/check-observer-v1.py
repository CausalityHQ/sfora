"""Stdlib-only checks of the actual injected gate and original lifecycle nodes."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location('observer', folder/'init-admission-observe-v1.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
source = Path(sys.argv[1])
raw = source.read_bytes()
tree, proof = observer.transform(raw, str(source))
original = ast.parse(raw, filename=str(source))
gpu = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_run')
run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
gpu_block = next(n for n in gpu.body if isinstance(n, ast.Try))
run_block = next(n for n in run.body if isinstance(n, ast.Try))
static = ast.literal_eval(next(n.value for n in original.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'STATIC_KEYS' for t in n.targets)))
for before, after in zip(original.body, tree.body, strict=True):
    if not isinstance(before, ast.FunctionDef) or before.name not in ('gpu_run', 'run'):
        assert observer.dump(before, True) == observer.dump(after, True)
assert proof['exact_inverse_including_attributes'] and proof['original_exit_calls'] == 1


def compile_probe(text, block, namespace):
    module = ast.parse(text)
    module.body[0].body.append(copy.deepcopy(block))
    module.body[0].body.append(ast.Raise(exc=ast.Call(func=ast.Name(id='AssertionError', ctx=ast.Load()),
        args=[ast.Constant('update or receipt tail reached')], keywords=[]), cause=None))
    exec(compile(ast.fix_missing_locations(module), '<stdlib-observation-check>', 'exec'), namespace)


class TensorMetadata:
    shape = (2, 3)
    dtype = 'float32'
    device = 'cuda:0'
    layout = 'strided'
    requires_grad = False

    def stride(self):
        return (3, 1)


class BrokenJSON:
    def dumps(self, *args, **kwargs):
        raise OSError('synthetic report sink failure')


def exercise(false_index=None, broken_report=False):
    events = []
    flags = {}
    args = SimpleNamespace(phase='mechanics', arm='control', seed=179061)
    ident = {'static_sha256': 'static', 'scope': {'arm': 'control'},
             'schedule_provenance_sha256': 'schedule'}
    qualified = {'identity': copy.deepcopy(ident), 'initial_raw_unit_packed_sha256': 'witness',
                 'common_input_raw_unit_packed_sha256': 'common', 'common_statistics_sha256': 'statistics'}
    witness = 'witness'
    context = {'args': args, 'initial': {k: {'tensor': TensorMetadata()} for k in static},
               'common_input_raw_unit_packed_sha256': 'common', 'common_statistics_sha256': 'statistics'}
    for key in ('initial_static_sha256', 'common_initial_sha256', 'initial_A_sha256',
                'initial_C_sha256', 'mu_train_sha256', 'mu_train_provenance_sha256'):
        context[key] = key
    state = {k: {'tensor': TensorMetadata()} for k in static}
    state.update(device='cuda', counter=0)
    if false_index == 0:
        ident['static_sha256'] = 'different'
    elif false_index == 1:
        ident['scope'] = {'arm': 'candidate'}
    elif false_index == 2:
        ident['schedule_provenance_sha256'] = 'different'
    elif false_index == 3:
        witness = 'different'
    elif false_index == 4:
        context['common_input_raw_unit_packed_sha256'] = 'different'
    elif false_index == 5:
        context['common_statistics_sha256'] = 'different'

    def require(condition, message):
        if message == observer.MESSAGE:
            events.append('original-require')
        if not condition:
            raise ValueError(message)

    def release(ctx, live):
        assert ctx is context and live is state
        events.append('release-state')
        live.clear()

    def release_scope(ctx):
        assert not state
        events.append('release-scope')
        ctx['initial'].clear()

    def exit_rehash(ctx):
        assert not state and not ctx['initial']
        events.append('original-uncached-exit')

    @contextlib.contextmanager
    def timed(ctx, name):
        events.append(name)
        yield

    reader = SimpleNamespace(bound_file=lambda *a: None)
    api = SimpleNamespace(audit_origins=lambda *a, **k: events.append('origin-audit'))
    reference = SimpleNamespace(admit_cgroup=lambda *a: events.append('after-cgroup'))
    source_driver = SimpleNamespace(numerical_flags=lambda: flags, cgroup_memory=lambda: {'path': '/fake'})
    context.update(legacy={'source_driver': source_driver, 'original': SimpleNamespace(FlatAdmission=lambda: reader),
                           'origins': {'files': {}}, 'selected': {'genuine': {'reference': reference}}},
                   nearest=SimpleNamespace(native_source_api=lambda c: api),
                   old=SimpleNamespace(zero_events=lambda v: events.append('zero-events')), guards={})
    namespace = {'torch': SimpleNamespace(Tensor=TensorMetadata, cuda=SimpleNamespace(
        is_initialized=lambda: True, max_memory_allocated=lambda: 0)), 'STATIC_KEYS': static,
        'json': BrokenJSON() if broken_report else json, 'require': require, 'release': release,
        'release_scope': release_scope, 'exit_rehash': exit_rehash, 'timed': timed, 'time': time,
        'resource': resource, 'policy': lambda phase: {'seconds': 600}}
    compile_probe('def gpu_probe(context, state, ident, initial_witness, qualified):\n    args=context["args"]\n',
                  gpu_block, namespace)
    namespace['gpu_run'] = lambda c: namespace['gpu_probe'](c, state, ident, witness, qualified)
    compile_probe('def run_probe(context):\n    args=context["args"]\n    legacy=context["legacy"]\n'
                  '    source=legacy["source_driver"]\n    flags={}\n    before={"path":"/fake"}\n'
                  '    started=time.perf_counter()\n    unit="fake"\n', run_block, namespace)
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            namespace['run_probe'](context)
    except ValueError as error:
        assert str(error) == (observer.STOP if false_index is None else observer.MESSAGE)
    else:
        raise AssertionError('diagnostic must always reject before updates')
    assert events.count('original-require') == events.count('release-state') == 1
    assert events.count('release-scope') == events.count('original-uncached-exit') == 1
    assert events.index('release-state') < events.index('release-scope') < events.index('original-uncached-exit')
    assert events.count('origin-audit') == 1 and events.count('zero-events') == 2
    assert not state and not context['initial']
    if not broken_report:
        report = json.loads(output.getvalue())
        assert [v['equal'] for v in report['conjuncts']] == [i != false_index for i in range(6)]
        assert report['static_metadata']['head']['members']['tensor']['shape'] == [2, 3]
        assert report['counter'] == 0 and report['production_pass'] is False
    assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)


started = time.perf_counter()
for index in (None, 0, 1, 2, 3, 4, 5):
    exercise(index)
    exercise(index, broken_report=True)
try:
    observer.transform(raw + b'\n', str(source))
except ValueError:
    pass
else:
    raise AssertionError('source pin tamper accepted')
assert time.perf_counter() - started < 10
print(json.dumps({'checks': 14, 'each_conjunct_false': 'verified', 'all_six_true': 'stops before updates',
                  'report_sink_failure': 'unchanged gate, disposal and original uncached exit verified',
                  'exact_AST_inverse': True, 'native_imports': False,
                  'elapsed_seconds': time.perf_counter()-started}))
