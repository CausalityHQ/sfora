#!/usr/bin/env python3
"""Stdlib-only symbolic falsifier for connected_residual_readout.raw_features.

A fake `torch` records an expression tree plus the autograd-reachable leaves of every
result (detach / no_grad cut the graph). The ACTUAL source text of the two detaching
functions runs against it (RED), then the connected helper (GREEN). Native Torch
numerical and autograd parity is UNRUN here and root-owned.
"""
import ast
import hashlib
import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
PINS = {  # master 125c9200; a change here means re-review the helper against the new source
    'prototype_residual_readout.py': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68',
    'quadratic_readout.py': '12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6',
}
FULLFEATURE_SEGMENT = 'ecb24f950f28abaf1eb81f808aab7f5d39bb6d45034a6c0068ef95df7397be46'
FULLFEATURE_FILES = ('train_siglip2_compact_ranking.py', 'train_siglip2_identity_diversity.py')

G = {}


class Dev:
    def __init__(self, kind):
        self.type = kind

    def __str__(self):
        return self.type


class T:
    """Symbolic tensor: `reach` = trainable leaves connected through the autograd graph."""
    dtype = 'torch.float32'
    layout = 'torch.strided'

    def __init__(self, expr, shape, reach=(), leaf=True, finite=True, dev='cpu'):
        self.expr, self.shape, self.reach = expr, tuple(shape), frozenset(reach)
        self.leaf, self.finite, self.device = leaf, finite, Dev(dev)

    requires_grad = property(lambda s: bool(s.reach))
    is_leaf = property(lambda s: s.leaf)
    grad_fn = property(lambda s: None if s.leaf or not s.reach else 'grad_fn')

    def detach(self):
        return op('detach', self.shape, self, cut=True)

    def float(self):
        return op('float', self.shape, self)

    def to(self, device=None):
        return op('to', self.shape, self, dev=str(device))

    def __add__(self, other):
        return op('add', max(self.shape, other.shape, key=len), self, other)

    def __sub__(self, other):
        return op('sub', max(self.shape, other.shape, key=len), self, other)


class P(T):
    def __init__(self, name, shape, trainable=True, **kw):
        super().__init__(name, shape, {name} if trainable else (), **kw)


def op(name, shape, *args, cut=False, dev=None):
    G['log'].append(G['ac'])
    ts = [a for a in args if isinstance(a, T)]
    reach = frozenset().union(*(t.reach for t in ts)) if G['grad'] and not cut else frozenset()
    return T((name, *(t.expr for t in ts)), shape, reach, leaf=False, dev=dev or str(ts[0].device),
             finite=all(t.finite for t in ts))


class Head:
    def __init__(self):
        self.center = T('head.center', (1152,))
        self.tensors = [self.center]

    def __call__(self, x):
        return op('head', (x.shape[0], 128), x)

    def down(self, t):
        return op('down', (t.shape[0], 32), t)


@contextmanager
def _grad(enabled):
    old, G['grad'] = G['grad'], enabled
    try:
        yield
    finally:
        G['grad'] = old


@contextmanager
def _autocast(device_type, enabled=True):
    old, G['ac'] = G['ac'], enabled
    G['calls'].append((device_type, enabled))
    try:
        yield
    finally:
        G['ac'] = old


def _fake_torch():
    F = ModuleType('torch.nn.functional')
    F.normalize = lambda x, dim: op('normalize', x.shape, x)
    F.linear = lambda x, w: op('linear', (x.shape[0], w.shape[0]), x, w)
    nn = ModuleType('torch.nn')
    nn.Parameter, nn.functional = P, F
    torch = ModuleType('torch')
    torch.nn, torch.no_grad, torch.autocast = nn, lambda: _grad(False), _autocast
    torch.cat = lambda ts, dim: op('cat', (ts[0].shape[0], sum(t.shape[1] for t in ts)), *ts)
    torch.isfinite = lambda v: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: v.finite))
    return {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': F}


def load(name):
    spec = importlib.util.spec_from_file_location('_t_' + name[:-3], HERE / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def req(condition, message):
    if not condition:
        raise ValueError(message)


def segment(path, name):
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


READOUT, QUAD, CONN = load('prototype_residual_readout.py'), load('quadratic_readout.py'), load('connected_residual_readout.py')
PRIM = SimpleNamespace(_require=QUAD._require, _check_tensor=QUAD._check_tensor, _check_features=QUAD._check_features,
                       _finite=QUAD._finite,
                       _check_base=lambda base, device: [QUAD._check_tensor(t, t.shape, device, frozen=True) or t
                                                         for t in base.tensors])
_ns = {'require': req, 'parameter_roles': lambda arm: None}
exec(segment(HERE / FULLFEATURE_FILES[0], 'fullfeature_raw_features'), _ns)
FULL = _ns['fullfeature_raw_features']


def scenario(**over):
    G.update(grad=True, ac=None, log=[], calls=[])
    kw = dict(features=T('features', (3, 1152), {'features'}), head=Head(), A=P('A', (128, 160)),
              means={'linear': T('means.linear', (32,)), 'concat': T('means.concat', (160,))},
              C=P('C', (128, 1152)), mu_train=T('mu', (1152,)), primitive=PRIM, readout=READOUT)
    kw.update(over)
    return kw


def src_readout(kw):  # source 1: detach inside no_grad
    return READOUT.raw_features(kw['features'], kw['head'], kw['A'], kw['means'], 'concat', PRIM)


def src_full(kw):  # source 2: + features.detach().float() - mu_train C term
    return FULL(kw['features'], kw['head'], kw['A'], kw['means'], kw['C'], kw['mu_train'], 'control', PRIM, READOUT)


def canon(e, zero=()):
    """Erase detach (forward no-op); with C exactly zero drop its additive term."""
    if isinstance(e, str):
        return e
    if e[0] == 'detach':
        return canon(e[1], zero)
    args = [canon(a, zero) for a in e[1:]]
    if e[0] == 'linear' and args[1] in zero:
        return 'ZERO'
    if e[0] == 'add' and args[1] == 'ZERO':
        return args[0]
    return (e[0], *args)


def green(fn):
    """All connected-readout claims at once; a mutant must break at least one."""
    want, want_zero = canon(src_full(scenario()).expr), canon(src_readout(scenario()).expr)
    try:
        raw = fn(**scenario())
    except ValueError:
        return False
    return (raw.reach == {'features', 'A', 'C'} and canon(raw.expr) == want and
            canon(raw.expr, {'C'}) == want_zero and set(G['log']) == {False} and set(G['calls']) == {('cpu', False)})


def mutant(old, new):
    text = (HERE / 'connected_residual_readout.py').read_text()
    assert text.count(old) == 1, old
    ns = {}
    exec(compile(text.replace(old, new), 'mutant', 'exec'), ns)
    return ns['raw_features']


class ConnectedResidualReadout(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, _fake_torch())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_source_pins(self):
        for name, digest in PINS.items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        for name in FULLFEATURE_FILES:
            seg = segment(HERE / name, 'fullfeature_raw_features')
            self.assertEqual(hashlib.sha256(seg.encode()).hexdigest(), FULLFEATURE_SEGMENT, name)

    def test_detach_red(self):
        self.assertEqual(src_readout(scenario()).reach, {'A'})
        self.assertEqual(src_full(scenario()).reach, {'A', 'C'})  # features unreachable: the defect

    def test_connected_green(self):
        self.assertTrue(green(CONN.raw_features))
        self.assertEqual(CONN.raw_features(**scenario(features=T('f', (3, 1152)))).reach, {'A', 'C'})

    def test_concat_order_preserved(self):
        cat = next(n for n in _walk(canon(CONN.raw_features(**scenario()).expr)) if n[0] == 'cat')
        self.assertEqual((cat[1][0], cat[2][0]), ('down', 'head'))  # [Z32, H0raw128]

    def test_zero_and_nonzero_c_forward_equivalence(self):
        raw = CONN.raw_features(**scenario())
        self.assertEqual(canon(raw.expr), canon(src_full(scenario()).expr))  # nonzero C
        zero = canon(raw.expr, {'C'})
        self.assertEqual(zero, canon(src_readout(scenario()).expr))  # exact-zero C == readout alone
        self.assertNotEqual(canon(raw.expr), zero)
        self.assertIn('C', raw.reach)  # a zero C still receives gradient

    def test_mutants_are_caught(self):
        base = 'raw = h0 + F.linear(phi, A)\n        raw = raw + F.linear(x - mu_train, C)'
        for old, new in (
                ('x = features.float()', 'x = features.detach().float()'),
                (base, 'raw = h0 + (F.linear(phi, A) + F.linear(x - mu_train, C))'),
                ('F.linear(x - mu_train, C)', 'F.linear(x, C)'),
                ('F.linear(x - mu_train, C)', 'F.linear(F.normalize(x, dim=1) - mu_train, C)'),
                ('enabled=False', 'enabled=True')):
            self.assertFalse(green(mutant(old, new)), new)

    def test_cuda_device_with_cpu_means(self):
        head = Head()
        head.center = T('head.center', (1152,), dev='cuda')
        head.tensors = [head.center]
        cuda = dict(features=T('features', (3, 1152), {'features'}, dev='cuda'), head=head, A=P('A', (128, 160), dev='cuda'),
                    C=P('C', (128, 1152), dev='cuda'), mu_train=T('mu', (1152,), dev='cuda'))
        self.assertEqual(CONN.raw_features(**scenario(**cuda)).reach, {'features', 'A', 'C'})

    def test_invalid_roles(self):
        thawed = Head()
        thawed.tensors[0] = T('head.center', (1152,), {'x'})
        no_linear = {'concat': T('m', (160,))}
        cases = {
            'C frozen': dict(C=T('C', (128, 1152))), 'C not Parameter': dict(C=T('C', (128, 1152), {'C'})),
            'C non-leaf': dict(C=_nonleaf()), 'C shape': dict(C=P('C', (128, 1151))),
            'C nonfinite': dict(C=P('C', (128, 1152), finite=False)),
            'mu trainable': dict(mu_train=T('mu', (1152,), {'mu'})), 'mu shape': dict(mu_train=T('mu', (1151,))),
            'A frozen': dict(A=P('A', (128, 160), trainable=False)), 'A linear width': dict(A=P('A', (128, 32))),
            'means trainable': dict(means={'linear': T('l', (32,)), 'concat': T('m', (160,), {'m'})}),
            'means keys': dict(means=no_linear), 'head thawed': dict(head=thawed),
            'features width': dict(features=T('f', (3, 1151), {'f'})),
            'features nonfinite': dict(features=T('f', (3, 1152), {'f'}, finite=False)),
            'features device': dict(features=T('f', (3, 1152), {'f'}, dev='cuda')),
            'device kind': dict(features=T('f', (3, 1152), {'f'}, dev='mps'))}
        for name, over in cases.items():
            with self.assertRaises(ValueError, msg=name):
                CONN.raw_features(**scenario(**over))
        kw = scenario()
        G['grad'] = False  # no_grad/inference caller would silently disconnect
        with self.assertRaises(ValueError):
            CONN.raw_features(**kw)


def _nonleaf():
    c = P('C', (128, 1152))
    c.leaf = False
    return c


def _walk(e):
    if isinstance(e, tuple):
        yield e
        for a in e[1:]:
            yield from _walk(a)


if __name__ == '__main__':
    unittest.main()
