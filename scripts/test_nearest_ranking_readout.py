#!/usr/bin/env python3
"""Stdlib symbolic/admission checks; native Torch parity and gradients UNRUN."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import quadratic_readout as primitive
from test_quadratic_readout import rejects, source_base


class Device(str):
    @property
    def type(self):
        return self.split(':')[0]


class Tensor:
    """Symbolic operations record connectivity, not native numerical gradients."""

    def __init__(self, shape, expr, device='cpu', requires_grad=False,
                 finite=True, nonzero=True):
        self.shape, self.expr = shape, expr
        self.device, self.dtype = Device(device), 'torch.float32'
        self.requires_grad = requires_grad
        self.grad_fn = object() if requires_grad else None
        self.layout, self.is_leaf = 'torch.strided', not requires_grad
        self.finite, self.nonzero = finite, nonzero

    def result(self, shape, expr, other=None, device=None):
        return Tensor(shape, expr, device or self.device,
                      self.requires_grad or (other is not None and other.requires_grad),
                      self.finite and (other is None or other.finite), self.nonzero)

    def float(self):
        return self.result(self.shape, ('float', self.expr))

    def to(self, *, device):
        return self.result(self.shape, ('to', self.expr, str(device)), device=device)

    def detach(self):
        return Tensor(self.shape, ('detach', self.expr), self.device,
                      finite=self.finite, nonzero=self.nonzero)

    def __add__(self, other):
        return self.result(self.shape, ('add', self.expr, other.expr), other)

    def __sub__(self, other):
        return self.result(self.shape, ('sub', self.expr, other.expr), other)

    def __rmul__(self, scalar):
        return self.result(self.shape, ('mul', scalar, self.expr))

    def __gt__(self, scalar):
        assert scalar == 0
        return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: self.nonzero))


def normalize(value, *, dim):
    assert dim == 1
    return value.result(value.shape, ('normalize', value.expr, dim))


def linear(value, weight, bias=None):
    expr = ('linear', value.expr, weight.expr)
    if bias is not None:
        expr += (bias.expr,)
    return value.result((value.shape[0], weight.shape[0]), expr, weight)


class Autocast:
    def __init__(self, device, *, enabled):
        assert device in ('cpu', 'cuda') and enabled is False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fake_torch():
    torch, nn, functional = (ModuleType(name) for name in
                             ('torch', 'torch.nn', 'torch.nn.functional'))
    functional.normalize, functional.linear = normalize, linear
    nn.functional, torch.nn = functional, nn
    torch.autocast = Autocast
    torch.isfinite = lambda value: SimpleNamespace(
        all=lambda: SimpleNamespace(item=lambda: value.finite))
    torch.linalg = SimpleNamespace(vector_norm=lambda value, *, dim: value.result(
        (value.shape[0],), ('norm', value.expr, dim)))

    def cat(values, *, dim):
        assert dim == 1 and values[0].shape[0] == values[1].shape[0]
        return values[0].result((values[0].shape[0], sum(v.shape[1] for v in values)),
                                ('cat', *(v.expr for v in values), dim), values[1])
    torch.cat = cat
    return {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional}


def inputs(scripts, device='cpu', gradient=True):
    base = source_base(scripts)
    base.params = {name: Tensor(value.shape, name, device)
                   for name, value in base.params.items()}
    base.buffers = {name: Tensor(value.shape, name, device)
                    for name, value in base.buffers.items()}
    base.center = base.buffers['center']
    base.primary = lambda x: linear(x, base.params['primary.weight'], base.params['primary.bias'])
    base.down = lambda x: linear(x, base.params['down.weight'])
    base.up = lambda x: linear(x, base.params['up.weight'])
    # These methods belong to the stdlib-compiled standin, never the source module.
    base.forward.__func__.__globals__['F'] = SimpleNamespace(normalize=normalize)
    type(base).__call__ = lambda self, x: self.forward(x)
    return [Tensor((2, 1152), 'features', device, gradient), base,
            Tensor((128, 160), 'A', device),
            {'linear': Tensor((32,), 'linear_mean'),
             'concat': Tensor((160,), 'concat_mean')}, primitive]


def check_forward(function, args):
    raw = function(*args)
    # Independently specified source expression: no outer normalization of x.
    x = ('float', 'features')
    unit = ('normalize', ('float', x), 1)
    centered = ('sub', unit, 'center')
    h0 = ('add', ('linear', unit, 'primary.weight', 'primary.bias'),
          ('linear', ('mul', .5, ('linear', centered, 'down.weight')), 'up.weight'))
    z = ('linear', ('sub', ('normalize', x, 1), 'center'), 'down.weight')
    phi = ('sub', ('cat', z, h0, 1), ('to', 'concat_mean', str(args[0].device)))
    assert raw.expr == ('add', h0, ('linear', phi, 'A')), 'source arithmetic/order differs'
    assert raw.requires_grad == args[0].requires_grad, 'feature graph disconnected'
    assert raw.shape == (2, 128) and raw.dtype == 'torch.float32'
    assert all(not value.requires_grad and value.grad_fn is None
               for value in [args[2], *args[3].values(), *args[1].params.values(),
                             *args[1].buffers.values()]), 'owned coefficients changed'


def check_ast(source, scripts):
    tree = ast.parse(source)
    assert len([n for n in tree.body if isinstance(n, ast.FunctionDef)]) == 1
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)
    assert not any(isinstance(n, ast.Attribute) and n.attr in
                   {'detach', 'no_grad', 'cpu', 'fit_means'} for n in ast.walk(tree))
    original = ast.parse((scripts / 'prototype_residual_readout.py').read_text())
    source_raw = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'raw_features')
    head = ast.parse((scripts / 'train_siglip2_cached_readout.py').read_text())
    forward = next(n for n in ast.walk(head) if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    def has(node, expr):
        expected = ast.dump(ast.parse(expr, mode='eval').body)
        return any(ast.dump(n) == expected for n in ast.walk(node))
    assert has(source_raw, 'features.detach().float()')
    assert has(source_raw, 'base(x).detach()')
    assert has(source_raw, 'base.down(F.normalize(x, dim=1) - base.center)')
    assert has(forward, 'F.normalize(source.float(), dim=1)')


def main():
    assert __debug__, 'optimized checks forbidden'
    assert 'torch' not in sys.modules, 'native Torch must stay unimported'
    scripts = Path(__file__).resolve().parent
    path = scripts / 'nearest_ranking_readout.py'
    source = path.read_text()
    check_ast(source, scripts)
    namespace = {}
    exec(compile(source, str(path), 'exec'), namespace)
    assert 'torch' not in sys.modules, 'module import must stay lazy'
    modules = fake_torch()
    sys.modules.update(modules)
    try:
        function = namespace['raw_features']
        # RED: detached feature mutation must fail the connected expression check.
        changed = source.replace('x = features.float()', 'x = features.detach().float()')
        assert changed != source
        mutant = {}
        exec(compile(changed, '<detach-mutant>', 'exec'), mutant)
        rejects(lambda: check_forward(mutant['raw_features'], inputs(scripts)), AssertionError)
        # RED: a no_grad context is prohibited even if the standin cannot model it.
        changed = source.replace('with torch.autocast(device.type, enabled=False):',
                                 'with torch.autocast(device.type, enabled=False), torch.no_grad():')
        assert changed != source
        rejects(lambda: check_ast(changed, scripts), AssertionError)
        print('RED: detach behavior and no_grad AST mutations rejected')
        for device in ('cpu', 'cuda:0'):
            for gradient in (False, True):
                for dtype in ('torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float64'):
                    args = inputs(scripts, device, gradient)
                    args[0].dtype = dtype
                    check_forward(function, args)
            args = inputs(scripts, device)
            for value in args[3].values():
                value.device = Device(device)
            check_forward(function, args)
        mutations = [
            lambda a: setattr(a[0], 'shape', (0, 1152)),
            lambda a: setattr(a[0], 'shape', (2, 1151)),
            lambda a: setattr(a[0], 'dtype', 'torch.int64'),
            lambda a: setattr(a[0], 'layout', 'torch.sparse_coo'),
            lambda a: setattr(a[0], 'device', Device('mps')),
            lambda a: setattr(a[0], 'finite', False),
            lambda a: setattr(a[0], 'nonzero', False),
            lambda a: setattr(a[2], 'shape', (128, 32)),
            lambda a: setattr(a[2], 'dtype', 'torch.float16'),
            lambda a: setattr(a[2], 'device', Device('cuda:1')),
            lambda a: setattr(a[2], 'requires_grad', True),
            lambda a: setattr(a[2], 'grad_fn', object()),
            lambda a: setattr(a[2], 'finite', False),
            lambda a: a[3].pop('linear'),
            lambda a: a[3].update(extra=Tensor((1,), 'extra')),
            lambda a: setattr(a[3]['concat'], 'shape', (32,)),
            lambda a: setattr(a[3]['linear'], 'dtype', 'torch.float64'),
            lambda a: setattr(a[3]['concat'], 'device', Device('cuda:1')),
            lambda a: setattr(a[3]['concat'], 'requires_grad', True),
            lambda a: setattr(a[3]['linear'], 'grad_fn', object()),
            lambda a: setattr(a[3]['linear'], 'finite', False),
            lambda a: setattr(a[1].params['down.weight'], 'requires_grad', True),
            lambda a: setattr(a[1].buffers['center'], 'grad_fn', object()),
            lambda a: setattr(a[1].params['up.weight'], 'shape', (127, 32)),
            lambda a: setattr(a[1].params['primary.bias'], 'finite', False),
        ]
        for mutate in mutations:
            args = inputs(scripts)
            mutate(args)
            rejects(lambda: function(*args))
        print('GREEN: symbolic source order, connected/inference paths, frozen role admission')
    finally:
        for name in modules:
            sys.modules.pop(name)
    assert 'torch' not in sys.modules
    print('Native Torch CPU/CUDA numerical parity and autograd qualification: UNRUN (root-owned)')


if __name__ == '__main__':
    main()
