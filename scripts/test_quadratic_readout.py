#!/usr/bin/env python3
"""Stdlib admission/AST falsifiers only; no Torch or numerical qualification."""

if not __debug__:
    raise SystemExit("checks require assertions; optimized mode is forbidden")

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError("invalid metadata admitted")


def tensor(dims, location="cpu", **changes):
    return SimpleNamespace(**{
        "shape": dims, "dtype": "torch.float32", "device": location,
        "layout": "torch.strided", "requires_grad": False,
        "grad_fn": None, "is_leaf": True, **changes,
    })


def source_base(scripts, arm="control", filename=None):
    """Compile original method closures, without importing/executing Torch."""
    path = scripts / "train_siglip2_cached_readout.py"
    source = ast.parse(path.read_text())
    factory = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "head_from")
    residual = next(n for n in factory.body if isinstance(n, ast.ClassDef))
    residual.bases = []
    residual.body = [n for n in residual.body if isinstance(n, ast.FunctionDef) and n.name != "__init__"]
    factory.body = [residual, ast.Return(value=ast.Call(func=ast.Name(id="Residual", ctx=ast.Load()), args=[], keywords=[]))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
                 filename or str(path), "exec"), namespace)
    base = namespace["head_from"](arm)
    base.params = {
        "primary.weight": tensor((128, 1152)), "primary.bias": tensor((128,)),
        "down.weight": tensor((32, 1152)), "up.weight": tensor((128, 32)),
    }
    base.buffers = {"center": tensor((1152,)), "preactivation_std": tensor(())}
    base.named_parameters = lambda: base.params.items()
    base.named_buffers = lambda: base.buffers.items()
    return base


def metadata_checks(q, scripts):
    base = source_base(scripts)
    q._check_base(base, "cpu")
    rejects(lambda: q._check_base(source_base(scripts, "candidate"), "cpu"))
    rejects(lambda: q._check_base(source_base(scripts, filename="different_factory.py"), "cpu"))
    for mutation in (
        lambda b: b.params.pop("primary.bias"),
        lambda b: b.params.update(extra=tensor((1,))),
        lambda b: setattr(b.params["up.weight"], "requires_grad", True),
        lambda b: setattr(b.params["down.weight"], "shape", (31, 1152)),
        lambda b: setattr(b.params["primary.weight"], "dtype", "torch.float16"),
        lambda b: setattr(b.params["primary.bias"], "device", "cuda:0"),
        lambda b: b.buffers.pop("preactivation_std"),
        lambda b: b.buffers.update(extra=tensor(())),
        lambda b: setattr(b.buffers["center"], "requires_grad", True),
        lambda b: setattr(b.buffers["center"], "grad_fn", object()),
        lambda b: setattr(b.buffers["preactivation_std"], "shape", (1,)),
        lambda b: setattr(b, "forward", lambda x: x),
        lambda b: setattr(b, "residual", lambda x: x),
    ):
        changed = source_base(scripts)
        mutation(changed)
        rejects(lambda: q._check_base(changed, "cpu"))

    for arm in ("control", "candidate"):
        q._check_arm(arm)
    for arm in ("quadratic", "gelu", "", None, []):
        rejects(lambda: q._check_arm(arm))

    q._check_features(tensor((6355, 1152)), "cpu", train=True)
    for changes in ({"shape": (6354, 1152)}, {"shape": (6355, 1024)},
                    {"dtype": "torch.float16"}, {"device": "cuda:0"},
                    {"layout": "torch.sparse_coo"}):
        rejects(lambda: q._check_features(tensor((6355, 1152), **changes), "cpu", train=True))
    for dtype in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"):
        q._check_features(tensor((16, 1152), dtype=dtype, requires_grad=True), "cpu")
    for shape in ((0, 1152), (16, 1024), (1152,), (1, 16, 1152)):
        rejects(lambda: q._check_features(tensor(shape), "cpu"))
    rejects(lambda: q._check_features(tensor((16, 1152), dtype="torch.int64"), "cpu"))

    weight = tensor((128, 32), requires_grad=True)
    q._check_weight(weight, "cpu")
    for changes in ({"shape": (32, 128)}, {"dtype": "torch.float16"},
                    {"device": "cuda:0"}, {"requires_grad": False},
                    {"is_leaf": False}, {"grad_fn": object()}):
        rejects(lambda: q._check_weight(tensor((128, 32), **{"requires_grad": True, **changes}), "cpu"))

    means = {"linear": tensor((32,)), "quadratic": tensor((32,))}
    q._check_means(means, "cpu")
    q._check_means(means, "cuda:0")
    q._check_means({k: tensor((32,), "cuda:0") for k in means}, "cuda:0")
    for mutation in (
        lambda m: m.pop("linear"), lambda m: m.update(extra=tensor((32,))),
        lambda m: setattr(m["linear"], "device", "cuda:0"),
        lambda m: setattr(m["quadratic"], "device", "cuda:1"),
        lambda m: setattr(m["linear"], "shape", (1, 32)),
        lambda m: setattr(m["quadratic"], "dtype", "torch.float64"),
        lambda m: setattr(m["linear"], "requires_grad", True),
        lambda m: setattr(m["quadratic"], "grad_fn", object()),
    ):
        changed = copy.deepcopy(means)
        mutation(changed)
        rejects(lambda: q._check_means(changed, "cuda:0"))


def math_scope_checks(tree):
    """Falsify scope/formula changes; actual native parity remains UNRUN."""
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    def expression(source):
        return ast.dump(ast.parse(source, mode="eval").body)
    def has(function, source):
        return any(ast.dump(n) == expression(source) for n in ast.walk(functions[function]))

    assert has("fit_means", "F.normalize(canonical_train.detach().float(), dim=1)")
    assert has("fit_means", "base.down(F.normalize(x, dim=1) - base.center)")
    assert has("fit_means", "z.mean(dim=0).detach()")
    assert has("fit_means", "z.square().mean(dim=0).detach()")
    assert has("raw_features", "base(x).detach()")
    assert has("raw_features", "base.down(F.normalize(x, dim=1) - base.center)")
    assert has("raw_features", "z if arm == 'control' else z.square()")
    assert has("raw_features", "h0 + F.linear(phi.detach(), A)")
    assert has("new_weight", "torch.nn.Parameter(torch.zeros((128, 32), dtype=torch.float32, device=device))")
    fit = next(n for n in functions["fit_means"].body if isinstance(n, ast.With))
    assert [n.targets[0].id for n in fit.body if isinstance(n, ast.Assign)] == ["x", "z", "means"]
    raw = functions["raw_features"]
    scopes = [n for n in ast.walk(raw) if isinstance(n, ast.With)]
    frozen = next(n for n in scopes if any(ast.dump(i.context_expr) == expression("torch.no_grad()") for i in n.items))
    x = next(n for n in frozen.body if isinstance(n, ast.Assign) and n.targets[0].id == "x")
    assert ast.dump(x.value) == expression("features.detach().float()")
    assert any(ast.dump(n) == expression("base(x).detach()") for n in ast.walk(frozen))
    assert not any(ast.dump(n) == expression("F.linear(phi.detach(), A)") for n in ast.walk(frozen))
    for name in ("fit_means", "raw_features"):
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "autocast" and
                   any(k.arg == "enabled" and isinstance(k.value, ast.Constant) and k.value.value is False
                       for k in n.keywords) for n in ast.walk(functions[name]))
        assert not any(isinstance(n, ast.Attribute) and n.attr in
                       {"cuda", "std", "gelu", "clamp", "set_float32_matmul_precision", "manual_seed"}
                       for n in ast.walk(functions[name]))


def source_order_checks(scripts):
    """Original TRAIN normalizes before the control head normalizes again."""
    source = ast.parse((scripts / "train_siglip2_genuine_views.py").read_text())
    functions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    def contains(node, expression):
        expected = ast.dump(ast.parse(expression, mode="eval").body)
        return any(ast.dump(n) == expected for n in ast.walk(node))
    assert contains(functions["training_features"], "normalize_nonzero(features)")
    assert contains(functions["normalize_nonzero"], "F.normalize(value, dim=1)")
    source = ast.parse((scripts / "train_siglip2_cached_readout.py").read_text())
    factory = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "head_from")
    residual = next(n for n in factory.body if isinstance(n, ast.ClassDef))
    forward = next(n for n in residual.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert contains(forward, "F.normalize(source.float(), dim=1)")
    assert contains(forward, "self.primary(unit) + self.residual(unit)")


def main():
    scripts = Path(__file__).resolve().parent
    path = scripts / "quadratic_readout.py"
    assert path.is_file(), "quadratic readout primitive is missing"
    tree = ast.parse(path.read_text(), filename=str(path))
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                   any(a.name == "torch" or a.name.startswith("torch.") for a in n.names)
                   for n in tree.body), "Torch must stay lazy"
    spec = importlib.util.spec_from_file_location("quadratic_readout", path)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    assert "torch" not in sys.modules, "stdlib checks imported Torch"
    metadata_checks(q, scripts)
    source_order_checks(scripts)
    math_scope_checks(tree)
    for old, new in (("F.normalize(canonical_train.detach().float(), dim=1)", "canonical_train.detach().float()"),
                     ("x = features.detach().float()", "x = F.normalize(features.detach().float(), dim=1)"),
                     ("z.square()", "z"), ("base(x).detach()", "base.primary(x).detach()"),
                     ("enabled=False", "enabled=True")):
        changed = ast.parse(path.read_text().replace(old, new))
        rejects(lambda: math_scope_checks(changed), AssertionError)
    assert "torch" not in sys.modules, "stdlib checks imported Torch"
    print("PASS: stdlib metadata admission and AST math scopes; native/numerical qualification UNRUN")


if __name__ == "__main__":
    main()
