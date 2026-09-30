#!/usr/bin/env python3
"""Stdlib-only admission checks with standins; no tensor/layout/RNG qualification."""
if not __debug__:
    raise SystemExit("Checks require assertions")

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Weight:
    def __init__(self, device="cpu", trainable=False):
        self.shape = (128, 4096)
        self.dtype = "float32"
        self.device = device
        self.requires_grad = trainable
        self.finite = True
        self.nonzero = False


class Parameters:
    def __init__(self, members):
        self.members = members

    def parameters(self):
        return iter(p for _, p in self.members)


class AdamW:
    def __init__(self, groups):
        self.defaults = {"lr": 1e-3, "weight_decay": .05, "betas": (.9, .999), "eps": 1e-8}
        self.param_groups = [{**self.defaults, **group} for group in groups]
        self.state = {}

    def add_param_group(self, group):
        self.param_groups.append({**self.defaults, **group})


PATH = Path(__file__).with_name("token_residual_readout.py")
TREE = ast.parse(PATH.read_text(), filename=str(PATH))


class RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


NAMESPACE = {
    "torch": SimpleNamespace(Tensor=Weight, float32="float32", optim=SimpleNamespace(AdamW=AdamW),
                             isfinite=lambda w: SimpleNamespace(all=lambda: w.finite),
                             count_nonzero=lambda w: w.nonzero),
    "coverage": SimpleNamespace(parameters=lambda model, head, classifier:
                                model.members + head.members + [("classifier", classifier)]),
    "new_weight": Weight,
}
SELECTED = ast.Module(body=[node for node in TREE.body if isinstance(node, ast.FunctionDef)
                           and node.name in ("validate_weight", "_validate_members", "attach")], type_ignores=[])
exec(compile(RemoveImports().visit(SELECTED), str(PATH), "exec"), NAMESPACE)


def fresh():
    model = Parameters([(f"vision.{i}", Weight(trainable=True)) for i in range(205)])
    head = Parameters([(f"compact_head.{name}", Weight(trainable=True)) for name in ("weight", "bias")])
    classifier = Weight(trainable=True)
    return {"model": model, "head": head, "classifier": classifier, "counter": 0,
            "bank": object(), "scaler": object(),
            "params": model.members + head.members + [("classifier", classifier)],
            "optimizer": AdamW([{"params": list(model.parameters()), "lr": 1e-5},
                                {"params": list(head.parameters()), "lr": 1e-4},
                                {"params": [classifier], "lr": 1e-4}])}


class Admission(unittest.TestCase):
    def test_weight_authority(self):
        validate = NAMESPACE["validate_weight"]
        validate(Weight(), "control")
        nonzero = Weight()
        nonzero.nonzero = True
        validate(nonzero, "candidate")  # Serialized candidate need not require gradients.
        for arm in ("foreign", None):
            with self.subTest(arm=arm), self.assertRaises(AssertionError):
                validate(Weight(), arm)
        for weight in (None, object(), nonzero):
            with self.subTest(weight=weight), self.assertRaises(AssertionError):
                validate(weight, "control")
        for name, value in (("shape", (4096, 128)), ("shape", (128, 1024)),
                            ("dtype", "float16"), ("dtype", "float64"), ("finite", False)):
            bad = Weight()
            setattr(bad, name, value)
            for arm in ("control", "candidate"):
                with self.subTest(name=name, value=value, arm=arm), self.assertRaises(AssertionError):
                    validate(bad, arm)

    def test_attach_roles_order_and_original_state_preserved(self):
        for arm, count in (("control", 208), ("candidate", 209)):
            state = fresh()
            untouched = {k: state[k] for k in ("model", "head", "classifier", "bank", "scaler", "optimizer")}
            groups = list(state["optimizer"].param_groups)
            snapshots = [{**group, "params": list(group["params"])} for group in groups]
            params = list(state["params"])
            weight = NAMESPACE["attach"](state, arm)
            self.assertIs(state["residual"], weight)
            self.assertEqual(weight.requires_grad, arm == "candidate")
            self.assertEqual(len(state["params"]), count)
            self.assertEqual(state["params"][:208], params)
            self.assertEqual(state["optimizer"].param_groups[:3], snapshots)
            for index, group in enumerate(groups):
                self.assertIs(state["optimizer"].param_groups[index], group)
            for key, value in untouched.items():
                self.assertIs(state[key], value)
            self.assertFalse(state["optimizer"].state)
            self.assertEqual(state["counter"], 0)
            if arm == "candidate":
                self.assertEqual(state["params"][-1], ("residual", weight))
                self.assertEqual(state["optimizer"].param_groups[-1], {
                    **state["optimizer"].defaults, "lr": 1e-4, "params": [weight]})
            else:
                self.assertFalse(any(p is weight for _, p in state["params"]))
            with self.assertRaises(AssertionError):
                NAMESPACE["attach"](state, arm)

    def test_stale_foreign_and_wrong_members_rejected_before_attachment(self):
        mutations = (
            lambda s: s.update(counter=1),
            lambda s: s.update(counter=True),
            lambda s: s.update(counter=0.0),
            lambda s: s["optimizer"].state.update({0: {"step": 1}}),
            lambda s: s.update(optimizer=object()),
            lambda s: s["params"].pop(),
            lambda s: s["params"].reverse(),
            lambda s: s["optimizer"].param_groups[0]["params"].reverse(),
            lambda s: s["optimizer"].param_groups[0]["params"].append(s["classifier"]),
            lambda s: s["optimizer"].param_groups.append({"params": []}),
            lambda s: s["optimizer"].param_groups[0].update(lr=1e-4),
            lambda s: s["model"].members.pop(),
            lambda s: s["optimizer"].param_groups[1]["params"].insert(
                0, s["optimizer"].param_groups[0]["params"].pop()),
        )
        for index, mutate in enumerate(mutations):
            state = fresh()
            mutate(state)
            with self.subTest(mutation=index), self.assertRaises(AssertionError):
                NAMESPACE["attach"](state, "candidate")
            self.assertNotIn("residual", state)
        state = fresh()
        with self.assertRaises(AssertionError):
            NAMESPACE["attach"](state, "foreign")
        self.assertNotIn("residual", state)


if __name__ == "__main__":
    unittest.main()
