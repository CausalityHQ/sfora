"""Small stdlib check of dense-boundary parameter and buffer authority."""
import importlib.util

assert importlib.util.find_spec("large_dense_boundary") is not None, "dense boundary qualifier is missing"
import large_dense_boundary as dense

class Parameter:
    def __init__(self):
        self.requires_grad = True

class Module:
    def __init__(self, parameters):
        self.parameters = parameters
    def requires_grad_(self, enabled):
        for value in self.parameters:
            value.requires_grad = enabled

class Vision:
    def __init__(self):
        self.values = {"embeddings.weight": Parameter(), **{f"encoder.layers.{i}.weight": Parameter() for i in range(24)}}
        self.embeddings = Module([self.values["embeddings.weight"]])
        self.encoder = type("Encoder", (), {"layers": [Module([self.values[f"encoder.layers.{i}.weight"]]) for i in range(24)]})()
    def requires_grad_(self, enabled):
        Module(list(self.values.values())).requires_grad_(enabled)
    def named_parameters(self):
        return self.values.items()
    def state_dict(self):
        return {**self.values, "encoder.layers.9.buffer": "frozen", "encoder.layers.10.buffer": "trainable"}

vision = Vision()
dense.configure(vision)
assert [n for n,p in vision.named_parameters() if p.requires_grad] == [f"encoder.layers.{i}.weight" for i in range(10,24)]
frozen = dense.frozen_state(vision)
assert "encoder.layers.9.buffer" in frozen and "encoder.layers.10.buffer" not in frozen
assert "encoder.layers.10.weight" not in frozen
vision.values["encoder.layers.9.weight"].requires_grad = True
try:
    dense.verify_boundary(vision)
except AssertionError:
    pass
else:
    raise AssertionError("wrong boundary accepted")
print("PASS dense10 boundary and named-buffer authority; stdlib only, not native qualification")
