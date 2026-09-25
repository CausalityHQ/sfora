"""The source ceiling uses held queries and packed inverse norms."""

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "diagnose_inshop_siglip2_source_ceiling.py"
SPEC = importlib.util.spec_from_file_location("diagnose_inshop_siglip2_source_ceiling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_held_query_self_exclusion_and_packed_inverse_norm() -> None:
    values = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0, 1])
    assert MODULE.recall_at_1(values, labels, (0,), (0, 1, 2), device=torch.device("cpu")) == 1
    assert (
        MODULE.recall_at_1(
            values,
            labels,
            (0,),
            (0, 1, 2),
            torch.tensor([1.0, 0.25, 1.0]),
            device=torch.device("cpu"),
        )
        == 0
    )
