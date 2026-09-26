"""The unseen-gallery split is class-disjoint and schedule prefixes pair."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "preflight_inshop_siglip2_unseen_gallery",
    SCRIPTS / "preflight_inshop_siglip2_unseen_gallery.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_and_schedule_prefix() -> None:
    labels = tuple(name for i in range(40) for name in (str(i), str(i))) + ("singleton",)
    fit, held = MODULE.split(labels)
    assert "singleton" in {labels[i] for i in fit}
    assert {labels[i] for i in fit}.isdisjoint(labels[i] for i in held)
    fit_labels = tuple(labels[i] for i in fit)
    assert MODULE.schedule(fit_labels, 10) == MODULE.schedule(fit_labels, 20)[:10]


def test_seeded_schedule_pairs_and_changes_across_seeds() -> None:
    labels = tuple(name for i in range(40) for name in (str(i), str(i)))
    fit = tuple(labels[i] for i in MODULE.split(labels)[0])
    first = MODULE.schedule(fit, 10, seed=179024)
    assert first == MODULE.schedule(fit, 20, seed=179024)[:10]
    assert first != MODULE.schedule(fit, 10, seed=179025)
