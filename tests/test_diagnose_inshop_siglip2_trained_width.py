"""The trained-width diagnostic excludes self and resolves exact ties by ordinal."""

import json
import subprocess
import sys
from pathlib import Path


def test_float_top1_self_exclusion_and_ties() -> None:
    script = Path(__file__).parents[1] / "scripts/diagnose_inshop_siglip2_trained_width.py"
    result = subprocess.run(
        [sys.executable, str(script), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"hits": [1, 1, 0, 0]}
