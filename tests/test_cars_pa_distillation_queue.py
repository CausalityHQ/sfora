import subprocess
from pathlib import Path

QUEUE = Path("scripts/run_priority_queue_v47.sh")


def test_cars_confirmation_queue_is_digest_locked_and_deterministic() -> None:
    source = QUEUE.read_text(encoding="utf-8")

    assert "d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a" in source
    assert "080a45b8c14460d43b6f5f1d352f10854adb0d6c8d434fc6d2f02f2dbd501b02" in source
    assert 'for seed in 0 1 2 3 4 5; do' in source
    assert 'run_confirmation_arm "proxy_anchor" "auto" "${seed}"' in source
    assert 'run_confirmation_arm "pa_distill" "pa_distill" "${seed}"' in source
    assert source.index('run_confirmation_arm "proxy_anchor"') < source.index(
        'run_confirmation_arm "pa_distill"'
    )
    assert "--deterministic" in source
    assert 'config.get("deterministic") is True' in source
    assert "PREFLIGHT_ONLY" in source


def test_cars_confirmation_queue_has_valid_bash_syntax() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(QUEUE)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
