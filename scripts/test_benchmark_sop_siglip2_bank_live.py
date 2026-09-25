"""The paired live probe balances arm order and reports latency in milliseconds."""

from benchmark_sop_siglip2_bank_live import paired_order, summarize_ns


def test_paired_order_balances_position_every_four_blocks() -> None:
    assert [paired_order(i) for i in range(4)] == [
        ("control", "bank"),
        ("bank", "control"),
        ("bank", "control"),
        ("control", "bank"),
    ]


def test_summary_keeps_ns_to_ms_units_and_count() -> None:
    summary = summarize_ns([1_000_000, 2_000_000, 3_000_000, 4_000_000])
    assert summary["calls"] == 4
    assert summary["p50_ms"] == 2.5
    assert summary["mean_ms"] == 2.5
