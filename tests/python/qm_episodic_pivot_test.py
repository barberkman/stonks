"""Tests for Setup 2 — episodic pivot (gap bar, long)."""

from qm_episodic_pivot import QMEpisodicPivotStrategy
from qm_helpers import bar, buys, run, sells, to_klines, uptrend

# Wide-range warmup so the ADR-based epWithin gate has room (close ~20 at the gap).
WARMUP = uptrend(51, start=10.0, step=0.2, spread=0.03)


def _gap_bar(o=21.0, hi=22.0, lo=21.4, c=21.9, vol=2000.0):
    return bar(c, vol=vol, o=o, hi=hi, lo=lo)


def test_ep_fires_on_gap_volume_strong_close():
    ctx = run(QMEpisodicPivotStrategy(), to_klines("AAA", WARMUP + [_gap_bar()]))
    assert len(buys(ctx)) == 1 and len(sells(ctx)) == 0


def test_ep_rejects_small_gap():
    # Open only 0.25% above the prior close of 20.0.
    ctx = run(QMEpisodicPivotStrategy(), to_klines("AAA", WARMUP + [_gap_bar(o=20.05)]))
    assert ctx.orders == []


def test_ep_rejects_without_volume():
    ctx = run(QMEpisodicPivotStrategy(), to_klines("AAA", WARMUP + [_gap_bar(vol=1000.0)]))
    assert ctx.orders == []


def test_ep_rejects_weak_close():
    # Close below the bar mid-point fails the strong-close gate.
    ctx = run(QMEpisodicPivotStrategy(), to_klines("AAA", WARMUP + [_gap_bar(c=21.0, lo=20.9)]))
    assert ctx.orders == []


def test_ep_rejects_risk_too_wide():
    # Low far below the close -> close-to-low risk exceeds the ADR stop distance.
    ctx = run(QMEpisodicPivotStrategy(), to_klines("AAA", WARMUP + [_gap_bar(lo=18.0)]))
    assert ctx.orders == []
