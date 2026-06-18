"""Tests for Setup 1 — momentum breakout (long)."""

import pytest

from qm_breakout import QMBreakoutStrategy
from qm_helpers import bar, buys, run, sells, to_klines, uptrend

BASE = [48, 47, 46, 47, 48, 47, 48, 48]  # 8-bar shallow base under the ~50.5 pivot


def _setup(break_close=51.0, break_vol=2000.0):
    """51-bar ramp to ~50, an 8-bar base, then a break bar (the last row)."""
    rows = uptrend(51, start=10.0, step=0.8)
    rows += [bar(c, spread=0.012) for c in BASE]
    rows.append(bar(break_close, vol=break_vol, hi=break_close + 0.6, lo=48.6, o=49.0))
    return rows


def test_breakout_fires_on_clean_setup():
    ctx = run(QMBreakoutStrategy(), to_klines("AAA", _setup()))
    assert len(buys(ctx)) == 1 and len(sells(ctx)) == 0


def test_breakout_rejects_when_close_below_pivot():
    # Break bar closes at 49 (< ~50.5 pivot) — no trigger.
    ctx = run(QMBreakoutStrategy(), to_klines("AAA", _setup(break_close=49.0)))
    assert ctx.orders == []


def test_breakout_rejects_deep_pullback():
    rows = uptrend(51, start=10.0, step=0.8)
    rows += [bar(c, spread=0.012) for c in [40, 35, 30, 28, 30, 32, 30, 31]]  # >40% retrace
    rows.append(bar(51.0, vol=2000.0, hi=51.6, lo=48.6, o=49.0))
    ctx = run(QMBreakoutStrategy(), to_klines("AAA", rows))
    assert ctx.orders == []


def test_breakout_rejects_without_volume_expansion():
    # Break bar volume == baseline (< 1.3x avg) and useBOVol is on by default.
    ctx = run(QMBreakoutStrategy(), to_klines("AAA", _setup(break_vol=1000.0)))
    assert ctx.orders == []


def test_breakout_rejects_with_insufficient_history():
    ctx = run(QMBreakoutStrategy(), to_klines("AAA", uptrend(30)))
    assert ctx.orders == []


def test_breakout_rejects_when_not_trending():
    ctx = run(QMBreakoutStrategy(), to_klines("AAA", [bar(20.0) for _ in range(60)]))
    assert ctx.orders == []


def test_breakout_stops_out():
    rows = _setup()
    rows.append(bar(50.0, hi=50.5, lo=49.7))  # fill bar, holds above the stop
    rows.append(bar(48.5, hi=49.5, lo=48.0))  # low pierces the stop -> exit
    s = QMBreakoutStrategy()
    ctx = run(s, to_klines("AAA", rows))
    assert len(buys(ctx)) == 1 and len(sells(ctx)) >= 1
    assert s.pos == {}  # flat after the stop


def test_breakout_partial_then_exit():
    rows = _setup()
    rows.append(bar(51.5, hi=52.0, lo=50.8))  # fill bar, holds
    rows.append(bar(53.0, hi=53.5, lo=52.0))  # tags the 2R target -> partial + BE
    rows += [bar(c) for c in [52, 51, 50, 49]]  # fades back through breakeven -> exit
    s = QMBreakoutStrategy()
    ctx = run(s, to_klines("AAA", rows))
    assert len(buys(ctx)) == 1
    assert len(sells(ctx)) == 2  # partial, then remainder
    assert sells(ctx)[0].quantity == pytest.approx(buys(ctx)[0].quantity * 0.5)
    assert s.pos == {}
