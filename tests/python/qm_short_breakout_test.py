"""Tests for Setup 1c — short breakout (breakdown), mirror of the long breakout."""

from qm_short_breakout import QMShortBreakoutStrategy
from qm_helpers import bar, buys, downtrend, run, sells, to_klines

BASE = [11.5, 12.0, 11.5, 11.0, 11.5, 11.0, 11.5, 11.0]  # shallow bounce over the ~9.9 pivot


def _setup(break_close=9.5, break_vol=2000.0):
    """51-bar decline to ~10, an 8-bar bounce base, then a breakdown bar (last row)."""
    rows = downtrend(51, start=50.0, step=0.8)
    rows += [bar(c, spread=0.012) for c in BASE]
    rows.append(bar(break_close, vol=break_vol, hi=11.0, lo=break_close - 0.4, o=11.0))
    return rows


def test_short_breakout_fires_on_clean_setup():
    ctx = run(QMShortBreakoutStrategy(), to_klines("AAA", _setup()))
    assert len(sells(ctx)) == 1 and len(buys(ctx)) == 0  # sell-to-open, no cover yet


def test_short_breakout_rejects_when_close_above_pivot():
    ctx = run(QMShortBreakoutStrategy(), to_klines("AAA", _setup(break_close=11.0)))
    assert ctx.orders == []


def test_short_breakout_rejects_without_volume_expansion():
    ctx = run(QMShortBreakoutStrategy(), to_klines("AAA", _setup(break_vol=1000.0)))
    assert ctx.orders == []


def test_short_breakout_rejects_when_not_downtrending():
    ctx = run(QMShortBreakoutStrategy(), to_klines("AAA", [bar(20.0) for _ in range(60)]))
    assert ctx.orders == []


def test_short_breakout_covers_on_stop():
    rows = _setup()
    rows.append(bar(9.6, hi=10.0, lo=9.3))   # fill bar, holds below the stop
    rows.append(bar(12.0, hi=13.0, lo=11.5))  # high pierces the stop -> cover
    s = QMShortBreakoutStrategy()
    ctx = run(s, to_klines("AAA", rows))
    assert len(sells(ctx)) == 1 and len(buys(ctx)) >= 1  # entry sell, then cover buy
    assert s.pos == {}
