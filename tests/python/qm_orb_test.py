"""Tests for Setup 1b — opening range breakout (intraday, long).

ORB is intraday-only: a session is one calendar day, and the opening range is the
high of its first `orb_bars` bars. On daily data each bar is its own session, so
the strategy must emit nothing; on intraday data it must fire.
"""

from qm_orb import QMORBStrategy
from qm_helpers import HOUR, buys, run, to_klines, uptrend


def _intraday(days=6, per_day=5, start=10.0, step=0.5):
    """Rising bars, `per_day` per calendar day, so sessions have multiple bars."""
    rows, stamps = [], []
    k = 0
    for d in range(days):
        for b in range(per_day):
            c = start + k * step
            rows.append((c, c * 1.01, c * 0.99, c, 1000.0))
            stamps.append(d * 86_400_000 + b * HOUR)
            k += 1
    from stonks.testing import FakeKLine
    return [FakeKLine(ts, "AAA", *r) for ts, r in zip(stamps, rows)]


def test_orb_produces_no_signals_on_daily_data():
    # Daily bars that DO pass the universe filter — ORB still emits nothing because
    # each day is a one-bar session (the documented no-op).
    ctx = run(QMORBStrategy(), to_klines("AAA", uptrend(60)))
    assert ctx.orders == []


def test_orb_fires_on_intraday_breakout():
    ctx = run(QMORBStrategy(), _intraday())
    assert len(buys(ctx)) >= 1
