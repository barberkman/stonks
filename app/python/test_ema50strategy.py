"""Pure-Python unit tests for ema50strategy.py (mirror of ema50strategy.h).

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q

The C++ suite pins the 1%-of-equity sizing on its side
(ema50strategy_test.cpp::EntrySizedToOnePercentOfEquity); this pins the same
convention on the Python mirror so the two ports can't silently drift apart
again.
"""

import pytest

from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine

from ema50strategy import EMA50Strategy

DAY = 86_400_000


def _flat_then_cross(cross_close=105.0, after_close=None):
    """50 warmup bars at 100 (seeding the EMA at exactly 100), then a crossover
    bar, then optionally one more bar."""
    closes = [100.0] * 50 + [cross_close]
    if after_close is not None:
        closes.append(after_close)
    return [
        FakeKLine(i * DAY, "TEST", c, c + 0.5, c - 0.5, c, 1000.0)
        for i, c in enumerate(closes)
    ]


def _run(bars):
    ctx = FakeContext(bars)
    s = EMA50Strategy()
    s.on_start(ctx)
    for _ in {b.timestamp for b in bars}:
        ctx.advance()
        s.on_tick(ctx)
    return ctx, s


def test_entry_sized_to_one_percent_of_equity():
    ctx, _ = _run(_flat_then_cross(105.0))

    assert len(ctx.orders) == 1
    entry = ctx.orders[0]
    assert entry.side == OrderSide.Buy
    # Mirrors the C++ port exactly: equity() * POSITION_FRACTION / close.
    assert entry.quantity == pytest.approx(100_000.0 * 0.01 / 105.0)


def test_no_order_before_the_ema_is_seeded():
    ctx, _ = _run([
        FakeKLine(i * DAY, "TEST", 100.0 + i, 100.5 + i, 99.5 + i, 100.0 + i, 1000.0)
        for i in range(49)   # one bar short of the seed
    ])
    assert ctx.orders == []


def test_downside_cross_sells_the_held_quantity():
    ctx, _ = _run(_flat_then_cross(105.0, after_close=90.0))

    assert len(ctx.orders) == 2
    entry, exit_ = ctx.orders
    assert entry.side == OrderSide.Buy
    assert exit_.side == OrderSide.Sell
    assert exit_.quantity == pytest.approx(entry.quantity)   # closes exactly what it holds
