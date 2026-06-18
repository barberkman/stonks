"""Shared builders for the Qullamaggie strategy tests.

Bars are crafted as (open, high, low, close, volume) rows on a daily clock and run
through a FakeContext one timestamp at a time, mirroring the engine's per-tick
dispatch. Assertions are made against the emitted FakeOrders.
"""

from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine

DAY = 86_400_000
HOUR = 3_600_000


def bar(c, vol=1000.0, spread=0.01, o=None, hi=None, lo=None):
    """One OHLCV row centered on close `c`, with overridable o/h/l."""
    return (
        c if o is None else o,
        c * (1 + spread) if hi is None else hi,
        c * (1 - spread) if lo is None else lo,
        c,
        vol,
    )


def to_klines(symbol, rows, start=0, step=DAY):
    """Turn (o,h,l,c,v) rows into FakeKLines stamped `start, start+step, ...`."""
    return [FakeKLine(start + i * step, symbol, *r) for i, r in enumerate(rows)]


def run(strategy, bars):
    """Drive `strategy` over `bars`: advance once per distinct timestamp, tick."""
    ctx = FakeContext(bars, cash=100_000.0)
    strategy.on_start(ctx)
    for _ in sorted({b.timestamp for b in bars}):
        ctx.advance()
        strategy.on_tick(ctx)
    return ctx


def buys(ctx):
    return [o for o in ctx.orders if o.side == OrderSide.Buy]


def sells(ctx):
    return [o for o in ctx.orders if o.side == OrderSide.Sell]


def uptrend(n, start=10.0, step=0.8, vol=1000.0, spread=0.01):
    return [bar(start + i * step, vol=vol, spread=spread) for i in range(n)]


def downtrend(n, start=50.0, step=0.8, vol=1000.0, spread=0.01):
    return [bar(start - i * step, vol=vol, spread=spread) for i in range(n)]
