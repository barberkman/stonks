"""Pure-Python unit tests for qmsignals.py (the Qullamaggie port).

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q

`app/python` is the module-search root the engine uses at runtime, so
`from qmsignals import ...` resolves the same way the C++ `PythonStrategy`
loader would.
"""

import numpy as np
import pytest

from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine

from qmsignals import (
    QMSignalsStrategy,
    adr_pct,
    gain_pct,
    highest,
    lowest,
    sma,
)

DAY = QMSignalsStrategy.MS_PER_DAY


# ─── Stateless TA helpers ────────────────────────────────────────────────────
def test_sma():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert sma(a, 2) == pytest.approx(3.5)
    assert sma(a, 4) == pytest.approx(2.5)
    assert sma(a, 5) is None       # not enough history
    assert sma(a, 0) is None       # non-positive window


def test_adr_pct():
    high = np.array([2.0, 4.0])
    low = np.array([1.0, 2.0])
    # mean of 100*(2/1-1)=100 and 100*(4/2-1)=100
    assert adr_pct(high, low, 2) == pytest.approx(100.0)
    assert adr_pct(high, low, 3) is None


def test_gain_pct():
    close = np.array([100.0, 105.0, 110.0])
    assert gain_pct(close, 1) == pytest.approx(100.0 * (110.0 / 105.0 - 1.0))
    assert gain_pct(close, 2) == pytest.approx(10.0)
    assert gain_pct(close, 3) is None       # needs n+1 samples


def test_highest_lowest():
    a = np.array([1.0, 5.0, 3.0, 2.0])
    assert highest(a, 2) == pytest.approx(3.0)
    assert highest(a, 3) == pytest.approx(5.0)
    assert lowest(a, 2) == pytest.approx(2.0)
    assert lowest(a, 5) is None


# ─── Trade levels ────────────────────────────────────────────────────────────
def test_long_levels():
    s = QMSignalsStrategy()
    # adr_stop = 100*(1 - 1*2/100) = 98; LoD (97) is lower, so stop stays 98.
    stop, sell = s.long_levels(entry=100.0, adr=2.0, bar_low=97.0)
    assert stop == pytest.approx(98.0)
    assert sell == pytest.approx(104.0)     # 100 + 2*(100-98)
    # A higher LoD tightens the stop up to the bar low.
    stop, sell = s.long_levels(entry=100.0, adr=2.0, bar_low=99.0)
    assert stop == pytest.approx(99.0)
    assert sell == pytest.approx(102.0)


def test_short_levels():
    s = QMSignalsStrategy()
    # adr_stop = 100*(1 + 1*2/100) = 102; bar high (103) is higher, stop stays 102.
    stop, sell = s.short_levels(entry=100.0, adr=2.0, bar_high=103.0)
    assert stop == pytest.approx(102.0)
    assert sell == pytest.approx(96.0)      # 100 - 2*(102-100)


# ─── Integration via FakeContext ─────────────────────────────────────────────
def _run(bars):
    ctx = FakeContext(bars)
    s = QMSignalsStrategy()
    s.on_start(ctx)
    for _ in {b.timestamp for b in bars}:
        ctx.advance()
        s.on_tick(ctx)
    return ctx, s


def _breakout_bars(symbol="TEST"):
    """56 bars: warmup rise, a 40-bar tight base peaking early (pivot 142 at
    index 18), then a final bar that closes above the pivot on a volume spike —
    everything the long `breakout` gate needs."""
    bars = []
    closes = [100.0 + i * (35.0 / 14) for i in range(15)]   # 0..14: 100 -> 135
    closes += [140.0] * 40                                   # 15..54: flat base
    closes.append(143.0)                                     # 55: breakout
    for i, c in enumerate(closes):
        if i < 15:                          # warmup
            high, low, vol = c + 0.5, c - 0.5, 1000.0
        elif i == 55:                       # breakout bar
            high, low, vol = 143.5, 141.0, 2000.0
        elif i == 18:                       # the pivot high, early in the base
            high, low, vol = 142.0, 139.0, 1000.0
        else:                               # rest of the base, below the pivot
            high, low, vol = 141.0, 139.0, 1000.0
        open_ = closes[i - 1] if i > 0 else c   # no gap (keeps episodic_pivot out)
        bars.append(FakeKLine(i * DAY, symbol, open_, high, low, c, vol))
    return bars


def _short_breakout_bars(symbol="TEST"):
    """Mirror of `_breakout_bars`: warmup fall, a 40-bar base troughing early
    (pivot low 98 at index 18), then a final bar that closes below the trough."""
    bars = []
    closes = [140.0 - i * (35.0 / 14) for i in range(15)]   # 0..14: 140 -> 105
    closes += [100.0] * 40                                   # 15..54: flat base
    closes.append(97.0)                                      # 55: breakdown
    for i, c in enumerate(closes):
        if i < 15:                          # warmup
            high, low, vol = c + 0.5, c - 0.5, 1000.0
        elif i == 55:                       # breakdown bar
            high, low, vol = 99.0, 96.5, 2000.0
        elif i == 18:                       # the pivot low, early in the base
            high, low, vol = 101.0, 98.0, 1000.0
        else:                               # rest of the base, above the trough
            high, low, vol = 101.0, 99.0, 1000.0
        open_ = closes[i - 1] if i > 0 else c
        bars.append(FakeKLine(i * DAY, symbol, open_, high, low, c, vol))
    return bars


def _flat_bars(symbol="TEST"):
    """A dead-flat, low-volume series: no setup should fire."""
    return [
        FakeKLine(i * DAY, symbol, 100.0, 100.5, 99.5, 100.0, 1000.0)
        for i in range(56)
    ]


def test_breakout_fires_long_bracket():
    ctx, s = _run(_breakout_bars())

    sigs = s.last_signals("TEST")
    assert [x.setup for x in sigs] == ["breakout"]
    sig = sigs[0]

    assert len(ctx.orders) == 3
    entry, stop, tp = ctx.orders
    # Long bracket: Buy entry, Sell stop, Sell take-profit.
    assert entry.side == OrderSide.Buy and entry.price == pytest.approx(sig.entry)
    assert stop.side == OrderSide.Sell and stop.price == pytest.approx(sig.stop)
    assert tp.side == OrderSide.Sell and tp.price == pytest.approx(sig.sell)
    # The protective legs are bracketed under the entry's OrderID.
    assert entry.parent is None
    assert stop.parent == entry.id
    assert tp.parent == entry.id
    # Pivot high was 142; stop below entry; take-profit above.
    assert sig.entry == pytest.approx(142.0)
    assert sig.stop < sig.entry < sig.sell


def test_short_breakout_fires_short_bracket():
    ctx, s = _run(_short_breakout_bars())

    sigs = s.last_signals("TEST")
    assert [x.setup for x in sigs] == ["short_breakout"]
    sig = sigs[0]

    assert len(ctx.orders) == 3
    entry, stop, tp = ctx.orders
    # Short bracket: Sell entry, Buy stop, Buy take-profit.
    assert entry.side == OrderSide.Sell and entry.price == pytest.approx(sig.entry)
    assert stop.side == OrderSide.Buy and stop.price == pytest.approx(sig.stop)
    assert tp.side == OrderSide.Buy and tp.price == pytest.approx(sig.sell)
    # The protective legs are bracketed under the entry's OrderID.
    assert entry.parent is None
    assert stop.parent == entry.id
    assert tp.parent == entry.id
    assert sig.entry == pytest.approx(98.0)
    assert sig.sell < sig.entry < sig.stop      # short: stop above, target below


def test_no_signal_stays_flat():
    ctx, s = _run(_flat_bars())
    assert ctx.orders == []
    assert s.last_signals("TEST") == []
