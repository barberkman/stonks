"""Pure-Python unit tests for qmsignals.py (the Qullamaggie port).

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q

`app/python` is the module-search root the engine uses at runtime, so
`from qmsignals import ...` resolves the same way the C++ `PythonStrategy`
loader would.
"""

import numpy as np
import pytest

from stonks import OrderSide, OrderStatus
from stonks.testing import FakeContext, FakeKLine, FakePosition

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
def _run(bars, strategy=None):
    ctx = FakeContext(bars)
    s = strategy or QMSignalsStrategy()
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
    # Long bracket: Buy stop-entry, Sell stop-loss, Sell take-profit limit.
    assert entry.side == OrderSide.Buy and entry.price == pytest.approx(sig.entry)
    assert stop.side == OrderSide.Sell and stop.price == pytest.approx(sig.stop)
    assert tp.side == OrderSide.Sell and tp.price == pytest.approx(sig.sell)
    # Stop orders for entry and SL (they wait for their trigger); limit for the TP.
    assert entry.order_type == "stop"
    assert stop.order_type == "stop"
    assert tp.order_type == "limit"
    # Protective legs are reduce-only: an orphaned leg may never open a position.
    assert entry.reduce_only is False
    assert stop.reduce_only is True
    assert tp.reduce_only is True
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
    # Short bracket: Sell stop-entry, Buy stop-loss, Buy take-profit limit.
    assert entry.side == OrderSide.Sell and entry.price == pytest.approx(sig.entry)
    assert stop.side == OrderSide.Buy and stop.price == pytest.approx(sig.stop)
    assert tp.side == OrderSide.Buy and tp.price == pytest.approx(sig.sell)
    # Stop orders for entry and SL (they wait for their trigger); limit for the TP.
    assert entry.order_type == "stop"
    assert stop.order_type == "stop"
    assert tp.order_type == "limit"
    # The protective legs are bracketed under the entry's OrderID.
    assert entry.parent is None
    assert stop.parent == entry.id
    assert tp.parent == entry.id
    assert sig.entry == pytest.approx(98.0)
    assert sig.sell < sig.entry < sig.stop      # short: stop above, target below


def test_does_not_fire_while_position_is_open():
    bars = _breakout_bars()
    ctx = FakeContext(bars)
    ctx.positions["TEST"] = FakePosition(quantity=5.0, price=130.0)
    s = QMSignalsStrategy()
    s.on_start(ctx)
    for _ in {b.timestamp for b in bars}:
        ctx.advance()
        s.on_tick(ctx)
    assert ctx.orders == []                                       # gated
    assert [x.setup for x in s.last_signals("TEST")] == ["breakout"]   # scanner still fired


def test_opposite_side_signal_suppressed_not_netted_while_in_a_trade():
    bars = _breakout_bars()
    ctx = FakeContext(bars)
    ctx.positions["TEST"] = FakePosition(quantity=-5.0, price=150.0)   # short vs a long signal
    s = QMSignalsStrategy()
    s.on_start(ctx)
    for _ in {b.timestamp for b in bars}:
        ctx.advance()
        s.on_tick(ctx)
    assert ctx.orders == []


def _run_with_close_at(bars, close_tick, cooldown):
    """Hold a fake position on every tick before `close_tick`, then drop it —
    the strategy detects the close there and starts its cooldown."""
    ctx = FakeContext(bars)
    s = QMSignalsStrategy()
    s.cooldown_bars = cooldown
    s.on_start(ctx)
    for tick in range(len({b.timestamp for b in bars})):
        if tick < close_tick:
            ctx.positions["TEST"] = FakePosition(quantity=5.0, price=130.0)
        else:
            ctx.positions.pop("TEST", None)
        ctx.advance()
        s.on_tick(ctx)
    return ctx


def test_cooldown_after_close_suppresses_the_signal():
    # Close detected at tick 53; cooldown 5 still has bars left on the signal tick (55).
    assert _run_with_close_at(_breakout_bars(), 53, 5).orders == []


def test_fires_again_after_cooldown_elapses():
    # Cooldown 1 expires by the signal tick: the bracket goes out.
    assert len(_run_with_close_at(_breakout_bars(), 53, 1).orders) == 3


def test_replaces_stale_pending_entry_on_new_signal():
    bars = _breakout_bars()
    bars.append(FakeKLine(56 * DAY, "TEST", 143.0, 145.5, 142.5, 145.0, 2000.0))
    s = QMSignalsStrategy()
    s.min_base_days = 0                     # let the very next bar re-qualify
    ctx, s = _run(bars, s)

    assert len(ctx.orders) == 6             # two full brackets
    entry1, sl1, tp1, entry2, sl2, tp2 = ctx.orders
    # The unfilled first bracket was cancelled wholesale before the second went out.
    assert entry1.status == OrderStatus.Cancelled
    assert sl1.status == OrderStatus.Cancelled
    assert tp1.status == OrderStatus.Cancelled
    assert entry2.status == OrderStatus.Open
    assert entry1.price != entry2.price     # fresh levels, not a duplicate


def test_reanchors_bracket_when_entry_fill_gaps_past_the_plan():
    bars = _breakout_bars()
    # A benign extra tick (low volume, no new breakout) on which the strategy
    # observes the gapped fill and re-anchors.
    bars.append(FakeKLine(56 * DAY, "TEST", 143.0, 143.5, 142.5, 143.0, 1000.0))
    ctx = FakeContext(bars)
    s = QMSignalsStrategy()
    s.on_start(ctx)
    n = len({b.timestamp for b in bars})
    for tick in range(n):
        ctx.advance()
        if tick == n - 1:
            # Simulate the broker: the stop-entry gapped and filled 3% above plan.
            entry = ctx.orders[0]
            entry.status = OrderStatus.Filled
            ctx.positions["TEST"] = FakePosition(quantity=entry.quantity,
                                                 price=entry.price * 1.03,
                                                 entry_id=entry.id)
        s.on_tick(ctx)

    assert len(ctx.orders) == 5                 # bracket + two re-anchored legs
    entry, sl, tp, new_sl, new_tp = ctx.orders
    assert sl.status == OrderStatus.Cancelled   # originals replaced
    assert tp.status == OrderStatus.Cancelled
    assert new_sl.order_type == "stop" and new_sl.reduce_only is True
    assert new_tp.order_type == "limit" and new_tp.reduce_only is True
    assert new_sl.parent == entry.id and new_tp.parent == entry.id
    assert new_sl.price == pytest.approx(sl.price * 1.03)
    assert new_tp.price == pytest.approx(tp.price * 1.03)
    assert new_sl.quantity == pytest.approx(entry.quantity)


def test_entry_leverage_formula():
    s = QMSignalsStrategy()
    assert s.entry_leverage(100.0, 95.0, True) == pytest.approx(19.0)    # Lmax=20 -> step below
    assert s.entry_leverage(100.0, 94.0, True) == pytest.approx(16.0)    # 16.67 -> 16
    assert s.entry_leverage(100.0, 106.0, False) == pytest.approx(16.0)  # short mirror
    assert s.entry_leverage(100.0, 99.5, True) == pytest.approx(125.0)   # capped at max_leverage
    assert s.entry_leverage(100.0, 40.0, True) == pytest.approx(1.0)     # wide stop -> no leverage


def test_entry_leverage_maintenance_margin():
    s = QMSignalsStrategy()
    s.maint_margin = 0.004
    assert s.entry_leverage(100.0, 95.0, True) == pytest.approx(18.0)


def test_entry_uses_computed_risk_leverage():
    ctx, s = _run(_breakout_bars())

    assert len(ctx.orders) == 3
    entry, stop, tp = ctx.orders
    sig = s.last_signals("TEST")[0]
    assert sig.setup == "breakout"

    # Risk-sized quantity and the isolated leverage that keeps liquidation past the stop.
    expected_qty = 100_000.0 * s.risk_fraction / abs(sig.entry - sig.stop)
    expected_lev = s.entry_leverage(sig.entry, sig.stop, True)
    assert expected_lev > 1.0

    assert entry.side == OrderSide.Buy
    assert entry.quantity == pytest.approx(expected_qty)
    assert entry.leverage == pytest.approx(expected_lev)
    # Protective legs: same quantity, default 1x leverage (ignored on closes).
    assert stop.quantity == pytest.approx(expected_qty)
    assert tp.quantity == pytest.approx(expected_qty)
    assert stop.leverage == pytest.approx(1.0)
    assert tp.leverage == pytest.approx(1.0)


def test_no_signal_stays_flat():
    ctx, s = _run(_flat_bars())
    assert ctx.orders == []
    assert s.last_signals("TEST") == []
