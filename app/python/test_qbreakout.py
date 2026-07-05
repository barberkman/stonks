"""Pure-Python unit tests for qbreakout.py (the Qullamaggie breakout strategy).

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q

`app/python` is the module-search root the engine uses at runtime, so
`from qbreakout import ...` resolves the same way the C++ `PythonStrategy`
loader would.

`FakeContext` records orders but does not simulate fills or P&L (equity == cash),
so the integration tests assert on the *orders the strategy places* and the
shadow-ledger transitions it drives off the fed bars — the multi-bar management
state machine (freeroll / scale-out / trail / supply-shoot / liquidation) is
exercised directly against `_manage`/`_reconcile` with a seeded position.
"""

import numpy as np
import pytest

from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine

from qbreakout import (
    QBreakoutStrategy,
    adr_pct,
    avg_dollar_vol,
    ema,
    gain_pct,
    highest,
    lowest,
    sma,
)

DAY = QBreakoutStrategy.MS_PER_DAY


# ─── Stateless TA helpers ────────────────────────────────────────────────────
def test_sma():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert sma(a, 2) == pytest.approx(3.5)
    assert sma(a, 4) == pytest.approx(2.5)
    assert sma(a, 5) is None
    assert sma(a, 0) is None


def test_ema():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    # seed = SMA(1,2) = 1.5; then 3 -> 2.5; then 4 -> 3.5 (alpha = 2/3).
    assert ema(a, 2) == pytest.approx(3.5)
    # A flat series collapses to its level.
    assert ema(np.array([10.0, 10.0, 10.0]), 2) == pytest.approx(10.0)
    assert ema(a, 5) is None


def test_adr_pct():
    high = np.array([2.0, 4.0])
    low = np.array([1.0, 2.0])
    # mean of 100*(2/1-1)=100 and 100*(4/2-1)=100
    assert adr_pct(high, low, 2) == pytest.approx(100.0)
    assert adr_pct(high, low, 3) is None


def test_avg_dollar_vol():
    close = np.array([10.0, 20.0])
    vol = np.array([2.0, 3.0])
    # mean of 10*2=20 and 20*3=60
    assert avg_dollar_vol(close, vol, 2) == pytest.approx(40.0)
    assert avg_dollar_vol(close, vol, 3) is None


def test_gain_pct():
    close = np.array([100.0, 105.0, 110.0])
    assert gain_pct(close, 1) == pytest.approx(100.0 * (110.0 / 105.0 - 1.0))
    assert gain_pct(close, 2) == pytest.approx(10.0)
    assert gain_pct(close, 3) is None


def test_highest_lowest():
    a = np.array([1.0, 5.0, 3.0, 2.0])
    assert highest(a, 2) == pytest.approx(3.0)
    assert highest(a, 3) == pytest.approx(5.0)
    assert lowest(a, 2) == pytest.approx(2.0)
    assert lowest(a, 5) is None


# ─── Sizing (Step 2) ─────────────────────────────────────────────────────────
def test_size_position_risk_based():
    s = QBreakoutStrategy()
    # risk=10 -> lev=10, liquidation lands exactly at the stop; risk-sized qty
    # (0.5% of 100k over a $10 stop) is well under the 25% cap.
    qty, lev = s.size_position(100.0, 90.0, 100_000.0)
    assert lev == pytest.approx(10.0)
    assert qty == pytest.approx(0.005 * 100_000.0 / 10.0)   # 50


def test_size_position_notional_cap():
    s = QBreakoutStrategy()
    # A tight $1 stop wants 500 units; the 25%-of-equity cap clips it to 250.
    qty, lev = s.size_position(100.0, 99.0, 100_000.0)
    assert lev == pytest.approx(100.0)
    assert qty == pytest.approx(0.25 * 100_000.0 / 100.0)   # 250


def test_size_position_leverage_capped():
    s = QBreakoutStrategy()
    # A 0.5% stop needs 200x; leverage clamps to max_leverage and the effective
    # stop (and risk) widen to match.
    qty, lev = s.size_position(100.0, 99.5, 100_000.0)
    assert lev == pytest.approx(125.0)
    assert qty == pytest.approx(0.25 * 100_000.0 / 100.0)   # cap still binds


# ─── Leadership rank ─────────────────────────────────────────────────────────
def test_leaders():
    s = QBreakoutStrategy()
    s.rs_top_frac = 0.5
    assert s.leaders({"A": 10.0, "B": 5.0, "C": 1.0}) == {"A", "B"}
    assert s.leaders({"A": 3.0}) == {"A"}          # a lone symbol always leads
    assert s.leaders({}) == set()
    s.rs_top_frac = 1.0
    assert s.leaders({"A": 10.0, "B": 5.0, "C": 1.0}) == {"A", "B", "C"}


# ─── Integration via FakeContext ─────────────────────────────────────────────
def _run(bars, strategy=None):
    ctx = FakeContext(bars)
    s = strategy or QBreakoutStrategy()
    s.on_start(ctx)
    for _ in {b.timestamp for b in bars}:
        ctx.advance()
        s.on_tick(ctx)
    return ctx, s


WARM = 204   # warmup bars 0..203, then a 40-bar base (204..243), breakout at 244


def _base_bars(symbol="TEST"):
    """245 bars that satisfy every entry gate for `symbol`: a long rising warmup
    (for the 200-SMA stack), a 40-bar tight base peaking early (pivot high 142.5
    at index 207), then a final bar closing above the pivot on a volume spike.
    ADR over the last 20 bars clears 2.25%."""
    bars = []
    for i in range(WARM):                              # warmup: 50 -> 139
        c = 50.0 + (139.0 - 50.0) * i / (WARM - 1)
        o = 50.0 + (139.0 - 50.0) * (i - 1) / (WARM - 1) if i > 0 else c
        bars.append(FakeKLine(i * DAY, symbol, o, c * 1.02, c * 0.995, c, 1000.0))
    for j in range(40):                                # base 204..243 at 140
        idx = WARM + j
        if j == 3:                                     # pivot high early in base
            hi, lo = 142.5, 138.5
        else:                                          # rest below the pivot
            hi, lo = 141.5, 138.0
        bars.append(FakeKLine(idx * DAY, symbol, 140.0, hi, lo, 140.0, 1000.0))
    bars.append(FakeKLine((WARM + 40) * DAY, symbol,   # 244: breakout bar
                          140.0, 143.5, 140.5, 143.0, 2000.0))
    return bars


def _flat_bars(symbol="TEST"):
    """A dead-flat, low-range series: the MA stack never separates and ADR never
    clears 2.25%, so nothing should fire."""
    return [
        FakeKLine(i * DAY, symbol, 100.0, 100.5, 99.5, 100.0, 1000.0)
        for i in range(245)
    ]


def test_breakout_fires_long_entry():
    ctx, s = _run(_base_bars())

    assert s.last_action("TEST") == "entry"
    assert len(ctx.orders) == 1
    entry = ctx.orders[0]
    assert entry.side == OrderSide.Buy
    assert entry.parent is None
    # Pivot 142.5, LoD stop 140.5 -> risk 2.0: leverage 71.25, quantity clipped
    # to the 25%-of-equity cap.
    assert entry.leverage == pytest.approx(142.5 / 2.0)
    assert entry.quantity == pytest.approx(0.25 * 100_000.0 / 142.5)
    # Ledger holds a pending entry until the next bar fills it.
    assert s.position("TEST")["state"] == "pending"


def test_no_signal_stays_flat():
    ctx, s = _run(_flat_bars())
    assert ctx.orders == []
    assert s.position("TEST") is None


def test_freeroll_sells_half_at_1r():
    # After the entry fills, a bar that prints +1R triggers the freeroll half-sell
    # while price stays above both EMAs (no scale/exit) and is an up bar (no shoot).
    bars = _base_bars()
    bars.append(FakeKLine((WARM + 41) * DAY, "TEST", 143.0, 146.0, 142.5, 145.0, 1000.0))
    ctx, s = _run(bars)

    assert [o.side for o in ctx.orders] == [OrderSide.Buy, OrderSide.Sell]
    buy, sell = ctx.orders
    assert sell.quantity == pytest.approx(0.5 * buy.quantity)
    assert s.position("TEST")["freerolled"] is True
    assert s.last_action("TEST") == "freeroll"


def test_liquidation_never_sells_while_flat():
    # The entry fills, then the next bar craters through the leverage stop. The
    # ledger must clear (engine liquidated it) and place NO sell — a sell while
    # flat would open an accidental short.
    bars = _base_bars()
    bars.append(FakeKLine((WARM + 41) * DAY, "TEST", 143.0, 143.0, 100.0, 101.0, 3000.0))
    bars.append(FakeKLine((WARM + 42) * DAY, "TEST", 101.0, 101.5, 100.0, 100.5, 1000.0))
    ctx, s = _run(bars)

    assert [o.side for o in ctx.orders] == [OrderSide.Buy]   # entry only, no sell
    assert s.position("TEST") is None
    assert s.last_action("TEST") == "liquidated"


# ─── Shadow ledger (_reconcile) ──────────────────────────────────────────────
def _seed_pending(s, symbol="TEST", qty=100.0, lev=50.0):
    s.pos[symbol] = {
        "state": "pending", "order_qty": qty, "order_lev": lev,
        "entry_est": 100.0, "intended_stop": 98.0, "qty": 0.0, "remaining": 0.0,
        "entry_fill": 0.0, "leverage": lev, "effective_stop": 0.0, "R": 0.0,
        "freerolled": False, "scaled_10": False, "pending_exits": 0.0,
    }


def _seed_open(s, symbol="TEST", qty=100.0, remaining=None, entry_fill=100.0,
               R=2.0, freerolled=False, scaled_10=False, pending_exits=0.0):
    s.pos[symbol] = {
        "state": "open", "order_qty": qty, "order_lev": 50.0,
        "entry_est": entry_fill, "intended_stop": entry_fill - R, "qty": qty,
        "remaining": qty if remaining is None else remaining,
        "entry_fill": entry_fill, "leverage": 50.0,
        "effective_stop": entry_fill - R, "R": R,
        "freerolled": freerolled, "scaled_10": scaled_10,
        "pending_exits": pending_exits,
    }


def _fresh():
    s = QBreakoutStrategy()
    s.on_start(FakeContext([]))
    return s


def test_reconcile_fills_pending_entry():
    s = _fresh()
    _seed_pending(s, qty=100.0, lev=50.0)
    s._reconcile("TEST", o=100.0, h=101.0, l=99.0)   # fill at open 100
    pos = s.position("TEST")
    assert pos["state"] == "open"
    assert pos["entry_fill"] == pytest.approx(100.0)
    assert pos["remaining"] == pytest.approx(100.0)
    assert pos["effective_stop"] == pytest.approx(100.0 * (1.0 - 1.0 / 50.0))  # 98
    assert pos["R"] == pytest.approx(2.0)


def test_reconcile_liquidates_below_stop():
    s = _fresh()
    _seed_open(s, entry_fill=100.0, R=2.0)           # effective_stop 98
    s._reconcile("TEST", o=100.0, h=100.0, l=97.0)   # low pierces the stop
    assert s.position("TEST") is None
    assert s.last_action("TEST") == "liquidated"


def test_reconcile_applies_pending_exits():
    s = _fresh()
    _seed_open(s, qty=100.0, remaining=100.0, pending_exits=40.0)
    s._reconcile("TEST", o=100.0, h=101.0, l=99.0)   # exit fills, stays open
    pos = s.position("TEST")
    assert pos["remaining"] == pytest.approx(60.0)
    assert pos["pending_exits"] == 0.0

    _seed_open(s, qty=100.0, remaining=100.0, pending_exits=100.0)
    s._reconcile("TEST", o=100.0, h=101.0, l=99.0)   # exit flattens -> gone
    assert s.position("TEST") is None


# ─── Management state machine (_manage) ──────────────────────────────────────
def test_manage_freeroll():
    s = _fresh()
    ctx = FakeContext([])
    _seed_open(s, qty=100.0, entry_fill=100.0, R=2.0)
    # high 103 >= 100 + 1R; close 102 above both EMAs; up bar.
    s._manage(ctx, "TEST", o=100.0, h=103.0, l=100.0, c=102.0, v=1000.0,
              prev_low=99.0, avgvol=1000.0, e10=95.0, e21=94.0, adr=2.0)
    assert len(ctx.orders) == 1
    assert ctx.orders[0].side == OrderSide.Sell
    assert ctx.orders[0].quantity == pytest.approx(50.0)      # half of 100
    assert s.position("TEST")["freerolled"] is True


def test_manage_scale_out_below_10ema():
    s = _fresh()
    ctx = FakeContext([])
    _seed_open(s, qty=100.0, remaining=100.0, freerolled=True)
    # close below the 10-EMA but above the 21-EMA -> quarter scale-out.
    s._manage(ctx, "TEST", o=101.0, h=101.0, l=99.0, c=100.0, v=1000.0,
              prev_low=99.0, avgvol=1000.0, e10=105.0, e21=95.0, adr=2.0)
    assert ctx.orders[0].quantity == pytest.approx(25.0)      # quarter of 100
    assert s.position("TEST")["scaled_10"] is True


def test_manage_exit_remainder_below_21ema():
    s = _fresh()
    ctx = FakeContext([])
    _seed_open(s, qty=100.0, remaining=60.0, freerolled=True, scaled_10=True)
    s._manage(ctx, "TEST", o=101.0, h=101.0, l=99.0, c=100.0, v=1000.0,
              prev_low=99.0, avgvol=1000.0, e10=105.0, e21=105.0, adr=2.0)
    assert ctx.orders[0].quantity == pytest.approx(60.0)      # dumps the rest


def test_manage_supply_shoot_dumps_all():
    s = _fresh()
    ctx = FakeContext([])
    _seed_open(s, qty=100.0, remaining=80.0, freerolled=True)
    # wide-range down bar (6.3% vs adr 2.0), 3x volume, undercuts prior low.
    s._manage(ctx, "TEST", o=100.0, h=101.0, l=95.0, c=96.0, v=3000.0,
              prev_low=97.0, avgvol=1000.0, e10=90.0, e21=90.0, adr=2.0)
    assert ctx.orders[0].quantity == pytest.approx(80.0)
    assert s.last_action("TEST") == "supply_shoot"


def test_manage_holds_when_calm():
    s = _fresh()
    ctx = FakeContext([])
    _seed_open(s, qty=100.0, entry_fill=100.0, R=2.0)
    # below the +1R target, above both EMAs, quiet up bar: no exit.
    s._manage(ctx, "TEST", o=100.0, h=101.0, l=100.0, c=100.5, v=1000.0,
              prev_low=99.0, avgvol=1000.0, e10=98.0, e21=97.0, adr=2.0)
    assert ctx.orders == []
    assert s.position("TEST")["remaining"] == pytest.approx(100.0)
