"""Pure-Python unit tests for ttfm.py (the TTrades Fractal Model).

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest app/python/ -q

The model's core law is an AND gate: a weekly C2 closure (sweep the prior week's
extreme, close back inside) AND a daily CISD (delivery flip) must both confirm,
same direction. These tests pin that gate and its filters. Regime robustness /
order well-formedness is covered by the shared sweep in test_strategy_smoke.py.

Timeframes are derived from a single daily feed, so scenarios are daily bars laid
on Monday-start weeks: day 4 (1970-01-05) is a Monday, so bars spaced one day from
there fall on clean Mon–Sun weeks.
"""

import numpy as np
import pytest

from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine

from conftest import entry_orders, children_of, run_all

from ttfm import TTFMStrategy, week_index, day_of_week, weekly_bars, MS_PER_DAY

MON = 4   # epoch day 4 = Monday 1970-01-05


# ─── Scenario builders ────────────────────────────────────────────────────────
def template_week(H, L):
    """Seven daily bars spanning exactly [L, H] (high on bar 2, low on bar 4),
    everything else inside — an inert week that never sweeps an identical neighbor."""
    m = (H + L) / 2.0
    return [
        (m, m + 2, m - 2, m + 1),
        (m + 1, m + 3, m - 1, m),
        (m, H, m, H - 2),        # week high = H
        (H - 2, H - 1, m, m),
        (m, m + 1, L, L + 4),    # week low = L
        (L + 4, m, L + 2, m - 1),
        (m - 1, m + 1, m - 3, m),
    ]


# A bullish setup week: sweep the prior week's low to 88, red delivery into it,
# a displaced reclaim (CISD), a bullish FVG at (92, 95), then a close back into it.
SETUP_LONG = [
    (92, 92, 88, 89),   # b0 swing low 88 (red delivery candle)
    (89, 96, 89, 95),   # b1 CISD: closes above the run's open (92) with a big body
    (95, 99, 95, 98),   # b2 impulse leg → FVG gap (hi[b0]=92, lo[b2]=95)
    (98, 98, 92, 94),   # b3 retrace: close 94 back inside the FVG (the trigger)
]

# Bearish mirror: a red down-move precedes the green sweep of the prior week's
# high to 112 (the lead-in bounds the delivery run so its open is 108), a
# displaced break down (CISD), a bearish FVG at (103, 108), then a close into it.
SHORT_SETUP = [
    (109, 110, 107, 108),   # lead-in: red down-move (bounds the green run at b0)
    (108, 112, 108, 111),   # b0 swing high 112 (green sweep) → run open = 108
    (111, 111, 101, 102),   # b1 CISD: closes below the run's open (108) with a big body
    (99, 103, 97, 98),      # b2 impulse leg → FVG gap (hi[b2]=103, lo[b0]=108)
    (102, 107, 102, 106),   # b3 retrace: close 106 back inside the FVG (the trigger)
]


def _arrays(rows, start_day=MON):
    op = np.array([r[0] for r in rows], dtype=np.float64)
    hi = np.array([r[1] for r in rows], dtype=np.float64)
    lo = np.array([r[2] for r in rows], dtype=np.float64)
    cl = np.array([r[3] for r in rows], dtype=np.float64)
    ts = np.array([(start_day + i) * MS_PER_DAY for i in range(len(rows))], dtype=np.int64)
    return op, hi, lo, cl, ts


def _bars(rows, symbol="AAA", start_day=MON):
    return [FakeKLine((start_day + i) * MS_PER_DAY, symbol,
                      float(o), float(h), float(l), float(c), 1000.0)
            for i, (o, h, l, c) in enumerate(rows)]


# ─── Calendar helpers ─────────────────────────────────────────────────────────
def test_week_index_rolls_on_monday():
    assert week_index(4 * MS_PER_DAY) == week_index(10 * MS_PER_DAY)   # Mon..Sun same week
    assert week_index(11 * MS_PER_DAY) == week_index(4 * MS_PER_DAY) + 1   # next Monday


def test_day_of_week_maps_monday_to_zero():
    assert day_of_week(4 * MS_PER_DAY) == 0     # 1970-01-05 Monday
    assert day_of_week(5 * MS_PER_DAY) == 1     # Tuesday
    assert day_of_week(3 * MS_PER_DAY) == 6     # 1970-01-04 Sunday


def test_weekly_bars_aggregates_ohlc_and_extreme_indices():
    op, hi, lo, cl, ts = _arrays(template_week(110, 90) * 2)
    weeks = weekly_bars(ts, op, hi, lo, cl)
    assert len(weeks) == 2
    for wk in weeks:
        assert wk["high"] == pytest.approx(110.0)
        assert wk["low"] == pytest.approx(90.0)
    # High is bar 2, low is bar 4 of each 7-bar week.
    assert weeks[0]["high_i"] == 2 and weeks[0]["low_i"] == 4
    assert weeks[1]["high_i"] == 9 and weeks[1]["low_i"] == 11


# ─── The core-law gate (_signal, evaluated on the trigger bar) ────────────────
def _sig(rows):
    return TTFMStrategy()._signal(*_arrays(rows))


def test_core_law_long_fires_on_the_poi_close():
    sig = _sig(template_week(110, 90) * 3 + SETUP_LONG)
    assert sig is not None
    is_long, entry, stop, target = sig
    assert is_long is True
    assert entry == pytest.approx(94.0)
    assert stop < 88.0                       # just beyond the protected swing (88)
    assert target == pytest.approx(110.0)    # buy-side DOL = prior-week high
    assert (target - entry) >= 2.0 * (entry - stop)   # the 2R-to-DOL gate holds


def test_core_law_short_fires_on_the_poi_close():
    sig = _sig(template_week(110, 90) * 3 + SHORT_SETUP)
    assert sig is not None
    is_long, entry, stop, target = sig
    assert is_long is False
    assert entry == pytest.approx(106.0)
    assert stop > 112.0                      # just beyond the protected swing high (112)
    assert target == pytest.approx(90.0)     # sell-side DOL = prior-week low


def test_no_cisd_no_trade():
    # Weekly sweep to 88, but delivery never flips (nothing closes above the run open).
    setup = [(92, 93, 88, 89), (89, 91, 88, 90), (90, 92, 89, 91), (91, 92, 89, 90)]
    assert _sig(template_week(110, 90) * 3 + setup) is None


def test_no_htf_closure_no_trade():
    # A daily up-move with NO weekly sweep behind it (low 92 never takes out 90).
    setup = [(100, 101, 92, 93), (93, 99, 92, 98), (98, 102, 98, 101), (101, 103, 100, 102)]
    assert _sig(template_week(110, 90) * 3 + setup) is None


def test_displacement_filter_rejects_slow_cisd():
    rows = template_week(110, 90) * 3 + SETUP_LONG
    assert TTFMStrategy()._signal(*_arrays(rows)) is not None       # default passes
    s = TTFMStrategy()
    s.min_displacement = 100.0                                      # demand an impossible body
    assert s._signal(*_arrays(rows)) is None


def test_two_r_gate_skips_when_dol_is_closer_than_2r():
    # Prior-week high 100 puts the DOL under 2R from the ~94 entry → skip.
    assert _sig(template_week(100, 90) * 3 + SETUP_LONG) is None
    # Prior-week high 110 puts it beyond 2R → take it.
    assert _sig(template_week(110, 90) * 3 + SETUP_LONG) is not None


def test_c4_stand_aside_blocks_a_late_entry():
    late = SETUP_LONG + [(94, 95, 93, 94)] * 3        # trigger now >4 bars past the swing
    assert _sig(template_week(110, 90) * 3 + late) is None
    assert _sig(template_week(110, 90) * 3 + SETUP_LONG) is not None


def test_eq_filter_skips_entries_past_equilibrium():
    # FVG straddles EQ (100); the trigger closes at 101, above it.
    setup = [(96, 96, 88, 90), (90, 103, 90, 102), (102, 105, 102, 104), (104, 104, 97, 101)]
    rows = template_week(110, 90) * 3 + setup
    blocked = TTFMStrategy()
    blocked.partial_rr = 0.1                     # take the 2R gate out of the way
    assert blocked._signal(*_arrays(rows)) is None
    allowed = TTFMStrategy()
    allowed.partial_rr = 0.1
    allowed.require_eq_filter = False
    assert allowed._signal(*_arrays(rows)) is not None


# ─── End-to-end through the engine (bracket placement) ────────────────────────
def test_long_setup_places_a_long_bracket():
    bars = _bars(template_week(110, 90) * 10 + SETUP_LONG)
    ctx = run_all(TTFMStrategy(), bars)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.side == OrderSide.Buy and entry.order_type == "market"

    kids = children_of(ctx, entry.id)
    assert len(kids) == 3
    for k in kids:
        assert k.side == OrderSide.Sell and k.reduce_only
    sl = next(k for k in kids if k.order_type == "stop")
    tps = sorted((k for k in kids if k.order_type == "limit"), key=lambda o: o.price)
    assert sl.price < 88.0                       # stop beyond the protected swing
    assert tps[0].price == pytest.approx(94.0 + 2.0 * (94.0 - sl.price))   # TP1 at 2R
    assert tps[1].price == pytest.approx(110.0)  # TP2 rides to the DOL


def test_short_setup_places_a_short_bracket():
    bars = _bars(template_week(110, 90) * 10 + SHORT_SETUP)
    ctx = run_all(TTFMStrategy(), bars)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.side == OrderSide.Sell and entry.order_type == "market"

    kids = children_of(ctx, entry.id)
    assert len(kids) == 3
    for k in kids:
        assert k.side == OrderSide.Buy and k.reduce_only
    sl = next(k for k in kids if k.order_type == "stop")
    assert sl.price > 112.0                       # stop beyond the protected swing high
    assert min(k.price for k in kids if k.order_type == "limit") == pytest.approx(90.0)


def test_flat_market_stays_out():
    # Ten identical weeks never sweep each other → no bias → no orders.
    bars = _bars(template_week(110, 90) * 10)
    ctx = run_all(TTFMStrategy(), bars)
    assert entry_orders(ctx) == []
