"""Unit tests for the qullamaggie momentum swing port (qmmomentumswing.py).

FakeContext never fills orders on its own; fills are simulated by flipping
the order status and setting ctx.positions, mirroring how the broker reports
them. Engine fill mechanics (stop trigger prices, bracket OCO, settle
rounds) are pinned by the C++ suite under tests/core/.

Two execution models are covered (docstring deviation 3):
  - use_bo_vol=True (default): VIRTUAL arms — nothing parked; the strategy
    evaluates the pine's break+volume fill condition itself and enters at
    market. Low-volume crossings produce no orders at all.
  - use_bo_vol=False: the resting-stop model — a real stop order parks at
    the level with its bracket and fills on the first touch."""

import numpy as np
import pytest

import stonks
from stonks import OrderSide, OrderStatus
from stonks.testing import FakeContext, FakeKLine, FakePosition

from qmmomentumswing import QMMomentumSwingStrategy, max_safe_leverage

MS = 3_600_000  # 1h bars: 24 per UTC day
BUF = 1.0 + 5.0 / 10_000.0  # default entry_buffer_bps
FEE = 5.0 / 10_000.0        # default taker_fee_bps as a fraction


def risked(entry_order, stop_order, entry_ref=None):
    """Dollars lost on a stop-out including both taker fee legs (notes §2).
    Market entries have no order price; pass their signal anchor."""
    entry_px = entry_order.price if entry_order.price is not None else entry_ref
    return entry_order.quantity * (abs(entry_px - stop_order.price)
                                   + entry_px * FEE
                                   + stop_order.price * FEE)


def bars_from(symbol, rows, start_ts=MS):
    return [FakeKLine(start_ts + i * MS, symbol, o, h, l, c, v)
            for i, (o, h, l, c, v) in enumerate(rows)]


def start_run(rows, symbol, start_ts=MS, **overrides):
    ctx = FakeContext(bars_from(symbol, rows, start_ts))
    strategy = QMMomentumSwingStrategy()
    for name, value in overrides.items():
        setattr(strategy, name, value)
    strategy.on_start(ctx)
    return ctx, strategy


def drive(strategy, ctx, ticks):
    for _ in range(ticks):
        ctx.advance()
        strategy.on_tick(ctx)


def entries(ctx):
    return [o for o in ctx.orders if not o.reduce_only]


def children(ctx, entry_id):
    return [o for o in ctx.orders if o.parent == entry_id]


def adr20(rows, idx):
    """The strategy's ADR on bar `idx`: SMA(100 x (high/low - 1), 20)."""
    window = rows[idx - 19: idx + 1]
    return np.mean([100.0 * (h / l - 1.0) for (_, h, l, _, _) in window])


# ─── Bar builders ─────────────────────────────────────────────────────────────

def ramp_rows(n=44, start=100.0, step=1.01):
    """Steady trend: every bar opens at the prior close (no gaps -> no EP)."""
    rows = []
    c = start
    for _ in range(n):
        o = c
        c = o * step
        hi, lo = max(o, c), min(o, c)
        rows.append((o, hi * 1.003, lo * 0.997, c, 1000.0))
    return rows


def flag_rows(prev_close, n, first_move=0.99, drift=1.0005):
    """Tight consolidation under the prior extreme: one small counter-move,
    then a drift too small to take the pivot out."""
    rows = []
    c = prev_close
    for j in range(n):
        o = c
        c = o * (first_move if j == 0 else drift)
        rows.append((o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000.0))
    return rows


def armed_bo_rows(flag_bars=3):
    """ramp + flag: with defaults the BO arms on the 3rd flag bar. Returns
    (rows, level)."""
    rows = ramp_rows(50)
    rows += flag_rows(rows[-1][3], flag_bars)
    return rows, max(r[1] for r in rows) * BUF


# ─── Virtual arms (use_bo_vol on, the default) ────────────────────────────────

def test_virtual_bo_churn_regression(capsys):
    """Low-volume crossings of the armed level place NOTHING (the old code
    booked a bail round trip per bar); the first volume-confirmed break
    enters at market with pine-anchored SL/TP."""
    rows, level = armed_bo_rows()
    o = rows[-1][3]
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 1000.0))  # skip 1
    for _ in range(2):                                    # skips 2-3: held above
        o = rows[-1][3]
        rows.append((o, o * 1.002, o * 0.998, o * 1.001, 1000.0))
    o = rows[-1][3]
    rows.append((o, o * 1.003, o * 0.995, o * 1.002, 5000.0))           # volume bar
    sym = "VA1"
    ctx, s = start_run(rows, sym)
    drive(s, ctx, len(rows) - 1)
    assert ctx.orders == []                               # zero orders while skipping
    out = capsys.readouterr().out
    assert "BO LONG arm virtual" in out
    assert out.count("skipped (vol") == 3

    drive(s, ctx, 1)                                      # the volume-confirmed break
    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "market" and e.side == OrderSide.Buy

    fill = rows[-1]
    entry_ref = max(fill[0], level)                       # open is above the level
    assert entry_ref == fill[0]
    kids = children(ctx, e.id)
    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")
    # LOD stop: the break bar's low beats the ADR stop here (deviation 4)
    adr = adr20(rows, len(rows) - 1)
    expected_stop = min(max(entry_ref * (1 - adr / 100.0), fill[2]),
                        entry_ref * 0.999)
    assert expected_stop == fill[2]                       # genuinely the bar low
    assert sl.price == pytest.approx(expected_stop, rel=1e-9)
    assert tp.price == pytest.approx(entry_ref + 2.0 * (entry_ref - sl.price),
                                     rel=1e-9)
    assert sl.reduce_only and tp.reduce_only
    assert risked(e, sl, entry_ref) == pytest.approx(ctx.equity() * 0.01, rel=1e-9)
    assert e.leverage == max_safe_leverage(entry_ref, sl.price, 0.0, 100)
    assert "BO LONG enter market" in capsys.readouterr().out


def test_virtual_expiry_boundary_fills_on_last_valid_bar():
    rows, level = armed_bo_rows()                         # arm on tick 53
    c = rows[-1][3]
    for _ in range(9):                                    # setup dies, level untouched
        o = c
        c = o * 0.94
        rows.append((o, o * 1.0005, c * 0.997, c, 1000.0))
    o = c
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 5000.0))  # tick 63
    ctx, s = start_run(rows, "VA2")
    drive(s, ctx, len(rows))                              # cross at armed+10: fills
    es = entries(ctx)
    assert len(es) == 1 and es[0].order_type == "market"


def test_virtual_expiry_boundary_expires_one_bar_later(capsys):
    rows, level = armed_bo_rows()
    c = rows[-1][3]
    for _ in range(10):
        o = c
        c = o * 0.94
        rows.append((o, o * 1.0005, c * 0.997, c, 1000.0))
    o = c
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 5000.0))  # tick 64
    ctx, s = start_run(rows, "VA3")
    drive(s, ctx, len(rows))                              # armed+11: expired first
    assert ctx.orders == []
    assert "BO order expired unfilled" in capsys.readouterr().out


def test_virtual_arm_survives_ep_hold():
    """pine keeps boArmed while inLong: an EP hold blocks fills (canEnter)
    but neither cancels the arm nor resets its timer; the arm can fill
    right after the exit."""
    rows, level = armed_bo_rows()                         # arm on tick 53
    prev = rows[-1][3]
    o = prev * 1.006                                      # EP: gap, volume, strong close
    c = o * 1.003
    rows.append((o, c * 1.0005, o * 0.999, c, 5000.0))    # tick 54, high < level
    assert rows[-1][1] < level
    o = rows[-1][3]
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 5000.0))  # 55: held
    o = rows[-1][3]
    rows.append((o, o * 1.0005, level * 0.996, level * 0.997, 1000.0))  # 56: exit bar
    o = rows[-1][3]
    rows.append((o, level * 1.002, o * 0.999, level * 1.001, 5000.0))   # 57: refill
    sym = "VA4"
    ctx, s = start_run(rows, sym)
    drive(s, ctx, 54)
    es = entries(ctx)
    assert len(es) == 1                                   # the EP market entry
    ep = es[0]
    assert ep.order_type == "market"

    ep.status = OrderStatus.Filled                        # engine: fills next open
    ctx.positions[sym] = FakePosition(ep.quantity, level * 0.999, entry_id=ep.id)
    drive(s, ctx, 1)                                      # 55: crossing while held
    assert len(entries(ctx)) == 1                         # canEnter blocks the arm

    for k in children(ctx, ep.id):                        # bracket closes the EP
        k.status = OrderStatus.Cancelled
    del ctx.positions[sym]
    drive(s, ctx, 1)                                      # 56: exitedNow blocks
    assert len(entries(ctx)) == 1
    drive(s, ctx, 1)                                      # 57: arm still live -> fill
    es = entries(ctx)
    assert len(es) == 2
    assert es[1].order_type == "market" and es[1].side == OrderSide.Buy


def test_virtual_level_rearm_prints_without_orders(capsys):
    rows, _ = armed_bo_rows()                             # L1 on tick 53
    c = rows[-1][3]
    for _ in range(2):                                    # new leg up -> higher pivot
        o = c
        c = o * 1.02
        rows.append((o, c * 1.003, o * 0.997, c, 1000.0))
    rows += flag_rows(rows[-1][3], 3)                     # L2 on tick 58
    ctx, s = start_run(rows, "VA5")
    drive(s, ctx, len(rows))
    assert ctx.orders == []                               # virtual: nothing parked
    out = capsys.readouterr().out
    assert out.count("BO LONG arm virtual") == 2


def test_virtual_short_bo_mirror():
    rows = []
    c = 100.0
    for _ in range(45):                                   # downtrend leg
        o = c
        c = o * 0.99
        rows.append((o, o * 1.003, c * 0.997, c, 1000.0))
    for j in range(6):                                    # weak bounce above the trough
        o = c
        c = o * (1.01 if j == 0 else 0.9995)
        rows.append((o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000.0))
    level = min(r[2] for r in rows) * (1.0 - 5.0 / 10_000.0)
    o = rows[-1][3]
    rows.append((o, o * 1.0005, level * 0.998, level * 0.999, 5000.0))  # breakdown
    ctx, s = start_run(rows, "VA6", enable_short_bo=True)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "market" and e.side == OrderSide.Sell

    fill = rows[-1]
    entry_ref = min(fill[0], level)                       # open is above the level
    assert entry_ref == level
    kids = children(ctx, e.id)
    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")
    # short LOD: the break bar's high beats the ADR stop (deviation 4)
    adr = adr20(rows, len(rows) - 1)
    expected_stop = max(min(entry_ref * (1 + adr / 100.0), fill[1]),
                        entry_ref * 1.001)
    assert expected_stop == fill[1]                       # genuinely the bar high
    assert sl.side == OrderSide.Buy
    assert sl.price == pytest.approx(expected_stop, rel=1e-9)
    assert tp.price == pytest.approx(entry_ref - 2.0 * (sl.price - entry_ref),
                                     rel=1e-9)
    assert risked(e, sl, entry_ref) == pytest.approx(ctx.equity() * 0.01, rel=1e-9)


def test_virtual_orb_skip_fill_and_rejection(capsys):
    """Hourly bars: ORB breaks before the 51-bar volume warmup are skipped;
    the first confirmable break enters at market; a broker rejection burns
    the opportunity (orb_taken, deviation 14) so a later break is inert."""
    rows = [(100.0, 100.3, 99.7, 100.0, 1000.0)] * 47     # days 0-1: flat
    c = 100.0
    for _ in range(3):                                    # day 2 (ticks 48-50): ramp
        o = c
        c = o * 1.01
        rows.append((o, c * 1.003, o * 0.997, c, 1000.0))
    o = c
    rows.append((o, c * 1.02, o * 0.997, c * 1.015, 5000.0))   # tick 51: volume break
    o = rows[-1][3]
    rows.append((o, o * 1.02, o * 0.997, o * 1.01, 5000.0))    # tick 52: second break
    ctx, s = start_run(rows, "VA7", enable_orb=True, mom_len=5)
    drive(s, ctx, 50)
    assert ctx.orders == []                               # warmup breaks only skipped
    assert capsys.readouterr().out.count("ORB break") >= 1

    drive(s, ctx, 1)                                      # tick 51: fill
    es = entries(ctx)
    assert len(es) == 1 and es[0].order_type == "market"
    es[0].status = OrderStatus.Rejected                   # broker: not enough cash
    drive(s, ctx, 1)                                      # tick 52: orb_taken holds
    assert len(entries(ctx)) == 1
    assert "ORB entry rejected by broker" in capsys.readouterr().out


def test_virtual_bo_beats_ep_same_bar(capsys):
    rows, level = armed_bo_rows()
    prev = rows[-1][3]
    o = prev * 1.006                                      # EP gap bar that ALSO breaks
    c = level * 1.002
    rows.append((o, c * 1.0005, o * 0.999, c, 5000.0))
    ctx, s = start_run(rows, "VA8")
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    out = capsys.readouterr().out
    assert "BO LONG enter market" in out and "EP LONG" not in out
    # numbers anchor to the pine's entry_ref = max(open, level) = level
    fill = rows[-1]
    entry_ref = max(fill[0], level)
    assert entry_ref == level
    sl = next(k for k in children(ctx, es[0].id) if k.order_type == "stop")
    adr = adr20(rows, len(rows) - 1)
    expected_stop = min(max(entry_ref * (1 - adr / 100.0), fill[2]),
                        entry_ref * 0.999)
    assert sl.price == pytest.approx(expected_stop, rel=1e-9)


def test_virtual_no_double_entry_while_in_flight():
    rows, level = armed_bo_rows()
    o = rows[-1][3]
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 5000.0))  # break 1
    o = rows[-1][3]
    rows.append((o, o * 1.003, o * 0.998, o * 1.002, 5000.0))           # break 2
    sym = "VA9"
    ctx, s = start_run(rows, sym)
    drive(s, ctx, len(rows) - 1)
    es = entries(ctx)
    assert len(es) == 1                                   # entry placed, still Open
    drive(s, ctx, 1)                                      # in flight: no double entry
    assert len(entries(ctx)) == 1

    es[0].status = OrderStatus.Filled                     # now it fills
    ctx.positions[sym] = FakePosition(es[0].quantity, level, entry_id=es[0].id)
    drive(s, ctx, 0)
    assert len(entries(ctx)) == 1


def test_virtual_rejected_entry_consumes_arm_then_rearms(capsys):
    rows, level = armed_bo_rows()
    o = rows[-1][3]
    rows.append((o, level * 1.001, o * 0.999, level * 0.998, 5000.0))   # break bar
    rows += flag_rows(rows[-1][3], 3)                     # base rebuilds under new pivot
    sym = "VA10"
    ctx, s = start_run(rows, sym)
    drive(s, ctx, 54)                                     # entry placed on the break
    e = entries(ctx)[0]
    e.status = OrderStatus.Rejected
    drive(s, ctx, 1)
    assert "BO entry rejected by broker" in capsys.readouterr().out
    assert len(entries(ctx)) == 1                         # arm was consumed (dev 14)
    drive(s, ctx, 2)                                      # 3rd flag bar re-arms
    out = capsys.readouterr().out
    assert "BO LONG arm virtual" in out
    assert len(entries(ctx)) == 1                         # armed only, nothing placed


# ─── BO: momentum breakout, resting-stop model (use_bo_vol off) ───────────────

def test_bo_arms_stop_entry_with_bracket(capsys):
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 6)
    ctx, s = start_run(rows, "BO1", use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "stop" and e.side == OrderSide.Buy
    assert e.status == OrderStatus.Open

    peak_high = max(r[1] for r in rows)
    assert e.price == pytest.approx(peak_high * BUF, rel=1e-12)

    kids = children(ctx, e.id)
    assert len(kids) == 2
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.side == OrderSide.Sell and sl.reduce_only and sl.price < e.price
    assert tp.side == OrderSide.Sell and tp.reduce_only and tp.price > e.price
    assert sl.quantity == e.quantity and tp.quantity == e.quantity

    # stop = entry x (1 - adr%) off the armed level, using the arm bar's ADR
    arm_idx = 46  # third flag bar: sincePk hits min_base_bars
    adr = adr20(rows, arm_idx)
    assert sl.price == pytest.approx(min(e.price * (1 - adr / 100),
                                         e.price * 0.999), rel=1e-9)
    # target at target_rr R; a stop-out loses exactly risk_fraction of
    # equity including both fee legs (notes §2 risk mode)
    assert tp.price - e.price == pytest.approx(2.0 * (e.price - sl.price), rel=1e-9)
    assert risked(e, sl) == pytest.approx(ctx.equity() * 0.01, rel=1e-9)

    # max-safe isolated leverage on the entry only (notes §9)
    assert e.leverage == max_safe_leverage(e.price, sl.price, 0.0, 100)
    assert 1 < e.leverage < 100
    assert sl.leverage == 1.0 and tp.leverage == 1.0

    out = capsys.readouterr().out
    assert "BO LONG arm stop-entry" in out and "valid 10 bars" in out
    assert f"L {int(e.leverage)}x margin" in out


def test_bo_rearm_keeps_resting_order_alive_past_expiry():
    # The setup holds for 14 flag bars; the pine re-arms every bar, so the
    # 10-bar expiry timer keeps resetting and the order never lapses. The
    # level never changes either, so no cancel/replace churn.
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 14)
    ctx, s = start_run(rows, "BO2", use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    assert es[0].status == OrderStatus.Open


def test_bo_arm_expires_after_order_bars(capsys):
    # Setup fades right after arming (trend collapses, level untouched by
    # FakeContext): the order must be cancelled order_bars after the last arm.
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 3)
    c = rows[-1][3]
    for _ in range(14):  # crash: universe gates fail immediately
        o = c
        c = o * 0.94
        rows.append((o, o * 1.0005, c * 0.997, c, 1000.0))
    ctx, s = start_run(rows, "BO3", use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    assert es[0].status == OrderStatus.Cancelled
    assert all(k.status == OrderStatus.Cancelled for k in children(ctx, es[0].id))
    assert "BO order expired unfilled" in capsys.readouterr().out


def test_bo_level_change_cancels_and_replaces():
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 3)      # arms at the first pivot
    c = rows[-1][3]
    for _ in range(2):                     # new leg up -> higher pivot
        o = c
        c = o * 1.02
        rows.append((o, c * 1.003, o * 0.997, c, 1000.0))
    rows += flag_rows(rows[-1][3], 3)      # re-arms at the higher pivot
    ctx, s = start_run(rows, "BO4", use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 2
    first, second = es
    assert first.status == OrderStatus.Cancelled
    assert all(k.status == OrderStatus.Cancelled for k in children(ctx, first.id))
    assert second.status == OrderStatus.Open
    assert second.price > first.price
    assert second.price == pytest.approx(max(r[1] for r in rows) * BUF, rel=1e-12)


def test_fill_position_gate_and_rearm_after_close():
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 14)
    sym = "BO5"
    # resting-stop model: this test exercises position gating on real fills
    ctx, s = start_run(rows, sym, use_bo_vol=False)
    drive(s, ctx, 47)                      # arm happens on tick 47

    e = entries(ctx)[0]
    assert e.status == OrderStatus.Open
    # simulate the broker: entry fills, position opens
    e.status = OrderStatus.Filled
    ctx.positions[sym] = FakePosition(e.quantity, e.price, entry_id=e.id)
    drive(s, ctx, 4)
    assert len(entries(ctx)) == 1          # position gates every setup

    # bracket closes the position: flat again, children OCO-cancelled
    for k in children(ctx, e.id):
        k.status = OrderStatus.Cancelled
    del ctx.positions[sym]
    drive(s, ctx, 1)                       # exit bar: pine's canEnter blocks
    assert len(entries(ctx)) == 1
    drive(s, ctx, 1)                       # setup still holds -> fresh arm
    es = entries(ctx)
    assert len(es) == 2 and es[1].status == OrderStatus.Open


# ─── EP: episodic pivot (gap, long) ───────────────────────────────────────────

def ep_rows():
    quiet = [(100.0, 100.3, 99.75, 100.0, 1000.0)] * 55
    o = 102.0                              # +2% gap over the prior close
    lo = o * 0.999
    c = o * 1.003                          # strong close, tiny bar risk
    hi = c * 1.0005
    return quiet + [(o, hi, lo, c, 5000.0)]


def test_ep_market_entry_with_lod_stop(capsys):
    rows = ep_rows()
    ctx, s = start_run(rows, "EP1")
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "market" and e.side == OrderSide.Buy

    o, hi, lo, c, v = rows[-1]
    kids = children(ctx, e.id)
    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")
    # ADR stop is wider than the signal-bar low here -> LOD stop wins
    assert sl.price == pytest.approx(lo, rel=1e-12)
    assert tp.price == pytest.approx(c + 2.0 * (c - lo), rel=1e-9)
    assert sl.reduce_only and tp.reduce_only
    # market entry has no order price; risk is anchored to the signal close
    assert e.quantity == pytest.approx(
        ctx.equity() * 0.01 / ((c - lo) + c * FEE + lo * FEE), rel=1e-9)
    # sub-1% stop distance -> L_max > 100 -> capped at max_leverage
    assert e.leverage == 100

    out = capsys.readouterr().out
    assert "EP LONG enter market" in out and "L 100x margin" in out


def test_no_signal_while_in_position():
    rows = ep_rows()
    ctx, s = start_run(rows, "EP2")
    ctx.positions["EP2"] = FakePosition(1.0, 100.0)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


# ─── Universe gates ───────────────────────────────────────────────────────────

def test_min_price_gate_blocks_everything():
    rows = [(o / 100, h / 100, l / 100, c / 100, v) for (o, h, l, c, v) in ep_rows()]
    ctx, s = start_run(rows, "GT1")        # prices ~1 < min_price 5
    drive(s, ctx, len(rows))
    assert ctx.orders == []


def test_min_gain_gate_blocks_bo():
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 6)
    ctx, s = start_run(rows, "GT2", min_gain=1e9)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


# ─── MA-stack gate (require_ma_stack — pine's requireStack) ───────────────────

def broken_stack_long_rows():
    """Old high plateau -> deep decline -> short sharp V-recovery + flag. A long
    BO arms (recent gain, close and the 10-SMA above the 20-SMA) but the 20-SMA
    still sits below the 50-SMA because the deep decline is still inside the
    50-bar window, so the sma10>sma20>sma50 stack is NOT aligned."""
    rows = []
    c = 200.0
    for _ in range(40):                    # high plateau: lifts the 50-SMA
        o = c
        rows.append((o, o * 1.003, o * 0.997, c, 1000.0))
    for _ in range(40):                    # deep decline
        o = c
        c = o * 0.985
        rows.append((o, o * 1.003, c * 0.997, c, 1000.0))
    for _ in range(12):                    # short sharp V-recovery: 10>20, 20<50
        o = c
        c = o * 1.02
        rows.append((o, c * 1.003, o * 0.997, c, 1000.0))
    rows += flag_rows(rows[-1][3], 6)      # tight flag under the recovery pivot
    return rows


def broken_stack_short_rows():
    """Mirror of broken_stack_long_rows: low plateau -> rally -> short sharp
    decline + weak bounce. A SHORT_BO arms (recent loss, close and the 10-SMA
    below the 20-SMA) but the 20-SMA still sits above the 50-SMA because the
    rally is still inside the 50-bar window, so the sma10<sma20<sma50 stack is
    NOT aligned."""
    rows = []
    c = 100.0
    for _ in range(40):                    # low plateau: holds the 50-SMA down
        o = c
        rows.append((o, o * 1.003, o * 0.997, c, 1000.0))
    for _ in range(40):                    # rally: lifts the 20-SMA above the 50
        o = c
        c = o * 1.015
        rows.append((o, c * 1.003, o * 0.997, c, 1000.0))
    for _ in range(12):                    # short sharp decline: 10<20, 20>50
        o = c
        c = o * 0.97
        rows.append((o, o * 1.003, c * 0.997, c, 1000.0))
    rows += flag_rows(rows[-1][3], 6, first_move=1.01, drift=0.9995)  # weak bounce
    return rows


def short_bo_rows():
    """Downtrend leg then a weak bounce above the trough -> a SHORT_BO arms."""
    rows = []
    c = 100.0
    for _ in range(44):
        o = c
        c = o * 0.99
        rows.append((o, o * 1.003, c * 0.997, c, 1000.0))
    for j in range(6):
        o = c
        c = o * (1.01 if j == 0 else 0.9995)
        rows.append((o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000.0))
    return rows


def para_rows():
    """Parabolic run then a first red bar -> a PARA short fires."""
    rows = [(100.0, 100.4, 99.7, 100.0, 1000.0)] * 25
    c = 100.0
    for _ in range(5):
        o = c
        c = o * 1.025
        rows.append((o, c * 1.002, o * 0.998, c, 1000.0))
    o = c
    c = o * 0.99
    rows.append((o, o * 1.001, c * 0.998, c, 1000.0))
    return rows


def test_ma_stack_allows_aligned_long():
    # steady 60-bar uptrend -> sma10>sma20>sma50 holds; the flag arms BO
    # (60 bars is under the 200-bar warmup the old sma200 stack required)
    rows = ramp_rows(60)
    rows += flag_rows(rows[-1][3], 6)
    ctx, s = start_run(rows, "MS0", require_ma_stack=True, use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    assert es[0].order_type == "stop" and es[0].side == OrderSide.Buy
    assert es[0].status == OrderStatus.Open


def test_ma_stack_allows_aligned_short():
    # steady 60-bar downtrend -> sma10<sma20<sma50 holds; the base arms SHORT_BO
    rows = ramp_rows(60, start=200.0, step=0.99)
    rows += flag_rows(rows[-1][3], 6, first_move=1.01, drift=0.9995)
    ctx, s = start_run(rows, "MS9", enable_short_bo=True, require_ma_stack=True,
                       use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    assert es[0].order_type == "stop" and es[0].side == OrderSide.Sell
    assert es[0].status == OrderStatus.Open


def test_ma_stack_blocks_unaligned_long():
    rows = broken_stack_long_rows()
    # filter off: the recovery flag arms a long BO
    ctx, s = start_run(rows, "MS1", use_bo_vol=False)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) >= 1
    # filter on: 200-SMA sits above the fast SMAs -> stack broken -> BO blocked
    ctx, s = start_run(rows, "MS2", require_ma_stack=True, use_bo_vol=False)
    drive(s, ctx, len(rows))
    assert entries(ctx) == []


def test_ma_stack_gates_short_bo():
    rows = broken_stack_short_rows()
    ctx, s = start_run(rows, "MS3", enable_short_bo=True, use_bo_vol=False)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) >= 1          # filter off: SHORT_BO arms
    # filter on: 20-SMA sits above the 50-SMA -> inverse stack broken -> gated
    ctx, s = start_run(rows, "MS4", enable_short_bo=True, require_ma_stack=True,
                       use_bo_vol=False)
    drive(s, ctx, len(rows))
    assert entries(ctx) == []


def test_ma_stack_does_not_gate_ep():
    # pine scope: EP rides only the liquidity gate, so require_ma_stack must NOT
    # touch it — it fires whether the filter is off or on.
    rows = ep_rows()
    ctx, s = start_run(rows, "MS5")
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) == 1          # filter off: EP fires
    ctx, s = start_run(rows, "MS6", require_ma_stack=True)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) == 1          # filter on: EP STILL fires (not gated)


def test_ma_stack_does_not_gate_para():
    # pine scope: PARA rides only the liquidity gate — unaffected by the filter.
    rows = para_rows()
    ctx, s = start_run(rows, "MS7", enable_para=True)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) == 1          # filter off: PARA fires
    ctx, s = start_run(rows, "MS8", enable_para=True, require_ma_stack=True)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) == 1          # filter on: PARA STILL fires (not gated)


# ─── SHORT_BO: breakdown (mirror), resting-stop model ─────────────────────────

def test_short_bo_arms_sell_stop_with_bracket():
    rows = short_bo_rows()
    ctx, s = start_run(rows, "SB1", enable_short_bo=True, use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "stop" and e.side == OrderSide.Sell
    assert e.status == OrderStatus.Open

    base_low = min(r[2] for r in rows)
    assert e.price == pytest.approx(base_low * (1.0 - 5.0 / 10_000.0), rel=1e-12)

    kids = children(ctx, e.id)
    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")
    assert sl.side == OrderSide.Buy and sl.price > e.price
    assert tp.side == OrderSide.Buy and tp.price < e.price
    assert e.price - tp.price == pytest.approx(2.0 * (sl.price - e.price), rel=1e-9)
    assert sl.reduce_only and tp.reduce_only
    assert risked(e, sl) == pytest.approx(ctx.equity() * 0.01, rel=1e-9)
    # short branch of the L_max formula: E / (S(1+m) - E)
    assert e.leverage == max_safe_leverage(e.price, sl.price, 0.0, 100)
    assert 1 < e.leverage < 100


# ─── PARA: parabolic first-red-bar short ──────────────────────────────────────

def test_para_short_market_entry_no_target(capsys):
    rows = para_rows()
    ctx, s = start_run(rows, "PS1", enable_para=True)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "market" and e.side == OrderSide.Sell

    kids = children(ctx, e.id)
    assert len(kids) == 1                  # stop-loss only: the pine has no PARA target
    sl = kids[0]
    assert sl.order_type == "stop" and sl.side == OrderSide.Buy and sl.reduce_only
    stop_ref = max(r[1] for r in rows[-3:])
    assert sl.price == pytest.approx(stop_ref * BUF, rel=1e-12)
    assert not any(o.order_type == "limit" for o in ctx.orders)
    close = rows[-1][3]
    assert e.quantity == pytest.approx(
        ctx.equity() * 0.01 / ((sl.price - close) + close * FEE + sl.price * FEE),
        rel=1e-9)
    assert e.leverage == max_safe_leverage(close, sl.price, 0.0, 100)

    out = capsys.readouterr().out
    assert "PARA SHORT enter market" in out and "TP none" in out


# ─── ORB: opening range breakout, resting-stop model ──────────────────────────

def test_orb_arms_then_lapses_with_setup(capsys):
    # start at 18:00 UTC so bar index 30 lands exactly on a day boundary
    rows = [(100.0, 100.3, 99.7, 100.0, 1000.0)] * 30   # flat: universe false
    rows.append((100.0, 101.303, 99.7, 101.0, 1000.0))  # day-2 opening bar
    rows.append((101.0, 102.31603, 100.697, 102.01, 1000.0))  # arm expected here
    rows.append((102.01, 102.061005, 96.61877215, 96.9095, 1000.0))  # trend dies
    ctx, s = start_run(rows, "OR1", start_ts=18 * MS, enable_orb=True,
                       use_bo_vol=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "stop" and e.side == OrderSide.Buy
    assert e.price == pytest.approx(101.303 * BUF, rel=1e-12)
    assert e.status == OrderStatus.Cancelled   # universe lapsed on the crash bar

    out = capsys.readouterr().out
    assert "ORB LONG arm stop-entry" in out
    assert "ORB order cancelled (setup lapsed)" in out


# ─── Sizing / leverage math ───────────────────────────────────────────────────

def test_max_safe_leverage_formula():
    # long: L_raw = E/(E - S(1-m)); exact integers step one down (notes §9)
    assert max_safe_leverage(100.0, 99.0, 0.0, 100) == 99     # raw exactly 100
    assert max_safe_leverage(100.0, 98.5, 0.0, 100) == 66     # raw 66.67
    assert max_safe_leverage(100.0, 99.5, 0.0, 100) == 100    # raw 200 -> cap
    assert max_safe_leverage(100.0, 55.0, 0.0, 100) == 2      # raw 2.22
    assert max_safe_leverage(100.0, 50.0, 0.0, 100) == 1      # raw exactly 2 -> 1
    assert max_safe_leverage(100.0, 1.0, 0.0, 100) == 1       # raw 1.01, floor >= 1
    # short: L_raw = E/(S(1+m) - E)
    assert max_safe_leverage(100.0, 101.0, 0.0, 100) == 99    # raw exactly 100
    # maintenance margin pulls L_max down: 100/(100 - 99*0.996) = 71.63
    assert max_safe_leverage(100.0, 99.0, 0.004, 100) == 71


def test_rejected_entry_prints_and_rearms(capsys):
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 14)
    ctx, s = start_run(rows, "RJ1", use_bo_vol=False)
    drive(s, ctx, 47)                      # arm happens on tick 47

    e = entries(ctx)[0]
    e.status = OrderStatus.Rejected        # broker: margin + fee > free cash
    drive(s, ctx, 1)
    assert "entry rejected by broker" in capsys.readouterr().out
    es = entries(ctx)                      # setup still holds -> fresh arm
    assert len(es) == 2 and es[1].status == OrderStatus.Open


# ─── Discovery metadata ───────────────────────────────────────────────────────

def test_param_and_indicator_specs_resolve():
    specs = stonks.param_specs(QMMomentumSwingStrategy)
    assert len(specs) == len(QMMomentumSwingStrategy.params)
    names = {p["name"] for p in specs}
    assert {"enable_bo", "enable_orb", "enable_short_bo", "enable_ep",
            "enable_para", "risk_fraction", "target_rr", "taker_fee_bps",
            "maintenance_margin_pct", "max_leverage", "use_bo_vol",
            "bo_vol_mult"} <= names
    inds = stonks.indicator_specs(QMMomentumSwingStrategy)
    assert {i["name"] for i in inds} == {"order_level", "stop_level",
                                         "target_level"}
