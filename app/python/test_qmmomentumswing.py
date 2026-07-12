"""Unit tests for the qullamaggie momentum swing port (qmmomentumswing.py).

FakeContext never fills orders on its own; fills are simulated by flipping
the order status and setting ctx.positions, mirroring how the broker reports
them. Engine fill mechanics (stop trigger prices, bracket OCO, settle
rounds) are pinned by the C++ suite under tests/core/."""

import numpy as np
import pytest

import stonks
from stonks import OrderSide, OrderStatus
from stonks.testing import FakeContext, FakeKLine, FakePosition

from qmmomentumswing import QMMomentumSwingStrategy, max_safe_leverage

MS = 3_600_000  # 1h bars: 24 per UTC day
BUF = 1.0 + 5.0 / 10_000.0  # default entry_buffer_bps
FEE = 5.0 / 10_000.0        # default taker_fee_bps as a fraction


def risked(entry_order, stop_order):
    """Dollars lost on a stop-out including both taker fee legs (notes §2)."""
    return entry_order.quantity * (abs(entry_order.price - stop_order.price)
                                   + entry_order.price * FEE
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


# ─── BO: momentum breakout (long) ─────────────────────────────────────────────

def test_bo_arms_stop_entry_with_bracket(capsys):
    rows = ramp_rows()
    rows += flag_rows(rows[-1][3], 6)
    ctx, s = start_run(rows, "BO1")
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
    window = rows[arm_idx - 19: arm_idx + 1]
    adr = np.mean([100.0 * (h / l - 1.0) for (_, h, l, _, _) in window])
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
    ctx, s = start_run(rows, "BO2")
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
    ctx, s = start_run(rows, "BO3")
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
    ctx, s = start_run(rows, "BO4")
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
    # volume gate off: this test exercises position gating, and its fill
    # bar has flat volume that would otherwise trigger the bail path
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
    """Old high plateau -> long decline -> recent recovery + flag. A long BO
    arms (recent gain, rising 10/20 SMAs) but the 200-SMA still sits above the
    fast SMAs because the old plateau is still inside its window, so the full
    sma10>sma20>sma50>sma200 stack is NOT aligned."""
    rows = []
    c = 200.0
    for _ in range(70):                    # plateau: pins the 200-SMA high
        o = c
        rows.append((o, o * 1.003, o * 0.997, c, 1000.0))
    for _ in range(70):                    # decline to ~100
        o = c
        c = o * 0.99
        rows.append((o, o * 1.003, c * 0.997, c, 1000.0))
    for _ in range(60):                    # recovery to ~130
        o = c
        c = o * 1.005
        rows.append((o, c * 1.003, o * 0.997, c, 1000.0))
    rows += flag_rows(rows[-1][3], 6)      # tight flag under the recovery pivot
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
    # steady 210-bar uptrend -> sma10>sma20>sma50>sma200 holds; the flag arms BO
    rows = ramp_rows(210)
    rows += flag_rows(rows[-1][3], 6)
    ctx, s = start_run(rows, "MS0", require_ma_stack=True)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    assert es[0].order_type == "stop" and es[0].side == OrderSide.Buy
    assert es[0].status == OrderStatus.Open


def test_ma_stack_blocks_unaligned_long():
    rows = broken_stack_long_rows()
    # filter off (default): the recovery flag arms a long BO
    ctx, s = start_run(rows, "MS1")
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) >= 1
    # filter on: 200-SMA sits above the fast SMAs -> stack broken -> BO blocked
    ctx, s = start_run(rows, "MS2", require_ma_stack=True)
    drive(s, ctx, len(rows))
    assert entries(ctx) == []


def test_ma_stack_gates_short_bo():
    rows = short_bo_rows()
    ctx, s = start_run(rows, "MS3", enable_short_bo=True)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) >= 1          # filter off: SHORT_BO arms
    # filter on: <200 bars -> inverse stack invalid -> the short universe is gated
    ctx, s = start_run(rows, "MS4", enable_short_bo=True, require_ma_stack=True)
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


# ─── SHORT_BO: breakdown (mirror) ─────────────────────────────────────────────

def test_short_bo_arms_sell_stop_with_bracket():
    rows = []
    c = 100.0
    for _ in range(44):                    # downtrend leg
        o = c
        c = o * 0.99
        rows.append((o, o * 1.003, c * 0.997, c, 1000.0))
    for j in range(6):                     # weak bounce above the trough
        o = c
        c = o * (1.01 if j == 0 else 0.9995)
        rows.append((o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000.0))
    ctx, s = start_run(rows, "SB1", enable_short_bo=True)
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
    rows = [(100.0, 100.4, 99.7, 100.0, 1000.0)] * 25
    c = 100.0
    for _ in range(5):                     # parabolic run: 5 straight up-closes
        o = c
        c = o * 1.025
        rows.append((o, c * 1.002, o * 0.998, c, 1000.0))
    o = c
    c = o * 0.99                           # first red bar
    rows.append((o, o * 1.001, c * 0.998, c, 1000.0))
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


# ─── ORB: opening range breakout (UTC-day session) ────────────────────────────

def test_orb_arms_then_lapses_with_setup(capsys):
    # start at 18:00 UTC so bar index 30 lands exactly on a day boundary
    rows = [(100.0, 100.3, 99.7, 100.0, 1000.0)] * 30   # flat: universe false
    rows.append((100.0, 101.303, 99.7, 101.0, 1000.0))  # day-2 opening bar
    rows.append((101.0, 102.31603, 100.697, 102.01, 1000.0))  # arm expected here
    rows.append((102.01, 102.061005, 96.61877215, 96.9095, 1000.0))  # trend dies
    ctx, s = start_run(rows, "OR1", start_ts=18 * MS, enable_orb=True)
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


# ─── Volume gate: fill-check-and-bail (pine useBOVol) ─────────────────────────

def test_bo_volume_gate_bails_and_rearms(capsys):
    rows = ramp_rows(50)                       # >=51 bars so avgVol50[1] exists
    rows += flag_rows(rows[-1][3], 3)          # arm on the 3rd flag bar (tick 53)
    level = max(r[1] for r in rows) * BUF
    o = rows[-1][3]
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 1000.0))   # break, NO volume
    o2 = rows[-1][3]
    rows.append((o2, o2 * 1.0005, o2 * 0.998, o2 * 0.999, 1000.0))       # bail fills here
    o3 = rows[-1][3]
    rows.append((o3, level * 1.002, o3 * 0.999, level * 1.001, 5000.0))  # break WITH volume
    sym = "VG1"
    ctx, s = start_run(rows, sym)
    drive(s, ctx, 53)
    e1 = entries(ctx)[0]
    assert e1.status == OrderStatus.Open

    # engine: the low-volume break bar fills the resting stop
    ctx.advance()
    e1.status = OrderStatus.Filled
    ctx.positions[sym] = FakePosition(e1.quantity, e1.price, entry_id=e1.id)
    s.on_tick(ctx)

    bails = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(bails) == 1
    assert bails[0].side == OrderSide.Sell and bails[0].parent == e1.id
    assert bails[0].quantity == pytest.approx(e1.quantity)
    es = entries(ctx)
    assert len(es) == 2                        # re-armed at the same level
    e2 = es[1]
    assert e2.status == OrderStatus.Open
    assert e2.price == pytest.approx(e1.price, rel=1e-12)
    assert "break lacked volume" in capsys.readouterr().out

    # engine: the bail market order fills at the next open -> flat, old
    # bracket OCO-cancelled
    bails[0].status = OrderStatus.Filled
    del ctx.positions[sym]
    for k in children(ctx, e1.id):
        if k.status == OrderStatus.Open:
            k.status = OrderStatus.Cancelled
    drive(s, ctx, 1)
    assert e2.status == OrderStatus.Open       # re-arm survives the exit tick

    # engine: the volume-confirmed re-break fills the re-armed stop
    ctx.advance()
    e2.status = OrderStatus.Filled
    ctx.positions[sym] = FakePosition(e2.quantity, e2.price, entry_id=e2.id)
    s.on_tick(ctx)
    assert len([o for o in ctx.orders
                if o.order_type == "market" and o.reduce_only]) == 1  # no second bail
    assert len(entries(ctx)) == 2              # held: no extra arm either


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
    ctx, s = start_run(rows, "RJ1")
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
