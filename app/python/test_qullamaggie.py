"""Unit tests for the Qullamaggie complete-ruleset strategy (qullamaggie.py).

FakeContext never fills orders on its own; fills are simulated by flipping
the order status and setting ctx.positions, mirroring how the broker reports
them. Engine fill mechanics (stop trigger prices, bracket OCO, settle
rounds) are pinned by the C++ suite under tests/core/."""

import numpy as np
import pytest

import stonks
from stonks import OrderSide, OrderStatus
from stonks.testing import FakeContext, FakeKLine, FakePosition

from qullamaggie import QullamaggieStrategy, Trade, ema_last, segments

MS = 86_400_000             # daily bars
BUF = 1.0 + 10.0 / 10_000.0  # default entry_buffer_bps
FEE = 5.0 / 10_000.0         # default taker_fee_bps as a fraction


def risked(entry_px, stop_px, qty):
    """Dollars lost on a stop-out including both taker fee legs (notes §2)."""
    return qty * (abs(entry_px - stop_px) + entry_px * FEE + stop_px * FEE)


def bars_from(symbol, rows, start_ts=MS):
    return [FakeKLine(start_ts + i * MS, symbol, o, h, l, c, v)
            for i, (o, h, l, c, v) in enumerate(rows)]


def start_run(universe, cash=100_000.0, **overrides):
    """universe: {symbol: rows} merged into one multi-symbol FakeContext."""
    bars = []
    for symbol, rows in universe.items():
        bars += bars_from(symbol, rows)
    bars.sort(key=lambda b: b.timestamp)
    ctx = FakeContext(bars, cash=cash)
    strategy = QullamaggieStrategy()
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


def reduce_markets(ctx):
    return [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]


def fill(ctx, order, symbol, qty=None, price=None):
    """Simulate the broker filling an entry order into a position."""
    order.status = OrderStatus.Filled
    q = order.quantity if qty is None else qty
    p = order.price if price is None else price
    signed = q if order.side == OrderSide.Buy else -q
    ctx.positions[symbol] = FakePosition(signed, p, entry_id=order.id)


# ─── Bar builders ─────────────────────────────────────────────────────────────

def path_rows(steps, start=10.0, spread=0.03, vol=5e6):
    """One bar per close multiplier; every bar opens at the prior close
    (no gaps -> no EP)."""
    rows = []
    c = start
    for m in steps:
        o = c
        c = o * m
        rows.append((o, max(o, c) * (1 + spread), min(o, c) * (1 - spread), c, vol))
    return rows


def leg_flag_rows(quiet=100, leg=25, flag=10, leg_step=1.015, price=10.0,
                  vol=5e6, quiet_spread=0.03, leg_spread=0.03,
                  flag_spread=0.025, flag_tighten=True, flag_drift=1.0):
    """Quiet base -> 30%+ leg into a pivot -> a `flag`-bar consolidation with
    (by default) tightening ranges. The BO geometry first qualifies when the
    flag reaches bo_base_min (10) bars, i.e. on the file's last tick when
    flag=10."""
    rows = path_rows([1.0] * quiet, start=price, spread=quiet_spread, vol=vol)
    c = price
    for _ in range(leg):
        o = c
        c = o * leg_step
        rows.append((o, c * (1 + leg_spread), o * (1 - leg_spread), c, vol))
    for j in range(flag):
        sp = flag_spread * (1.0 - 0.4 * j / max(flag - 1, 1)) if flag_tighten \
            else flag_spread
        o = c
        c = o * flag_drift
        rows.append((o, max(o, c) * (1 + sp), min(o, c) * (1 - sp), c, vol))
    return rows


def gap_rows(quiet_rows, gap=1.12, hi_mult=1.03, lo_mult=0.995, close_mult=1.02,
             vol=5e6):
    """Append an EP gap bar to a prepared quiet history."""
    o = quiet_rows[-1][3] * gap
    return quiet_rows + [(o, o * hi_mult, o * lo_mult, o * close_mult, vol)]


def para_rows(quiet=40, ups=6, up_step=1.08, price=10.0, vol=5e6):
    """Flat base -> `ups` straight up closes with tight bars -> first down
    close. Extend with fades for the cover leg."""
    rows = path_rows([1.0] * quiet, start=price, vol=vol)
    c = price
    for _ in range(ups):
        o = c
        c = o * up_step
        rows.append((o, c, o, c, vol))       # high = close, low = open
    o = c
    c = o * 0.98
    rows.append((o, o, c, c, vol))
    return rows


def adr_of(rows, n=20):
    return float(np.mean([100.0 * (h / l - 1.0) for (_, h, l, _, _) in rows[-n:]]))


# ─── BO: scan gates + arming ──────────────────────────────────────────────────

def test_bo_arms_resting_entry_with_bracket_and_sizing(capsys):
    rows = leg_flag_rows()
    ctx, s = start_run({"BO1": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "stop" and e.side == OrderSide.Buy
    assert e.status == OrderStatus.Open
    assert e.leverage == 1.0

    pivot = max(r[1] for r in rows)
    assert e.price == pytest.approx(pivot * BUF, rel=1e-12)

    kids = children(ctx, e.id)
    assert len(kids) == 1                    # protective stop only, no TP
    sl = kids[0]
    assert sl.order_type == "stop" and sl.side == OrderSide.Sell
    assert sl.reduce_only and sl.quantity == e.quantity

    # initial stop = entry x (1 - ADR%) using the arm bar's ADR20
    adr = adr_of(rows)
    assert sl.price == pytest.approx(min(e.price * (1 - adr / 100.0),
                                         e.price * 0.999), rel=1e-9)
    # a stop-out loses exactly risk_fraction of equity incl. both fee legs
    assert risked(e.price, sl.price, e.quantity) == pytest.approx(
        ctx.equity() * 0.005, rel=1e-9)

    out = capsys.readouterr().out
    assert "BO LONG arm stop-entry" in out
    assert "rank 100" in out and "valid 5 bars" in out


def test_bo_base_geometry_rejects():
    # base shorter than bo_base_min never qualifies
    rows = leg_flag_rows(flag=9)
    ctx, s = start_run({"GM1": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, len(rows))
    assert ctx.orders == []

    # a deep intrabase flush breaches bo_max_retrace
    rows = leg_flag_rows()
    pivot = max(r[1] for r in rows)
    peak = rows[-1][3]
    rows[-5] = (peak, peak * 1.02, pivot * 0.70, peak, 5e6)
    ctx, s = start_run({"GM2": rows}, use_regime=False, scan_every=1,
                       bo_require_tightening=False)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


def test_bo_tightening_gate():
    # constant flag ranges: last-5 mean == first-5 mean -> not tightening
    rows = leg_flag_rows(flag_tighten=False)
    ctx, s = start_run({"TG1": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, len(rows))
    assert ctx.orders == []
    # contracting ranges pass
    rows = leg_flag_rows(flag_tighten=True)
    ctx, s = start_run({"TG2": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) == 1


def test_scan_price_gate():
    rows = leg_flag_rows(price=0.02)         # closes ~0.03 < $1
    ctx, s = start_run({"PG1": rows}, use_regime=False, scan_every=1,
                       min_dollar_volume=0.0)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


def test_scan_dollar_volume_gate():
    rows = leg_flag_rows(vol=1e4)            # ~$150k/day << $10M floor
    ctx, s = start_run({"DV1": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


def test_scan_adr_gate_blocks_and_allows():
    rows = leg_flag_rows(leg_spread=0.008, flag_spread=0.005, flag_tighten=False)
    assert adr_of(rows) < 4.0
    ctx, s = start_run({"AD1": rows}, use_regime=False, scan_every=1,
                       bo_require_tightening=False)
    drive(s, ctx, len(rows))
    assert ctx.orders == []                  # ADR 1.9% < min_adr 4%
    ctx, s = start_run({"AD2": rows}, use_regime=False, scan_every=1,
                       bo_require_tightening=False, min_adr=1.0)
    drive(s, ctx, len(rows))
    assert len(entries(ctx)) == 1            # same bars pass a loosened floor


def test_scan_trend_gate_ema10_vs_ema20():
    rows = leg_flag_rows(flag_drift=0.98, flag_spread=0.02, flag_tighten=False)
    closes = np.array([r[3] for r in rows])
    assert ema_last(closes, 10) <= ema_last(closes, 20)   # premise: trend broken
    ctx, s = start_run({"TR1": rows}, use_regime=False, scan_every=1,
                       bo_require_tightening=False)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


def test_scan_rank_cross_sectional():
    # two 2%/day rockets outrank the flag stock in every window -> blocked
    uni = {"TGT": leg_flag_rows(),
           "H1": path_rows([1.02] * 135), "H2": path_rows([1.02] * 135),
           "F1": path_rows([1.0] * 135), "F2": path_rows([1.0] * 135)}
    ctx, s = start_run(uni, use_regime=False, scan_every=1)
    drive(s, ctx, 135)
    assert ctx.orders == []                  # TGT rank 50 < 90; rockets have no base
    # against flat peers only, TGT ranks 100 -> arms
    uni = {"TGT": leg_flag_rows(),
           "F1": path_rows([1.0] * 135), "F2": path_rows([1.0] * 135),
           "F3": path_rows([1.0] * 135), "F4": path_rows([1.0] * 135)}
    ctx, s = start_run(uni, use_regime=False, scan_every=1)
    drive(s, ctx, 135)
    es = entries(ctx)
    assert len(es) == 1 and es[0].symbol == "TGT"


def test_scan_weekly_cadence():
    # geometry first qualifies on tick 136 (136 % 5 != 0): the arm must wait
    # for the next scan tick, 140
    rows = leg_flag_rows(quiet=101, flag=14)
    assert len(rows) == 140
    ctx, s = start_run({"WK1": rows}, use_regime=False)
    drive(s, ctx, 139)
    assert ctx.orders == []
    drive(s, ctx, 1)
    assert len(entries(ctx)) == 1


# ─── BO: resting-order lifecycle ──────────────────────────────────────────────

def test_bo_ttl_expires_after_setup_lapses(capsys):
    # the flag outgrows bo_base_max on tick 168; the last re-arm was tick 167,
    # so the order dies order_bars (5) ticks later
    rows = leg_flag_rows(flag=48, flag_tighten=False)
    ctx, s = start_run({"TTL": rows}, use_regime=False, scan_every=1,
                       bo_require_tightening=False)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    assert es[0].status == OrderStatus.Cancelled
    assert all(k.status == OrderStatus.Cancelled for k in children(ctx, es[0].id))
    assert "BO order expired unfilled" in capsys.readouterr().out


def test_bo_rejected_entry_prints_and_rearms(capsys):
    rows = leg_flag_rows(flag=12)
    ctx, s = start_run({"RJ1": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, 135)                       # arm on tick 135

    e = entries(ctx)[0]
    e.status = OrderStatus.Rejected          # broker: margin + fee > free cash
    drive(s, ctx, 1)
    assert "entry rejected by broker" in capsys.readouterr().out
    es = entries(ctx)                        # setup still holds -> fresh arm
    assert len(es) == 2 and es[1].status == OrderStatus.Open


def test_bo_fill_and_collapse_round_trip(capsys):
    rows = leg_flag_rows(flag=11)
    ctx, s = start_run({"RT1": rows}, use_regime=False, scan_every=1)
    drive(s, ctx, 135)

    e = entries(ctx)[0]
    ctx.advance()
    e.status = OrderStatus.Filled            # filled AND stopped intrabar: no position
    s.on_tick(ctx)
    assert "filled and stopped out within the same bar" in capsys.readouterr().out
    assert s.trades == {}
    es = entries(ctx)                        # setup still holds -> re-arm
    assert len(es) == 2 and es[1].status == OrderStatus.Open


# ─── Management: day-low stop, partial, breakeven, trail ──────────────────────

def lifecycle_rows(extra):
    """leg_flag base (arms on tick 135) plus `extra` management bars:
    fill bar, down, down, up (partial), crash (trail), cleanup."""
    rows = leg_flag_rows()
    peak = rows[-1][3]
    post = [
        (peak, peak * 1.09, peak * 0.999, peak * 1.05, 5e6),
        (peak * 1.05, peak * 1.055, peak * 1.035, peak * 1.041, 5e6),
        (peak * 1.041, peak * 1.045, peak * 1.030, peak * 1.034, 5e6),
        (peak * 1.034, peak * 1.048, peak * 1.032, peak * 1.042, 5e6),
        (peak * 1.042, peak * 1.043, peak * 0.888, peak * 0.896, 5e6),
        (peak * 0.896, peak * 0.90, peak * 0.89, peak * 0.896, 5e6),
    ]
    return rows + post[:extra], peak


def armed_and_filled(extra, symbol="LC1"):
    rows, peak = lifecycle_rows(extra)
    ctx, s = start_run({symbol: rows}, use_regime=False, scan_every=1)
    drive(s, ctx, 135)
    e = entries(ctx)[0]
    sl0 = children(ctx, e.id)[0]
    ctx.advance()                            # tick 136: the breakout/fill bar
    fill(ctx, e, symbol)
    s.on_tick(ctx)
    return ctx, s, e, sl0, peak


def test_bo_fill_tightens_stop_to_day_low():
    ctx, s, e, sl0, peak = armed_and_filled(extra=1)

    assert sl0.status == OrderStatus.Cancelled
    live = [o for o in ctx.orders if o.order_type == "stop" and o.reduce_only
            and o.status == OrderStatus.Open]
    assert len(live) == 1
    sl1 = live[0]
    assert sl1.parent == e.id and sl1.side == OrderSide.Sell
    assert sl1.price == pytest.approx(peak * 0.999, rel=1e-12)   # fill bar's low
    assert sl1.quantity == pytest.approx(e.quantity)
    assert sl1.price > sl0.price             # tightened, never loosened


def test_partial_on_first_up_close_then_breakeven():
    ctx, s, e, _, peak = armed_and_filled(extra=4)
    drive(s, ctx, 2)                         # two down closes: no partial yet
    assert reduce_markets(ctx) == []
    drive(s, ctx, 1)                         # first up close on bar 3 after fill

    parts = reduce_markets(ctx)
    assert len(parts) == 1
    assert parts[0].side == OrderSide.Sell and parts[0].parent == e.id
    assert parts[0].quantity == pytest.approx(e.quantity / 3.0)

    live = [o for o in ctx.orders if o.order_type == "stop" and o.reduce_only
            and o.status == OrderStatus.Open]
    assert len(live) == 1                    # day-low stop replaced by breakeven
    be = live[0]
    assert be.price == pytest.approx(e.price, rel=1e-12)
    assert be.quantity == pytest.approx(e.quantity * 2.0 / 3.0)
    assert be.parent == e.id


def test_partial_forced_at_max_bars():
    rows = leg_flag_rows()
    peak = rows[-1][3]
    closes = [1.08, 1.075, 1.071, 1.067, 1.063, 1.059]
    o = peak
    post = []
    for c in closes:
        post.append((o, max(o, peak * c) * 1.002, min(o, peak * c) * 0.998,
                     peak * c, 5e6))
        o = peak * c
    sym = "FP1"
    ctx, s = start_run({sym: rows + post}, use_regime=False, scan_every=1)
    drive(s, ctx, 135)
    e = entries(ctx)[0]
    ctx.advance()
    fill(ctx, e, sym)
    s.on_tick(ctx)

    drive(s, ctx, 4)                         # four straight down closes
    assert reduce_markets(ctx) == []
    drive(s, ctx, 1)                         # bar 5 after the fill: forced
    parts = reduce_markets(ctx)
    assert len(parts) == 1
    assert parts[0].quantity == pytest.approx(e.quantity / 3.0)


def test_trail_exit_full_remainder_and_cleanup(capsys):
    ctx, s, e, _, peak = armed_and_filled(extra=6, symbol="TE1")
    drive(s, ctx, 3)                         # through the partial on the up close
    partial = reduce_markets(ctx)[0]
    partial.status = OrderStatus.Filled
    ctx.positions["TE1"] = FakePosition(e.quantity * 2.0 / 3.0, e.price,
                                        entry_id=e.id)
    drive(s, ctx, 1)                         # crash bar closes below the 10-EMA

    exits = [o for o in reduce_markets(ctx) if o is not partial]
    assert len(exits) == 1
    assert exits[0].quantity == pytest.approx(e.quantity * 2.0 / 3.0)
    assert exits[0].parent == e.id
    assert "trail exit" in capsys.readouterr().out

    exits[0].status = OrderStatus.Filled     # broker: exit fills, flat
    del ctx.positions["TE1"]
    drive(s, ctx, 1)
    assert "TE1" not in s.trades
    be = [o for o in ctx.orders if o.order_type == "stop" and o.reduce_only
          and o.price == pytest.approx(e.price, rel=1e-12)]
    assert all(o.status == OrderStatus.Cancelled for o in be)
    assert "position closed" in capsys.readouterr().out


# ─── EP: episodic pivot ───────────────────────────────────────────────────────

def test_ep_market_entry_stop_at_gap_low():
    rows = gap_rows(path_rows([1.0] * 60, spread=0.02, vol=1e6))
    ctx, s = start_run({"EP1": rows}, use_regime=False, ep_neglect_lookback=50)
    drive(s, ctx, len(rows))

    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "market" and e.side == OrderSide.Buy
    assert e.leverage == 1.0

    o, hi, lo, c, v = rows[-1]
    kids = children(ctx, e.id)
    assert len(kids) == 1
    sl = kids[0]
    assert sl.order_type == "stop" and sl.reduce_only
    assert sl.price == pytest.approx(lo, rel=1e-12)      # gap-day low
    assert e.quantity == pytest.approx(
        ctx.equity() * 0.005 / ((c - lo) + c * FEE + lo * FEE), rel=1e-9)


def test_ep_skip_rules():
    quiet = path_rows([1.0] * 60, spread=0.02, vol=1e6)
    # (a) stop wider than 1 x ADR: low of the gap bar is too far away
    rows = gap_rows(quiet, lo_mult=0.94)
    ctx, s = start_run({"EPa": rows}, use_regime=False, ep_neglect_lookback=50,
                       enable_breakout=False)
    drive(s, ctx, len(rows))
    assert ctx.orders == []
    # (b) not neglected: it already rallied 60%+ into the gap
    rows = gap_rows(path_rows([1.01] * 60, spread=0.02, vol=1e6))
    ctx, s = start_run({"EPb": rows}, use_regime=False, ep_neglect_lookback=50,
                       enable_breakout=False)
    drive(s, ctx, len(rows))
    assert ctx.orders == []
    # (c) weak close: red gap bar
    rows = gap_rows(quiet, lo_mult=0.985, close_mult=0.99)
    ctx, s = start_run({"EPc": rows}, use_regime=False, ep_neglect_lookback=50,
                       enable_breakout=False)
    drive(s, ctx, len(rows))
    assert ctx.orders == []


# ─── PARA: parabolic short ────────────────────────────────────────────────────

def test_para_short_entry_stop_and_cover():
    base = para_rows()
    fades = path_rows([0.97] * 4, start=base[-1][3], spread=0.0, vol=5e6)
    rows = base + [(r[0], r[0], r[3], r[3], r[4]) for r in fades]
    sym = "PS1"

    # off by default: the same bars produce nothing
    ctx, s = start_run({sym: rows}, use_regime=False)
    drive(s, ctx, len(base))
    assert ctx.orders == []

    ctx, s = start_run({sym: rows}, use_regime=False, enable_para=True)
    drive(s, ctx, len(base))
    es = entries(ctx)
    assert len(es) == 1
    e = es[0]
    assert e.order_type == "market" and e.side == OrderSide.Sell

    stop_ref = max(r[1] for r in base[-3:])
    close = base[-1][3]
    kids = children(ctx, e.id)
    assert len(kids) == 1
    sl = kids[0]
    assert sl.order_type == "stop" and sl.side == OrderSide.Buy and sl.reduce_only
    assert sl.price == pytest.approx(stop_ref, rel=1e-12)
    assert e.quantity == pytest.approx(
        ctx.equity() * 0.005 / ((sl.price - close) + close * FEE + sl.price * FEE),
        rel=1e-9)

    ctx.advance()                            # short fills at the first fade's open
    fill(ctx, e, sym, price=close)
    s.on_tick(ctx)
    drive(s, ctx, 3)                         # fades sink the close to the SMA10
    covers = [o for o in reduce_markets(ctx) if o.side == OrderSide.Buy]
    assert len(covers) == 1
    assert covers[0].quantity == pytest.approx(e.quantity)
    assert covers[0].parent == e.id


# ─── Market-regime filter ─────────────────────────────────────────────────────

def test_regime_blocks_new_entries():
    # a falling market symbol drags the synthetic index down: no new entries
    uni = {"TGT": leg_flag_rows(), "MKT": path_rows([0.99] * 135)}
    ctx, s = start_run(uni, scan_every=1, rank_pct=0.0)
    drive(s, ctx, 135)
    assert ctx.orders == []
    assert not s.regime_on
    # a rising market symbol keeps the index trending: the arm goes through
    uni = {"TGT": leg_flag_rows(), "MKT": path_rows([1.01] * 135)}
    ctx, s = start_run(uni, scan_every=1, rank_pct=0.0)
    drive(s, ctx, 135)
    assert len(entries(ctx)) == 1
    assert s.regime_on


def test_regime_flip_cancels_resting_entry(capsys):
    uni = {"TGT": leg_flag_rows(flag=30),
           "MKT": path_rows([1.01] * 140 + [0.98] * 15)}
    ctx, s = start_run(uni, scan_every=1, rank_pct=0.0)
    drive(s, ctx, 155)

    es = entries(ctx)
    assert len(es) == 1                      # armed once while the regime was on
    assert es[0].status == OrderStatus.Cancelled
    out = capsys.readouterr().out
    assert "regime OFF" in out and "cancelled (regime off)" in out


def test_management_runs_while_regime_off():
    # regime is OFF (warmup + falling market) yet the held trade still gets
    # its partial and breakeven move
    tgt = path_rows([1.0, 1.0, 1.01, 1.01, 1.01, 1.01], spread=0.01)
    uni = {"TGT": tgt, "MKT": path_rows([0.99] * 6)}
    ctx, s = start_run(uni, scan_every=1)
    s.trades["TGT"] = Trade("BO", "long", entry_id=901, stop_id=902,
                            fill_px=10.0, stop_px=9.4, adr_pct=5.0)
    ctx.positions["TGT"] = FakePosition(90.0, 10.0, entry_id=901)
    drive(s, ctx, 6)

    assert not s.regime_on
    assert entries(ctx) == []                # no new entries while off
    parts = reduce_markets(ctx)
    assert len(parts) == 1 and parts[0].quantity == pytest.approx(30.0)
    be = [o for o in ctx.orders if o.order_type == "stop"]
    assert len(be) == 1
    assert be[0].price == pytest.approx(10.0) and be[0].quantity == pytest.approx(60.0)


# ─── Sizing / capacity ────────────────────────────────────────────────────────

def test_sizing_clamps():
    ctx = FakeContext([], cash=100_000.0)
    s = QullamaggieStrategy()
    s.on_start(ctx)
    # tight stop: the 25%-of-equity notional cap binds before the risk budget
    assert s._size(ctx, 10.0, 9.9) == pytest.approx(0.25 * 100_000.0 / 10.0)
    # wide stop: the risk fraction binds
    rpu = 1.0 + 10.0 * FEE + 9.0 * FEE
    assert s._size(ctx, 10.0, 9.0) == pytest.approx(100_000.0 * 0.005 / rpu)
    # degenerate input sizes to zero
    assert s._size(ctx, 0.0, -1.0) == 0.0

    s2 = QullamaggieStrategy()
    s2.risk_fraction = 5.0                   # force the free-cash clamp to bind
    s2.max_position_pct = 10.0
    ctx2 = FakeContext([], cash=1_000.0)
    s2.on_start(ctx2)
    assert s2._size(ctx2, 10.0, 9.0) == pytest.approx(
        0.99 * 1_000.0 / (10.0 * (1.0 + FEE)))


def test_max_positions_cap():
    uni = {"AAA": leg_flag_rows(leg_step=1.017),
           "BBB": leg_flag_rows(leg_step=1.012)}
    ctx, s = start_run(uni, use_regime=False, scan_every=1, rank_pct=0.0,
                       max_positions=1)
    drive(s, ctx, 135)
    es = entries(ctx)
    assert len(es) == 1 and es[0].symbol == "AAA"    # better rank arms first

    ctx, s = start_run(uni, use_regime=False, scan_every=1, rank_pct=0.0)
    drive(s, ctx, 135)
    assert {o.symbol for o in entries(ctx)} == {"AAA", "BBB"}


# ─── Helpers / discovery metadata ─────────────────────────────────────────────

def test_segments_and_ema_helpers():
    ts = np.array([1, 2, 3, 2, 3, 3], dtype=np.int64)   # A:1,2,3  B:2,3  C:3
    starts, ends = segments(ts)
    assert starts.tolist() == [0, 3, 5] and ends.tolist() == [2, 4, 5]

    flat = np.full(120, 5.0)
    assert ema_last(flat, 10) == pytest.approx(5.0)
    rising = 100.0 * np.cumprod(np.full(120, 1.01))
    assert ema_last(rising, 10) > ema_last(rising, 20)
    assert ema_last(np.arange(5, dtype=float), 10) is None


def test_param_and_indicator_specs_resolve():
    specs = stonks.param_specs(QullamaggieStrategy)
    assert len(specs) == len(QullamaggieStrategy.params)
    names = {p["name"] for p in specs}
    assert {"enable_breakout", "enable_ep", "enable_para", "use_regime",
            "risk_fraction", "max_positions", "max_position_pct",
            "partial_fraction", "partial_min_bars", "partial_max_bars",
            "trail_len", "trail_use_ema", "rank_pct",
            "min_dollar_volume"} <= names
    inds = stonks.indicator_specs(QullamaggieStrategy)
    assert {i["name"] for i in inds} == {"order_level", "stop_level", "trail_ma"}
