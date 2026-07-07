"""Behavior tests, one per QM-family strategy: each crafts a minimal series
that proves the strategy's signature setup/management fires with the right
order shape (side / type / price / parent / reduce_only / quantity)."""

import pytest

from stonks import OrderSide, OrderStatus
from conftest import (
    MS4H,
    bars_from,
    children_of,
    entry_orders,
    fill_entry,
    fill_exit,
    run_all,
    start_run,
    tick,
)


def uptrend_rows(n, start=100.0, step=0.003, vol=1000.0):
    rows = []
    px = start
    for _ in range(n):
        o = px
        c = px * (1.0 + step)
        rows.append((o, c * 1.005, o * 0.995, c, vol))
        px = c
    return rows


def downtrend_rows(n, start=100.0, step=0.003, vol=1000.0):
    rows = []
    px = start
    for _ in range(n):
        o = px
        c = px * (1.0 - step)
        rows.append((o, o * 1.005, c * 0.995, c, vol))
        px = c
    return rows


def flat_rows(n, px, vol=1000.0, spread=0.004):
    return [(px, px * (1.0 + spread), px * (1.0 - spread), px, vol) for _ in range(n)]


# ─── 1. qmliteral — arm at the pivot, bracket shape, time partial + BE ────────

def test_qmliteral_arms_pivot_bracket_then_time_partial_moves_stop_to_breakeven():
    from qmliteral import QMLiteralStrategy

    rows = uptrend_rows(70)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))          # peak bar: pivot high 130
    rows.append((128.0, 128.5, 123.0, 126.0, 900.0))                 # pullback 1
    rows.append((126.0, 127.0, 122.8, 125.5, 900.0))                 # pullback 2
    rows.append((125.3, 126.0, 122.5, 125.0, 900.0))                 # pullback 3 -> arms here
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))                # break bar, volume expands
    rows += flat_rows(9, 131.0)                                      # quiet hold above the trail

    strategy = QMLiteralStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)                                          # through pullback 3

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "stop"
    assert entry.side == OrderSide.Buy
    assert entry.price == pytest.approx(130.0 * (1.0 + 5.0 / 10_000.0))

    kids = children_of(ctx, entry.id)
    assert len(kids) == 2
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.side == OrderSide.Sell and sl.reduce_only
    assert tp.side == OrderSide.Sell and tp.reduce_only
    assert sl.quantity == pytest.approx(entry.quantity)              # SL stays full-size
    assert tp.quantity == pytest.approx(entry.quantity / 2.0)        # half partial at 2R
    assert sl.price < entry.price < tp.price
    assert tp.price == pytest.approx(entry.price + 2.0 * (entry.price - sl.price))

    # Broker fills the resting stop at its level; the trade then ages 6 bars
    # without reaching 2R, so the time partial fires and the stop moves to BE.
    fill_entry(ctx, entry)
    tick(ctx, strategy, 10)

    partials = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(partials) == 1
    assert partials[0].side == OrderSide.Sell
    assert partials[0].quantity == pytest.approx(entry.quantity / 2.0)
    assert ctx.order(tp.id).status == OrderStatus.Cancelled          # replaced by the time partial
    assert ctx.order(sl.id).status == OrderStatus.Cancelled          # replaced by the BE stop

    be_stops = [o for o in ctx.orders
                if o.order_type == "stop" and o.reduce_only and o.id != sl.id]
    assert len(be_stops) == 1
    assert be_stops[0].price == pytest.approx(entry.price)           # breakeven at the fill
    assert be_stops[0].quantity == pytest.approx(entry.quantity)


# ─── 4. qmepisodic — gap + volume + strong close fires a market bracket ───────

def test_qmepisodic_gap_bar_fires_market_entry_with_lod_stop_and_2r_target():
    from qmepisodic import QMEpisodicStrategy

    rows = flat_rows(65, 100.0, spread=0.006)
    rows.append((101.0, 102.5, 101.2, 102.0, 2000.0))   # 1% gap, 2x volume, strong close

    strategy = QMEpisodicStrategy()
    ctx = run_all(strategy, bars_from("TST", rows))

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "market"                  # the pine buys the gap-bar close
    assert entry.side == OrderSide.Buy

    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.price == pytest.approx(101.2)              # signal-bar low beats the ADR stop
    assert tp.price == pytest.approx(102.0 + 2.0 * (102.0 - 101.2))
    assert sl.quantity == pytest.approx(entry.quantity)
    assert tp.quantity == pytest.approx(entry.quantity / 2.0)
    assert sl.reduce_only and tp.reduce_only
    assert sl.parent == entry.id and tp.parent == entry.id


# ─── 3. qmbreakoutpure — dry base, full 3R bracket, no management ever ────────

def test_qmbreakoutpure_places_full_size_3r_bracket_and_never_manages():
    from qmbreakoutpure import QMBreakoutPureStrategy

    rows = uptrend_rows(70)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))          # pivot high 130
    rows.append((128.0, 128.5, 123.0, 126.0, 500.0))                 # dry pullback 1
    rows.append((126.0, 127.0, 122.8, 125.5, 500.0))                 # dry pullback 2
    rows.append((125.3, 126.0, 122.5, 125.0, 500.0))                 # dry pullback 3 -> arms
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))                # break bar, volume expands
    rows += flat_rows(7, 131.0)

    strategy = QMBreakoutPureStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "stop"
    assert entry.price == pytest.approx(130.0 * (1.0 + 5.0 / 10_000.0))
    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.quantity == pytest.approx(entry.quantity)
    assert tp.quantity == pytest.approx(entry.quantity)              # FULL take-profit
    assert tp.price == pytest.approx(entry.price + 3.0 * (entry.price - sl.price))

    # After the fill nothing else ever happens: the bracket is the trade.
    fill_entry(ctx, entry)
    tick(ctx, strategy, 8)
    assert len(ctx.orders) == 3                                      # entry + SL + TP, nothing more
    assert ctx.order(sl.id).status == OrderStatus.Open               # never moved to breakeven
    assert ctx.order(tp.id).status == OrderStatus.Open
    assert not any(o.order_type == "market" for o in ctx.orders)     # no partials, no trail exits


# ─── 2. qmcloseconfirm — intrabar poke ignored; close-through fires market ────

def test_qmcloseconfirm_ignores_wick_poke_and_enters_market_on_close_through():
    from qmcloseconfirm import QMCloseConfirmStrategy

    rows = uptrend_rows(70)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))          # pivot high 130
    rows.append((128.0, 128.5, 123.0, 126.0, 900.0))
    rows.append((126.0, 127.0, 122.8, 125.5, 900.0))
    rows.append((125.3, 126.0, 122.5, 125.0, 900.0))
    # Wick poke: high crosses the 130.065 level (a resting stop WOULD fill),
    # but the close stays under it -> close-confirm does nothing.
    rows.append((125.0, 130.5, 124.0, 128.0, 2000.0))
    rows.append((128.0, 129.5, 126.0, 128.0, 900.0))                 # digestion under the poke high
    rows.append((128.0, 129.0, 126.5, 128.2, 900.0))
    rows.append((128.2, 129.0, 126.8, 128.0, 900.0))
    poke_pivot = 130.5 * (1.0 + 5.0 / 10_000.0)
    rows.append((128.0, 131.9, 127.8, 131.5, 2000.0))                # CLOSE clears the (new) pivot
    rows.append((131.0, 132.0, 130.8, 131.8, 1000.0))                # fill bar
    rows.append((130.0, 130.0, 119.5, 120.0, 1000.0))                # crash through the stop level

    strategy = QMCloseConfirmStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 75)                                          # through the wick-poke bar
    assert entry_orders(ctx) == []                                   # the poke triggered nothing

    tick(ctx, strategy, 4)                                           # digestion + close-through
    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "market"
    assert entry.side == OrderSide.Buy
    assert 131.5 >= poke_pivot                                       # scenario sanity
    assert not any(o.order_type in ("stop", "limit") for o in ctx.orders)   # nothing ever rests

    # Close-driven stop: fill, then a bar CLOSING under the internal stop
    # level produces a reduce-only market flatten at the next open.
    fill_entry(ctx, entry, price=131.0)
    tick(ctx, strategy, 2)
    exits = [o for o in ctx.orders if o.reduce_only]
    assert len(exits) == 1
    assert exits[0].order_type == "market"
    assert exits[0].side == OrderSide.Sell
    assert exits[0].quantity == pytest.approx(entry.quantity)
    assert not any(o.order_type in ("stop", "limit") for o in ctx.orders)


# ─── 6. qmshortbo — downtrend base arms a sell-stop breakdown bracket ─────────

def test_qmshortbo_arms_sell_stop_under_base_low_with_mirrored_bracket():
    from qmshortbo import QMShortBOStrategy

    rows = downtrend_rows(70)
    last = rows[-1][3]
    rows.append((last, last * 1.003, 75.0, 76.0, 1500.0))            # trough: base low 75
    rows.append((76.0, 77.5, 75.3, 77.0, 900.0))                     # bounce 1
    rows.append((77.0, 78.0, 75.5, 76.8, 900.0))                     # bounce 2
    rows.append((76.8, 77.8, 75.2, 76.5, 900.0))                     # bounce 3 -> arms

    strategy = QMShortBOStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "stop"
    assert entry.side == OrderSide.Sell
    assert entry.price == pytest.approx(75.0 * (1.0 - 5.0 / 10_000.0))

    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.side == OrderSide.Buy and tp.side == OrderSide.Buy
    assert sl.reduce_only and tp.reduce_only
    assert sl.quantity == pytest.approx(entry.quantity)
    assert tp.quantity == pytest.approx(entry.quantity / 2.0)
    assert tp.price < entry.price < sl.price                         # mirrored geometry
    assert tp.price == pytest.approx(entry.price - 2.0 * (sl.price - entry.price))


def qm_base_rows(pull_vol=900.0):
    """The shared QM tight-flag scenario: 70-bar uptrend, pivot high 130,
    three digestion bars -> a qualified base armed on the last row."""
    rows = uptrend_rows(70)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))
    rows.append((128.0, 128.5, 123.0, 126.0, pull_vol))
    rows.append((126.0, 127.0, 122.8, 125.5, pull_vol))
    rows.append((125.3, 126.0, 122.5, 125.0, pull_vol))
    return rows


# ─── 9. qmatr — chandelier stop ratchets up, and is the only exit ─────────────

def test_qmatr_chandelier_replaces_stop_monotonically_upward():
    from qmatr import QMATRStrategy

    rows = qm_base_rows()
    rows.append((125.0, 132.0, 125.0, 131.5, 1000.0))    # rising highs after the fill
    rows.append((131.5, 136.0, 131.0, 135.5, 1000.0))
    rows.append((135.5, 139.0, 135.0, 138.5, 1000.0))

    strategy = QMATRStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    kids = children_of(ctx, entry.id)
    assert len(kids) == 1                                # a stop-loss only: no take-profit
    assert kids[0].order_type == "stop"
    assert 0.0 < kids[0].price < entry.price
    assert not any(o.order_type == "limit" for o in ctx.orders)

    fill_entry(ctx, entry)
    tick(ctx, strategy, 3)                               # three rising-high bars

    stops = [o for o in ctx.orders if o.order_type == "stop" and o.reduce_only]
    assert len(stops) == 4                               # initial + one ratchet per rising bar
    prices = [o.price for o in sorted(stops, key=lambda o: o.id)]
    assert all(b > a for a, b in zip(prices, prices[1:]))   # only ever upward
    assert all(o.status == OrderStatus.Cancelled for o in stops[:-1] for o in [ctx.order(o.id)])
    assert ctx.order(sorted(stops, key=lambda o: o.id)[-1].id).status == OrderStatus.Open
    assert all(o.quantity == pytest.approx(entry.quantity) for o in stops)


# ─── 10. qmrunner — no partial ever; EMA10 close-break flattens in full ───────

def test_qmrunner_full_exit_on_ema10_close_break_and_untouched_hard_stop():
    from qmrunner import QMRunnerStrategy

    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))    # break bar, volume expands
    rows.append((131.0, 131.5, 130.5, 131.0, 1000.0))    # holds above the EMA10
    rows.append((131.0, 131.4, 130.6, 131.1, 1000.0))
    rows.append((131.0, 131.2, 121.5, 122.0, 1000.0))    # closes under the EMA10

    strategy = QMRunnerStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert len(children_of(ctx, entry.id)) == 1          # hard stop only, no TP

    fill_entry(ctx, entry)
    tick(ctx, strategy, 4)

    exits = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(exits) == 1                               # one full flatten, never a partial
    assert exits[0].quantity == pytest.approx(entry.quantity)
    stops = [o for o in ctx.orders if o.order_type == "stop" and o.reduce_only]
    assert len(stops) == 1                               # the hard stop was never re-placed
    assert not any(o.order_type == "limit" for o in ctx.orders)


# ─── 11. qmthirds — two thirds rest as TPs; BE stop stays FULL size ───────────

def test_qmthirds_tp1_fill_moves_stop_to_breakeven_still_full_size():
    from qmthirds import QMThirdsStrategy

    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))    # break bar, volume expands
    rows += flat_rows(2, 131.0)

    strategy = QMThirdsStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tps = sorted((o for o in kids if o.order_type == "limit"), key=lambda o: o.price)
    assert len(tps) == 2
    risk = entry.price - sl.price
    assert sl.quantity == pytest.approx(entry.quantity)
    assert tps[0].price == pytest.approx(entry.price + 2.0 * risk)
    assert tps[1].price == pytest.approx(entry.price + 4.0 * risk)
    assert tps[0].quantity == pytest.approx(entry.quantity / 3.0)
    assert tps[1].quantity == pytest.approx(entry.quantity / 3.0)

    fill_entry(ctx, entry)
    tick(ctx, strategy, 1)                               # fill bar (volume passes the scratch check)
    fill_exit(ctx, tps[0])                               # first third banks at 2R
    tick(ctx, strategy, 1)

    assert ctx.order(sl.id).status == OrderStatus.Cancelled
    be_stops = [o for o in ctx.orders
                if o.order_type == "stop" and o.reduce_only and o.id != sl.id]
    assert len(be_stops) == 1
    assert be_stops[0].price == pytest.approx(entry.price)          # breakeven
    assert be_stops[0].quantity == pytest.approx(entry.quantity)    # STILL full size
    assert ctx.order(tps[1].id).status == OrderStatus.Open          # 4R third still working


# ─── 12. qmpullback — buy the 10-SMA reclaim inside the base ─────────────────

def test_qmpullback_buys_reclaim_with_dip_low_stop_and_pivot_target():
    from qmpullback import QMPullbackStrategy

    rows = uptrend_rows(70)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))          # pivot high 130
    rows.append((128.0, 128.3, 120.9, 121.5, 900.0))                 # dip under the 10-SMA
    rows.append((121.5, 122.0, 120.5, 121.0, 900.0))
    rows.append((121.0, 121.5, 120.2, 120.8, 900.0))                 # dip low 120.2
    rows.append((120.8, 125.5, 120.7, 125.0, 1000.0))                # closes back above -> buy

    strategy = QMPullbackStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 75)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "market"                              # reclaim buys the next open
    assert entry.side == OrderSide.Buy

    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.price == pytest.approx(120.2)                          # the dip low
    assert tp.price == pytest.approx(130.0)                          # the pivot is the first sell
    assert sl.quantity == pytest.approx(entry.quantity)
    assert tp.quantity == pytest.approx(entry.quantity / 2.0)


# ─── 5. qmorb — session-scoped resting entry, flatten at the rollover ─────────

def test_qmorb_arms_over_opening_range_and_flattens_at_day_rollover():
    from qmorb import QMORBStrategy

    rows = uptrend_rows(65)                                          # 4h bars: 6 per UTC day
    rows.append((121.5, 125.0, 121.4, 124.0, 1500.0))                # i=65: first bar of a new day
    rows.append((124.0, 124.5, 123.6, 124.2, 1000.0))                # i=66: range complete -> arm
    rows += [(124.0, 124.5, 123.5, 124.0, 1000.0)] * 4               # rest of the session
    rows.append((124.0, 124.3, 123.8, 124.0, 1000.0))                # i=71: next day -> flatten

    strategy = QMORBStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 67)

    entries = sorted(entry_orders(ctx), key=lambda o: o.id)
    assert entries                                                   # earlier sessions may have armed too
    live = entries[-1]
    assert live.order_type == "stop"
    assert live.price == pytest.approx(125.0 * (1.0 + 5.0 / 10_000.0))   # today's OR high + buffer
    assert all(o.status == OrderStatus.Cancelled for o in entries[:-1])  # session-scoped TTL
    assert len(children_of(ctx, live.id)) == 1                       # stop-loss only, no TP

    fill_entry(ctx, live)
    tick(ctx, strategy, 5)                                           # through the day rollover

    exits = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(exits) == 1                                           # the session-end flatten
    assert exits[0].side == OrderSide.Sell
    assert exits[0].quantity == pytest.approx(live.quantity)


# ─── 7. qmparabolic — first red bar shorts; first up-close covers ─────────────

def test_qmparabolic_shorts_first_red_bar_and_covers_on_up_close():
    from qmparabolic import QMParabolicStrategy

    rows = flat_rows(55, 100.0)
    rows.append((100.0, 103.2, 99.9, 103.0, 1200.0))                 # parabolic leg
    rows.append((103.0, 106.3, 102.9, 106.1, 1200.0))
    rows.append((106.1, 109.5, 106.0, 109.3, 1200.0))
    rows.append((109.3, 112.85, 109.2, 112.6, 1200.0))
    rows.append((112.6, 112.8, 110.5, 111.0, 1300.0))                # first red bar -> short
    rows.append((111.0, 112.5, 110.8, 112.0, 1000.0))                # up-close (detection bar)
    rows.append((112.0, 113.5, 111.8, 113.0, 1000.0))                # up-close -> cover fires

    strategy = QMParabolicStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 60)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "market"                              # pine shorts the close
    assert entry.side == OrderSide.Sell
    kids = children_of(ctx, entry.id)
    assert len(kids) == 1                                            # a stop above, NO take-profit
    assert kids[0].order_type == "stop"
    assert kids[0].side == OrderSide.Buy
    assert kids[0].price == pytest.approx(112.85 * (1.0 + 5.0 / 10_000.0))
    assert not any(o.order_type == "limit" for o in ctx.orders)

    fill_entry(ctx, entry, price=111.0)
    tick(ctx, strategy, 2)                                           # up-close bar covers

    covers = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(covers) == 1
    assert covers[0].side == OrderSide.Buy
    assert covers[0].quantity == pytest.approx(entry.quantity)


# ─── 8. qmfullsuite — a bar valid for BOTH breakout and EP takes the BO ───────

def test_qmfullsuite_priority_prefers_breakout_arm_over_episodic_pivot():
    from qmfullsuite import QMFullSuiteStrategy
    from qmepisodic import QMEpisodicStrategy

    rows = qm_base_rows()
    # A textbook EP bar printed while the breakout base is still armed:
    # 1% gap, 2x volume, strong close — and the 130 pivot still intact above.
    rows.append((126.3, 127.8, 126.4, 127.5, 2000.0))
    bars = bars_from("TST", rows)

    # Sanity: the EP gate alone DOES fire on this exact series.
    ep_ctx = run_all(QMEpisodicStrategy(), bars_from("TST", rows))
    assert any(o.order_type == "market" for o in entry_orders(ep_ctx))

    strategy = QMFullSuiteStrategy()
    ctx = run_all(strategy, bars)
    # With all five setups on, earlier bars legitimately arm ORB attempts and
    # even a parabolic short (the first red bar after the 130 spike) — each
    # replaced by the next, fresher signal. The one WORKING entry at the end
    # must be the breakout arm; the EP (a market BUY) never took the slot.
    entries = sorted(entry_orders(ctx), key=lambda o: o.id)
    live = [o for o in entries if o.status == OrderStatus.Open]
    assert len(live) == 1
    assert live[0].order_type == "stop"                              # the BO arm owns the slot
    assert live[0].side == OrderSide.Buy
    assert live[0].price == pytest.approx(130.0 * (1.0 + 5.0 / 10_000.0))
    assert all(o.status == OrderStatus.Cancelled for o in entries if o.id != live[0].id)
    assert not any(o.order_type == "market" and o.side == OrderSide.Buy
                   for o in entry_orders(ctx))                       # EP never fired
