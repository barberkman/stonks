"""Behavior tests, one per Darvas-family strategy: each walks the box state
machine through a crafted sequence and asserts the trade orders' shape."""

import pytest

from stonks import OrderSide, OrderStatus
from conftest import (
    bars_from,
    children_of,
    entry_orders,
    fill_entry,
    fill_exit,
    run_all,
    start_run,
    tick,
)


def box_walk_rows():
    """5 flat bars, a peak (top 110), three lower highs with a higher low on
    the third -> box [100, 110] completes on the 9th bar; a 10th inside bar
    keeps it active. rows: (open, high, low, close, volume)."""
    rows = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 5
    rows.append((100.0, 110.0, 100.0, 105.0, 1000.0))   # peak: top candidate 110
    rows.append((105.0, 108.0, 101.0, 104.0, 1000.0))   # lower high 1 -> state 2
    rows.append((104.0, 107.0, 102.0, 104.0, 1000.0))   # lower high 2 -> state 3
    rows.append((104.0, 106.0, 103.0, 104.0, 1000.0))   # higher low   -> state 5 (box)
    rows.append((104.0, 107.0, 104.0, 105.0, 1000.0))   # inside bar: box stays active
    return rows


def second_box_rows():
    """A break bar out of the first box, then a fresh, higher box [108, 120]
    that completes on the last row (for ratchet assertions)."""
    return [
        (105.0, 112.0, 108.0, 111.0, 1000.0),   # breaks the [100,110] box upward
        (111.0, 120.0, 110.0, 118.0, 1000.0),   # new top candidate 120
        (118.0, 118.0, 111.0, 114.0, 1000.0),   # lower high 1
        (114.0, 117.0, 112.0, 114.0, 1000.0),   # lower high 2
        (114.0, 116.0, 113.0, 114.0, 1000.0),   # higher low -> box [108, 120]
    ]


# ─── 13. darvasclassic — box arms a stop bracket; later box ratchets the SL ───

def test_darvasclassic_arms_at_box_top_and_ratchets_stop_under_next_box():
    from darvasclassic import DarvasClassicStrategy

    rows = box_walk_rows() + second_box_rows()
    strategy = DarvasClassicStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 10)                      # through the inside bar

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "stop"
    assert entry.side == OrderSide.Buy
    assert entry.price == pytest.approx(110.0 * (1.0 + 5.0 / 10_000.0))

    kids = children_of(ctx, entry.id)
    assert len(kids) == 1                        # a stop-loss and nothing else: no TP
    sl = kids[0]
    assert sl.order_type == "stop"
    assert sl.side == OrderSide.Sell and sl.reduce_only
    assert sl.price == pytest.approx(100.0)      # the box bottom
    assert sl.quantity == pytest.approx(entry.quantity)

    # The break bar fills the resting stop; the next box [108, 120] completes
    # five bars later and ratchets the stop up under its bottom.
    fill_entry(ctx, entry)
    tick(ctx, strategy, 5)

    assert len(entry_orders(ctx)) == 1           # in-position: the new box arms nothing
    assert ctx.order(sl.id).status == OrderStatus.Cancelled
    ratchets = [o for o in ctx.orders
                if o.order_type == "stop" and o.reduce_only and o.id != sl.id]
    assert len(ratchets) == 1
    assert ratchets[0].price == pytest.approx(108.0)
    assert ratchets[0].quantity == pytest.approx(entry.quantity)
    assert not any(o.order_type == "limit" for o in ctx.orders)   # never a take-profit


# ─── 14. darvasstrict — a broken higher-low streak delays the box ─────────────

def test_darvasstrict_requires_three_consecutive_higher_lows():
    from darvasstrict import DarvasStrictStrategy

    rows = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 5
    rows.append((100.0, 110.0, 100.0, 105.0, 1000.0))   # top candidate 110
    rows.append((105.0, 108.0, 101.0, 104.0, 1000.0))   # lower high 1
    rows.append((104.0, 107.0, 102.0, 104.0, 1000.0))   # lower high 2
    rows.append((104.0, 106.0, 103.0, 104.0, 1000.0))   # higher low: streak 1 (classic would box HERE)
    rows.append((104.0, 105.0, 99.0, 100.0, 1000.0))    # undercut: streak resets, top holds
    rows.append((100.0, 104.0, 100.0, 102.0, 1000.0))   # higher low: streak 1 again
    rows.append((102.0, 103.5, 100.5, 102.0, 1000.0))   # streak 2
    rows.append((102.0, 103.0, 101.0, 102.0, 1000.0))   # streak 3 -> box [99, 110]

    strategy = DarvasStrictStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 10)                              # through the streak-breaking undercut
    assert entry_orders(ctx) == []                       # one higher low is NOT enough here

    tick(ctx, strategy, 3)                               # the fresh 3-bar streak completes
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(110.0 * (1.0 + 5.0 / 10_000.0))
    sl = children_of(ctx, entries[0].id)[0]
    assert sl.price == pytest.approx(99.0)               # the rolling low the streak beat


# ─── 15. darvasshort — sell-stop under the bottom; ratchet DOWN over boxes ────

def test_darvasshort_arms_breakdown_and_ratchets_cover_stop_down():
    from darvasshort import DarvasShortStrategy

    rows = box_walk_rows()
    rows.append((104.0, 105.0, 95.0, 96.0, 1000.0))     # breakdown of [100, 110]
    rows.append((96.0, 99.0, 94.0, 95.0, 1000.0))       # lower high 1 (top candidate 105)
    rows.append((95.0, 98.0, 94.5, 95.5, 1000.0))       # lower high 2
    rows.append((95.5, 97.0, 95.0, 96.0, 1000.0))       # higher low -> lower box [94, 105]

    strategy = DarvasShortStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 10)

    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "stop"
    assert entry.side == OrderSide.Sell
    assert entry.price == pytest.approx(100.0 * (1.0 - 5.0 / 10_000.0))
    sl = children_of(ctx, entry.id)[0]
    assert sl.side == OrderSide.Buy
    assert sl.price == pytest.approx(110.0)             # the box top caps the short

    fill_entry(ctx, entry)
    tick(ctx, strategy, 4)                              # breakdown + the lower box forms

    assert ctx.order(sl.id).status == OrderStatus.Cancelled
    ratchets = [o for o in ctx.orders
                if o.order_type == "stop" and o.reduce_only and o.id != sl.id]
    assert len(ratchets) == 1
    assert ratchets[0].price == pytest.approx(105.0)    # the NEW box's top, lower
    assert not any(o.order_type == "limit" for o in ctx.orders)


# ─── 16. darvasboth — the box END is the (market) signal, both directions ─────

def test_darvasboth_market_enters_in_the_break_direction_with_box_bracket():
    from darvasboth import DarvasBothStrategy

    # (a) upward break: buy, stop at the bottom, target one height above.
    rows_up = box_walk_rows()
    rows_up.append((105.0, 112.0, 104.0, 111.0, 1000.0))
    ctx = run_all(DarvasBothStrategy(), bars_from("TST", rows_up))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].order_type == "market"            # a resting order would be fiction
    assert entries[0].side == OrderSide.Buy
    kids = children_of(ctx, entries[0].id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.price == pytest.approx(100.0)
    assert tp.price == pytest.approx(111.0 + 10.0)      # close + one box height
    assert sl.quantity == pytest.approx(entries[0].quantity)
    assert tp.quantity == pytest.approx(entries[0].quantity)

    # (b) downward break: sell, stop at the top, target one height below.
    rows_dn = box_walk_rows()
    rows_dn.append((104.0, 106.0, 95.0, 96.0, 1000.0))
    ctx = run_all(DarvasBothStrategy(), bars_from("TST", rows_dn))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].order_type == "market"
    assert entries[0].side == OrderSide.Sell
    kids = children_of(ctx, entries[0].id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.price == pytest.approx(110.0)
    assert tp.price == pytest.approx(96.0 - 10.0)


# ─── 17. darvasvolume — no volume, no trade ───────────────────────────────────

def test_darvasvolume_takes_only_the_expanded_volume_break():
    from darvasvolume import DarvasVolumeStrategy

    prelude = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 45   # history for the 50-bar volume mean

    quiet = prelude + box_walk_rows()
    quiet.append((105.0, 112.0, 104.0, 111.0, 1000.0))      # closes above 110, volume flat
    ctx = run_all(DarvasVolumeStrategy(), bars_from("TST", quiet))
    assert entry_orders(ctx) == []                          # the book says walk away

    loud = prelude + box_walk_rows()
    loud.append((105.0, 112.0, 104.0, 111.0, 2500.0))       # same break, 2.5x volume
    ctx = run_all(DarvasVolumeStrategy(), bars_from("TST", loud))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].order_type == "market"
    sl = children_of(ctx, entries[0].id)[0]
    assert sl.price == pytest.approx(100.0)


# ─── 18. darvasboxrisk — stop at the box mid, target the measured move ────────

def test_darvasboxrisk_brackets_with_midpoint_stop_and_measured_move_target():
    from darvasboxrisk import DarvasBoxRiskStrategy

    ctx = run_all(DarvasBoxRiskStrategy(), bars_from("TST", box_walk_rows()))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.price == pytest.approx(110.0 * (1.0 + 5.0 / 10_000.0))
    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.price == pytest.approx(105.0)             # (110 + 100) / 2
    assert tp.price == pytest.approx(120.0)             # top + one box height
    assert sl.quantity == pytest.approx(entry.quantity)
    assert tp.quantity == pytest.approx(entry.quantity)


# ─── 19. darvastight — only the coiled box qualifies ──────────────────────────

def spread_flats(n, px, spread, vol=1000.0):
    return [(px, px * (1.0 + spread), px * (1.0 - spread), px, vol)] * n


def test_darvastight_rejects_wide_box_and_arms_tight_one():
    from darvastight import DarvasTightStrategy

    wide = spread_flats(20, 100.0, 0.012)                       # height ~11 vs ~3.2% ADR
    wide.append((100.0, 110.0, 99.9, 105.0, 1000.0))            # top candidate 110
    wide.append((105.0, 108.0, 101.0, 104.0, 1000.0))
    wide.append((104.0, 107.0, 102.0, 104.0, 1000.0))
    wide.append((104.0, 106.0, 103.0, 104.0, 1000.0))           # box [98.8, 110] completes
    ctx = run_all(DarvasTightStrategy(), bars_from("TST", wide))
    assert entry_orders(ctx) == []                              # too sloppy to be a spring

    tight = spread_flats(20, 100.0, 0.012)
    tight.append((100.0, 102.5, 99.9, 102.0, 1000.0))           # top candidate 102.5
    tight.append((102.0, 102.2, 100.5, 101.5, 1000.0))
    tight.append((101.5, 102.0, 100.8, 101.2, 1000.0))
    tight.append((101.2, 101.8, 101.0, 101.4, 1000.0))          # box [98.8, 102.5], height 3.7
    ctx = run_all(DarvasTightStrategy(), bars_from("TST", tight))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(102.5 * (1.0 + 5.0 / 10_000.0))
    sl = children_of(ctx, entries[0].id)[0]
    assert sl.price == pytest.approx(100.0 * (1.0 - 0.012))     # the rolling-low bottom


# ─── 20. darvastrend — the same box, taken only with the tide ─────────────────

def test_darvastrend_ignores_box_in_downtrend_and_arms_in_uptrend():
    from darvastrend import DarvasTrendStrategy
    from test_qm_family import uptrend_rows, downtrend_rows

    falling = downtrend_rows(55, step=0.006)                    # -9% over the momentum window
    falling.append((71.8, 79.0, 71.5, 75.0, 1000.0))
    falling.append((75.0, 77.5, 72.0, 74.0, 1000.0))
    falling.append((74.0, 76.5, 72.5, 74.0, 1000.0))
    falling.append((74.0, 75.5, 73.0, 74.0, 1000.0))            # box completes, tide is out
    ctx = run_all(DarvasTrendStrategy(), bars_from("TST", falling))
    assert entry_orders(ctx) == []

    rising = uptrend_rows(55)
    rising.append((117.9, 125.0, 117.5, 121.0, 1000.0))
    rising.append((121.0, 123.0, 118.0, 120.0, 1000.0))
    rising.append((120.0, 122.0, 118.5, 119.5, 1000.0))
    rising.append((119.5, 121.0, 118.8, 119.8, 1000.0))         # box completes, tide is in
    ctx = run_all(DarvasTrendStrategy(), bars_from("TST", rising))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(125.0 * (1.0 + 5.0 / 10_000.0))


# ─── 21. darvasrebreak — one retry at the same level, then wait for a new box ─

def test_darvasrebreak_rearms_once_after_stopout_then_requires_new_box():
    from darvasrebreak import DarvasRebreakStrategy

    rows = box_walk_rows() + second_box_rows()
    strategy = DarvasRebreakStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 10)                              # box [100,110] arms attempt 1

    entries = entry_orders(ctx)
    assert len(entries) == 1
    e1 = entries[0]
    assert e1.price == pytest.approx(110.0 * (1.0 + 5.0 / 10_000.0))
    sl1 = children_of(ctx, e1.id)[0]

    fill_entry(ctx, e1)
    tick(ctx, strategy, 1)                               # fill detected
    fill_exit(ctx, sl1)                                  # shakeout: stop-out #1
    tick(ctx, strategy, 1)

    entries = sorted(entry_orders(ctx), key=lambda o: o.id)
    assert len(entries) == 2                             # re-armed immediately...
    e2 = entries[-1]
    assert e2.price == pytest.approx(e1.price)           # ...at the SAME level
    sl2 = children_of(ctx, e2.id)[0]
    assert sl2.price == pytest.approx(100.0)

    fill_entry(ctx, e2)
    tick(ctx, strategy, 1)
    fill_exit(ctx, sl2)                                  # stop-out #2: the box is retired
    tick(ctx, strategy, 1)
    assert len(entry_orders(ctx)) == 2                   # no third attempt at this box

    tick(ctx, strategy, 1)                               # the NEW box [108, 120] completes
    entries = sorted(entry_orders(ctx), key=lambda o: o.id)
    assert len(entries) == 3                             # fresh box, fresh budget
    assert entries[-1].price == pytest.approx(120.0 * (1.0 + 5.0 / 10_000.0))
    assert children_of(ctx, entries[-1].id)[0].price == pytest.approx(108.0)
