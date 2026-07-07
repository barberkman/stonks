"""Behavior tests, one per combined QM x Darvas strategy."""

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
from test_qm_family import downtrend_rows, flat_rows, qm_base_rows, uptrend_rows
from test_darvas_family import box_walk_rows, spread_flats


def rising_box_rows(warmup=65):
    """An uptrend then a Darvas box [~120.9, 128] right at its highs — the
    box top IS the 40-bar pivot, and the QM up-universe holds throughout."""
    rows = uptrend_rows(warmup)
    last = rows[-1][3]
    rows.append((last, 128.0, last * 0.998, 124.5, 1000.0))          # top candidate 128
    rows.append((124.5, 126.5, 121.8, 123.5, 1000.0))
    rows.append((123.5, 125.5, 122.2, 123.2, 1000.0))
    rows.append((123.2, 124.5, 122.5, 123.4, 1000.0))                # box completes
    return rows


def falling_box_rows(warmup=60):
    """A downtrend, then a bounce box [68.4, 74]. The pine machine never
    resets in a monotonic downtrend (state 4, top frozen at the very first
    high), so the scenario explicitly completes and BREAKS that giant box
    first; only then can the bounce box form under a fresh 74 reference.
    The QM up-universe fails throughout (deep negative momentum); the
    down-universe holds."""
    rows = downtrend_rows(warmup, step=0.006)                        # ends ~69.7
    rows.append((69.72, 69.93, 69.58, 69.79, 1000.0))                # higher low: the giant box completes
    rows.append((69.79, 70.00, 68.90, 69.00, 1000.0))                # flush under it: giant box breaks down
    rows.append((69.00, 74.00, 68.80, 70.00, 1000.0))                # fresh reference high 74
    rows.append((70.00, 71.50, 68.50, 69.00, 1000.0))                # lower high 1
    rows.append((69.00, 70.50, 68.40, 68.80, 1000.0))                # lower high 2
    rows.append((68.80, 69.80, 68.60, 68.90, 1000.0))                # higher low -> box [68.4, 74]
    return rows


# ─── 22. qmdarvasbase — box as the flag, QM gates + QM exit engine ────────────

def test_qmdarvasbase_arms_gated_box_with_qm_bracket_and_tighter_stop():
    from qmdarvasbase import QMDarvasBaseStrategy

    ctx = run_all(QMDarvasBaseStrategy(), bars_from("TST", rising_box_rows()))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_type == "stop"
    assert entry.price == pytest.approx(128.0 * (1.0 + 5.0 / 10_000.0))
    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")         # QM exits: a TP exists
    assert sl.price > 120.9 + 0.5                                 # ADR stop BEAT the box bottom
    assert tp.quantity == pytest.approx(entry.quantity / 2.0)     # half partial at 2R
    assert tp.price == pytest.approx(entry.price + 2.0 * (entry.price - sl.price))

    ctx = run_all(QMDarvasBaseStrategy(), bars_from("TST", falling_box_rows()))
    assert entry_orders(ctx) == []                                # universe gate holds it out


# ─── 24. qmdarvasuniverse — same gates, but Darvas' own exit engine ───────────

def test_qmdarvasuniverse_arms_gated_box_with_pure_darvas_exits():
    from qmdarvasuniverse import QMDarvasUniverseStrategy

    rows = rising_box_rows()
    box_bottom = min(rows[i][2] for i in (-5, -4, -3, -2))        # the rolling low at completion
    ctx = run_all(QMDarvasUniverseStrategy(), bars_from("TST", rows))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    entry = entries[0]
    kids = children_of(ctx, entry.id)
    assert len(kids) == 1                                         # stop-loss only, NO take-profit
    assert kids[0].order_type == "stop"
    assert kids[0].price == pytest.approx(box_bottom)             # the box bottom, not an ADR stop
    assert not any(o.order_type == "limit" for o in ctx.orders)

    ctx = run_all(QMDarvasUniverseStrategy(), bars_from("TST", falling_box_rows()))
    assert entry_orders(ctx) == []


# ─── 23. qmdarvasconsensus — the two ceilings must agree ──────────────────────

def test_qmdarvasconsensus_requires_pivot_and_box_top_agreement():
    from qmdarvasconsensus import QMDarvasConsensusStrategy

    # Agreement: the box top IS the 40-bar pivot (128 == 128).
    ctx = run_all(QMDarvasConsensusStrategy(), bars_from("TST", rising_box_rows()))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(128.0 * (1.0 + 5.0 / 10_000.0))
    assert any(o.order_type == "limit" for o in children_of(ctx, entries[0].id))

    # Disagreement: an old 140 spike still owns the 40-bar pivot when a much
    # lower box completes -> the flag is valid, the box is valid, but the two
    # ceilings are ~10% apart and nothing arms.
    rows = [(100.0, 100.4, 99.6, 100.0, 1000.0)] * 45
    rows.append((100.0, 140.0, 99.5, 100.2, 1000.0))     # spike wick; closes stay flat
    rows.append((100.2, 101.5, 99.7, 100.2, 1000.0))     # lower highs under 140...
    rows.append((100.2, 101.0, 99.8, 100.2, 1000.0))
    rows.append((100.2, 100.9, 99.9, 100.2, 1000.0))     # ...box [~99.5, 140] completes (no arm:
    rows.append((100.2, 100.8, 99.0, 99.5, 1000.0))      # flat closes fail the gain gate), breaks down
    px = 99.5
    for _ in range(20):                                  # a real uptrend restores the universe
        o = px
        px = px * 1.003
        rows.append((o, px * 1.005, o * 0.995, px, 1000.0))
    last = rows[-1][3]
    rows.append((last, last * 1.02, last * 0.997, last * 1.01, 1000.0))   # top candidate ~107
    rows.append((last * 1.01, last * 1.015, last * 1.0, last * 1.008, 1000.0))
    rows.append((last * 1.008, last * 1.012, last * 1.002, last * 1.006, 1000.0))
    rows.append((last * 1.006, last * 1.010, last * 1.004, last * 1.007, 1000.0))  # low box completes
    ctx = run_all(QMDarvasConsensusStrategy(), bars_from("TST", rows))
    assert entry_orders(ctx) == []                       # ceilings ~24% apart: no consensus


# ─── 25. qmdarvasepbox — the gap must clear the PRIOR bar's active box ────────

def test_qmdarvasepbox_takes_gap_over_box_and_ignores_boxless_gap():
    from qmdarvasepbox import QMDarvasEPBoxStrategy

    with_box = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 55 + box_walk_rows()
    with_box.append((110.5, 112.5, 110.2, 112.0, 2500.0))   # gaps OVER the [100,110] box top
    ctx = run_all(QMDarvasEPBoxStrategy(), bars_from("TST", with_box))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].order_type == "market"
    kids = children_of(ctx, entries[0].id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.price == pytest.approx(100.0)                 # the box bottom is the stop
    assert tp.price == pytest.approx(112.0 + 2.0 * (112.0 - 100.0))
    assert tp.quantity == pytest.approx(entries[0].quantity / 2.0)

    boxless = [(100.0, 100.4, 99.6, 100.0, 1000.0)] * 65
    boxless.append((101.0, 102.5, 100.8, 102.0, 2500.0))    # the same-quality gap, no box under it
    ctx = run_all(QMDarvasEPBoxStrategy(), bars_from("TST", boxless))
    assert entry_orders(ctx) == []


# ─── 27. qmdarvasshort — box breakdown gated by the QM downtrend universe ─────

def test_qmdarvasshort_arms_breakdown_bracket_in_downtrend_universe():
    from qmdarvasshort import QMDarvasShortStrategy

    ctx = run_all(QMDarvasShortStrategy(), bars_from("TST", falling_box_rows()))
    # The giant downtrend box arms once and is cancelled when it breaks; the
    # bounce box [68.4, 74] owns the live order.
    entries = sorted(entry_orders(ctx), key=lambda o: o.id)
    live = [o for o in entries if o.status == OrderStatus.Open]
    assert len(live) == 1
    entry = live[0]
    assert entry.order_type == "stop"
    assert entry.side == OrderSide.Sell
    assert entry.price == pytest.approx(68.4 * (1.0 - 5.0 / 10_000.0))
    assert all(o.status == OrderStatus.Cancelled for o in entries if o.id != entry.id)
    kids = children_of(ctx, entry.id)
    sl = next(o for o in kids if o.order_type == "stop")
    tp = next(o for o in kids if o.order_type == "limit")
    assert sl.side == OrderSide.Buy and tp.side == OrderSide.Buy
    assert sl.price == pytest.approx(74.0)                  # the box top caps the risk
    assert tp.quantity == pytest.approx(entry.quantity / 2.0)
    assert tp.price == pytest.approx(entry.price - 2.0 * (sl.price - entry.price))

    ctx = run_all(QMDarvasShortStrategy(), bars_from("TST", rising_box_rows()))
    assert entry_orders(ctx) == []                          # up-universe: no shorts


# ─── 26. qmdarvasexit — the runner's stop ratchets under new boxes, no MA ─────

def test_qmdarvasexit_partial_then_box_ratchet_replaces_stop():
    from qmdarvasexit import QMDarvasExitStrategy

    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))       # break bar, volume expands
    rows.append((131.0, 136.0, 130.8, 135.5, 1000.0))       # the runner runs...
    rows.append((135.5, 141.0, 135.0, 140.0, 1000.0))
    rows.append((140.0, 145.0, 139.5, 143.0, 1000.0))       # new top candidate 145
    rows.append((143.0, 144.0, 140.0, 142.0, 1000.0))
    rows.append((142.0, 143.5, 140.5, 142.0, 1000.0))
    rows.append((142.0, 143.0, 141.0, 142.0, 1000.0))       # box [135, 145] completes

    strategy = QMDarvasExitStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)
    entry = entry_orders(ctx)[0]
    fill_entry(ctx, entry)
    tick(ctx, strategy, 7)                                  # partial (6 bars) + the new box

    partials = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(partials) == 1                               # the time partial; no MA trail exit
    assert partials[0].quantity == pytest.approx(entry.quantity / 2.0)
    live_stops = [o for o in ctx.orders
                  if o.order_type == "stop" and o.reduce_only
                  and o.status == OrderStatus.Open]
    assert len(live_stops) == 1
    assert live_stops[0].price == pytest.approx(135.0)      # the new box's bottom, above BE
    assert live_stops[0].quantity == pytest.approx(entry.quantity)


# ─── 28. qmdarvasregime — the live dial picks the engine; the tag manages ─────

def test_qmdarvasregime_dispatches_by_adr_and_manages_by_stored_tag():
    from qmdarvasregime import QMDarvasRegimeStrategy

    # (a) fast tape -> QM engine: pivot arm WITH a take-profit leg.
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))       # fill bar
    rows += [(131.0, 131.05, 130.95, 131.0, 1000.0)] * 7    # dead-quiet bars: live ADR% flips low
    strategy = QMDarvasRegimeStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)
    entries = entry_orders(ctx)
    assert len(entries) == 1
    qm_entry = entries[0]
    assert qm_entry.price == pytest.approx(130.0 * (1.0 + 5.0 / 10_000.0))
    assert any(o.order_type == "limit" for o in children_of(ctx, qm_entry.id))

    # (c) the live regime flips to quiet mid-trade, but the stored "qm" tag
    # still runs QM management: the 6-bar time partial + breakeven fire.
    fill_entry(ctx, qm_entry)
    tick(ctx, strategy, 8)
    partials = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(partials) == 1
    assert partials[0].quantity == pytest.approx(qm_entry.quantity / 2.0)
    be_stops = [o for o in ctx.orders
                if o.order_type == "stop" and o.reduce_only
                and o.status == OrderStatus.Open]
    assert len(be_stops) == 1
    assert be_stops[0].price == pytest.approx(qm_entry.price)   # breakeven, a QM-only move

    # (b) quiet tape -> Darvas engine: box arm with NO take-profit.
    quiet = spread_flats(65, 100.0, 0.002)
    quiet.append((100.0, 102.0, 99.9, 101.2, 1000.0))       # top candidate 102
    quiet.append((101.2, 101.8, 100.2, 101.0, 1000.0))
    quiet.append((101.0, 101.5, 100.4, 100.9, 1000.0))
    quiet.append((100.9, 101.3, 100.5, 100.95, 1000.0))     # box [99.8, 102] completes
    ctx = run_all(QMDarvasRegimeStrategy(), bars_from("TST", quiet))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(102.0 * (1.0 + 5.0 / 10_000.0))
    kids = children_of(ctx, entries[0].id)
    assert len(kids) == 1                                   # Darvas mode: stop only
    assert kids[0].price == pytest.approx(100.0 * (1.0 - 0.002))


# ─── 29. qmdarvasfirstbox — ignition opens the window; the first box arms ─────

def test_qmdarvasfirstbox_arms_first_box_after_ignition_only():
    from qmdarvasfirstbox import QMDarvasFirstBoxStrategy

    rows = flat_rows(65, 100.0)
    rows.append((101.0, 102.5, 101.2, 102.0, 2500.0))       # ignition: gap + volume + strong close
    rows.append((102.0, 104.0, 101.8, 103.0, 1000.0))       # top candidate 104
    rows.append((103.0, 103.8, 102.0, 102.8, 1000.0))
    rows.append((102.8, 103.5, 102.2, 102.8, 1000.0))
    rows.append((102.8, 103.2, 102.4, 102.9, 1000.0))       # the FIRST post-ignition box
    ctx = run_all(QMDarvasFirstBoxStrategy(), bars_from("TST", rows))
    entries = entry_orders(ctx)
    assert len(entries) == 1
    assert entries[0].order_type == "stop"
    assert entries[0].price == pytest.approx(104.0 * (1.0 + 5.0 / 10_000.0))
    kids = children_of(ctx, entries[0].id)
    sl = next(o for o in kids if o.order_type == "stop")
    assert sl.price == pytest.approx(101.2)                 # rolling-low box bottom

    # The identical box WITHOUT an ignition arms nothing: the box alone is
    # not a thesis here.
    rows = flat_rows(65, 100.0)
    rows.append((100.0, 104.0, 99.8, 103.0, 1000.0))
    rows.append((103.0, 103.8, 102.0, 102.8, 1000.0))
    rows.append((102.8, 103.5, 102.2, 102.8, 1000.0))
    rows.append((102.8, 103.2, 102.4, 102.9, 1000.0))
    ctx = run_all(QMDarvasFirstBoxStrategy(), bars_from("TST", rows))
    assert entry_orders(ctx) == []


# ─── 30. qmdarvasboxtrail — the structure tripwire beats the slow EMA ─────────

def test_qmdarvasboxtrail_runner_exits_on_box_bottom_violation_before_ema():
    from qmdarvasboxtrail import QMDarvasBoxTrailStrategy

    rows = uptrend_rows(80)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))   # pivot high 130
    rows.append((128.0, 128.5, 123.0, 126.0, 900.0))
    rows.append((126.0, 127.0, 122.8, 125.5, 900.0))
    rows.append((125.3, 126.0, 122.5, 125.0, 900.0))          # base armed
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))         # break bar
    rows += flat_rows(5, 131.0)                               # quiet: time partial due
    rows.append((131.0, 133.0, 130.8, 132.0, 1000.0))         # top candidate 133
    rows.append((132.0, 132.8, 130.9, 131.5, 1000.0))
    rows.append((131.5, 132.5, 131.0, 131.5, 1000.0))
    rows.append((131.5, 132.2, 131.2, 131.6, 1000.0))         # box completes: bottom ~130.48
    rows.append((131.5, 131.8, 128.8, 129.0, 1000.0))         # closes UNDER the box bottom

    strategy = QMDarvasBoxTrailStrategy()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 84)
    entry = entry_orders(ctx)[0]
    fill_entry(ctx, entry)
    tick(ctx, strategy, 10)                                   # through the box, before violation

    exits = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(exits) == 1                                    # only the time partial so far

    tick(ctx, strategy, 1)                                    # the violating close
    exits = [o for o in ctx.orders if o.order_type == "market" and o.reduce_only]
    assert len(exits) == 2                                    # the structure tripwire flattened it
    assert exits[-1].quantity == pytest.approx(entry.quantity)
    # The slow EMA sat far below the violating close: the box leg fired, not the EMA.
    ema_plots = [p.value for p in ctx.plots if p.name == "slow_ema"]
    assert ema_plots and ema_plots[-1] < 129.0