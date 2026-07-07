"""Behavior tests for qmswing — the 1:1 signal-printer port of the pine.

The strategy never places orders, so every assertion reads `strategy.signals`
(Signal records) and `ctx.plots`, never `ctx.orders`."""

import pytest

import stonks
from conftest import MS4H, bars_from, make_bars, run_all, start_run, tick
from qmswing import QMSwingStrategy


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


def qm_base_rows(pull_vol=900.0):
    """The family's shared tight-flag scenario: 70-bar uptrend, pivot high
    130, three digestion bars -> a qualified base armed on the last row."""
    rows = uptrend_rows(70)
    last = rows[-1][3]
    rows.append((last, 130.0, last * 0.998, 128.0, 1500.0))
    rows.append((128.0, 128.5, 123.0, 126.0, pull_vol))
    rows.append((126.0, 127.0, 122.8, 125.5, pull_vol))
    rows.append((125.3, 126.0, 122.5, 125.0, pull_vol))
    return rows


def qmswing(**overrides):
    strategy = QMSwingStrategy()
    strategy.print_signals = False   # keep pytest output lean; signals always record
    for name, value in overrides.items():
        setattr(strategy, name, value)
    return strategy


def kinds(strategy, kind):
    return [s for s in strategy.signals if s.kind == kind]


def ts_of(i):
    return MS4H * (i + 1)   # bars_from stamps row i at start_ts + i * interval


BO_LEVEL = 130.0 * (1.0 + 5.0 / 10_000.0)


# ─── 1. Advance notice: the armed level prints BEFORE the break fills ─────────

def test_bo_arms_and_prints_advance_notice_before_fill(capsys):
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))                # break bar
    strategy = qmswing(print_signals=True)
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 74)                                          # through the arm bar

    arms = kinds(strategy, "arm_bo")
    assert arms, "the tight flag must arm the resting buy-stop"
    assert arms[-1].price == pytest.approx(BO_LEVEL)
    assert arms[-1].ttl_bars == 10
    assert not [s for s in strategy.signals if s.kind.startswith("entry")]
    assert "BO buy-stop armed" in capsys.readouterr().out            # printed, not just recorded

    tick(ctx, strategy, 1)                                           # the break bar
    entries = kinds(strategy, "entry_bo")
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(BO_LEVEL)
    assert arms[-1].timestamp < entries[0].timestamp                 # notice strictly precedes the fill
    assert ctx.orders == [] and ctx.positions == {}


# ─── 2. Resting-stop fill convention: max(open, level) ────────────────────────

def test_bo_fill_price_is_max_of_open_and_level():
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))                # opens below the level
    below = qmswing()
    run_all(below, bars_from("TST", rows))
    assert kinds(below, "entry_bo")[0].price == pytest.approx(BO_LEVEL)

    rows = qm_base_rows()
    rows.append((132.0, 133.5, 131.0, 133.0, 2000.0))                # gaps through the level
    gapped = qmswing()
    run_all(gapped, bars_from("TST", rows))
    assert kinds(gapped, "entry_bo")[0].price == pytest.approx(132.0)


# ─── 3. wait_for_close: a wick poke is not a break ────────────────────────────

def test_wait_for_close_ignores_wick_poke_and_fires_on_close_through():
    rows = qm_base_rows()
    rows.append((125.0, 130.5, 124.0, 128.0, 2000.0))                # high pokes the level, close under
    rows.append((128.0, 129.5, 126.0, 128.0, 900.0))                 # digestion under the poke high
    rows.append((128.0, 129.0, 126.5, 128.2, 900.0))
    rows.append((128.2, 129.0, 126.8, 128.0, 900.0))                 # re-arms at the new pivot
    rows.append((128.0, 131.9, 127.8, 131.5, 2000.0))                # the CLOSE clears it
    strategy = qmswing(wait_for_close=True)
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 75)                                          # through the poke bar
    assert kinds(strategy, "entry_bo") == []                         # the poke filled nothing

    tick(ctx, strategy, 4)
    entries = kinds(strategy, "entry_bo")
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(130.5 * (1.0 + 5.0 / 10_000.0))


# ─── 4. Break-bar volume gate blocks the fill, the order stays armed ──────────

def test_vol_break_gate_blocks_weak_volume_break_bar():
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 900.0))                 # breaks on weak volume
    rows.append((131.0, 132.0, 130.5, 131.8, 2000.0))                # volume expands -> fills
    strategy = qmswing()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 75)                                          # through the weak break
    assert kinds(strategy, "entry_bo") == []
    assert [s for s in kinds(strategy, "arm_bo") if s.timestamp == ts_of(74)]   # still armed

    tick(ctx, strategy, 1)
    entries = kinds(strategy, "entry_bo")
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(131.0)                  # gapped past the level -> the open


# ─── 5. EP: exact watch thresholds one bar ahead, then entry at the close ─────

def test_ep_watch_level_then_entry_at_the_close():
    rows = flat_rows(65, 100.0, spread=0.006)
    rows.append((101.0, 102.5, 101.2, 102.0, 2000.0))                # 1% gap, 2x volume, strong close
    strategy = qmswing()
    ctx = start_run(strategy, bars_from("TST", rows))
    tick(ctx, strategy, 65)                                          # the bar BEFORE the gap

    watches = kinds(strategy, "watch_ep")
    assert watches
    assert watches[-1].price == pytest.approx(100.0 * (1.0 + 0.5 / 100.0))
    assert watches[-1].vol_threshold == pytest.approx(1.3 * 1000.0)
    assert kinds(strategy, "entry_ep") == []

    tick(ctx, strategy, 1)
    entries = kinds(strategy, "entry_ep")
    assert len(entries) == 1
    e = entries[0]
    assert e.price == pytest.approx(102.0)                           # the pine buys the gap-bar close
    assert e.stop == pytest.approx(101.2)                            # gap-bar low beats the ADR stop
    assert e.target == pytest.approx(102.0 + 2.0 * (102.0 - 101.2))
    assert watches[-1].timestamp < e.timestamp


# ─── 6. Partial at 2R -> breakeven -> trail exit ──────────────────────────────

def test_partial_at_target_then_breakeven_then_trail_exit():
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))                # fill @ the level
    rows.append((131.0, 138.0, 130.8, 131.5, 1000.0))                # tags the 2R target
    rows += flat_rows(27, 131.0)                                     # holds above BE, EMA20 rises
    rows.append((130.9, 131.0, 130.15, 130.2, 1000.0))               # closes below the trail
    strategy = qmswing()
    run_all(strategy, bars_from("TST", rows))

    entry = kinds(strategy, "entry_bo")[0]
    partials = kinds(strategy, "partial")
    assert len(partials) == 1
    assert partials[0].reason == "target"
    assert partials[0].price == pytest.approx(entry.target)
    assert partials[0].stop == pytest.approx(entry.price)            # stop moved to breakeven
    exits = kinds(strategy, "exit")
    assert len(exits) == 1
    assert exits[0].reason == "trail"
    assert exits[0].price == pytest.approx(130.2)
    assert exits[0].timestamp > partials[0].timestamp


# ─── 7. Hard stop exit reports the stop level itself ──────────────────────────

def test_hard_stop_exit_reports_the_stop_level():
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 124.8, 131.0, 2000.0))                # fill bar
    rows.append((130.0, 130.5, 119.5, 120.0, 1000.0))                # crashes through the stop
    strategy = qmswing()
    run_all(strategy, bars_from("TST", rows))

    entry = kinds(strategy, "entry_bo")[0]
    exits = kinds(strategy, "exit")
    assert len(exits) == 1
    assert exits[0].reason == "stop"
    assert exits[0].price == pytest.approx(entry.stop)
    assert kinds(strategy, "partial") == []


# ─── 8. Same-bar collapse: filled then closed through the fresh stop ──────────

def test_same_bar_collapse_bails_immediately():
    rows = qm_base_rows()
    rows.append((125.0, 131.5, 118.0, 118.5, 2000.0))                # breaks, then collapses
    strategy = qmswing()
    run_all(strategy, bars_from("TST", rows))

    entries = kinds(strategy, "entry_bo")
    exits = kinds(strategy, "exit")
    assert len(entries) == 1 and len(exits) == 1
    assert exits[0].reason == "collapse"
    assert exits[0].timestamp == entries[0].timestamp                # same bar
    assert not strategy.state["TST"]["in_long"]


# ─── 9. Priority: a bar valid for both BO fill and EP takes the BO ────────────

def test_priority_bo_fill_beats_ep_on_the_same_bar():
    rows = qm_base_rows()
    rows.append((129.9, 130.6, 129.8, 130.4, 2000.0))                # textbook EP that ALSO breaks

    lone = qmswing(enable_bo=False)                                  # sanity: EP alone fires here
    run_all(lone, bars_from("TST", rows))
    assert len(kinds(lone, "entry_ep")) == 1

    strategy = qmswing()
    run_all(strategy, bars_from("TST", rows))
    entries = kinds(strategy, "entry_bo")
    assert len(entries) == 1
    assert entries[0].price == pytest.approx(BO_LEVEL)
    assert kinds(strategy, "entry_ep") == []                         # EP never got the bar


# ─── 10. ORB / SBO / PARA stay silent with the pine's default toggles ─────────

@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("regime", ["trending", "choppy", "gappy"])
def test_orb_sbo_para_stay_silent_by_default(regime, direction):
    strategy = qmswing()
    run_all(strategy, make_bars(["AAA", "BBB"], 300, regime, seed=7, direction=direction))
    off_kinds = {"arm_orb", "arm_so", "watch_para", "entry_orb", "entry_sbo", "entry_para"}
    assert not [s for s in strategy.signals if s.kind in off_kinds]


# ─── 11. ORB: per-UTC-day range, one attempt, fill consumes it ────────────────

def test_orb_arms_per_utc_day_and_fill_consumes_the_attempt():
    rows = uptrend_rows(65)                                          # 4h bars: 6 per UTC day
    rows.append((121.5, 125.0, 121.4, 124.0, 1500.0))                # row 65: first bar of a new day
    rows.append((124.0, 124.5, 123.6, 124.2, 1000.0))                # row 66: range complete
    rows += [(124.0, 124.5, 123.5, 124.0, 1000.0)] * 4               # rows 67-70: no break
    rows.append((124.0, 124.3, 123.8, 124.1, 1000.0))                # row 71: NEXT day's first bar
    rows.append((124.1, 124.6, 123.9, 124.4, 1000.0))                # row 72: new range complete
    rows.append((124.4, 126.0, 124.3, 125.8, 2000.0))                # row 73: breaks -> entry
    rows.append((125.8, 126.0, 125.2, 125.6, 1000.0))                # row 74: attempt consumed
    strategy = qmswing(enable_orb=True, enable_bo=False)
    run_all(strategy, bars_from("TST", rows))

    arms = kinds(strategy, "arm_orb")
    day1 = [s for s in arms if ts_of(65) <= s.timestamp <= ts_of(70)]
    assert day1
    assert day1[0].timestamp == ts_of(66)                            # active once the range completes
    assert day1[0].price == pytest.approx(125.0 * (1.0 + 5.0 / 10_000.0))
    assert not [s for s in arms if s.timestamp == ts_of(71)]         # new day: range re-forms
    day2 = [s for s in arms if s.timestamp >= ts_of(72)]
    assert day2 and day2[0].price == pytest.approx(124.3 * (1.0 + 5.0 / 10_000.0))
    assert [s for s in arms if s.timestamp == ts_of(73)]             # pine plots the level on its fill bar

    entries = kinds(strategy, "entry_orb")
    assert len(entries) == 1
    assert entries[0].timestamp == ts_of(73)
    assert entries[0].price == pytest.approx(124.4)                  # gapped past the level -> the open
    assert not [s for s in arms if s.timestamp == ts_of(74)]         # taken: dark for the rest of the day


# ─── 12. SBO mirror: arm, fill at min(open, level), partial -> BE -> trail ────

def test_sbo_mirror_arm_entry_partial_breakeven_trail_cover():
    rows = downtrend_rows(70)
    last = rows[-1][3]
    rows.append((last, last * 1.003, 75.0, 76.0, 1500.0))            # trough: base low 75
    rows.append((76.0, 77.5, 75.3, 77.0, 900.0))                     # bounce 1
    rows.append((77.0, 78.0, 75.5, 76.8, 900.0))                     # bounce 2
    rows.append((76.8, 77.8, 75.2, 76.5, 900.0))                     # bounce 3 -> arms
    rows.append((76.5, 76.8, 74.5, 74.6, 2000.0))                    # breaks the base low
    rows.append((74.6, 74.8, 71.0, 72.0, 1000.0))                    # tags the 2R target
    rows += flat_rows(22, 72.0)                                      # holds under BE, EMA20 falls
    rows.append((72.0, 73.4, 71.9, 73.3, 1000.0))                    # closes above the trail
    strategy = qmswing(enable_sbo=True)
    run_all(strategy, bars_from("TST", rows))

    level = 75.0 * (1.0 - 5.0 / 10_000.0)
    arms = kinds(strategy, "arm_so")
    assert arms and arms[0].price == pytest.approx(level)
    entries = kinds(strategy, "entry_sbo")
    assert len(entries) == 1
    e = entries[0]
    assert e.price == pytest.approx(level)                           # opened above: fills at the level
    assert e.stop > e.price                                          # mirrored geometry
    assert e.target == pytest.approx(e.price - 2.0 * (e.stop - e.price))
    partials = kinds(strategy, "partial")
    assert len(partials) == 1
    assert partials[0].side == "short" and partials[0].reason == "target"
    assert partials[0].stop == pytest.approx(e.price)                # breakeven = min(stop, entry)
    exits = kinds(strategy, "exit")
    assert len(exits) == 1
    assert exits[0].reason == "trail" and exits[0].side == "short"
    assert exits[0].price == pytest.approx(73.3)


# ─── 13. PARA: watch on the streak, short the first red bar, cover the bounce ─

def test_para_watch_then_first_red_bar_short_and_bounce_cover():
    rows = flat_rows(62, 100.0)
    rows.append((100.0, 103.2, 99.9, 103.0, 1200.0))                 # parabolic leg
    rows.append((103.0, 106.3, 102.9, 106.1, 1200.0))
    rows.append((106.1, 109.5, 106.0, 109.3, 1200.0))
    rows.append((109.3, 112.85, 109.2, 112.6, 1200.0))               # 4 straight up-closes
    rows.append((112.6, 112.8, 110.5, 111.0, 1300.0))                # first red bar -> short
    rows.append((111.0, 112.5, 110.8, 112.0, 1000.0))                # up-close -> cover
    strategy = qmswing(enable_para=True)
    run_all(strategy, bars_from("TST", rows))

    watches = kinds(strategy, "watch_para")
    assert watches
    assert watches[-1].timestamp == ts_of(65)                        # the last up-leg bar
    assert watches[-1].price == pytest.approx(112.6)                 # a close below this triggers
    entries = kinds(strategy, "entry_para")
    assert len(entries) == 1
    e = entries[0]
    assert e.timestamp == ts_of(66)
    assert e.price == pytest.approx(111.0)                           # the pine shorts the close
    assert e.stop == pytest.approx(112.85 * (1.0 + 5.0 / 10_000.0))
    assert e.target is None                                          # the parabolic track has no target
    exits = kinds(strategy, "exit")
    assert len(exits) == 1
    assert exits[0].reason == "bounce"
    assert exits[0].price == pytest.approx(112.0)


# ─── 14. Pure signal printer: never an order, only declared plots ─────────────

@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("regime", ["trending", "choppy", "gappy"])
def test_never_places_orders_and_plots_only_declared_series(regime, direction):
    strategy = qmswing(enable_orb=True, enable_sbo=True, enable_para=True)
    ctx = run_all(strategy, make_bars(["AAA", "BBB"], 300, regime, seed=7, direction=direction))
    assert ctx.orders == []
    assert ctx.positions == {}
    assert strategy.signals                                          # it is not silent
    assert {p.name for p in ctx.plots} <= set(QMSwingStrategy.indicators)


# ─── 15. Every pine input is exposed as a GUI param ───────────────────────────

def test_every_pine_input_is_a_gui_param():
    specs = stonks.param_specs(QMSwingStrategy)
    assert len(specs) == 41                                          # 40 pine inputs + print_signals
    assert {s["type"] for s in specs} <= {"bool", "int", "float"}
