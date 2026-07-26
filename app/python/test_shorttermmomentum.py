"""Unit tests for the short-term cross-sectional momentum strategy.

FakeContext never fills orders on its own; a fill is simulated by setting
ctx.positions, mirroring what the broker would report on the next tick. Engine
fill mechanics (next-open market fills, reduce-only, insufficient-cash
rejection) are pinned by the C++ suite under tests/core/."""

import pandas as pd
import pytest

import stonks
from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine, FakePosition

from shorttermmomentum import ShortTermMomentumStrategy

WINDOW = ShortTermMomentumStrategy.window
TOP = ShortTermMomentumStrategy.top

# Twelve names ranked by how fast they climb: L is strongest, A weakest.
STEPS = {chr(ord("A") + i): 0.1 * (i + 1) for i in range(12)}
RANKED_DESC = sorted(STEPS, key=lambda s: -STEPS[s])
# A one-bar reversal that flips the ranking: the six leaders drop ~16% and the
# six laggards gain ~20%, both inside the daily band so nothing is booked as 0%.
JUMP = {s: (-20.0 if s in RANKED_DESC[:6] else 20.0) for s in STEPS}


def sessions(count, start="2025-01-02"):
    """`count` consecutive weekday timestamps in ms, like a daily equity feed."""
    return [int(d.value // 1_000_000)
            for d in pd.bdate_range(start=start, periods=count, tz="UTC")]


JAN = sessions(WINDOW + 1, start="2025-01-02")
FEB = sessions(3, start="2025-02-03")


def feed(phases, first=100.0):
    """[({symbol: per-bar step}, timestamps)] -> one merged, time-sorted feed.
    Each phase continues a symbol's price where the previous phase left it; the
    step applies from a phase's first bar, so a phase change is a visible move."""
    prices = {}
    bars = []
    for steps, timestamps in phases:
        for symbol, step in steps.items():
            price = prices.get(symbol, first)
            for ts in timestamps:
                price += step
                bars.append(FakeKLine(ts, symbol, price, price, price, price, 1_000.0))
            prices[symbol] = price
    bars.sort(key=lambda b: b.timestamp)
    return bars


def start_run(bars, cash=100_000.0, **overrides):
    ctx = FakeContext(bars, cash=cash)
    strategy = ShortTermMomentumStrategy()
    for name, value in overrides.items():
        setattr(strategy, name, value)
    strategy.on_start(ctx)
    return ctx, strategy


def drive(strategy, ctx, ticks):
    for _ in range(ticks):
        ctx.advance()
        strategy.on_tick(ctx)


def close_at(ctx, symbol):
    """That symbol's close on the tick the context is currently on."""
    return next(b.close for b in ctx._bars
                if b.symbol == symbol and b.timestamp == ctx.now())


def fill(ctx, order):
    """Report `order` as filled at the current close, like the broker would."""
    if order.side == OrderSide.Buy:
        ctx.positions[order.symbol] = FakePosition(quantity=order.quantity,
                                                   price=close_at(ctx, order.symbol))
    else:
        ctx.positions.pop(order.symbol, None)


def buys(ctx):
    return [o for o in ctx.orders if o.side == OrderSide.Buy]


def sells(ctx):
    return [o for o in ctx.orders if o.side == OrderSide.Sell]


def test_no_orders_before_a_full_window():
    ctx, strategy = start_run(feed([(STEPS, JAN[:-1])]))
    drive(strategy, ctx, WINDOW)
    assert ctx.orders == []


def test_first_rebalance_buys_the_top_names_on_the_warmup_bar():
    ctx, strategy = start_run(feed([(STEPS, JAN)]))
    drive(strategy, ctx, WINDOW + 1)

    # Nothing held, so nothing to fund from: the entries go out immediately.
    assert sorted(o.symbol for o in buys(ctx)) == sorted(RANKED_DESC[:TOP])
    assert sells(ctx) == []


def test_equal_weight_sizing_within_the_cash_budget():
    ctx, strategy = start_run(feed([(STEPS, JAN)]), cash=100_000.0)
    drive(strategy, ctx, WINDOW + 1)

    # cash_buffer holds back 2%, so the cash cap binds before the equity/top one.
    budget = 100_000.0 * (1.0 - ShortTermMomentumStrategy.cash_buffer) / TOP
    for order in buys(ctx):
        assert order.quantity * close_at(ctx, order.symbol) == pytest.approx(budget)
    assert all(o.leverage == 1.0 and not o.reduce_only for o in buys(ctx))


def test_goes_to_cash_when_fewer_names_than_slots_are_rankable():
    thin = {s: STEPS[s] for s in RANKED_DESC[:TOP - 1]}
    ctx, strategy = start_run(feed([(thin, JAN)]))
    drive(strategy, ctx, WINDOW + 1)
    assert ctx.orders == []


def test_partial_window_names_are_not_ranked():
    # A latecomer that would out-rank everything, but has only three bars.
    bars = feed([({s: STEPS[s] for s in RANKED_DESC[:TOP]}, JAN)])
    bars += feed([({"NEW": 50.0}, JAN[-3:])])
    bars.sort(key=lambda b: b.timestamp)
    ctx, strategy = start_run(bars)
    drive(strategy, ctx, WINDOW + 1)

    assert "NEW" not in {o.symbol for o in buys(ctx)}
    assert len(buys(ctx)) == TOP


def test_corporate_action_sized_move_is_booked_as_zero_return():
    # A flat name that gets a 3x "gain" from a bonus issue on the last bar: on
    # raw close ratios it ranks first, on cleaned returns it does not rank at all.
    bars = feed([({s: STEPS[s] for s in RANKED_DESC[:TOP]}, JAN)])
    split = feed([({"SPLIT": 0.0}, JAN)])
    split[-1] = FakeKLine(JAN[-1], "SPLIT", 300.0, 300.0, 300.0, 300.0, 1_000.0)
    ctx, strategy = start_run(sorted(bars + split, key=lambda b: b.timestamp))
    drive(strategy, ctx, WINDOW + 1)

    assert "SPLIT" not in {o.symbol for o in buys(ctx)}


def test_holds_the_book_between_month_boundaries():
    # JAN ends on 2025-01-30; one more January bar must not re-decide anything,
    # even though the ranking would have rotated by then.
    ctx, strategy = start_run(feed([(STEPS, JAN), (JUMP, sessions(1, "2025-01-31"))]))
    drive(strategy, ctx, WINDOW + 1)
    for order in buys(ctx):
        fill(ctx, order)
    placed = len(ctx.orders)

    drive(strategy, ctx, 1)
    assert len(ctx.orders) == placed


def test_month_change_exits_dropouts_and_defers_entries_one_bar():
    # The first February bar reverses the ranking, so part of the book rotates.
    ctx, strategy = start_run(feed([(STEPS, JAN), (JUMP, FEB)]))
    drive(strategy, ctx, WINDOW + 1)
    for order in buys(ctx):
        fill(ctx, order)
    held = {o.symbol for o in buys(ctx)}
    before = len(ctx.orders)

    drive(strategy, ctx, 1)          # first February bar: exits only
    exits = sells(ctx)
    assert exits, "a month change with a rotated ranking must close the dropouts"
    assert all(o.reduce_only and o.symbol in held for o in exits)
    assert [o for o in ctx.orders[before:] if o.side == OrderSide.Buy] == []

    for order in exits:
        fill(ctx, order)
    before = len(ctx.orders)
    drive(strategy, ctx, 1)          # next bar: the entries, now funded
    assert [o for o in ctx.orders[before:] if o.side == OrderSide.Buy]


def test_exit_quantity_matches_the_held_position():
    ctx, strategy = start_run(feed([(STEPS, JAN), (JUMP, FEB)]))
    drive(strategy, ctx, WINDOW + 1)
    for order in buys(ctx):
        fill(ctx, order)
    drive(strategy, ctx, 1)

    assert sells(ctx)
    for order in sells(ctx):
        assert order.quantity == pytest.approx(abs(ctx.positions[order.symbol].quantity))


def test_a_liquidated_position_is_not_re_sold():
    ctx, strategy = start_run(feed([(STEPS, JAN), (JUMP, FEB)]))
    drive(strategy, ctx, WINDOW + 1)
    for order in buys(ctx):
        fill(ctx, order)
    ctx.positions.clear()            # the broker closed everything behind our back
    drive(strategy, ctx, 1)

    assert sells(ctx) == []


def test_params_are_declared_for_the_gui():
    names = {spec["name"] for spec in stonks.param_specs(ShortTermMomentumStrategy)}
    assert names == {"window", "top", "price_limit_pct", "cash_buffer"}
