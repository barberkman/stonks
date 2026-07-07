"""Shared pytest helpers for the app/python strategy suite.

Not a strategy file: discover_strategies() imports it, finds no Strategy
subclass, and skips it silently — it never shows up in the GUI list.

Three layers, matching how the tests use FakeContext:

  * bar builders — `bars_from` for hand-crafted behavior scenarios and
    `make_bars` for the smoke sweep's synthetic regimes (trending / choppy /
    gappy). Timestamps default to 4h spacing so UTC-day session logic (qmorb)
    is exercised: six bars per day.
  * run/fill helpers — advance-per-timestamp driving plus explicit
    `fill_entry` / `fill_exit` calls for behavior tests that need to walk a
    trade's lifecycle (FakeContext never fills orders on its own).
  * `run_with_broker` — the smoke sweep's driver: a deliberately minimal
    broker sim (`settle`) fills resting orders against each tick's bar BEFORE
    on_tick runs, mirroring the engine's settle-first timeline, so the
    strategies' in-position management paths actually execute.
"""

from typing import Dict, List, Optional

import numpy as np

from stonks import OrderSide, OrderStatus
from stonks.testing import FakeContext, FakeKLine, FakePosition

MS4H = 4 * 3_600_000
MS_PER_DAY = 86_400_000


def _ms(ts) -> int:
    if hasattr(ts, "to_millis"):
        return int(ts.to_millis())
    return int(ts)


# ─── Bar builders ─────────────────────────────────────────────────────────────

def bars_from(symbol: str, rows, start_ts: int = MS4H, interval_ms: int = MS4H) -> List[FakeKLine]:
    """Hand-crafted scenario bars: rows of (open, high, low, close, volume)."""
    return [
        FakeKLine(start_ts + i * interval_ms, symbol,
                  float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        for i, r in enumerate(rows)
    ]


def make_bars(symbols: List[str], n: int, regime: str, interval_ms: int = MS4H,
              seed: int = 0, direction: int = 1, start_price: float = 100.0) -> List[FakeKLine]:
    """Deterministic multi-symbol OHLCV in one of three synthetic regimes.

    trending — geometric drift (direction=+1 up / -1 down) with noise;
    choppy   — mean reversion around a slowly oscillating base (bases and
               boxes repeatedly form and fail);
    gappy    — trending noise plus a large open-vs-prior-close gap every 12
               bars (exercises gap/EP gates and gapped-fill edge cases).
    """
    bars: List[FakeKLine] = []
    for k, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + 7919 * k)
        base = start_price * (1.0 + 0.1 * k)
        prev_close = base
        for i in range(n):
            ts = (i + 1) * interval_ms
            open_ = prev_close
            if regime == "gappy" and i % 12 == 5:
                open_ = prev_close * (1.0 + float(rng.choice((-1.0, 1.0))) * rng.uniform(0.02, 0.05))
            noise = rng.normal(0.0, 0.005)
            if regime == "choppy":
                target = base * (1.0 + 0.06 * np.sin(i / 7.0))
                close = open_ + 0.4 * (target - open_) + open_ * noise
            else:
                close = open_ * (1.0 + 0.004 * direction + noise)
            close = max(float(close), 0.5)
            hi = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.004)))
            lo = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.004)))
            vol = 1_000.0 * (1.0 + abs(rng.normal(0.0, 0.3)))
            if rng.uniform() < 0.1:
                vol *= 3.0
            bars.append(FakeKLine(ts, sym, float(open_), float(hi), float(lo), close, float(vol)))
            prev_close = close
    bars.sort(key=lambda b: (_ms(b.timestamp), b.symbol))
    return bars


# ─── Run helpers ──────────────────────────────────────────────────────────────

def n_stamps(bars: List[FakeKLine]) -> int:
    return len({_ms(b.timestamp) for b in bars})


def start_run(strategy, bars: List[FakeKLine], cash: float = 100_000.0) -> FakeContext:
    ctx = FakeContext(bars, cash)
    strategy.on_start(ctx)
    return ctx


def tick(ctx: FakeContext, strategy, n: int = 1) -> None:
    for _ in range(n):
        ctx.advance()
        strategy.on_tick(ctx)


def run_all(strategy, bars: List[FakeKLine], cash: float = 100_000.0) -> FakeContext:
    ctx = start_run(strategy, bars, cash)
    tick(ctx, strategy, n_stamps(bars))
    return ctx


# ─── Order lookups for assertions ─────────────────────────────────────────────

def entry_orders(ctx: FakeContext, symbol: Optional[str] = None):
    """Orders that open a trade: unparented and not reduce-only."""
    return [o for o in ctx.orders
            if o.parent is None and not o.reduce_only
            and (symbol is None or o.symbol == symbol)]


def children_of(ctx: FakeContext, parent_id: int):
    return [o for o in ctx.orders if o.parent == parent_id]


# ─── Explicit fill simulation (behavior tests) ────────────────────────────────

def fill_entry(ctx: FakeContext, order, price: Optional[float] = None) -> FakePosition:
    """Broker fills an entry: status -> Filled, position opened at `price`
    (defaults to the order's own price — pass one for market orders/gaps)."""
    px = order.price if price is None else price
    assert px is not None, "market entries need an explicit fill price"
    signed = order.quantity if order.side == OrderSide.Buy else -order.quantity
    order.status = OrderStatus.Filled
    pos = FakePosition(quantity=signed, price=float(px), entry_id=order.id)
    ctx.positions[order.symbol] = pos
    return pos


def fill_exit(ctx: FakeContext, order, price: Optional[float] = None) -> None:
    """Broker fills a protective/exit order: shrink the position (clamped to
    what is held) and, once flat, cancel every other open order on the symbol
    — the broker's OCO subtree cancellation, coarsely."""
    pos = ctx.positions.get(order.symbol)
    assert pos is not None, "fill_exit needs an open position"
    order.status = OrderStatus.Filled
    qty = min(order.quantity, abs(pos.quantity))
    remaining = abs(pos.quantity) - qty
    if remaining <= 1e-9 * max(1.0, abs(pos.quantity)):
        del ctx.positions[order.symbol]
        for o in ctx.orders:
            if o.symbol == order.symbol and o.status == OrderStatus.Open:
                o.status = OrderStatus.Cancelled
    else:
        sign = 1.0 if pos.quantity > 0.0 else -1.0
        pos.quantity = sign * remaining


# ─── Minimal broker sim for the smoke sweep ───────────────────────────────────

_FILL_PRIORITY = {"market": 0, "stop": 1, "limit": 2}


def settle(ctx: FakeContext, bars_now: Dict[str, FakeKLine], watermark: int) -> None:
    """Fill pre-existing open orders (id <= watermark) against this tick's
    bars. Coarse mirror of BacktestBroker: market @ open; stop triggers on
    touch, fills at trigger-or-worse; limit fills at limit-or-better; children
    dormant until their parent is Filled; same-side adds Rejected; opposite
    fills clamp to the held quantity; a flat symbol cancels its remaining
    open orders (subtree OCO, coarsely). Rounds repeat until no progress so an
    entry fill can arm and fire its children within the same bar."""
    progressed = True
    while progressed:
        progressed = False
        candidates = sorted(
            (o for o in ctx.orders if o.id <= watermark and o.status == OrderStatus.Open),
            key=lambda o: (_FILL_PRIORITY[o.order_type], o.id))
        for o in candidates:
            if o.status != OrderStatus.Open:
                continue
            bar = bars_now.get(o.symbol)
            if bar is None:
                continue
            if o.parent is not None:
                parent = ctx.order(o.parent)
                if parent is None or parent.status != OrderStatus.Filled:
                    continue    # dormant child
            if o.order_type == "market":
                px = bar.open
            elif o.order_type == "stop":
                if o.side == OrderSide.Buy:
                    if bar.high < o.price:
                        continue
                    px = max(o.price, bar.open)
                else:
                    if bar.low > o.price:
                        continue
                    px = min(o.price, bar.open)
            else:   # limit
                if o.side == OrderSide.Buy:
                    if bar.low > o.price:
                        continue
                    px = min(o.price, bar.open)
                else:
                    if bar.high < o.price:
                        continue
                    px = max(o.price, bar.open)

            pos = ctx.positions.get(o.symbol)
            if pos is None:
                if o.reduce_only:
                    o.status = OrderStatus.Cancelled
                    progressed = True
                    continue
                signed = o.quantity if o.side == OrderSide.Buy else -o.quantity
                ctx.positions[o.symbol] = FakePosition(quantity=signed, price=float(px),
                                                       entry_id=o.id)
                o.status = OrderStatus.Filled
                progressed = True
            else:
                same_side = (pos.quantity > 0.0) == (o.side == OrderSide.Buy)
                if same_side:
                    o.status = OrderStatus.Rejected
                    progressed = True
                    continue
                o.status = OrderStatus.Filled
                progressed = True
                qty = min(o.quantity, abs(pos.quantity))
                remaining = abs(pos.quantity) - qty
                if remaining <= 1e-9 * max(1.0, abs(pos.quantity)):
                    del ctx.positions[o.symbol]
                    for other in ctx.orders:
                        if other.symbol == o.symbol and other.status == OrderStatus.Open:
                            other.status = OrderStatus.Cancelled
                else:
                    sign = 1.0 if pos.quantity > 0.0 else -1.0
                    pos.quantity = sign * remaining


def run_with_broker(strategy, bars: List[FakeKLine], cash: float = 100_000.0,
                    after_tick=None) -> FakeContext:
    """Drive a strategy over `bars` with the settle() mini-broker applied
    before each on_tick — the engine's timeline (settle, then strategy)."""
    ctx = FakeContext(bars, cash)
    strategy.on_start(ctx)
    by_ts: Dict[int, Dict[str, FakeKLine]] = {}
    for b in bars:
        by_ts.setdefault(_ms(b.timestamp), {})[b.symbol] = b
    watermark = 0
    for ts in sorted(by_ts):
        ctx.advance()
        settle(ctx, by_ts[ts], watermark)
        strategy.on_tick(ctx)
        watermark = max((o.id for o in ctx.orders), default=0)
        if after_tick is not None:
            after_tick(ctx, ts, by_ts[ts])
    return ctx
