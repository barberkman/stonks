# `app/strategies/` — strategy authoring

This folder holds trading strategies for the `stonks` backtest runner. This file tells you how to write one.

## Scope — read this first

**Hard rule: never create, edit, move, or delete any file outside `app/strategies/` (this folder). No exceptions, ever — not even if the user asks in passing, not even one-line "trivial" edits, not even to fix a build error you caused.**

Inside this folder you may only create or edit `.h` files. No `.cpp` files, no `CMakeLists.txt`, no tests in here either.

Off-limits, explicitly:

- `app/main.cpp` — the user wires strategies in themselves. Never touch it.
- `include/`, `src/`, `tests/`, `cmake/`, `tools/`, `app/data/` — entirely off-limits.
- Any `CMakeLists.txt` anywhere — strategies are header-only and need no build wiring.

If a task seems to require editing anything outside this folder — wiring, a missing library feature, a bug in the broker, anything — **stop, do nothing, and tell the user.** They will handle it. Do not "just this once" reach out of the folder.

## What a strategy is

A bare `struct` — no base class, no virtuals. It satisfies the `stonks::core::Strategy` concept by exposing up to three lifecycle methods:

- `on_start(auto& context)` — once, before the first bar. Optional.
- `on_tick(auto& context)` — once per **timestamp** (all symbols printing at that time arrive in one tick). **Required.**
- `on_stop(auto& context)` — once, after the last bar. Optional.

`context` is templated and duck-typed, so always take it as `auto&`. Store any per-strategy state as member variables on the struct.

Execution timing contract: on each tick the broker settles first (resting
orders fill against this bar, bracket children armed by a parent filling this
bar can fill this bar too, then liquidation is checked), *then* `on_tick`
runs — so `history(n)` already includes this bar's close and
`position()`/`order()` reflect every fill through it. Orders you place are
stamped with the current time and become eligible from the **next** bar,
never this one. Decide on the close, execute from the next bar.

## Skeleton

```cpp
#pragma once

struct MyStrategy
{
    void on_start(auto& context)
    {
        // one-time setup, e.g. read starting cash
    }

    void on_tick(auto& context)
    {
        // One tick per timestamp: loop the symbols that printed today.
        for (const auto& s : context.history(20).series) {
            if (s.bars.size() < 20) continue;       // s.bars is this symbol's last 20 bars
            const double last_close = s.bars.close.back();
            // decide and place orders for s.symbol here
        }
    }

    void on_stop(auto& context)
    {
        // one-time teardown
    }
};
```

## `Context` API — everything you can call

From `include/stonks/core/context.h`:

| Method | Returns | Notes |
| --- | --- | --- |
| `now()` | `Timestamp` | current simulation time |
| `cash()` | `Balance` | available cash |
| `equity()` | `Balance` | total portfolio value (cash + margin + unrealized P&L) |
| `history(int count)` | `MarketWindow` | every symbol that printed at this timestamp, each with its last N bars (incl. today's) as column views; clamped to bars seen so far |
| `place_order(MarketOrderParams, parent = nullopt)` | `OrderID` | fills at the next bar's open |
| `place_order(LimitOrderParams, parent = nullopt)` | `OrderID` | rests until a bar's range reaches the price; fills at the price or better |
| `place_order(StopOrderParams, parent = nullopt)` | `OrderID` | dormant until the trigger is touched; fills at the trigger or worse (gap ⇒ the open). Use for stop-losses and breakout entries |
| `position(const Symbol&)` | `optional<Position>` | the open position (`quantity` signed, `price` = entry, `leverage`), or nullopt when flat |
| `order(OrderID)` | `optional<Order>` | a status snapshot (`Open` / `Filled` / `Rejected` / `Cancelled`); nullopt for unknown ids |
| `cancel_order(OrderID)` | `bool` | cancels a still-open order and its dormant bracket children |

Order placement is one call: `context.place_order(SomeParams{ ... })`. The broker stamps the order with the current time, queues it, and returns its `OrderID` — you never build or hold an `Order` yourself. **The OrderID is not a success flag**: rejected orders get ids too. Check `order(id)->status` on a later tick, or `position(symbol)`, to learn what happened.

Pass `parent = <entry's OrderID>` to chain an order as a bracket child: it stays dormant until the parent fills, becomes eligible from the parent's fill bar onward, and is OCO-cancelled when the position goes flat. Set `.reduce_only = true` on protective legs so an orphaned leg can only shrink a position, never open one. A bracket looks like:

```cpp
const auto entry = context.place_order(stonks::core::StopOrderParams{
    .symbol = sym,
    .side = stonks::core::OrderSide::Buy,
    .quantity = qty,
    .price = pivot,             // breakout entry at the signal level
    .leverage = lev,
});
context.place_order(stonks::core::StopOrderParams{
    .symbol = sym,
    .side = stonks::core::OrderSide::Sell,
    .quantity = qty,
    .price = stop,              // protective stop, below the pivot
    .reduce_only = true,
}, entry);
context.place_order(stonks::core::LimitOrderParams{
    .symbol = sym,
    .side = stonks::core::OrderSide::Sell,
    .quantity = qty,
    .price = target,            // take-profit
    .reduce_only = true,
}, entry);
```

## Types — the ones strategies use

From `include/stonks/core/types.h`, all in `namespace stonks::core`:

- Scalars: `Price`, `Volume`, `Balance`, `Quantity`, `Symbol`, `SymbolID`, `OrderID`, `TradeID`.
- `Timestamp` — has `operator<=>`, arithmetic, and `Timestamp::from_millis(...)`.
- `KLine { Timestamp timestamp; Symbol symbol; Price open, high, low, close; Volume volume; }`.
- `SeriesView { std::span<const std::int64_t> timestamp; std::span<const double> open, high, low, close, volume; }` — one symbol's column views into feed storage.
- `SymbolSeries { std::string_view symbol; SeriesView bars; }` and `MarketWindow { std::vector<SymbolSeries> series; }` — returned by `history()`, one `SymbolSeries` per symbol that printed this tick. Re-query each tick; don't cache a view.
- `Position { Quantity quantity; Price price; OrderID entry_id; double leverage; }` — returned by `position()`; `quantity` is signed (+long / −short), `price` is the actual entry fill.
- Strategies never construct or hold an `Order` — you pass the param structs below and get back an `OrderID`. `Order` snapshots come back from `order(id)` for status checks (`parent_id`, `status`, `reduce_only`, etc.).
- Enums: `OrderSide::{ Buy, Sell }`, `OrderType::{ Market, Limit, Stop }`, `OrderStatus::{ Open, Filled, Rejected, Cancelled }`, `TimeInForce::{ GTC }` (stored but inert — everything is GTC).
- Param structs (trailing fields default, so you can omit them):
  - `MarketOrderParams { Symbol symbol; OrderSide side; Quantity quantity; TimeInForce time_in_force = GTC; double leverage = 1.0; bool reduce_only = false; }`
  - `LimitOrderParams { Symbol symbol; OrderSide side; Quantity quantity; Price price; TimeInForce time_in_force = GTC; double leverage = 1.0; bool reduce_only = false; }`
  - `StopOrderParams { Symbol symbol; OrderSide side; Quantity quantity; Price price /*trigger*/; TimeInForce time_in_force = GTC; double leverage = 1.0; bool reduce_only = false; }`
- `leverage` matters only on the order that **opens** a position (isolated margin divisor); it is ignored on closing legs.

## Fill mechanics & no-lookahead

- **No lookahead.** `context.history(n)` returns only the symbols that printed at `now()`, each with its own bars up to and including `now()`. You cannot see future bars, and an order placed on a bar never fills against that same bar. Bracket children are the one refinement: once their parent fills they are eligible from the parent's fill bar onward, so a stop can protect the entry bar itself.
- **Market orders** fill at the **next bar's open** (never the bar the order was placed on).
- **Limit orders** fill only when a bar's range reaches the limit: a buy fills at `min(limit, open)` once `low <= limit`; a sell at `max(limit, open)` once `high >= limit`. Until then the order rests. Never worse than the limit.
- **Stop orders** stay dormant until the trigger is touched (`high >= trigger` for buys, `low <= trigger` for sells), then fill at the trigger or worse — a bar gapping through the trigger fills at the open. A "stop" written as a limit on the wrong side of the market fills immediately instead of waiting; always use `StopOrderParams` for stops.
- **Same-bar ties resolve by policy, not placement order**: markets, then stops, then limits (the default Conservative policy — protective stops beat profit targets on a double-touch bar).
- **Margin-collateralized, one position per symbol.** Opening posts `quantity * fill_price / leverage` of cash plus the entry fee; an unaffordable (or below-min-notional) order is **rejected** — it does not wait. While you hold a position, a **same-side** order is **rejected** (no adding — close first); an opposite-side order closes it (partly or fully), clamping an oversized close and never flipping. **Shorts are allowed.** Dust left by partial closes snaps to exactly flat.
- **Liquidation.** A position is force-closed once a bar's adverse extreme reaches `entry * (1 ∓ 1/leverage) / (1 ∓ m)` (`m` = configured maintenance-margin rate). A resting stop that fills on the crash bar pre-empts the liquidation. Because the *actual* fill anchors this price, a gapped entry can drag liquidation inside your stop — compare `position()->price` to your plan and re-anchor the legs (see `qmsignals.h`).
- **Fees** are charged per fill (maker for a limit filled at its own price, taker for everything that crosses; plus an optional flat amount) per the run's `BrokerConfig` — strategies don't control them, but sizing should assume they exist.

## Style & naming

The repo root `CLAUDE.md` is the source of truth. The points that bite strategy authors most:

- **Filenames: no underscores between words.** `meanreversion.h`, not `mean_reversion.h`.
- **Brace-init with inner spaces:** `Order{ ... }`, `MarketOrderParams{ ... }` — not `Order(...)` or `Order{...}`.
- **Acronyms uppercase:** `OrderID`, not `OrderId`. (`KLine` is title-case — the K is not an acronym.)
- C++20; library types are namespaced as `stonks::core::`, `stonks::broker::`, `stonks::datafeed::`.

See `ema50strategy.h` in this folder for a minimal fire-and-forget example, and `qmsignals.h` for the full managed pattern: brackets, one-trade-at-a-time gating with a cooldown, stale-entry replacement, and gap re-anchoring off `position()->price`.

Python-authored strategies use `pythonstrategy.h`, which loads a user-supplied Python class by module path. Sample strategies live in `app/python/`; the venv used at runtime is `app/python/.venv`. See `app/python/README.md` for the Python-side authoring story (same broker, same semantics, including a complete managed-bracket example). GUI-editable parameter declarations (`params` + `stonks.Param`) are a Python-only feature — C++ strategies keep compile-time fields set in `app/main.cpp`.

## After writing a strategy

Tell the user something like:

> Created `app/strategies/<name>.h`. The user wires this into `app/main.cpp` themselves — I have not touched that file or anything else outside `app/strategies/`.
