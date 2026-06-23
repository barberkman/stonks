# `app/strategies/` — strategy authoring

This folder holds trading strategies for the `stonks` backtest runner. This file tells you how to write one.

## Scope — read this first

**Hard rule: never create, edit, move, or delete any file outside `/home/baris/stonks/app/strategies/`. No exceptions, ever — not even if the user asks in passing, not even one-line "trivial" edits, not even to fix a build error you caused.**

Inside this folder you may only create or edit `.h` files. No `.cpp` files, no `CMakeLists.txt`, no tests in here either.

Off-limits, explicitly:

- `app/main.cpp` — the user wires strategies in themselves. Never touch it.
- `include/`, `src/`, `tests/`, `cmake/`, `apps/`, `app/data/` — entirely off-limits.
- Any `CMakeLists.txt` anywhere — strategies are header-only and need no build wiring.

If a task seems to require editing anything outside this folder — wiring, a missing library feature, a bug in the broker, anything — **stop, do nothing, and tell the user.** They will handle it. Do not "just this once" reach out of the folder.

## What a strategy is

A bare `struct` — no base class, no virtuals. It satisfies the `stonks::core::Strategy` concept by exposing up to three lifecycle methods:

- `on_start(auto& context)` — once, before the first bar. Optional.
- `on_tick(auto& context)` — once per bar. **Required.**
- `on_stop(auto& context)` — once, after the last bar. Optional.

`context` is templated and duck-typed, so always take it as `auto&`. Store any per-strategy state as member variables on the struct.

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
| `equity()` | `Balance` | total portfolio value (cash + positions) |
| `history(int count)` | `MarketWindow` | every symbol that printed at this timestamp, each with its last N bars (incl. today's) as column views; clamped to bars seen so far |
| `place_order(MarketOrderParams)` | `OrderID` | submit a market order to the broker (one call) |
| `place_order(LimitOrderParams)` | `OrderID` | submit a limit order to the broker (one call) |

Order placement is one call: `context.place_order(MarketOrderParams{ ... })` or `context.place_order(LimitOrderParams{ ... })`. The broker stamps the order with the current time, queues it, and returns its `OrderID` — you never build or hold an `Order` yourself.

## Types — the ones strategies use

From `include/stonks/core/types.h`, all in `namespace stonks::core`:

- Scalars: `Price`, `Volume`, `Balance`, `Quantity`, `Symbol`, `SymbolID`, `OrderID`, `TradeID`.
- `Timestamp` — has `operator<=>`, arithmetic, and `Timestamp::from_millis(...)`.
- `KLine { Timestamp timestamp; Symbol symbol; Price open, high, low, close; Volume volume; }`.
- `SeriesView { std::span<const std::int64_t> timestamp; std::span<const double> open, high, low, close, volume; }` — one symbol's column views into feed storage.
- `SymbolSeries { std::string_view symbol; SeriesView bars; }` and `MarketWindow { std::vector<SymbolSeries> series; }` — returned by `history()`, one `SymbolSeries` per symbol that printed this tick. Re-query each tick; don't cache a view.
- Strategies never construct or hold an `Order` — you pass the param structs below and get back an `OrderID`. `OrderStatus::{ Open, Filled, Rejected }` is an order's lifecycle state (inspected via the broker, not the strategy).
- Enums: `OrderSide::{ Buy, Sell }`, `OrderType::{ Market, Limit }`, `OrderStatus::{ Open, Filled, Rejected }`, `TimeInForce::{ GTC }`.
- Param structs (`time_in_force` defaults to `GTC`, so you can omit it):
  - `MarketOrderParams { Symbol symbol; OrderSide side; Quantity quantity; TimeInForce time_in_force = GTC; }`
  - `LimitOrderParams { Symbol symbol; OrderSide side; Quantity quantity; Price price; TimeInForce time_in_force = GTC; }`

Example:

```cpp
const stonks::core::OrderID id = context.place_order(stonks::core::MarketOrderParams{
    .symbol = "BTCUSDT",
    .side = stonks::core::OrderSide::Buy,
    .quantity = 0.01,
});
// The OrderID is rarely needed — most strategies ignore the return value.
```

## Fill mechanics & no-lookahead

- **No lookahead.** `context.history(n)` returns only the symbols that printed at `now()`, each with its own bars up to and including `now()`. You cannot see future bars, and an order placed on a bar never fills against that same bar.
- **Market orders** fill at the **next bar's open** (never the bar the order was placed on).
- **Limit orders** fill only when a later bar's range reaches the limit: a buy fills at `min(limit, open)` once `low <= limit`; a sell at `max(limit, open)` once `high >= limit`. Until then the order rests.
- **Cash-secured, one position per symbol.** Opening ties up `quantity * fill_price` of cash; an order you can't afford is **rejected** (it does not wait for cash to free up). While you hold a position in a symbol, a **same-side** order on it is **rejected** — there is no adding to a position, so close it first. An opposite-side order closes it (partly or fully); an oversized close clamps to what you hold and never flips. **Shorts are allowed** — selling with no position opens a short, cash-secured the same way a long is.

## Style & naming

The root `/home/baris/stonks/CLAUDE.md` is the source of truth. The points that bite strategy authors most:

- **Filenames: no underscores between words.** `meanreversion.h`, not `mean_reversion.h`.
- **Brace-init with inner spaces:** `Order{ ... }`, `MarketOrderParams{ ... }` — not `Order(...)` or `Order{...}`.
- **Acronyms uppercase:** `OrderID`, not `OrderId`. (`KLine` is title-case — the K is not an acronym.)
- C++20; library types are namespaced as `stonks::core::`, `stonks::broker::`, `stonks::datafeed::`.

See `ema50strategy.h` in this folder for a live example.

Python-authored strategies use `pythonstrategy.h`, which loads a user-supplied Python class by module path. Sample strategies live in `/home/baris/stonks/app/python/`; the venv used at runtime is `/home/baris/stonks/app/python/.venv`. See `/home/baris/stonks/app/python/README.md` for the Python-side authoring story.

## After writing a strategy

Tell the user something like:

> Created `app/strategies/<name>.h`. The user wires this into `app/main.cpp` themselves — I have not touched that file or anything else outside `app/strategies/`.
