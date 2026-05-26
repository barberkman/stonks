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
        const auto bars = context.klines(20);
        if (bars.size() < 20) return;
        // decide and place orders here
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
| `klines(int count)` | `std::vector<KLine>` | last N bars, clamped to `now()` |
| `klines(Timestamp start, std::optional<Timestamp> end = {})` | `std::vector<KLine>` | range query, clamped to `now()` |
| `make_market_order(MarketOrderParams)` | `Order` | builds the order — does **not** submit it |
| `make_limit_order(LimitOrderParams)` | `Order` | builds the order — does **not** submit it |
| `place_order(const Order&)` | `bool` | submits a built order to the broker |

Order placement is two-step: `make_*_order(...)` builds the `Order`, then `place_order(order)` submits it.

## Types — the ones strategies use

From `include/stonks/core/types.h`, all in `namespace stonks::core`:

- Scalars: `Price`, `Volume`, `Balance`, `Quantity`, `Symbol`, `OrderID`, `TradeID`.
- `Timestamp` — has `operator<=>`, arithmetic, and `Timestamp::from_millis(...)`.
- `KLine { Timestamp timestamp; Symbol symbol; Price open, high, low, close; Volume volume; }`.
- `Order` — opaque from the strategy's side after `make_*_order` returns it.
- Enums: `OrderSide::{ Buy, Sell }`, `OrderType::{ Market, Limit }`, `TimeInForce::{ GTC }`.
- Param structs:
  - `MarketOrderParams { Symbol symbol; OrderSide side; Quantity quantity; TimeInForce time_in_force; }`
  - `LimitOrderParams { Symbol symbol; OrderSide side; Quantity quantity; Price price; TimeInForce time_in_force; }`

Example:

```cpp
const auto order = context.make_market_order(stonks::core::MarketOrderParams{
    .symbol = "BTCUSDT",
    .side = stonks::core::OrderSide::Buy,
    .quantity = 0.01,
    .time_in_force = stonks::core::TimeInForce::GTC,
});
context.place_order(order);
```

## Fill mechanics & no-lookahead

- **No lookahead.** `context.klines(...)` is automatically clamped to `now()`. You cannot see future bars.
- **Market orders** fill at the next bar's close price when the broker ticks.
- **Limit orders** fill only if the limit price falls within the bar's `[low, high]` range.

## Style & naming

The root `/home/baris/stonks/CLAUDE.md` is the source of truth. The points that bite strategy authors most:

- **Filenames: no underscores between words.** `meanreversion.h`, not `mean_reversion.h`.
- **Brace-init with inner spaces:** `Order{ ... }`, `MarketOrderParams{ ... }` — not `Order(...)` or `Order{...}`.
- **Acronyms uppercase:** `OrderID`, not `OrderId`. (`KLine` is title-case — the K is not an acronym.)
- C++20; library types are namespaced as `stonks::core::`, `stonks::broker::`, `stonks::datafeed::`.

See `placeholderstrategy.h` in this folder for a live example.

## After writing a strategy

Tell the user something like:

> Created `app/strategies/<name>.h`. The user wires this into `app/main.cpp` themselves — I have not touched that file or anything else outside `app/strategies/`.
