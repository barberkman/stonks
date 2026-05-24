# Backtesting & Trading Engine — Core Design

A high-level design for the C++20 core of an event-driven, realistic, high-performance
backtesting and live-trading engine. Strategies are C++ for now; the design keeps the
later Python port cheap (a Python adapter just becomes one more type that satisfies the
strategy *concept*).

This document covers the **core parts only**: value types, market/order events, the
merged-timeline scheduler, the strategy interface, the `Context` broker handle, the
`FillModel`, the `Portfolio`, and the `Engine` loop. Things deliberately left as
extension points are listed at the end.

---

## 1. Architecture overview

Four layers with narrow interfaces between them:

```
  ┌─────────────┐   feeds market data    ┌──────────────────────────────┐
  │  Data layer │ ─────────────────────► │          Engine core         │
  │  DataFeed   │                        │  EventScheduler (merge)      │
  └─────────────┘                        │  FillModel  (matching)       │
                                         │  Portfolio  (accounting)     │
  ┌─────────────┐   on_bar/on_tick/...   │  Engine     (the loop)       │
  │  Strategy   │ ◄───────────────────►  │                              │
  │  (C++)      │   ctx.buy()/position() └──────────────────────────────┘
  └─────────────┘                                      │ equity curve, fills
                                                        ▼
                                              ┌──────────────────┐
                                              │  Analysis layer  │
                                              │  metrics, report │
                                              └──────────────────┘
```

Two design rules drive everything:

1. **Backtest == live.** Strategy code never knows which mode it is in. Only the
   *backends* swap: `HistoricalDataFeed` vs `LiveDataFeed`, `SimulatedFillModel` vs
   `LiveBroker`. The `Context` handed to the strategy is the single swap point.
2. **No lookahead, by construction.** The strategy can only ever touch data the engine
   has already pushed to it. It holds no dataset, cannot seek, and any order it submits
   carries a `submit_time`; the matching engine refuses to fill an order against data
   whose timestamp is not strictly greater than that `submit_time`.

### The per-event loop

Every event is processed in a fixed order. The order is what makes the engine
lookahead-free:

1. **Match** — resting orders (submitted at some `T' < T`) are matched against the new
   event at time `T`. Fills are applied to the portfolio.
2. **Notify fills** — strategy's `on_fill` is called for each fill.
3. **Mark** — last prices updated for mark-to-market.
4. **Dispatch** — the strategy sees the event via its typed callback and may enqueue new
   orders. Those orders get `submit_time = T`.
5. **Activate** — newly enqueued orders move into the open-order set. They are *not*
   matched against the current event (its time equals their `submit_time`, not greater).
6. **Record** — an equity-curve point is recorded at `T`.

A strategy therefore physically cannot act on bar `T` and fill at bar `T` — its order
fills against bar `T+1` at the earliest.

### Performance model

- **Single-threaded, deterministic engine.** Parallelism comes from running *many*
  backtests at once (parameter sweeps, walk-forward), each engine single-threaded with
  its own arena. Near-linear scaling, zero locks in the hot path.
- **No per-event heap allocation.** Events are POD types carried in a `std::variant`;
  the pending-order and fill buffers are `std::vector`s that are `clear()`-ed (capacity
  retained), never reallocated per event.
- **Static dispatch to the strategy.** The `Engine` is templated on the strategy type
  and constrained by concepts, so strategy callbacks inline into the loop. No virtual
  call on the hottest boundary.
- **Integer money and prices.** Timestamps are `int64` nanoseconds; prices are integer
  ticks; money is integer micro-units. Backtests are bit-reproducible.

---

## 2. Core value types

Strong types prevent unit mix-ups; defaulted `operator<=>` gives ordering for free.

```cpp
// core/types.hpp
#pragma once
#include <cstdint>
#include <compare>
#include <string>

struct Timestamp {                        // nanoseconds since epoch
    std::int64_t ns{};
    auto operator<=>(const Timestamp&) const = default;
};

struct Price {                            // integer number of ticks
    std::int64_t ticks{};
    auto operator<=>(const Price&) const = default;
};

struct Money {                            // integer micro-units of account currency
    std::int64_t micros{};
    auto operator<=>(const Money&) const = default;
    Money& operator+=(Money o) { micros += o.micros; return *this; }
    Money& operator-=(Money o) { micros -= o.micros; return *this; }
};
inline Money operator+(Money a, Money b) { return {a.micros + b.micros}; }
inline Money operator-(Money a, Money b) { return {a.micros - b.micros}; }

using Quantity = std::int64_t;            // scaled integer (1 for stocks; sub-units for crypto)
using SymbolId = std::uint32_t;           // dense handle — never a string in the hot path
using OrderId  = std::uint64_t;

enum class Side        : std::uint8_t { Buy, Sell };
enum class OrderType   : std::uint8_t { Market, Limit };
enum class TimeInForce : std::uint8_t { Day, GTC, IOC, FOK };
enum class AssetClass  : std::uint8_t { Equity, Crypto, Future, FX };
```

An `Instrument` carries everything needed to turn ticks into money, so the same engine
handles stocks, crypto and futures:

```cpp
// core/instrument.hpp
struct Instrument {
    SymbolId    id{};
    std::string symbol;                   // "AAPL", "BTC-USD"
    AssetClass  asset_class{};
    std::int64_t tick_value_micros{};     // money-micros that one Price tick is worth, per unit
    Quantity     min_qty{1};              // smallest tradable increment (scaled)
};

// Notional value of `qty` units (signed) traded at `price`.
inline Money notional(const Instrument& ins, Price price, Quantity qty) {
    return Money{ price.ticks * ins.tick_value_micros * qty };
}

class InstrumentTable {
public:
    SymbolId add(Instrument i) { i.id = next_++; table_.push_back(std::move(i)); return table_.back().id; }
    const Instrument& at(SymbolId id) const { return table_[id]; }
private:
    std::vector<Instrument> table_;
    SymbolId next_{0};
};
```

---

## 3. Market events

POD structs, one per data shape. A bar's `close_time` is the strategy's notion of "now"
when it receives that bar — the bar is delivered only once it is *closed*.

```cpp
// core/market_event.hpp
#include <variant>

struct Bar {
    SymbolId  symbol{};
    Timestamp open_time{};
    Timestamp close_time{};               // == strategy's "now" for this event
    Price open{}, high{}, low{}, close{};
    Quantity volume{};
};

struct Tick {
    SymbolId  symbol{};
    Timestamp time{};
    Price     price{};
    Quantity  size{};
    Side      aggressor{};
};

struct BookUpdate {
    SymbolId  symbol{};
    Timestamp time{};
    Price     bid{}, ask{};
    Quantity  bid_size{}, ask_size{};
    // depth levels omitted for brevity
};

using MarketEvent = std::variant<Bar, Tick, BookUpdate>;

// Uniform access to the timestamp of any market event.
inline Timestamp event_time(const MarketEvent& ev) {
    return std::visit([](const auto& e) {
        using E = std::decay_t<decltype(e)>;
        if constexpr (std::same_as<E, Bar>) return e.close_time;
        else                                return e.time;
    }, ev);
}
```

---

## 4. Orders and fills

```cpp
// core/order.hpp
struct Order {
    OrderId     id{};
    SymbolId    symbol{};
    Side        side{};
    OrderType   type{};
    TimeInForce tif{TimeInForce::Day};
    Quantity    quantity{};               // always positive; direction is `side`
    Price       limit_price{};            // ignored for Market orders
    Timestamp   submit_time{};            // THE anti-lookahead anchor
};

struct Fill {
    OrderId   order_id{};
    SymbolId  symbol{};
    Side      side{};
    Quantity  quantity{};
    Price     price{};
    Timestamp time{};
    Money     commission{};
};
```

---

## 5. The merged-timeline scheduler

Any number of feeds — each forward-only and internally time-sorted — are merged into one
chronological stream by a k-way merge over a min-heap. This single ordered timeline is
what prevents lookahead *across* feeds.

```cpp
// core/data_feed.hpp
#include <optional>

class DataFeed {
public:
    virtual ~DataFeed() = default;
    // Timestamp of the next event without consuming it; nullopt when exhausted.
    virtual std::optional<Timestamp> peek_time() const = 0;
    // Consume and return the next event. Precondition: peek_time() != nullopt.
    virtual MarketEvent next() = 0;
};
```

```cpp
// core/event_scheduler.hpp
#include <queue>
#include <vector>
#include <memory>

class EventScheduler {
    struct Entry {
        Timestamp next_time;
        DataFeed* feed;
    };
    struct Later {            // min-heap: earliest timestamp on top
        bool operator()(const Entry& a, const Entry& b) const {
            return a.next_time > b.next_time;
        }
    };
public:
    void add_feed(std::unique_ptr<DataFeed> feed) {
        if (auto t = feed->peek_time())
            heap_.push(Entry{*t, feed.get()});
        feeds_.push_back(std::move(feed));
    }

    // Earliest event across all feeds, or nullopt when every feed is exhausted.
    std::optional<MarketEvent> next() {
        if (heap_.empty()) return std::nullopt;
        Entry top = heap_.top();
        heap_.pop();
        MarketEvent ev = top.feed->next();
        if (auto t = top.feed->peek_time())          // re-insert with its new front time
            heap_.push(Entry{*t, top.feed});
        return ev;
    }
private:
    std::priority_queue<Entry, std::vector<Entry>, Later> heap_;
    std::vector<std::unique_ptr<DataFeed>> feeds_;
};
```

A concrete in-memory feed, handy for tests and examples:

```cpp
// data/vector_bar_feed.hpp
class VectorBarFeed : public DataFeed {
public:
    explicit VectorBarFeed(std::vector<Bar> bars) : bars_(std::move(bars)) {}

    std::optional<Timestamp> peek_time() const override {
        if (pos_ >= bars_.size()) return std::nullopt;
        return bars_[pos_].close_time;
    }
    MarketEvent next() override { return bars_[pos_++]; }
private:
    std::vector<Bar> bars_;
    std::size_t pos_{0};
};
```

In production this becomes a memory-mapped binary feed (zero-copy iteration) and a
`LiveDataFeed` backed by an exchange socket — both satisfy the same `DataFeed` interface.

---

## 6. The strategy interface

The strategy is a plain C++ type. It does **not** inherit from anything. Instead the
engine detects, at compile time, which callbacks the strategy defines, via concepts. A
tick-only strategy that never writes `on_bar` pays nothing — no code, no branch.

```cpp
// core/strategy.hpp
class Context;   // forward decl — defined in section 7

template <class S> concept HasOnStart =
    requires(S s, Context& c) { s.on_start(c); };
template <class S> concept HasOnStop =
    requires(S s, Context& c) { s.on_stop(c); };
template <class S> concept HasOnBar =
    requires(S s, const Bar& b, Context& c) { s.on_bar(b, c); };
template <class S> concept HasOnTick =
    requires(S s, const Tick& t, Context& c) { s.on_tick(t, c); };
template <class S> concept HasOnBook =
    requires(S s, const BookUpdate& u, Context& c) { s.on_book(u, c); };
template <class S> concept HasOnFill =
    requires(S s, const Fill& f, Context& c) { s.on_fill(f, c); };

// Minimum bar for any strategy: it must at least be movable and define one data callback.
template <class S> concept Strategy =
    std::movable<S> && (HasOnBar<S> || HasOnTick<S> || HasOnBook<S>);
```

The strategy receives data only through these callbacks. There is no `get_history(from, to)`
with a caller-chosen end — if a strategy wants history it keeps its own append-only
buffer (see `RollingMean` in section 11), which by construction holds only past values.

---

## 7. The `Context` (broker handle)

`Context` is the single object the strategy uses to act on the world. It is the
backtest/live swap point: in a backtest it enqueues into the engine's pending buffer; in
live trading the same surface forwards to a real broker API.

`submit()` only *enqueues* — it stamps the order with `now()` and returns. Nothing fills
during the callback.

```cpp
// core/context.hpp
class Portfolio;   // section 9

class Context {
public:
    Context(const Portfolio& pf, std::vector<Order>& pending,
            OrderId& next_id, const Timestamp& clock)
        : pf_(pf), pending_(pending), next_id_(next_id), clock_(clock) {}

    OrderId submit(SymbolId sym, Side side, OrderType type,
                   Quantity qty, Price limit = {}) {
        Order o;
        o.id          = next_id_++;
        o.symbol      = sym;
        o.side        = side;
        o.type        = type;
        o.quantity    = qty;
        o.limit_price = limit;
        o.submit_time = clock_;            // <-- anti-lookahead anchor: "now"
        pending_.push_back(o);
        return o.id;
    }

    // Convenience wrappers
    OrderId buy (SymbolId s, Quantity q)            { return submit(s, Side::Buy,  OrderType::Market, q); }
    OrderId sell(SymbolId s, Quantity q)            { return submit(s, Side::Sell, OrderType::Market, q); }
    OrderId limit_buy (SymbolId s, Quantity q, Price p) { return submit(s, Side::Buy,  OrderType::Limit, q, p); }
    OrderId limit_sell(SymbolId s, Quantity q, Price p) { return submit(s, Side::Sell, OrderType::Limit, q, p); }

    // Read-only state — reflects everything up to and including "now".
    Quantity  position(SymbolId s) const;          // -> pf_.position(s)
    Money     cash()   const;                      // -> pf_.cash()
    Money     equity() const;                      // -> pf_.equity()
    Timestamp now()    const { return clock_; }
private:
    const Portfolio&    pf_;
    std::vector<Order>& pending_;                  // reused across events; never reallocated
    OrderId&            next_id_;
    const Timestamp&    clock_;                    // engine-owned simulated/wall clock
};
```

The strategy never reads the system clock — `ctx.now()` is the one source of time, which
keeps scheduled logic identical between backtest and live.

---

## 8. The `FillModel`

All fill logic lives behind one interface. The simple model ships first; an L2
queue-position model is a later drop-in that touches neither the engine nor strategies.
The interface also carries the core anti-lookahead invariant.

```cpp
// core/fill_model.hpp
#include <span>

struct FillBatch {
    std::vector<Fill>    fills;            // reused buffers — clear(), don't reallocate
    std::vector<OrderId> completed;        // ids of fully-filled orders to retire

    void clear() { fills.clear(); completed.clear(); }
};

class FillModel {
public:
    virtual ~FillModel() = default;
    // Match resting orders against ONE market event; append results to `out`.
    // INVARIANT: an order may only fill against an event whose timestamp is
    //            strictly greater than order.submit_time.
    virtual void match(const MarketEvent& ev,
                       std::span<const Order> open_orders,
                       FillBatch& out) = 0;
};
```

```cpp
// fill/immediate_fill_model.hpp
#include <algorithm>

// Fills on bar closes: market orders at the next bar's open; limit orders when the
// bar's range crosses the limit. Optional fixed slippage per unit.
class ImmediateFillModel : public FillModel {
public:
    explicit ImmediateFillModel(Price slippage_ticks = {}) : slip_(slippage_ticks) {}

    void match(const MarketEvent& ev,
               std::span<const Order> open,
               FillBatch& out) override {
        const Bar* bar = std::get_if<Bar>(&ev);
        if (!bar) return;                          // this model only fills on bars

        for (const Order& o : open) {
            if (o.symbol != bar->symbol) continue;
            if (!(bar->close_time > o.submit_time))// <-- INVARIANT enforced here
                continue;
            std::optional<Price> px = price_for(o, *bar);
            if (!px) continue;

            out.fills.push_back(Fill{
                .order_id   = o.id,
                .symbol     = o.symbol,
                .side       = o.side,
                .quantity   = o.quantity,
                .price      = with_slippage(*px, o.side),
                .time       = bar->close_time,
                .commission = Money{0},            // plug a commission model here
            });
            out.completed.push_back(o.id);
        }
    }
private:
    std::optional<Price> price_for(const Order& o, const Bar& bar) const {
        if (o.type == OrderType::Market)
            return bar.open;                       // next bar's open after submission
        if (o.side == Side::Buy)
            return bar.low <= o.limit_price
                       ? std::optional{ std::min(bar.open, o.limit_price) } : std::nullopt;
        else
            return bar.high >= o.limit_price
                       ? std::optional{ std::max(bar.open, o.limit_price) } : std::nullopt;
    }
    Price with_slippage(Price p, Side s) const {
        return s == Side::Buy ? Price{p.ticks + slip_.ticks}
                              : Price{p.ticks - slip_.ticks};
    }
    Price slip_;
};
```

The future `OrderBookMatchingModel` consumes `BookUpdate` events, tracks queue position,
and emits partial fills — same `match()` signature, so it slots straight in.

---

## 9. The `Portfolio`

Positions, cash, realized/unrealized PnL, and the equity curve.

```cpp
// core/portfolio.hpp
#include <unordered_map>

struct PositionState {
    Quantity qty{0};                       // signed: + long, - short
    Price    avg_cost{0};                  // average entry price (ticks)
    Money    realized_pnl{0};
    Price    last_price{0};                // most recent mark
};

struct EquityPoint { Timestamp time; Money equity; };

class Portfolio {
public:
    Portfolio(Money starting_cash, const InstrumentTable& instr)
        : cash_(starting_cash), instr_(instr) {}

    void apply_fill(const Fill& f) {
        const Instrument& ins = instr_.at(f.symbol);
        PositionState& p = positions_[f.symbol];
        const Quantity fq = (f.side == Side::Buy ? +1 : -1) * f.quantity;

        // Cash always moves by the traded notional plus commission.
        cash_ -= notional(ins, f.price, fq);
        cash_ -= f.commission;

        const bool same_side = (p.qty == 0) || ((p.qty > 0) == (fq > 0));
        if (same_side) {
            // Growing the position: blend the average cost.
            const std::int64_t cost = p.avg_cost.ticks * p.qty + f.price.ticks * fq;
            p.qty += fq;
            if (p.qty != 0) p.avg_cost = Price{ cost / p.qty };
        } else {
            // Reducing / closing: realize PnL on the closed portion.
            const Quantity closed = std::min<Quantity>(std::llabs(fq), std::llabs(p.qty));
            const std::int64_t sign = (p.qty > 0) ? +1 : -1;
            const std::int64_t pnl_ticks =
                sign * (f.price.ticks - p.avg_cost.ticks) * closed;
            p.realized_pnl += Money{ pnl_ticks * ins.tick_value_micros };
            p.qty += fq;
            if (p.qty != 0 && ((p.qty > 0) != (p.qty - fq > 0)))
                p.avg_cost = f.price;      // flipped through zero -> remainder opens here
        }
    }

    void mark(SymbolId sym, Price last) { positions_[sym].last_price = last; }

    Quantity position(SymbolId s) const {
        auto it = positions_.find(s);
        return it == positions_.end() ? 0 : it->second.qty;
    }
    Money cash() const { return cash_; }

    Money equity() const {
        Money e = cash_;
        for (const auto& [sym, p] : positions_)
            e += notional(instr_.at(sym), p.last_price, p.qty);
        return e;
    }

    void record_equity(Timestamp t) { curve_.push_back({t, equity()}); }
    const std::vector<EquityPoint>& curve() const { return curve_; }

private:
    Money cash_;
    const InstrumentTable& instr_;
    std::unordered_map<SymbolId, PositionState> positions_;
    std::vector<EquityPoint> curve_;
};
```

---

## 10. The `Engine`

Templated on the strategy type, so callbacks inline. The `run()` loop is the per-event
ordering from section 1 made concrete.

```cpp
// core/engine.hpp
template <Strategy S>
class Engine {
public:
    Engine(S strategy, InstrumentTable instr,
           std::unique_ptr<FillModel> fill, Money starting_cash)
        : strategy_(std::move(strategy)),
          instr_(std::move(instr)),
          fill_(std::move(fill)),
          portfolio_(starting_cash, instr_) {}

    void add_feed(std::unique_ptr<DataFeed> f) { scheduler_.add_feed(std::move(f)); }

    void run() {
        Context ctx(portfolio_, pending_, next_order_id_, clock_);
        if constexpr (HasOnStart<S>) strategy_.on_start(ctx);

        while (auto ev = scheduler_.next()) {
            clock_ = event_time(*ev);

            // 1. Match resting orders against the NEW event (data strictly after submit).
            batch_.clear();
            fill_->match(*ev, open_orders_, batch_);
            for (const Fill& f : batch_.fills) {
                portfolio_.apply_fill(f);
                // 2. Notify the strategy of each fill.
                if constexpr (HasOnFill<S>) strategy_.on_fill(f, ctx);
            }
            retire_completed();

            // 3. Mark-to-market.
            mark(*ev);

            // 4. Strategy reacts; any orders it submits are stamped with clock_.
            dispatch(*ev, ctx);

            // 5. Activate new orders — NOT matched against the event just processed.
            for (Order& o : pending_) open_orders_.push_back(o);
            pending_.clear();

            // 6. Record an equity-curve point.
            portfolio_.record_equity(clock_);
        }
        if constexpr (HasOnStop<S>) strategy_.on_stop(ctx);
    }

    const Portfolio& portfolio() const { return portfolio_; }

private:
    void dispatch(const MarketEvent& ev, Context& ctx) {
        std::visit([&](const auto& e) {
            using E = std::decay_t<decltype(e)>;
            if constexpr (std::same_as<E, Bar>) {
                if constexpr (HasOnBar<S>)  strategy_.on_bar(e, ctx);
            } else if constexpr (std::same_as<E, Tick>) {
                if constexpr (HasOnTick<S>) strategy_.on_tick(e, ctx);
            } else if constexpr (std::same_as<E, BookUpdate>) {
                if constexpr (HasOnBook<S>) strategy_.on_book(e, ctx);
            }
        }, ev);
    }

    void mark(const MarketEvent& ev) {
        std::visit([&](const auto& e) {
            using E = std::decay_t<decltype(e)>;
            if constexpr (std::same_as<E, Bar>)        portfolio_.mark(e.symbol, e.close);
            else if constexpr (std::same_as<E, Tick>)  portfolio_.mark(e.symbol, e.price);
            // BookUpdate: mark at mid — omitted for brevity
        }, ev);
    }

    void retire_completed() {
        if (batch_.completed.empty()) return;
        auto done = [&](const Order& o) {
            return std::find(batch_.completed.begin(), batch_.completed.end(), o.id)
                   != batch_.completed.end();
        };
        std::erase_if(open_orders_, done);
    }

    S                          strategy_;
    InstrumentTable            instr_;
    std::unique_ptr<FillModel> fill_;
    Portfolio                  portfolio_;
    EventScheduler             scheduler_;
    std::vector<Order>         open_orders_;        // reused
    std::vector<Order>         pending_;            // reused
    FillBatch                  batch_;              // reused
    Timestamp                  clock_{};
    OrderId                    next_order_id_{1};
};
```

---

## 11. Putting it together

A small append-only indicator (note: it can only ever hold past values — lookahead is
impossible) and an SMA-crossover strategy:

```cpp
// example/sma_cross.hpp
class RollingMean {
public:
    explicit RollingMean(std::size_t window) : window_(window), buf_(window) {}
    void push(Price p) {
        sum_ += p.ticks - buf_[head_];
        buf_[head_] = p.ticks;
        head_ = (head_ + 1) % window_;
        if (count_ < window_) ++count_;
    }
    bool   ready() const { return count_ == window_; }
    double value() const { return static_cast<double>(sum_) / window_; }
private:
    std::size_t window_, head_{0}, count_{0};
    std::vector<std::int64_t> buf_;
    std::int64_t sum_{0};
};

struct SmaCross {
    SymbolId    sym;
    RollingMean fast{10};
    RollingMean slow{30};
    Quantity    target{100};

    void on_bar(const Bar& bar, Context& ctx) {
        fast.push(bar.close);
        slow.push(bar.close);
        if (!fast.ready() || !slow.ready()) return;

        const bool bullish = fast.value() > slow.value();
        const Quantity pos = ctx.position(sym);

        if (bullish && pos < target)        ctx.buy(sym, target - pos);
        else if (!bullish && pos > 0)       ctx.sell(sym, pos);
        // Orders enqueue now; they fill against the NEXT bar's open.
    }
};
```

```cpp
// example/main.cpp
int main() {
    InstrumentTable instr;
    const SymbolId aapl = instr.add(Instrument{
        .symbol = "AAPL", .asset_class = AssetClass::Equity,
        .tick_value_micros = 10'000,        // 1 tick = $0.01
        .min_qty = 1,
    });

    Engine engine{
        SmaCross{ .sym = aapl },
        std::move(instr),
        std::make_unique<ImmediateFillModel>(/*slippage*/ Price{1}),
        Money{ 100'000 * 1'000'000 },       // $100,000 starting cash
    };

    engine.add_feed(std::make_unique<VectorBarFeed>(load_aapl_bars()));
    engine.run();

    // Analysis layer consumes engine.portfolio().curve()
    print_report(engine.portfolio());
    return 0;
}
```

---

## 12. Extension points (intentionally not built yet)

- **`OrderBookMatchingModel`** — L2 queue-position simulation; a drop-in `FillModel`.
- **`LiveDataFeed` / `LiveBroker`** — same `DataFeed` / `Context` surfaces, real sockets.
- **Python strategy adapter** — one type satisfying the `Strategy` concept, forwarding
  callbacks across the language boundary; the engine is unchanged.
- **Synthetic data generators** — GARCH / regime-switching / block-bootstrap feeds that
  implement `DataFeed`.
- **Risk layer** — intercepts orders between `Context::submit` and `open_orders_`.
- **Reporting** — Sharpe, Sortino, max drawdown, turnover, trade log, benchmark compare.
- **Sweep harness** — runs N `Engine` instances across cores; each stays single-threaded.
- **Commission models** — pluggable, applied inside the `FillModel`.
