# Architecture diagrams

Mermaid renderings of the engine's structure and runtime flow. The class
diagrams cover the static design; the sequence diagrams cover what happens during
a run.

> Note: `Strategy`, `DataFeed`, and `Broker` are **compile-time C++20 concepts**,
> not virtual interfaces — they're drawn as interfaces here for clarity. The
> engine is a template instantiated for one concrete strategy/feed/broker combo,
> so there are no virtual calls on the hot path.

## 1. Core wiring (engine, context, concepts, implementations)

```mermaid
classDiagram
    class Engine {
        +run() void
    }
    class Context {
        -m_broker : BrokerT&
        -m_dataFeed : const DataFeedT&
        -m_clock : const Clock&
        +now() Timestamp
        +cash() Balance
        +equity() Balance
        +history(count) MarketWindow
        +place_order(MarketOrderParams) OrderID
        +place_order(LimitOrderParams) OrderID
        +place_order(StopOrderParams) OrderID
        +position(symbol) optional~Position~
        +order(id) optional~Order~
        +cancel_order(id) bool
    }
    class Clock {
        -m_timestamp : Timestamp
        +now() Timestamp
        +set(ts) void
    }
    class Strategy {
        <<concept>>
        +on_tick(ctx)*
        +on_start(ctx) optional
        +on_stop(ctx) optional
    }
    class DataFeed {
        <<concept>>
        +next_timestamp() optional~Timestamp~
        +advance() void
        +current_bars() vector~KLine~
        +window(count) MarketWindow
        +resolution() duration
        +size() size_t
    }
    class Broker {
        <<concept>>
        +cash() Balance
        +equity() Balance
        +place_order(MarketOrderParams) OrderID
        +place_order(LimitOrderParams) OrderID
        +place_order(StopOrderParams) OrderID
        +position(symbol) optional~Position~
        +cancel_order(id) bool
        +on_tick(bar) void
        +trades() map~TradeID,Trade~
        +orders() map~OrderID,Order~
    }
    class KLineFeed
    class BacktestBroker
    class EMA50Strategy
    class PythonStrategy

    note for Engine "template StrategyT, DataFeedT, BrokerT"
    note for Context "template BrokerT, DataFeedT"

    Engine *-- Strategy : owns (StrategyT)
    Engine *-- DataFeed : owns (DataFeedT)
    Engine *-- Broker : owns (BrokerT)
    Engine *-- Clock : owns
    Engine ..> Context : creates per run
    Context o-- Broker : ref
    Context o-- DataFeed : ref
    Context o-- Clock : ref
    KLineFeed ..|> DataFeed : satisfies
    BacktestBroker ..|> Broker : satisfies
    EMA50Strategy ..|> Strategy : satisfies
    PythonStrategy ..|> Strategy : satisfies
```

## 2. Core data types

```mermaid
classDiagram
    class Timestamp {
        +value : time_point
        +from_millis(ms) Timestamp
    }
    class KLine {
        +timestamp : Timestamp
        +symbol : Symbol
        +open : Price
        +high : Price
        +low : Price
        +close : Price
        +volume : Volume
    }
    class SeriesView {
        +timestamp : span~int64~
        +open : span~double~
        +high : span~double~
        +low : span~double~
        +close : span~double~
        +volume : span~double~
        +size() size_t
    }
    class SymbolSeries {
        +symbol : string_view
        +bars : SeriesView
    }
    class MarketWindow {
        +series : vector~SymbolSeries~
        +size() size_t
    }
    MarketWindow o-- SymbolSeries
    SymbolSeries o-- SeriesView
    class Order {
        +id : OrderID
        +parent_id : optional~OrderID~
        +timestamp : Timestamp
        +symbol : Symbol
        +side : OrderSide
        +type : OrderType
        +status : OrderStatus
        +price : optional~Price~
        +quantity : Quantity
        +time_in_force : TimeInForce
        +leverage : double
        +reduce_only : bool
    }
    class Trade {
        +id : TradeID
        +order_id : OrderID
        +timestamp : Timestamp
        +symbol : Symbol
        +side : OrderSide
        +quantity : Quantity
        +price : Price
        +liquidation : bool
        +fee : double
    }
    class Position {
        +quantity : Quantity signed
        +price : Price average entry
        +entry_id : OrderID
        +leverage : double
    }
    class MarketOrderParams {
        +symbol : Symbol
        +side : OrderSide
        +quantity : Quantity
        +time_in_force : TimeInForce
        +leverage : double
        +reduce_only : bool
    }
    class LimitOrderParams {
        +symbol : Symbol
        +side : OrderSide
        +quantity : Quantity
        +price : Price
        +time_in_force : TimeInForce
        +leverage : double
        +reduce_only : bool
    }
    class StopOrderParams {
        +symbol : Symbol
        +side : OrderSide
        +quantity : Quantity
        +price : Price trigger
        +time_in_force : TimeInForce
        +leverage : double
        +reduce_only : bool
    }
    class OrderSide {
        <<enumeration>>
        Buy
        Sell
    }
    class OrderType {
        <<enumeration>>
        Market
        Limit
        Stop
    }
    class OrderStatus {
        <<enumeration>>
        Open
        Filled
        Rejected
        Cancelled
    }
    class TimeInForce {
        <<enumeration>>
        GTC
    }

    KLine *-- Timestamp
    Order *-- Timestamp
    Trade *-- Timestamp
    Order ..> OrderSide
    Order ..> OrderType
    Order ..> OrderStatus
    Order ..> TimeInForce
    Order --> Order : parent_id brackets under
    Trade ..> OrderSide
    Trade --> Order : order_id refers to
    Position --> Order : entry_id refers to
```

## 3. Python boundary

The templated C++ `Context` can't bind to pybind11 directly, so a type-erased
`IContext` (bound to Python as `Context`) is implemented by `ContextAdapter`.

```mermaid
classDiagram
    class IContext {
        <<interface>>
        +now() Timestamp
        +cash() Balance
        +equity() Balance
        +history(count) MarketWindow
        +place_market_order(params, parent) OrderID
        +place_limit_order(params, parent) OrderID
        +place_stop_order(params, parent) OrderID
        +position(symbol) optional~Position~
        +order(id) optional~Order~
        +cancel_order(id) bool
    }
    class ContextAdapter {
        -m_ctx : Context&
    }
    class Context
    class PythonStrategy {
        -m_python : EmbeddedPython
        -m_py_instance : py::object
        -m_adapter : unique_ptr~IContext~
        +on_start(ctx)
        +on_tick(ctx)
        +on_stop(ctx)
    }
    class EmbeddedPython {
        +add_sys_path(path)
    }
    class PyStrategy {
        <<python>>
        on_tick(ctx)
    }

    ContextAdapter ..|> IContext : implements
    ContextAdapter o-- Context : wraps
    PythonStrategy *-- EmbeddedPython : owns
    PythonStrategy *-- IContext : owns adapter
    PythonStrategy ..> PyStrategy : instantiates and calls
    PyStrategy ..> IContext : uses as Context
```

## 4. Module dependencies

Everything depends inward on `core`; `core` depends on nothing else. New
strategies, feeds, and brokers plug into the concept slots.

```mermaid
graph TD
    app["app/strategies + app/python"] --> core
    datafeed["datafeed: KLineFeed"] --> core
    broker["broker: BacktestBroker"] --> core
    python["python: IContext, ContextAdapter, EmbeddedPython"] --> core
    pybind["python/stonks pkg + bindings"] --> python
    pybind --> core
    core["core: engine, context, clock, concepts, types"]
    style core fill:#f9f,stroke:#333
```

## 5. Sequence — a full backtest run

One strategy tick **per timestamp**: the engine first feeds every symbol's bar
at that timestamp to the broker (settle the past — fill prior orders, mark
prices), then calls the strategy **once** (decide the future). That ordering is
what stops an order from filling on the same timestamp it was placed.

```mermaid
sequenceDiagram
    participant Main as main()
    participant E as Engine
    participant C as Clock
    participant F as DataFeed
    participant B as Broker
    participant Ctx as Context
    participant S as Strategy

    Main->>E: run()
    E->>Ctx: build Context{broker, dataFeed, clock}
    opt on_start defined
        E->>S: on_start(ctx)
    end
    E->>B: cash() to starting_cash

    loop one timestamp per iteration
        E->>F: next_timestamp()
        F-->>E: ts
        E->>C: set(ts)
        E->>F: current_bars()
        F-->>E: all symbols' bars at ts
        loop each bar at ts
            E->>B: on_tick(bar)
            Note over B: settle in rounds (fills arm bracket children), then check liquidation
        end
        E->>B: equity()
        Note over E: append EquityPoint to the run's equity curve
        E->>S: on_tick(ctx)
        Note over S,Ctx: strategy loops the window, places orders
        E->>F: advance()
    end

    opt on_stop defined
        E->>S: on_stop(ctx)
    end
    E->>B: trades(), cash(), equity()
    Note over E: print report
    E-->>Main: return
```

## 6. Sequence — order lifecycle (the no-lookahead gate)

One order followed across the two bars it touches. The single check
`order.timestamp >= bar.timestamp` (reject) is the whole guarantee: an order
placed at bar N cannot fill until a bar strictly after N.

Bracket children are the one refinement: a child keeps its original placement
timestamp (always earlier than any bar its parent can fill on), so once the
parent fills, the same gate admits it from the parent's **fill bar** onward —
the broker settles each bar in rounds until no order makes progress, ordering
each round's candidates Market → Stop → Limit under the default Conservative
policy (protective stops before profit targets).

```mermaid
sequenceDiagram
    participant E as Engine
    participant Ctx as Context
    participant S as Strategy
    participant B as Broker

    rect rgb(235, 245, 255)
        Note over E,B: Bar N (clock = N)
        E->>B: on_tick(bar_N)
        Note over B: mark last_price; no order for it yet
        E->>S: on_tick(ctx)
        S->>Ctx: history(n)
        Note over Ctx: symbols printing at N, each bounded to <= N
        Ctx-->>S: MarketWindow (cannot see N+1)
        S->>Ctx: place_order(MarketOrderParams{...})
        Ctx->>B: place_order(params)
        Note over B: Order built + stamped (timestamp = now() = N), queued
    end

    rect rgb(235, 255, 235)
        Note over E,B: Bar N+1 (clock = N+1)
        E->>B: on_tick(bar_N+1)
        Note over B: try_fill: order.ts(N) >= bar.ts(N+1)? NO, gate passes
        Note over B: fill at bar_N+1.open; update cash/position; record Trade; dequeue
    end
```

## 7. Sequence — Python strategy detour

When the strategy is `PythonStrategy`, the engine's `on_tick(ctx)` expands into
this. The adapter is built once (lazily) and reused; the GIL is acquired per
call. From the engine's side it's indistinguishable from a C++ strategy.

```mermaid
sequenceDiagram
    participant E as Engine
    participant PS as PythonStrategy
    participant A as ContextAdapter
    participant Ctx as Context
    participant Py as Python class

    E->>PS: on_tick(ctx)
    opt first call only
        PS->>A: make_adapter(ctx)
        Note over A: wraps Context (B, F)
    end
    Note over PS: acquire GIL
    PS->>Py: py_instance.on_tick(adapter)
    Py->>A: ctx.history(50)
    A->>Ctx: history(50)
    Ctx-->>A: MarketWindow (per-symbol spans)
    A-->>Py: combined numpy columns (one DataFrame)
    Py->>A: ctx.place_market_order(...)
    A->>Ctx: place_order(params)
    Ctx-->>A: OrderID
    A-->>Py: OrderID (int)
    Py-->>PS: return
    Note over PS: release GIL
    PS-->>E: return
```
