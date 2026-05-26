# Project Brief — Backtesting and Live Trading Engine

This document describes, in plain language, what I am trying to build and how I want it
built. It is intended as context for an AI coding agent. It contains no code and no file
layout — only intent, requirements, and the reasoning behind them.

## What I am building

I am building a system for stock backtesting and live trading. I want it to be versatile
and to serve as a long-term foundation, not a throwaway script. The core of the system
is written in C++ and is responsible for everything that is performance-sensitive:
ingesting market data, running the simulation, matching orders, and tracking the
portfolio. Trading strategies are kept separate from the core. For now I will write
strategies in C++ as well, but I plan to add a Python strategy interface later, and the
core must be designed so that adding Python does not require changing the engine.

The system must not be tied to a single market. It has to handle different kinds of
instruments — equities, crypto, futures, and so on — and different kinds of market data:
traditional OHLCV bars, real trade-tick data, and order-book (Level 2) data. It should
also be able to generate realistic synthetic data for testing, and to produce
performance reports at the end of a run. Crucially, the same system must be usable for
live trading, not only for backtesting.

## The two things that matter most

Above everything else, I care about two properties: the system must be **realistic**,
and it must be **high-performance**. Every design choice should be judged against these
two goals first.

Realistic means the backtest reflects what would actually have happened. I do not want a
vectorized backtester that applies array math over the whole price series at once. That
style is fast, but it cannot model anything that depends on the ordering of events, and
it makes it far too easy to accidentally use information from the future. I want a
strictly event-driven engine that processes one market event at a time, in chronological
order, where a strategy only ever reacts to information it has already been given.

High-performance means the C++ core is written with genuine, deliberate attention to
performance. I am using C++20 and I want modern best practices applied throughout:
cache-friendly data layouts, no wasteful work on the hot path, and no unnecessary
allocations while events are being processed. The engine will process very large amounts
of tick data, so the per-event cost has to stay small.

## No lookahead bias — this is non-negotiable

A backtest that uses future information is worthless, and the most common way that
happens is subtle and accidental. I want lookahead bias to be impossible by
construction, not merely discouraged by guidelines. Even a badly written strategy must
not be able to peek into the future.

Concretely, this means a strategy should never hold the full dataset, should never be
able to seek forward, and should only ever receive data because the engine pushed an
event to it. Any order a strategy places must be timestamped at the exact moment it is
placed, and the engine must never fill that order against market data from at or before
that moment. A decision made on one bar must execute on a later bar, never on the same
one. If the design is right, a strategy author cannot create lookahead bias even if they
try.

## Backtest and live trading must share the same code

I do not want two separate code paths. The strategy code must not know, and must not
care, whether it is running on historical data or on a live feed. Only the backends
should differ: where market data comes from, and where orders are sent. Everything else
— the strategy, the engine, the accounting, the risk checks, the reporting — must be
exactly the same in both modes. Time should be handled the same way too: the strategy
must always ask the engine what time it is rather than reading the system clock, so that
scheduled logic behaves identically whether time is simulated or real.

## How I want strategies to work

A strategy should be simple to write and should focus only on decision logic. It should
receive market data through clear, separate entry points depending on the kind of data —
one for bars, one for ticks, one for order-book updates — and a strategy should only
have to deal with the kinds of data it actually uses.

A strategy should act by telling the engine what to do — for example, to buy or sell a
quantity of an instrument, or to place and cancel orders — rather than by returning a
list of signals. This imperative style mirrors how real trading works and keeps a single
consistent interface across backtesting and live trading. When a strategy places an
order, that order should not execute instantly inside the same step; it should take
effect against the next market data that arrives, which is also what keeps the system
free of lookahead.

## Market data

The engine must support OHLCV bars, real trade-tick data, and order-book data, and it
must support several instruments at once. It should be possible to run a strategy on
data from multiple instruments and even multiple data types together, always processed
in correct global time order. The storage and loading of market data should be efficient
enough to handle large tick datasets without becoming a bottleneck. A strategy should be
able to declare which instruments and which data resolution it needs, so it is not
forced to pay the cost of data it does not use.

## Realistic order execution

How orders get filled is one of the most important parts of realism, and it is also
something I want to be able to tune. The level of execution realism should be a dial.
With only OHLCV data, fills can only be approximate — something like filling at the next
bar's open with a slippage assumption. With real tick and order-book data, I want a much
more faithful simulation: an order that sits in the book, tracks its place in the queue,
fills partially as the market trades through its price, and accounts for market impact.

I do not need the most advanced version first. I want the execution logic to live behind
a clean boundary so the simple version can be built early and the order-book version can
be added later without disturbing the rest of the engine or any strategy.

## Synthetic data

The system should be able to generate synthetic market data so I can stress-test
strategies against conditions that may not appear in my historical data. Simple random
walks are not enough. I want synthetic data that is genuinely realistic — exhibiting
things like fat-tailed returns, volatility that clusters, changing market regimes, and
realistic intraday patterns. Resampling real data in a way that preserves its structure
is also acceptable. Synthetic data should plug into the engine the same way any other
data source does.

## Risk, reporting, and optimization

I want a risk layer that sits between the strategy and the market and is kept separate
from strategy logic. It should enforce constraints such as maximum position sizes and
exposure limits, and it should be able to halt trading entirely if losses pass a
threshold. The same risk layer should apply in both backtesting and live trading.

Every run should produce a proper performance report: the standard metrics such as
risk-adjusted returns, drawdowns, win rate, turnover, and exposure, together with the
equity curve, a log of trades, and a comparison against a simple benchmark.

I also want to be able to study strategies systematically — running many parameter
combinations and doing walk-forward analysis where a strategy is tuned on one period and
evaluated on the next. This needs the engine to be fast and to run many independent
backtests in parallel.

## How I want performance handled

A single backtest run should be single-threaded and fully deterministic. I do not want
concurrency inside one run; instead, parallelism should come from running many
independent backtests at the same time, one per processor core. Determinism is
important: the same inputs must always produce exactly the same results, so that
debugging and regression testing are reliable. To support this, money, prices, and time
should be represented with integers rather than floating-point values, which avoids
rounding drift and keeps results reproducible.

## How I want it built

I want the system built incrementally and in a disciplined order. The first goal should
be a minimal but complete end-to-end backtest — data in, a trivial strategy, an order,
a fill, portfolio accounting, and a result out — so that the overall design is proven
before any component is made sophisticated. Only after that should each part be
deepened.

Correctness must come before features. The guarantees that matter most — no lookahead
bias, and full determinism — should be proven by tests early, and no further work should
be layered on top until those tests pass. Everything should be well tested as it is
built, with particular attention to the accounting logic and the order-matching logic,
and with explicit tests that confirm a deliberately misbehaving strategy still cannot
gain access to future information.

In short: I want a realistic, high-performance, event-driven engine that I can trust;
that runs the same strategy code in backtesting and live trading; that handles many
instruments and many forms of market data; and that is built carefully enough to be a
durable foundation rather than a prototype.
