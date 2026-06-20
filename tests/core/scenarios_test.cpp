// End-to-end scenarios: scripted strategies driven through the full Engine
// against a real BacktestBroker. A small BrokerSpy holds a pointer to an
// externally-owned broker so we can inspect its trades and balances after
// engine.run() (Engine moves the broker in by value).

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stonks/broker/backtestbroker.h"
#include "stonks/core/engine.h"
#include "stonks/core/types.h"

#include "src/report.h"
#include "test_stubs.h"

namespace stonks::broker {

using namespace stonks::core;
using stonks::core::test::StubFeed;

namespace {

struct BrokerSpy
{
    BacktestBroker* impl{};

    Balance cash() const { return impl->cash(); }
    Balance equity() const { return impl->equity(); }
    const std::vector<Trade>& trades() const { return impl->trades(); }
    const std::vector<Order>& orders() const { return impl->orders(); }
    bool place_order(const Order& o) { return impl->place_order(o); }
    void on_tick(const KLine& bar) { impl->on_tick(bar); }
};
static_assert(Broker<BrokerSpy>);

struct CoutMute
{
    std::ostringstream sink;
    std::streambuf* old{ std::cout.rdbuf(sink.rdbuf()) };
    ~CoutMute() { std::cout.rdbuf(old); }
};

KLine bar(std::int64_t ms, const Symbol& sym, Price o, Price h, Price l, Price c)
{
    return KLine{ Timestamp::from_millis(ms), sym, o, h, l, c, Volume{ 1.0 } };
}

KLine bar(std::int64_t ms, Price o, Price h, Price l, Price c)
{
    return bar(ms, Symbol{ "X" }, o, h, l, c);
}

KLine flat(std::int64_t ms, Price p) { return bar(ms, p, p, p, p); }
KLine flat(std::int64_t ms, const Symbol& sym, Price p) { return bar(ms, sym, p, p, p, p); }

template <class StrategyT>
void run_engine(StrategyT strategy, std::vector<KLine> bars, BacktestBroker& broker)
{
    StubFeed feed;
    feed.bars = std::move(bars);
    BrokerSpy spy{ &broker };

    CoutMute mute;
    Engine engine{ std::move(strategy), std::move(feed), std::move(spy) };
    engine.run();
}

// --- Scripted strategies -----------------------------------------------------

// Buy once on the first tick, sell on the second tick.
struct BuyThenSell
{
    Quantity qty;
    Symbol symbol{ "X" };
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick == 0) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Buy, .quantity = qty,
            }));
        } else if (tick == 1) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Sell, .quantity = qty,
            }));
        }
        ++tick;
    }
};

// Buy once on first tick, hold forever.
struct BuyAndHold
{
    Quantity qty;
    Symbol symbol{ "X" };
    bool placed{ false };

    void on_tick(auto& ctx)
    {
        if (placed) { return; }
        ctx.place_order(ctx.make_market_order({
            .symbol = symbol, .side = OrderSide::Buy, .quantity = qty,
        }));
        placed = true;
    }
};

// Ladder: buy 1 on each of the first N ticks, then liquidate everything.
struct LadderThenLiquidate
{
    int rungs;
    Quantity per_rung;
    Symbol symbol{ "X" };
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick < rungs) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Buy, .quantity = per_rung,
            }));
        } else if (tick == rungs) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Sell, .quantity = per_rung * rungs,
            }));
        }
        ++tick;
    }
};

// Place a single limit buy on the first tick, then idle.
struct LimitBuyOnce
{
    Quantity qty;
    Price limit;
    Symbol symbol{ "X" };
    bool placed{ false };

    void on_tick(auto& ctx)
    {
        if (placed) { return; }
        ctx.place_order(ctx.make_limit_order({
            .symbol = symbol, .side = OrderSide::Buy,
            .quantity = qty, .price = limit,
        }));
        placed = true;
    }
};

// Buy symbol A on first A-bar; buy symbol B on first B-bar.
struct TwoSymbolBuyer
{
    Symbol a{ "A" };
    Symbol b{ "B" };
    Quantity qty_a;
    Quantity qty_b;
    bool bought_a{ false };
    bool bought_b{ false };

    void on_tick(auto& ctx)
    {
        if (!bought_a) {
            ctx.place_order(ctx.make_market_order({
                .symbol = a, .side = OrderSide::Buy, .quantity = qty_a,
            }));
            bought_a = true;
            return;
        }
        if (!bought_b) {
            ctx.place_order(ctx.make_market_order({
                .symbol = b, .side = OrderSide::Buy, .quantity = qty_b,
            }));
            bought_b = true;
        }
    }
};

// Sell on first tick without holding anything (short).
struct ShortOnceThenCover
{
    Quantity qty;
    Symbol symbol{ "X" };
    int cover_at;
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick == 0) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Sell, .quantity = qty,
            }));
        } else if (tick == cover_at) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Buy, .quantity = qty,
            }));
        }
        ++tick;
    }
};

// Place a buy on the very last bar — there is no next bar to fill against.
struct BuyOnLastBar
{
    int last_index;
    Quantity qty;
    Symbol symbol{ "X" };
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick == last_index) {
            ctx.place_order(ctx.make_market_order({
                .symbol = symbol, .side = OrderSide::Buy, .quantity = qty,
            }));
        }
        ++tick;
    }
};

// Place two buys on the first tick — they should both arrive before the next
// bar and process FIFO when cash permits.
struct TwoBuysSameBar
{
    Quantity q1;
    Quantity q2;
    Symbol symbol{ "X" };
    bool done{ false };

    void on_tick(auto& ctx)
    {
        if (done) { return; }
        ctx.place_order(ctx.make_market_order({
            .symbol = symbol, .side = OrderSide::Buy, .quantity = q1,
        }));
        ctx.place_order(ctx.make_market_order({
            .symbol = symbol, .side = OrderSide::Buy, .quantity = q2,
        }));
        done = true;
    }
};

// Capture lifecycle hook invocations.
struct LifecycleSpy
{
    int* starts;
    int* ticks;
    int* stops;

    void on_start(auto&) { ++(*starts); }
    void on_tick(auto&) { ++(*ticks); }
    void on_stop(auto&) { ++(*stops); }
};

// Buys one share of every symbol that printed, on every tick. Used to assert the
// no-lookahead property over a multi-bar, multi-symbol run.
struct BuyEachPrinterEveryTick
{
    void on_tick(auto& ctx)
    {
        for (const auto& s : ctx.history(1).series) {
            ctx.place_order(ctx.make_market_order({
                .symbol = Symbol{ s.symbol },
                .side = OrderSide::Buy,
                .quantity = Quantity{ 1.0 },
            }));
        }
    }
};

} // namespace

// --- Tests -------------------------------------------------------------------

TEST(Scenario, RoundTrip_BuyThenSell_TwoTradesAtNextBarOpens)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    run_engine(BuyThenSell{ Quantity{ 2.0 } }, {
        flat(1000, 100.0),   // tick 0: strategy places buy
        bar (2000, 110.0, 115.0, 105.0, 112.0),  // tick 1: buy fills @ open=110; strategy places sell
        bar (3000, 120.0, 125.0, 118.0, 122.0),  // tick 2: sell fills @ open=120
    }, broker);

    ASSERT_EQ(broker.trades().size(), 2u);

    const auto& t1 = broker.trades()[0];
    EXPECT_EQ(t1.id, TradeID{ 1 });
    EXPECT_EQ(t1.timestamp, Timestamp::from_millis(2000));
    EXPECT_EQ(t1.side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(t1.quantity, 2.0);
    EXPECT_DOUBLE_EQ(t1.price, 110.0);

    const auto& t2 = broker.trades()[1];
    EXPECT_EQ(t2.id, TradeID{ 2 });
    EXPECT_GT(t2.order_id, t1.order_id);
    EXPECT_EQ(t2.timestamp, Timestamp::from_millis(3000));
    EXPECT_EQ(t2.side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(t2.quantity, 2.0);
    EXPECT_DOUBLE_EQ(t2.price, 120.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 2 * 110.0 + 2 * 120.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());   // flat position
}

TEST(Scenario, BuyAndHold_OneTrade_EquityMarksToLastClose)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    run_engine(BuyAndHold{ Quantity{ 3.0 } }, {
        flat(1000, 100.0),   // tick 0: strategy places buy
        flat(2000, 110.0),   // tick 1: buy fills @ 110; close=110
        flat(3000, 130.0),   // tick 2: close=130
        flat(4000, 125.0),   // tick 3: close=125 — final mark
    }, broker);

    ASSERT_EQ(broker.trades().size(), 1u);
    const auto& t = broker.trades().front();
    EXPECT_EQ(t.timestamp, Timestamp::from_millis(2000));
    EXPECT_DOUBLE_EQ(t.price, 110.0);
    EXPECT_DOUBLE_EQ(t.quantity, 3.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 3 * 110.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash() + 3 * 125.0);
}

TEST(Scenario, LadderThenLiquidate_TradesAndPnL)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    // ticks 0..2: each places a buy. tick 3: places a sell of 3 units.
    // Buys fill at next bar's open: 110, 120, 130. Sell fills at 140.
    run_engine(LadderThenLiquidate{ /*rungs=*/3, /*per_rung=*/Quantity{ 1.0 } }, {
        flat(1000, 100.0),
        flat(2000, 110.0),   // buy 1 @ 110
        flat(3000, 120.0),   // buy 1 @ 120
        flat(4000, 130.0),   // buy 1 @ 130
        flat(5000, 140.0),   // sell 3 @ 140
    }, broker);

    ASSERT_EQ(broker.trades().size(), 4u);

    const auto& trades = broker.trades();
    EXPECT_EQ(trades[0].side, OrderSide::Buy);  EXPECT_DOUBLE_EQ(trades[0].price, 110.0); EXPECT_DOUBLE_EQ(trades[0].quantity, 1.0);
    EXPECT_EQ(trades[1].side, OrderSide::Buy);  EXPECT_DOUBLE_EQ(trades[1].price, 120.0); EXPECT_DOUBLE_EQ(trades[1].quantity, 1.0);
    EXPECT_EQ(trades[2].side, OrderSide::Buy);  EXPECT_DOUBLE_EQ(trades[2].price, 130.0); EXPECT_DOUBLE_EQ(trades[2].quantity, 1.0);
    EXPECT_EQ(trades[3].side, OrderSide::Sell); EXPECT_DOUBLE_EQ(trades[3].price, 140.0); EXPECT_DOUBLE_EQ(trades[3].quantity, 3.0);

    // Trade IDs strictly increasing 1..4
    for (std::size_t i = 0; i < trades.size(); ++i) {
        EXPECT_EQ(trades[i].id, TradeID{ i + 1 });
    }
    // Order IDs also strictly increasing (one order per buy, one for sell)
    for (std::size_t i = 1; i < trades.size(); ++i) {
        EXPECT_GT(trades[i].order_id, trades[i - 1].order_id);
    }

    const double pnl = -110.0 - 120.0 - 130.0 + 3 * 140.0;   // = 60
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 + pnl);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());        // flat
}

TEST(Scenario, LimitBuy_StaysQueuedThenFillsOnLaterBar)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // Limit @ 95. Bar 1: low=105, no fill. Bar 2: low=90 ≤ 95, fills at min(95, open=100) = 95.
    run_engine(LimitBuyOnce{ Quantity{ 1.0 }, Price{ 95.0 } }, {
        flat(1000, 100.0),                           // tick 0: places limit
        bar (2000, 110.0, 115.0, 105.0, 112.0),      // tick 1: low=105 > 95, no fill
        bar (3000, 100.0, 102.0, 90.0,  101.0),      // tick 2: low=90 ≤ 95, fill at 95
        flat(4000, 101.0),
    }, broker);

    ASSERT_EQ(broker.trades().size(), 1u);
    const auto& t = broker.trades().front();
    EXPECT_EQ(t.timestamp, Timestamp::from_millis(3000));
    EXPECT_DOUBLE_EQ(t.price, 95.0);
}

TEST(Scenario, LimitBuy_GapDown_FillsAtOpenBelowLimit)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // Limit @ 90. Bar 1 opens at 80, gap-down: fill at min(90, 80) = 80.
    run_engine(LimitBuyOnce{ Quantity{ 1.0 }, Price{ 90.0 } }, {
        flat(1000, 100.0),
        bar (2000, 80.0, 85.0, 75.0, 82.0),
        flat(3000, 82.0),
    }, broker);

    ASSERT_EQ(broker.trades().size(), 1u);
    EXPECT_DOUBLE_EQ(broker.trades().front().price, 80.0);
}

TEST(Scenario, MultiSymbol_PortfolioEquityCombinesLastPricePerSymbol)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    // Interleave A and B bars chronologically.
    run_engine(TwoSymbolBuyer{ /*a=*/Symbol{ "A" }, /*b=*/Symbol{ "B" },
                               Quantity{ 1.0 }, Quantity{ 2.0 } }, {
        flat(1000, Symbol{ "A" }, 100.0),   // tick 0: strategy buys A
        flat(2000, Symbol{ "B" }, 50.0),    // tick 1: A buy still in queue (sym mismatch); B bar — strategy buys B
        flat(3000, Symbol{ "A" }, 120.0),   // tick 2: A buy fills @ 120; A last=120
        flat(4000, Symbol{ "B" }, 60.0),    // tick 3: B buy fills @ 60; B last=60
        flat(5000, Symbol{ "A" }, 130.0),   // tick 4: A last=130 (B last still 60)
        flat(6000, Symbol{ "B" }, 55.0),    // tick 5: B last=55
    }, broker);

    ASSERT_EQ(broker.trades().size(), 2u);
    const auto& a_trade = broker.trades()[0];
    const auto& b_trade = broker.trades()[1];
    EXPECT_EQ(a_trade.symbol, "A");
    EXPECT_EQ(a_trade.timestamp, Timestamp::from_millis(3000));
    EXPECT_DOUBLE_EQ(a_trade.price, 120.0);
    EXPECT_EQ(b_trade.symbol, "B");
    EXPECT_EQ(b_trade.timestamp, Timestamp::from_millis(4000));
    EXPECT_DOUBLE_EQ(b_trade.price, 60.0);

    const double cash = 10'000.0 - 1 * 120.0 - 2 * 60.0;
    EXPECT_DOUBLE_EQ(broker.cash(), cash);
    // Equity uses each symbol's most recent close: A=130, B=55.
    EXPECT_DOUBLE_EQ(broker.equity(), cash + 1 * 130.0 + 2 * 55.0);
}

TEST(Scenario, ShortSellThenCover_NetsCorrectly)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // Sell 1 (no position) on tick 0 → fills next bar @ 110. Cover at tick 2 → fills next bar @ 90.
    // P&L = +110 - 90 = +20.
    run_engine(ShortOnceThenCover{ .qty = Quantity{ 1.0 }, .cover_at = 2 }, {
        flat(1000, 100.0),
        flat(2000, 110.0),   // sell fills @ 110; position = -1; equity = cash + (-1)*110
        flat(3000, 105.0),   // mark: equity = (1000+110) + (-1)*105 = 1005
        flat(4000, 90.0),    // cover buy fills @ 90; position = 0
        flat(5000, 95.0),    // mark with no position
    }, broker);

    ASSERT_EQ(broker.trades().size(), 2u);
    EXPECT_EQ(broker.trades()[0].side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(broker.trades()[0].price, 110.0);
    EXPECT_EQ(broker.trades()[1].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(broker.trades()[1].price, 90.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 + 110.0 - 90.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());
}

TEST(Scenario, OrderPlacedOnLastBar_NeverFills)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // 3 bars, strategy places its buy on tick 2 (the last). No next bar → no fill.
    run_engine(BuyOnLastBar{ /*last_index=*/2, Quantity{ 1.0 } }, {
        flat(1000, 100.0),
        flat(2000, 110.0),
        flat(3000, 120.0),
    }, broker);

    EXPECT_TRUE(broker.trades().empty());
    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0);
}

TEST(Scenario, TwoBuysSameBar_BothFillFIFO_WhenCashSufficient)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    run_engine(TwoBuysSameBar{ Quantity{ 1.0 }, Quantity{ 2.0 } }, {
        flat(1000, 100.0),
        flat(2000, 110.0),   // both fill @ 110
        flat(3000, 120.0),
    }, broker);

    ASSERT_EQ(broker.trades().size(), 2u);
    EXPECT_LT(broker.trades()[0].order_id, broker.trades()[1].order_id);
    EXPECT_DOUBLE_EQ(broker.trades()[0].quantity, 1.0);
    EXPECT_DOUBLE_EQ(broker.trades()[1].quantity, 2.0);
    EXPECT_DOUBLE_EQ(broker.trades()[0].price, 110.0);
    EXPECT_DOUBLE_EQ(broker.trades()[1].price, 110.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 110.0 - 220.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash() + 3 * 120.0);
}

TEST(Scenario, TwoBuysSameBar_SecondQueuesWhenCashRunsOut)
{
    BacktestBroker broker{ Balance{ 150.0 } };
    // Both buys placed on tick 0. Next bar opens at 100.
    // First buy: 1 unit * 100 = 100 ≤ 150 → fills. cash=50.
    // Second buy: 2 units * 100 = 200 > 50 → stays queued.
    // Later bar opens at 20: second buy now affordable: 2 * 20 = 40 → fills.
    run_engine(TwoBuysSameBar{ Quantity{ 1.0 }, Quantity{ 2.0 } }, {
        flat(1000, 100.0),
        flat(2000, 100.0),   // first fills @ 100, second stays queued
        flat(3000, 100.0),   // second still can't fill (200 > 50)
        flat(4000, 20.0),    // second fills @ 20
    }, broker);

    ASSERT_EQ(broker.trades().size(), 2u);
    EXPECT_DOUBLE_EQ(broker.trades()[0].price, 100.0);
    EXPECT_EQ(broker.trades()[0].timestamp, Timestamp::from_millis(2000));
    EXPECT_DOUBLE_EQ(broker.trades()[1].price, 20.0);
    EXPECT_EQ(broker.trades()[1].timestamp, Timestamp::from_millis(4000));

    EXPECT_DOUBLE_EQ(broker.cash(), 150.0 - 100.0 - 40.0);
}

TEST(Scenario, LifecycleHooks_StartOnceTickNStopOnce)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    int starts = 0, ticks = 0, stops = 0;
    LifecycleSpy spy{ &starts, &ticks, &stops };

    StubFeed feed;
    feed.bars = {
        flat(1000, 100.0),
        flat(2000, 110.0),
        flat(3000, 120.0),
    };
    BrokerSpy bspy{ &broker };
    CoutMute mute;
    Engine engine{ std::move(spy), std::move(feed), std::move(bspy) };
    engine.run();

    EXPECT_EQ(starts, 1);
    EXPECT_EQ(ticks, 3);
    EXPECT_EQ(stops, 1);
}

TEST(Scenario, IdenticalRunsProduceIdenticalTrades)
{
    BacktestBroker b1{ Balance{ 1'000.0 } };
    BacktestBroker b2{ Balance{ 1'000.0 } };

    const std::vector<KLine> bars = {
        flat(1000, 100.0),
        flat(2000, 110.0),
        flat(3000, 120.0),
        flat(4000, 125.0),
    };
    run_engine(BuyThenSell{ Quantity{ 1.0 } }, bars, b1);
    run_engine(BuyThenSell{ Quantity{ 1.0 } }, bars, b2);

    ASSERT_EQ(b1.trades().size(), b2.trades().size());
    for (std::size_t i = 0; i < b1.trades().size(); ++i) {
        EXPECT_EQ(b1.trades()[i], b2.trades()[i]);
    }
    EXPECT_DOUBLE_EQ(b1.cash(), b2.cash());
    EXPECT_DOUBLE_EQ(b1.equity(), b2.equity());
}

// The engine keeps the run history; the external reporter derives metrics from
// it. This exercises the whole path: scripted run -> engine accessors ->
// compute_metrics, including the order log (which only a real run can populate,
// since Order's constructor is private to Context).
TEST(Scenario, Report_DerivesMetricsFromEngineData)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    BrokerSpy spy{ &broker };

    StubFeed feed;
    feed.bars = {
        flat(1000, 100.0),   // tick 0: buy placed, no fill yet
        flat(2000, 110.0),   // buy fills @110; tick 1: sell placed
        flat(3000, 120.0),   // sell fills @120
    };

    Engine engine{ BuyThenSell{ Quantity{ 1.0 } }, std::move(feed), std::move(spy) };
    engine.run();

    const auto metrics = stonks::app::compute_metrics(stonks::app::ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = engine.bars_processed(),
        .trades = engine.trades(),
        .orders = engine.orders(),
        .equity_curve = engine.equity_curve(),
        .ending_cash = engine.cash(),
        .ending_equity = engine.equity(),
        .elapsed = std::chrono::milliseconds{ 5 },
    });

    EXPECT_EQ(metrics.bars_processed, 3u);
    EXPECT_EQ(metrics.orders_placed, 2u);          // buy + sell, both recorded
    EXPECT_EQ(metrics.trade_count, 2u);            // both fill
    EXPECT_DOUBLE_EQ(metrics.notional, 110.0 + 120.0);
    EXPECT_DOUBLE_EQ(metrics.ending_cash, 1'000.0 - 110.0 + 120.0);
    ASSERT_TRUE(metrics.return_pct.has_value());
    EXPECT_DOUBLE_EQ(*metrics.return_pct, 1.0);    // (1010 - 1000) / 1000
    EXPECT_EQ(metrics.closed_trades, 1u);          // buy@110 -> sell@120 round trip
    EXPECT_EQ(metrics.winning_trades, 1u);         // closed +10
    ASSERT_TRUE(metrics.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(*metrics.win_rate_pct, 100.0);
    ASSERT_TRUE(metrics.first_ts.has_value());
    EXPECT_EQ(*metrics.first_ts, Timestamp::from_millis(1000));
    EXPECT_EQ(*metrics.last_ts, Timestamp::from_millis(3000));

    std::ostringstream os;
    stonks::app::print_report(os, metrics);
    const std::string report = os.str();
    EXPECT_NE(report.find("=== Backtest report ==="), std::string::npos);
    EXPECT_NE(report.find("Orders placed:"), std::string::npos);
    EXPECT_NE(report.find("Win rate:        100.00 % (1/1)"), std::string::npos);
    EXPECT_NE(report.find("Elapsed:"), std::string::npos);
    EXPECT_NE(report.find("per bar:"), std::string::npos);
}

TEST(Scenario, NoLookahead_EveryFillIsStrictlyAfterItsPlacement_MultiSymbol)
{
    // The end-to-end no-lookahead invariant: across a multi-bar, multi-symbol
    // run, no trade may fill at or before the timestamp its order was placed on.
    // Ample cash so nothing lingers on funds — we are isolating the time gate.
    BacktestBroker broker{ Balance{ 1'000'000.0 } };
    BrokerSpy spy{ &broker };

    StubFeed feed;
    feed.bars = {
        bar(1000, Symbol{ "A" }, 10.0, 11.0, 9.0, 10.0),
        bar(1000, Symbol{ "B" }, 20.0, 21.0, 19.0, 20.0),
        bar(2000, Symbol{ "A" }, 12.0, 13.0, 11.0, 12.0),
        bar(2000, Symbol{ "B" }, 22.0, 23.0, 21.0, 22.0),
        bar(3000, Symbol{ "A" }, 14.0, 15.0, 13.0, 14.0),
        bar(3000, Symbol{ "B" }, 24.0, 25.0, 23.0, 24.0),
    };

    CoutMute mute;
    Engine engine{ BuyEachPrinterEveryTick{}, std::move(feed), std::move(spy) };
    engine.run();

    ASSERT_FALSE(broker.trades().empty());

    std::unordered_map<OrderID, Timestamp> placed_at;
    for (const auto& o : broker.orders()) { placed_at[o.id] = o.timestamp; }

    for (const auto& t : broker.trades()) {
        ASSERT_TRUE(placed_at.contains(t.order_id));
        EXPECT_GT(t.timestamp, placed_at[t.order_id])
            << "order " << t.order_id << " filled at or before the bar it was placed on";
    }

    // Sanity: an order placed at ts=1000 fills at the next bar's open (ts=2000),
    // never the same bar.
    const auto& first = broker.trades().front();
    EXPECT_EQ(first.timestamp, Timestamp::from_millis(2000));
    EXPECT_DOUBLE_EQ(first.price, 12.0);   // A's ts=2000 open
}

} // namespace stonks::broker
