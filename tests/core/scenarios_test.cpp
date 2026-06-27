// End-to-end scenarios: scripted strategies driven through the full Engine
// against a real BacktestBroker. A small BrokerSpy holds a pointer to an
// externally-owned broker so we can inspect its trades/balances after
// engine.run() (Engine moves the broker in by value).

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
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

// Forwards the new Broker concept to an externally-owned BacktestBroker, so a
// test can inspect that broker after Engine moves the spy in by value.
struct BrokerSpy
{
    BacktestBroker* impl{};

    Balance cash() const { return impl->cash(); }
    Balance equity() const { return impl->equity(); }
    const std::unordered_map<TradeID, Trade>& trades() const { return impl->trades(); }
    const std::unordered_map<OrderID, Order>& orders() const { return impl->orders(); }
    OrderID place_order(const MarketOrderParams& p, std::optional<OrderID> parent = std::nullopt) { return impl->place_order(p, parent); }
    OrderID place_order(const LimitOrderParams& p, std::optional<OrderID> parent = std::nullopt) { return impl->place_order(p, parent); }
    void on_tick(const KLine& bar) { impl->on_tick(bar); }
};
static_assert(Broker<BrokerSpy>);

// trades()/orders() are unordered_map; flatten + sort by id for ordered asserts.
std::vector<Trade> sorted_trades(const BacktestBroker& b)
{
    std::vector<Trade> v;
    v.reserve(b.trades().size());
    for (const auto& [id, t] : b.trades()) { v.push_back(t); }
    std::ranges::sort(v, {}, &Trade::id);
    return v;
}

std::vector<Order> sorted_orders(const BacktestBroker& b)
{
    std::vector<Order> v;
    v.reserve(b.orders().size());
    for (const auto& [id, o] : b.orders()) { v.push_back(o); }
    std::ranges::sort(v, {}, &Order::id);
    return v;
}

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
    Engine engine{ std::move(strategy), std::move(feed), std::move(spy), ProgressOutput::Silent };
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
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = qty });
        } else if (tick == 1) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Sell, .quantity = qty });
        }
        ++tick;
    }
};

// Buy once on the first tick, hold forever.
struct BuyAndHold
{
    Quantity qty;
    Symbol symbol{ "X" };
    bool placed{ false };

    void on_tick(auto& ctx)
    {
        if (placed) { return; }
        ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = qty });
        placed = true;
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
        ctx.place_order(LimitOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = qty, .price = limit });
        placed = true;
    }
};

// Buy symbol A on the first tick; buy symbol B on the next tick.
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
            ctx.place_order(MarketOrderParams{ .symbol = a, .side = OrderSide::Buy, .quantity = qty_a });
            bought_a = true;
            return;
        }
        if (!bought_b) {
            ctx.place_order(MarketOrderParams{ .symbol = b, .side = OrderSide::Buy, .quantity = qty_b });
            bought_b = true;
        }
    }
};

// Sell on the first tick without holding anything (short), then cover later.
struct ShortOnceThenCover
{
    Quantity qty;
    int cover_at;
    Symbol symbol{ "X" };
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick == 0) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Sell, .quantity = qty });
        } else if (tick == cover_at) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = qty });
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
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = qty });
        }
        ++tick;
    }
};

// Buy on tick 0, then attempt to buy the SAME symbol/side again on tick 1 while
// already holding — the broker must reject the second (one position per symbol).
struct BuyTwiceSameSide
{
    Quantity q0;
    Quantity q1;
    Symbol symbol{ "X" };
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick == 0) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = q0 });
        } else if (tick == 1) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = q1 });
        }
        ++tick;
    }
};

// Buy `held` on tick 0, then sell `oversell` (> held) on tick 2 — the close must
// clamp to the held quantity and never flip to a short.
struct BuyThenOversell
{
    Quantity held;
    Quantity oversell;
    Symbol symbol{ "X" };
    int tick{ 0 };

    void on_tick(auto& ctx)
    {
        if (tick == 0) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Buy, .quantity = held });
        } else if (tick == 2) {
            ctx.place_order(MarketOrderParams{ .symbol = symbol, .side = OrderSide::Sell, .quantity = oversell });
        }
        ++tick;
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

// Buys one share of every symbol that printed, on every tick. Asserts the
// no-lookahead property over a multi-bar, multi-symbol run.
struct BuyEachPrinterEveryTick
{
    void on_tick(auto& ctx)
    {
        for (const auto& s : ctx.history(1).series) {
            ctx.place_order(MarketOrderParams{ .symbol = Symbol{ s.symbol }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 } });
        }
    }
};

} // namespace

// --- Tests -------------------------------------------------------------------

TEST(Scenario, RoundTrip_BuyThenSell_TwoTradesAtNextBarOpens)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    run_engine(BuyThenSell{ Quantity{ 2.0 } }, {
        flat(1000, 100.0),                        // tick 0: places buy (stamped 1000)
        bar (2000, 110.0, 115.0, 105.0, 112.0),   // tick 1: buy fills @110; places sell
        bar (3000, 120.0, 125.0, 118.0, 122.0),   // tick 2: sell fills @120
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);

    EXPECT_EQ(trades[0].id, TradeID{ 1 });
    EXPECT_EQ(trades[0].timestamp, Timestamp::from_millis(2000));
    EXPECT_EQ(trades[0].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(trades[0].quantity, 2.0);
    EXPECT_DOUBLE_EQ(trades[0].price, 110.0);

    EXPECT_EQ(trades[1].id, TradeID{ 2 });
    EXPECT_GT(trades[1].order_id, trades[0].order_id);
    EXPECT_EQ(trades[1].timestamp, Timestamp::from_millis(3000));
    EXPECT_EQ(trades[1].side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(trades[1].price, 120.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 2 * 110.0 + 2 * 120.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());   // flat position
}

TEST(Scenario, BuyAndHold_OneTrade_EquityMarksToLastClose)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    run_engine(BuyAndHold{ Quantity{ 3.0 } }, {
        flat(1000, 100.0),
        flat(2000, 110.0),   // buy fills @110
        flat(3000, 130.0),
        flat(4000, 125.0),   // final mark
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].timestamp, Timestamp::from_millis(2000));
    EXPECT_DOUBLE_EQ(trades[0].price, 110.0);
    EXPECT_DOUBLE_EQ(trades[0].quantity, 3.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 3 * 110.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash() + 3 * 125.0);
}

TEST(Scenario, LimitBuy_StaysOpenThenFillsOnLaterBar)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // Limit @95. Bar 1: low=105, no fill. Bar 2: low=90 <= 95, fills at min(95, open=100)=95.
    run_engine(LimitBuyOnce{ Quantity{ 1.0 }, Price{ 95.0 } }, {
        flat(1000, 100.0),
        bar (2000, 110.0, 115.0, 105.0, 112.0),   // low=105 > 95 -> stays open
        bar (3000, 100.0, 102.0, 90.0,  101.0),   // low=90 <= 95 -> fill @95
        flat(4000, 101.0),
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].timestamp, Timestamp::from_millis(3000));
    EXPECT_DOUBLE_EQ(trades[0].price, 95.0);
}

TEST(Scenario, LimitBuy_GapDown_FillsAtOpenBelowLimit)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // Limit @90; bar opens at 80 (gap-down): fill at min(90, 80)=80.
    run_engine(LimitBuyOnce{ Quantity{ 1.0 }, Price{ 90.0 } }, {
        flat(1000, 100.0),
        bar (2000, 80.0, 85.0, 75.0, 82.0),
        flat(3000, 82.0),
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_DOUBLE_EQ(trades[0].price, 80.0);
}

TEST(Scenario, MultiSymbol_PortfolioEquityCombinesLastClosePerSymbol)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    run_engine(TwoSymbolBuyer{ .qty_a = Quantity{ 1.0 }, .qty_b = Quantity{ 2.0 } }, {
        flat(1000, Symbol{ "A" }, 100.0),   // tick 0: buys A (stamped 1000)
        flat(2000, Symbol{ "B" }, 50.0),    // tick 1: A order waits (symbol mismatch); buys B
        flat(3000, Symbol{ "A" }, 120.0),   // tick 2: A buy fills @120
        flat(4000, Symbol{ "B" }, 60.0),    // tick 3: B buy fills @60
        flat(5000, Symbol{ "A" }, 130.0),   // A last=130
        flat(6000, Symbol{ "B" }, 55.0),    // B last=55
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].symbol, "A");
    EXPECT_EQ(trades[0].timestamp, Timestamp::from_millis(3000));
    EXPECT_DOUBLE_EQ(trades[0].price, 120.0);
    EXPECT_EQ(trades[1].symbol, "B");
    EXPECT_EQ(trades[1].timestamp, Timestamp::from_millis(4000));
    EXPECT_DOUBLE_EQ(trades[1].price, 60.0);

    const double cash = 10'000.0 - 1 * 120.0 - 2 * 60.0;
    EXPECT_DOUBLE_EQ(broker.cash(), cash);
    EXPECT_DOUBLE_EQ(broker.equity(), cash + 1 * 130.0 + 2 * 55.0);   // A=130, B=55
}

TEST(Scenario, ShortSellThenCover_NetsCorrectly)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // Sell 1 (no position) -> short fills next bar @110; cover -> fills @90. P&L = +20.
    run_engine(ShortOnceThenCover{ .qty = Quantity{ 1.0 }, .cover_at = 2 }, {
        flat(1000, 100.0),
        flat(2000, 110.0),   // short fills @110
        flat(3000, 105.0),
        flat(4000, 90.0),    // cover fills @90
        flat(5000, 95.0),
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(trades[0].price, 110.0);
    EXPECT_EQ(trades[1].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(trades[1].price, 90.0);

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 + 20.0);   // collateral returned + 20 profit
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());  // flat
}

TEST(Scenario, OrderPlacedOnLastBar_NeverFills)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    run_engine(BuyOnLastBar{ .last_index = 2, .qty = Quantity{ 1.0 } }, {
        flat(1000, 100.0),
        flat(2000, 110.0),
        flat(3000, 120.0),
    }, broker);

    EXPECT_TRUE(broker.trades().empty());
    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0);
}

// NEW (replaces the old laddering test): one position per symbol — a same-side
// order against an open position is rejected, not stacked.
TEST(Scenario, SameSideAddIsRejected_NoStacking)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    run_engine(BuyTwiceSameSide{ Quantity{ 1.0 }, Quantity{ 5.0 } }, {
        flat(1000, 100.0),   // tick 0: buy 1 (stamped 1000)
        flat(2000, 110.0),   // tick 1: buy 1 fills @110; second buy 5 placed (stamped 2000)
        flat(3000, 120.0),   // tick 2: second buy hits existing long -> Rejected
    }, broker);

    const auto orders = sorted_orders(broker);
    ASSERT_EQ(orders.size(), 2u);
    EXPECT_EQ(orders[0].status, OrderStatus::Filled);     // first buy filled
    EXPECT_EQ(orders[1].status, OrderStatus::Rejected);   // same-side add rejected

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);                          // only the first opened
    EXPECT_DOUBLE_EQ(trades[0].quantity, 1.0);
    // Cash only reflects the single 1-unit open (the 5-unit add never settled).
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 1 * 110.0);
}

// NEW: an oversized close clamps to the held quantity and never flips to a short.
TEST(Scenario, OversizedClose_ClampsToHeld_NoFlip)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    run_engine(BuyThenOversell{ .held = Quantity{ 2.0 }, .oversell = Quantity{ 5.0 } }, {
        flat(1000, 100.0),   // tick 0: buy 2 (stamped 1000)
        flat(2000, 110.0),   // tick 1: buy fills @110
        flat(3000, 120.0),   // tick 2: sell 5 placed (stamped 3000)
        flat(4000, 130.0),   // tick 3: sell fills, clamped to 2 @130
        flat(5000, 140.0),   // mark — must be flat, not short
    }, broker);

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].quantity, 2.0);            // closed only the held 2, not 5
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 2 * 110.0 + 2 * 130.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());     // flat: no residual short despite overselling
}

TEST(Scenario, LifecycleHooks_StartOnceTickNStopOnce)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    int starts = 0, ticks = 0, stops = 0;

    StubFeed feed;
    feed.bars = { flat(1000, 100.0), flat(2000, 110.0), flat(3000, 120.0) };
    BrokerSpy spy{ &broker };
    Engine engine{ LifecycleSpy{ &starts, &ticks, &stops }, std::move(feed), std::move(spy), ProgressOutput::Silent };
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
        flat(1000, 100.0), flat(2000, 110.0), flat(3000, 120.0), flat(4000, 125.0),
    };
    run_engine(BuyThenSell{ Quantity{ 1.0 } }, bars, b1);
    run_engine(BuyThenSell{ Quantity{ 1.0 } }, bars, b2);

    EXPECT_EQ(sorted_trades(b1), sorted_trades(b2));
    EXPECT_DOUBLE_EQ(b1.cash(), b2.cash());
    EXPECT_DOUBLE_EQ(b1.equity(), b2.equity());
}

// Whole path: scripted run -> engine accessors -> compute_metrics -> print_report.
TEST(Scenario, Report_DerivesMetricsFromEngineData)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    BrokerSpy spy{ &broker };

    StubFeed feed;
    feed.bars = {
        flat(1000, 100.0),   // buy placed
        flat(2000, 110.0),   // buy fills @110; sell placed
        flat(3000, 120.0),   // sell fills @120
    };

    Engine engine{ BuyThenSell{ Quantity{ 1.0 } }, std::move(feed), std::move(spy), ProgressOutput::Silent };
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
    EXPECT_EQ(metrics.orders_placed, 2u);
    EXPECT_EQ(metrics.trade_count, 2u);
    EXPECT_DOUBLE_EQ(metrics.notional, 110.0 + 120.0);
    EXPECT_DOUBLE_EQ(metrics.ending_cash, 1'000.0 - 110.0 + 120.0);
    ASSERT_TRUE(metrics.return_pct.has_value());
    EXPECT_DOUBLE_EQ(*metrics.return_pct, 1.0);
    EXPECT_EQ(metrics.closed_trades, 1u);
    EXPECT_EQ(metrics.winning_trades, 1u);
    ASSERT_TRUE(metrics.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(*metrics.win_rate_pct, 100.0);
}

TEST(Scenario, NoLookahead_EveryFillIsStrictlyAfterItsPlacement_MultiSymbol)
{
    BacktestBroker broker{ Balance{ 1'000'000.0 } };   // ample cash: isolate the time gate
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

    Engine engine{ BuyEachPrinterEveryTick{}, std::move(feed), std::move(spy), ProgressOutput::Silent };
    engine.run();

    ASSERT_FALSE(broker.trades().empty());

    std::unordered_map<OrderID, Timestamp> placed_at;
    for (const auto& [id, o] : broker.orders()) { placed_at[id] = o.timestamp; }

    for (const auto& [tid, t] : broker.trades()) {
        ASSERT_TRUE(placed_at.contains(t.order_id));
        EXPECT_GT(t.timestamp, placed_at[t.order_id])
            << "order " << t.order_id << " filled at or before the bar it was placed on";
    }

    // The earliest fill: an order placed at ts=1000 fills at the next bar (ts=2000) open.
    const auto trades = sorted_trades(broker);
    EXPECT_EQ(trades.front().timestamp, Timestamp::from_millis(2000));
    EXPECT_DOUBLE_EQ(trades.front().price, 12.0);   // A's ts=2000 open
}

} // namespace stonks::broker
