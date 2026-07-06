// Intrabar fill ordering and same-bar bracket eligibility. One bar can satisfy
// several orders at once; the broker settles it in rounds (a parent's fill arms
// its children within the same bar) and resolves multi-touch ties by order
// kind — market first, then stop/limit per BrokerConfig::fill_policy — with
// placement order only breaking ties within a kind.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "stonks/broker/backtestbroker.h"
#include "stonks/core/types.h"

namespace stonks::broker {

using namespace stonks::core;

namespace {

KLine bar(std::int64_t ms, const Symbol& sym, Price o, Price h, Price l, Price c)
{
    return KLine{ Timestamp::from_millis(ms), sym, o, h, l, c, Volume{ 1.0 } };
}
KLine bar(std::int64_t ms, Price o, Price h, Price l, Price c)
{
    return bar(ms, Symbol{ "X" }, o, h, l, c);
}
KLine flat(std::int64_t ms, Price p) { return bar(ms, p, p, p, p); }

std::vector<Trade> sorted_trades(const BacktestBroker& b)
{
    std::vector<Trade> v;
    for (const auto& [id, t] : b.trades()) { v.push_back(t); }
    std::ranges::sort(v, {}, &Trade::id);
    return v;
}

} // namespace

// --- Double-touch policy -------------------------------------------------------

TEST(IntrabarFillPolicy, ConservativeDefaultPicksStopOverLimitOnDoubleTouch)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };   // default config = Conservative
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    // TP placed BEFORE the SL: the policy, not placement order, must decide.
    const auto tp = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 110.0 }, entry);
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 }, entry);
    broker.on_tick(flat(1000, 100.0));                     // entry fills @100, legs armed

    broker.on_tick(bar(2000, 100.0, 112.0, 94.0, 105.0));  // touches both 110 and 95
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Cancelled);   // OCO loser
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].price, 95.0);               // exits at the stop
    EXPECT_DOUBLE_EQ(broker.cash(), 9'995.0);              // 10'000 - 100 + 95
}

TEST(IntrabarFillPolicy, OptimisticPolicyPicksLimitOverStopOnDoubleTouch)
{
    BacktestBroker broker{ Balance{ 10'000.0 }, BrokerConfig{ .fill_policy = IntrabarFillPolicy::Optimistic } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto tp = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 110.0 }, entry);
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 }, entry);
    broker.on_tick(flat(1000, 100.0));

    broker.on_tick(bar(2000, 100.0, 112.0, 94.0, 105.0));  // same double-touch bar
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].price, 110.0);              // exits at the target
    EXPECT_DOUBLE_EQ(broker.cash(), 10'010.0);             // 10'000 - 100 + 110
}

// --- Same-bar bracket eligibility ------------------------------------------------

TEST(IntrabarFillPolicy, ChildStopFillsOnTheSameBarAsItsParentEntry)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 }, entry);

    // One bar fills the entry at the open AND breaches the stop's trigger: the
    // armed child settles in the same bar's later round.
    broker.on_tick(bar(2000, 100.0, 101.0, 94.0, 96.0));
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].timestamp, trades[1].timestamp);   // same bar
    EXPECT_DOUBLE_EQ(trades[0].price, 100.0);
    EXPECT_DOUBLE_EQ(trades[1].price, 95.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 9'995.0);
}

TEST(IntrabarFillPolicy, StopLossOnTheEntryBarPreemptsSameBarLiquidation)
{
    // The counterpart to BacktestBrokerLeverage.PositionOpenedAndLiquidatedOnTheSameBar:
    // with a protective stop in the bracket, the violent entry bar exits at the
    // stop (loss = stop distance) instead of liquidating at the bankruptcy
    // price (loss = the whole margin).
    BacktestBroker broker{ Balance{ 1'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 95.0 }, entry);

    broker.on_tick(bar(2000, 100.0, 101.0, 88.0, 90.0));   // entry @100 (B=90); low 88 breaches both 95 and 90
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_FALSE(trades[1].liquidation);                   // the stop got there first
    EXPECT_DOUBLE_EQ(trades[1].price, 95.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 950.0);                // -50, not the full -100 margin
}

TEST(IntrabarFillPolicy, GrandchildFillsInTheSameBarThroughACascade)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    const auto tp1 = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 5.0, .price = 105.0 }, entry);
    const auto tp2 = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 5.0, .price = 110.0 }, tp1);

    // Entry fills at the open; tp1 arms and fills; that arms the grandchild
    // tp2, which also fills — three settlement rounds, one bar.
    broker.on_tick(bar(2000, 100.0, 111.0, 99.0, 108.0));
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(tp1).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(tp2).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 3u);
    EXPECT_DOUBLE_EQ(trades[1].price, 105.0);
    EXPECT_DOUBLE_EQ(trades[2].price, 110.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'075.0);             // -1000 + 525 + 550
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());      // flat
}

// --- Market priority ---------------------------------------------------------------

TEST(IntrabarFillPolicy, MarketOrderFillsBeforeArmedTriggersRegardlessOfPlacementOrder)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 }, entry);
    broker.on_tick(flat(1000, 100.0));                     // entry fills @100, SL armed

    // A manual close placed AFTER the stop still executes first (at the open),
    // and the flattened position OCO-cancels the stop.
    const auto manual = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    broker.on_tick(bar(2000, 94.0, 96.0, 93.0, 95.0));     // the stop's trigger is also breached
    EXPECT_EQ(broker.orders().at(manual).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].price, 94.0);               // the open, not the stop's 95
    EXPECT_DOUBLE_EQ(broker.cash(), 9'994.0);              // 10'000 - 100 + 94
}

} // namespace stonks::broker
