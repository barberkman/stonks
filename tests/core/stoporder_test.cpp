// Stop-market order semantics: dormant until the market touches the trigger,
// then fills at the trigger or worse — a gap through the trigger fills at the
// open, the same convention as the liquidation fill. Same recipe as
// backtestbroker_test.cpp: place an order, then on_tick a later bar.

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

// --- Trigger & fill price ------------------------------------------------------

TEST(BacktestBrokerStop, BuyStopStaysOpenUntilHighReachesTrigger)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    const auto id = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 105.0 });

    broker.on_tick(bar(2000, 100.0, 104.9, 99.0, 104.0));    // high never touches 105
    broker.on_tick(bar(3000, 104.0, 104.5, 103.0, 104.0));
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
    EXPECT_TRUE(broker.trades().empty());

    broker.on_tick(bar(4000, 104.0, 106.0, 103.5, 105.5));   // touches 105 -> fills
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_DOUBLE_EQ(trades[0].price, 105.0);                // open below the trigger -> fill at the trigger
}

TEST(BacktestBrokerStop, BuyStopGapThroughTriggerFillsAtOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 105.0 });

    broker.on_tick(bar(2000, 108.0, 110.0, 107.0, 109.0));   // opens above the trigger
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_DOUBLE_EQ(trades[0].price, 108.0);                // worse of (trigger, open) = the open
}

TEST(BacktestBrokerStop, SellStopStaysOpenUntilLowReachesTrigger)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    const auto id = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 });

    broker.on_tick(bar(2000, 100.0, 102.0, 95.1, 101.0));    // low never touches 95
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
    EXPECT_TRUE(broker.trades().empty());

    broker.on_tick(bar(3000, 98.0, 99.0, 94.0, 94.5));       // touches 95 -> fills, opens a short
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_DOUBLE_EQ(trades[0].price, 95.0);                 // open above the trigger -> fill at the trigger
}

TEST(BacktestBrokerStop, SellStopGapThroughTriggerFillsAtOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 });

    broker.on_tick(bar(2000, 90.0, 92.0, 89.0, 91.0));       // opens below the trigger
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_DOUBLE_EQ(trades[0].price, 90.0);                 // worse of (trigger, open) = the open
}

// --- Validation & lookahead ----------------------------------------------------

TEST(BacktestBrokerStop, RejectsNonPositiveTriggerPrice)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto zero = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 0.0 });
    const auto negative = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = -5.0 });
    EXPECT_EQ(broker.orders().at(zero).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(negative).status, OrderStatus::Rejected);

    broker.on_tick(flat(1000, 100.0));
    EXPECT_TRUE(broker.trades().empty());
}

TEST(BacktestBrokerStop, StopDoesNotFillOnItsOwnBarTimestamp)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));                       // stamps m_now = 1000
    const auto id = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 99.0 });

    broker.on_tick(bar(1000, 100.0, 101.0, 98.0, 100.0));    // same timestamp as the stamp
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);

    broker.on_tick(bar(2000, 100.0, 101.0, 98.0, 100.0));    // strictly later -> eligible
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
}

// --- Position mechanics ----------------------------------------------------------

TEST(BacktestBrokerStop, StopEntryPostsMarginAtOrderLeverage)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .price = 100.0, .leverage = 4.0 });

    broker.on_tick(bar(2000, 100.0, 101.0, 99.0, 100.0));    // fills at the 100 trigger
    ASSERT_EQ(broker.trades().size(), 1u);
    EXPECT_DOUBLE_EQ(broker.cash(), 750.0);                  // margin = 10 * 100 / 4 posted
}

TEST(BacktestBrokerStop, ProtectiveSellStopRestsWhileMarketHoldsAboveIt)
{
    // The stop-as-limit regression: a stop-loss below the market must NOT be
    // instantly marketable the way a sell *limit* below the market is.
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 }, entry);

    broker.on_tick(flat(2000, 100.0));                       // entry fills at 100; SL armed
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);

    broker.on_tick(bar(3000, 101.0, 103.0, 96.0, 102.0));    // low stays above 95 -> SL rests
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Open);
    EXPECT_EQ(broker.trades().size(), 1u);                   // just the entry fill

    broker.on_tick(bar(4000, 98.0, 99.0, 94.0, 94.5));       // breaches 95 -> exits at the stop
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].price, 95.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 9'995.0);                // 10'000 - 100 + 95
}

TEST(BacktestBrokerStop, StopChildStaysDormantUntilParentFills)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    const auto entry = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 110.0 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 85.0 }, entry);

    broker.on_tick(bar(2000, 100.0, 105.0, 84.0, 100.0));    // child's trigger breached, parent unfilled
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Open);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Open);
    EXPECT_TRUE(broker.trades().empty());

    broker.on_tick(bar(3000, 109.0, 111.0, 108.0, 110.5));   // parent fills at its 110 trigger
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Open);

    broker.on_tick(bar(4000, 100.0, 101.0, 84.0, 90.0));     // armed stop exits at 85
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[0].price, 110.0);
    EXPECT_DOUBLE_EQ(trades[1].price, 85.0);
}

} // namespace stonks::broker
