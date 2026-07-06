// Fee mechanics: bps-of-notional (maker vs taker by how the fill happened)
// plus a flat per-fill amount, charged where cash moves, recorded on the Trade.
// Zero-fee defaults leave every other suite's cash arithmetic untouched.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "stonks/broker/backtestbroker.h"
#include "stonks/core/types.h"

namespace stonks::broker {

using namespace stonks::core;

namespace {

KLine bar(std::int64_t ms, Price o, Price h, Price l, Price c)
{
    return KLine{ Timestamp::from_millis(ms), Symbol{ "X" }, o, h, l, c, Volume{ 1.0 } };
}
KLine flat(std::int64_t ms, Price p) { return bar(ms, p, p, p, p); }

std::vector<Trade> sorted_trades(const BacktestBroker& b)
{
    std::vector<Trade> v;
    for (const auto& [id, t] : b.trades()) { v.push_back(t); }
    std::ranges::sort(v, {}, &Trade::id);
    return v;
}

constexpr BrokerConfig FEES{ .maker_fee_bps = 2.0, .taker_fee_bps = 5.0 };

} // namespace

TEST(BrokerFees, MarketOpenPaysTakerOnNotional)
{
    BacktestBroker broker{ Balance{ 10'000.0 }, FEES };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    broker.on_tick(flat(1000, 100.0));            // notional 1000 -> taker fee 0.50

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_DOUBLE_EQ(trades[0].fee, 0.5);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 1'000.0 - 0.5);
}

TEST(BrokerFees, RestingLimitPaysMakerCrossingLimitPaysTaker)
{
    // Resting: a buy limit at 90 on a bar opening above it fills AT the limit
    // (price came to the order) -> maker. Crossing: the same limit on a bar
    // opening below it fills at the open (it crossed on arrival) -> taker.
    BacktestBroker resting{ Balance{ 10'000.0 }, FEES };
    resting.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .price = 90.0 });
    resting.on_tick(bar(1000, 95.0, 96.0, 85.0, 92.0));   // fills @90
    ASSERT_EQ(resting.trades().size(), 1u);
    EXPECT_DOUBLE_EQ(sorted_trades(resting)[0].fee, 900.0 * 2.0 / 10'000.0);   // 0.18 maker

    BacktestBroker crossing{ Balance{ 10'000.0 }, FEES };
    crossing.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .price = 90.0 });
    crossing.on_tick(bar(1000, 85.0, 88.0, 84.0, 86.0));  // fills @85 (the open)
    ASSERT_EQ(crossing.trades().size(), 1u);
    EXPECT_DOUBLE_EQ(sorted_trades(crossing)[0].fee, 850.0 * 5.0 / 10'000.0);  // 0.425 taker
}

TEST(BrokerFees, StopFillPaysTaker)
{
    BacktestBroker broker{ Balance{ 10'000.0 }, FEES };
    broker.on_tick(flat(1000, 100.0));
    broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .price = 105.0 });
    broker.on_tick(bar(2000, 104.0, 106.0, 103.0, 105.5));   // fills @105
    ASSERT_EQ(broker.trades().size(), 1u);
    EXPECT_DOUBLE_EQ(sorted_trades(broker)[0].fee, 1'050.0 * 5.0 / 10'000.0);
}

TEST(BrokerFees, FlatFeeAddsOnTopAndAppliesPerFill)
{
    BacktestBroker broker{ Balance{ 10'000.0 },
                           BrokerConfig{ .taker_fee_bps = 5.0, .fee_per_fill = 1.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    broker.on_tick(flat(1000, 100.0));            // fee 0.50 + 1.00
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0 });
    broker.on_tick(flat(2000, 100.0));            // close: same notional, second flat fee

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[0].fee, 1.5);
    EXPECT_DOUBLE_EQ(trades[1].fee, 1.5);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 3.0);   // flat round trip: only the fees
}

TEST(BrokerFees, CloseFeeChargedOnFilledNotionalAfterClamp)
{
    BacktestBroker broker{ Balance{ 10'000.0 }, FEES };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(flat(1000, 100.0));
    // Oversized close clamps to the held 2.0; the fee is on the clamped notional.
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 5.0 });
    broker.on_tick(flat(2000, 110.0));
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].quantity, 2.0);
    EXPECT_DOUBLE_EQ(trades[1].fee, 2.0 * 110.0 * 5.0 / 10'000.0);
}

TEST(BrokerFees, AffordabilityGateIncludesTheFee)
{
    // Cash covers exactly the margin but not margin + fee: rejected.
    BacktestBroker broker{ Balance{ 100.0 }, FEES };
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(1000, 100.0));            // margin 100 + taker 0.05 > 100
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Rejected);
    EXPECT_DOUBLE_EQ(broker.cash(), 100.0);
    EXPECT_TRUE(broker.trades().empty());
}

TEST(BrokerFees, LiquidationPaysTakerFee)
{
    BacktestBroker broker{ Balance{ 1'000.0 }, FEES };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    broker.on_tick(flat(2000, 100.0));            // margin 100, entry fee 0.5, B = 90
    broker.on_tick(bar(3000, 95.0, 96.0, 89.0, 94.0));   // low 89 -> liquidated @90
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    ASSERT_TRUE(trades[1].liquidation);
    EXPECT_DOUBLE_EQ(trades[1].fee, 900.0 * 5.0 / 10'000.0);   // 0.45 on the forced notional
    // cash: 1000 - 100 - 0.5 (open) + 100 - 100 (margin back, pnl) - 0.45 = 899.05
    EXPECT_DOUBLE_EQ(broker.cash(), 899.05);
}

TEST(BrokerFees, ZeroFeeDefaultsChargeNothing)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };   // default config
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    broker.on_tick(flat(1000, 100.0));
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0 });
    broker.on_tick(flat(2000, 100.0));
    for (const auto& t : sorted_trades(broker)) { EXPECT_DOUBLE_EQ(t.fee, 0.0); }
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);
}

} // namespace stonks::broker
