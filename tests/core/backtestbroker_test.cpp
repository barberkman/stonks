#include <gtest/gtest.h>

#include "stonks/broker/backtestbroker.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"

#include "test_stubs.h"

namespace stonks::broker {

using namespace stonks::core;

namespace {

KLine bar(std::int64_t ms, Price open, Price high, Price low, Price close)
{
    return KLine{ Timestamp::from_millis(ms), Symbol{ "X" }, open, high, low, close, Volume{ 1.0 } };
}

template <class MakeOrder>
Order order_at(std::int64_t ms, MakeOrder&& make)
{
    test::StubBroker stub;
    test::StubFeed feed;
    Clock clock;
    Context<test::StubBroker, test::StubFeed> ctx{ stub, feed, clock };
    clock.set(Timestamp::from_millis(ms));
    return make(ctx);
}

} // namespace

TEST(BacktestBroker, ConstructorSetsCashAndStartsEmpty)
{
    BacktestBroker broker{ Balance{ 50'000.0 } };
    EXPECT_EQ(broker.cash(), Balance{ 50'000.0 });
    EXPECT_EQ(broker.equity(), Balance{ 50'000.0 });
}

TEST(BacktestBroker, PlaceOrderEnqueuesAndDoesNotChangeCash)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto o = order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
        });
    });
    EXPECT_TRUE(broker.place_order(o));
    EXPECT_EQ(broker.cash(), Balance{ 10'000.0 });
}

TEST(BacktestBroker, MarketBuyFillsAtNextBarOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto buy = order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 2.0 },
        });
    });
    broker.place_order(buy);

    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 220.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 112.0);
}

TEST(BacktestBroker, MarketSellFillsAtNextBarOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 2.0 },
        });
    }));
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));   // fills buy @ 110

    broker.place_order(order_at(2000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Sell, .quantity = Quantity{ 2.0 },
        });
    }));
    broker.on_tick(bar(3000, 120.0, 125.0, 115.0, 122.0));   // fills sell @ 120

    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 220.0 + 240.0);
}

TEST(BacktestBroker, LimitBuyFillsWhenLowReachesLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_limit_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
            .quantity = Quantity{ 1.0 }, .price = Price{ 95.0 },
        });
    }));
    // bar low=90 ≤ limit=95; fill at min(95, 100) = 95
    broker.on_tick(bar(2000, 100.0, 105.0, 90.0, 100.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 95.0);
}

TEST(BacktestBroker, LimitBuyGapsThroughFillsAtOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_limit_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
            .quantity = Quantity{ 1.0 }, .price = Price{ 90.0 },
        });
    }));
    // open=80 below limit=90; fill at min(90, 80) = 80
    broker.on_tick(bar(2000, 80.0, 85.0, 75.0, 82.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 80.0);
}

TEST(BacktestBroker, LimitBuyStaysOpenWhenLowAboveLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_limit_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
            .quantity = Quantity{ 1.0 }, .price = Price{ 95.0 },
        });
    }));
    // low=105 > limit=95; no fill
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);
}

TEST(BacktestBroker, LimitSellFillsWhenHighReachesLimit)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    // high=130 ≥ limit=125; fill at max(125, 100) = 125
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_limit_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Sell,
            .quantity = Quantity{ 1.0 }, .price = Price{ 125.0 },
        });
    }));
    broker.on_tick(bar(2000, 100.0, 130.0, 95.0, 120.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 + 125.0);
}

TEST(BacktestBroker, OrderTimestampedAtBarDoesNotFillAgainstThatBar)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    // Order placed at ts=2000, same as the bar's timestamp.
    broker.place_order(order_at(2000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
        });
    }));
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);

    // Next bar at ts=3000 — now order.timestamp < bar.timestamp; should fill.
    broker.on_tick(bar(3000, 120.0, 125.0, 115.0, 122.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 120.0);
}

TEST(BacktestBroker, InsufficientCashBuyStaysQueuedUntilAffordable)
{
    BacktestBroker broker{ Balance{ 50.0 } };
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
        });
    }));
    broker.on_tick(bar(2000, 100.0, 100.0, 100.0, 100.0));   // open=100, too expensive
    EXPECT_DOUBLE_EQ(broker.cash(), 50.0);

    broker.on_tick(bar(3000, 40.0, 40.0, 40.0, 40.0));       // open=40, affordable
    EXPECT_DOUBLE_EQ(broker.cash(), 10.0);
}

TEST(BacktestBroker, TradesStartEmpty)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    EXPECT_TRUE(broker.trades().empty());
}

TEST(BacktestBroker, SuccessfulFillAppendsOneTrade)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto order = order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 2.0 },
        });
    });
    broker.place_order(order);
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));

    ASSERT_EQ(broker.trades().size(), 1u);
    const auto& t = broker.trades().front();
    EXPECT_EQ(t.id, TradeID{ 1 });
    EXPECT_EQ(t.order_id, order.id);
    EXPECT_EQ(t.timestamp, Timestamp::from_millis(2000));
    EXPECT_EQ(t.symbol, "X");
    EXPECT_EQ(t.side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(t.quantity, 2.0);
    EXPECT_DOUBLE_EQ(t.price, 110.0);
}

TEST(BacktestBroker, TradeIdsStrictlyIncrease)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
        });
    }));
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));

    broker.place_order(order_at(2000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
        });
    }));
    broker.on_tick(bar(3000, 115.0, 120.0, 110.0, 118.0));

    ASSERT_EQ(broker.trades().size(), 2u);
    EXPECT_EQ(broker.trades()[0].id, TradeID{ 1 });
    EXPECT_EQ(broker.trades()[1].id, TradeID{ 2 });
}

TEST(BacktestBroker, SkippedFillsDoNotAdvanceTradeId)
{
    BacktestBroker broker{ Balance{ 50.0 } };   // not enough for first attempt

    // Place a buy that won't fill on the next bar (cash too low).
    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
        });
    }));
    broker.on_tick(bar(2000, 100.0, 100.0, 100.0, 100.0));   // skipped, cash too low
    EXPECT_TRUE(broker.trades().empty());

    // Place a limit buy that won't fill (limit not crossed).
    broker.place_order(order_at(2000, [](auto& ctx) {
        return ctx.make_limit_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
            .quantity = Quantity{ 1.0 }, .price = Price{ 30.0 },
        });
    }));
    broker.on_tick(bar(3000, 100.0, 100.0, 100.0, 100.0));   // skipped, limit not crossed
    EXPECT_TRUE(broker.trades().empty());

    // Finally fill the queued market buy on a bar where price is affordable.
    broker.on_tick(bar(4000, 40.0, 40.0, 40.0, 40.0));
    ASSERT_EQ(broker.trades().size(), 1u);
    EXPECT_EQ(broker.trades().front().id, TradeID{ 1 });   // first trade id, counter not advanced by skips
}

TEST(BacktestBroker, EquityTracksLastClosePerSymbol)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(bar(1000, 100.0, 100.0, 100.0, 100.0));   // last close = 100
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0);

    broker.place_order(order_at(1000, [](auto& ctx) {
        return ctx.make_market_order({
            .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 2.0 },
        });
    }));
    broker.on_tick(bar(2000, 110.0, 110.0, 110.0, 110.0));   // fill 2@110; close=110
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 110.0);

    broker.on_tick(bar(3000, 130.0, 130.0, 130.0, 130.0));   // close=130
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 130.0);
}

} // namespace stonks::broker
